# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Owned-cell assembly and ordinary Hex8 Step distributed lifecycle gates."""

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

from agentfem import assembly, state
from agentfem.elements._uniform_hex_dolfinx import UniformHexResidual
from agentfem.kernel import dofs
from agentfem.time.explicit import central_difference


def problem(comm, counts=(4, 2, 2), stiffness=None):
    partitioner = None
    if counts == (1, 1, 1) and comm.size > 1:
        # Deliberate empty rank, independent of ParMETIS on an edgeless graph.
        from dolfinx import graph

        def partitioner(comm, parts, types, cells):
            count = (
                cells.num_nodes
                if hasattr(cells, "num_nodes")
                else sum(array.size // 8 for array in cells)
            )
            result = graph.adjacencylist(np.zeros((count, 1), dtype=np.int32))
            return getattr(result, "_cpp_object", result)

    domain = mesh.create_box(
        comm,
        [[0, 0, 0], [1, 0.2, 0.2]],
        counts,
        cell_type=mesh.CellType.hexahedron,
        partitioner=partitioner,
    )
    u = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
    c = (
        np.diag([100.0, 100.0, 100.0, 50.0, 50.0, 50.0])
        if stiffness is None
        else stiffness
    )
    operator = UniformHexResidual(
        u, c, density=2, hourglass_modulus=50, hourglass_scale=0.1
    )
    return domain, u, operator


@pytest.mark.parametrize("counts", [(4, 2, 2), (1, 1, 1)])
def test_distributed_hex_owned_affine_force_mass_energy(counts):
    domain, u, operator = problem(MPI.COMM_WORLD, counts)
    u.interpolate(
        lambda x: np.vstack((0.01 * x[0] + 0.03 * x[2], 0.02 * x[1], -0.01 * x[2]))
    )
    v = ufl.TestFunction(u.function_space)
    expression = ufl.inner(100 * ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
    reference = assembly.assemble_vector(fem.form(expression))
    actual = operator.assemble_vector()
    try:
        np.testing.assert_allclose(actual.array, reference.array, atol=1e-13)
        force = actual.array.reshape(-1, 3)
        xyz = u.function_space.tabulate_dof_coordinates()[: len(force)]
        np.testing.assert_allclose(domain.comm.allreduce(force.sum(0)), 0, atol=1e-13)
        np.testing.assert_allclose(
            domain.comm.allreduce(np.cross(xyz, force).sum(0)), 0, atol=1e-13
        )
    finally:
        actual.destroy()
        reference.destroy()
    mass = assembly.assemble_lumped_mass(u.function_space, density=2)
    np.testing.assert_allclose(operator.mass_diagonal, mass, atol=1e-15)
    assert domain.comm.allreduce(operator.mass_diagonal[::3].sum()) == pytest.approx(
        0.08
    )
    energy = fem.assemble_scalar(
        fem.form(50 * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(u))) * ufl.dx)
    )
    result = operator.energies()
    assert result["strain_energy"] == pytest.approx(domain.comm.allreduce(energy))
    assert result["hourglass_energy"] < 1e-25
    assert np.isfinite(operator.stable_dt())


def test_distributed_hex_wave_matches_serial_owned_values():
    solutions = []
    for comm in (MPI.COMM_WORLD, MPI.COMM_SELF):
        _, u, operator = problem(comm)
        x = u.function_space.tabulate_dof_coordinates()[:, 0]
        u.x.array[::3] = 1e-4 * np.cos(np.pi * x)
        history = state.second_order_state(u)
        initial = operator.assemble_vector()
        try:
            dofs.assign_owned(history.a, -initial.array * operator.inv_mass)
        finally:
            initial.destroy()
        integrator = central_difference(state=history, mass=operator)
        for step in range(10):
            integrator.step(1e-4, time=(step + 1) * 1e-4, residual_operator=operator)
        size = u.function_space.dofmap.index_map.size_local
        xyz = u.function_space.tabulate_dof_coordinates()[:size]
        solutions.append(
            (
                xyz,
                u.x.array[: 3 * size].reshape(-1, 3).copy(),
                operator.energies(),
                operator.stable_dt(),
            )
        )
    xyz, values, energy, dt = solutions[0]
    serial_xyz, serial_values, serial_energy, serial_dt = solutions[1]
    for point, value in zip(xyz, values):
        match = np.flatnonzero(np.linalg.norm(serial_xyz - point, axis=1) < 1e-12)
        assert len(match) == 1
        np.testing.assert_allclose(value, serial_values[match[0]], atol=1e-15)
    assert energy == pytest.approx(serial_energy, abs=1e-20)
    assert dt == pytest.approx(serial_dt)


def test_rank_local_invalid_material_fails_collectively():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("requires multiple ranks")
    c = np.eye(6)
    if MPI.COMM_WORLD.rank == 1:
        c[0, 0] = -1
    with pytest.raises(RuntimeError, match="rank 1"):
        problem(MPI.COMM_WORLD, stiffness=c)


def test_rank_local_nonfinite_field_does_not_hang_reverse_scatter():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("requires multiple ranks")
    _, u, operator = problem(MPI.COMM_WORLD)
    if MPI.COMM_WORLD.rank == 1:
        u.x.array[0] = np.nan
    with pytest.raises(RuntimeError, match="collectively"):
        operator.assemble_vector()
    u.x.array[:] = 0
    vector = operator.assemble_vector()
    try:
        assert vector.norm(PETSc.NormType.NORM_INFINITY) == 0
    finally:
        vector.destroy()


def test_stability_rejects_rank_inconsistent_safety():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("requires multiple ranks")
    _, _, operator = problem(MPI.COMM_WORLD)
    with pytest.raises(ValueError, match="differs across ranks"):
        operator.stable_dt(safety=0.5 if MPI.COMM_WORLD.rank == 0 else 0.8)


@pytest.mark.parametrize("where", ["time_input", "kinematics"])
def test_explicit_rank_local_input_failure_restores_all_ranks(where):
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("requires multiple ranks")
    from agentfem import amplitudes, constraints, problems

    _, u, operator = problem(MPI.COMM_WORLD)
    history = state.second_order_state(u)
    enabled = [True]

    def amplitude(t):
        if enabled[0] and MPI.COMM_WORLD.rank == 1 and t > 1e-4:
            raise ValueError("injected rank-local amplitude failure")
        return 0.0

    prescribed = constraints.time_dependent_component_dirichlet(
        u,
        0,
        marker=lambda x: np.ones(x.shape[1], dtype=bool),
        amplitude=amplitudes.Amplitude("fault_probe", amplitude),
    )

    def load(t):
        if enabled[0] and MPI.COMM_WORLD.rank == 1 and t > 0:
            u.x.array[:] = 999
            raise ValueError("injected rank-local load failure")

    step = problems.explicit_dynamics(
        state=history,
        integrator=central_difference(state=history, mass=operator),
        residual=operator,
        dt=1e-4,
        steps=1,
        prescribed=(prescribed,) if where == "kinematics" else (),
        update_load=load if where == "time_input" else None,
        progress=False,
    )
    accepted = history.snapshot()
    with pytest.raises(RuntimeError, match="rank 1"):
        step._advance_one(1e-4)
    for name, values in accepted["fields"].items():
        np.testing.assert_array_equal(history.snapshot()["fields"][name], values)
    enabled[0] = False
    step._advance_one(1e-4)
    np.testing.assert_array_equal(u.x.array, 0)


def test_work_ledger_samples_collectively_when_only_one_rank_owns_constraint():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("requires multiple ranks")
    from agentfem import constraints, fracture

    _, u, operator = problem(MPI.COMM_WORLD)
    u.interpolate(lambda x: np.vstack((0.01 * x[0], 0 * x[0], 0 * x[0])))
    history = state.second_order_state(u)
    fixed = constraints.component_dirichlet(
        u,
        0,
        marker=lambda x: np.all(np.isclose(x, 0), axis=0),
        value=0,
    )
    ledger = fracture.DynamicEnergyLedger(
        energy=None,
        state=history,
        mass=operator.mass_diagonal,
        residual=operator,
        prescribed=(fixed,),
    )
    constrained = ledger._prescribed_dofs(u)
    counts = MPI.COMM_WORLD.allgather(len(constrained))
    assert min(counts) == 0 and sum(counts) == 1
    expected = operator.assemble_vector()
    try:
        _, _, force = ledger._sample(u, history.a)
        np.testing.assert_allclose(force[constrained], expected.array[constrained])
        free = np.setdiff1d(np.arange(len(force)), constrained)
        np.testing.assert_array_equal(force[free], 0)
    finally:
        expected.destroy()
    ledger.restore(
        dict(
            natural_load_work=1.0,
            prescribed_motion_work=2.0,
            initial_accounted_energy=3.0,
        )
    )
    assert ledger._prescribed_work == 2


def ordinary_step(*, prescribed=False, domain=None):
    from agentfem import (
        amplitudes,
        constitutive,
        elements,
        fields,
        loads,
        models,
        studies,
    )

    if domain is None:
        domain = mesh.create_box(
            MPI.COMM_WORLD,
            [[0, 0, 0], [1, 0.2, 0.2]],
            [4, 2, 2],
            cell_type=mesh.CellType.hexahedron,
        )
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(constitutive.isotropic_elastic(young=100, poisson=0, density=2))
    if prescribed:
        model.fix(
            u,
            on=lambda x: np.ones(x.shape[1], dtype=bool),
            components=0,
            value=amplitudes.Amplitude(
                "quadratic",
                lambda t: 0.5 * t * t,
                metadata={"coefficient": 0.5, "power": 2},
            ),
        )
    else:
        model.load(loads.body_force((2, 0, 0), domain=domain, target=u))
    return model.step(
        target=u,
        element_policy=elements.uniform_strain_hex8(
            hourglass_modulus=50,
            hourglass_scale=0.1,
        ),
        dt=1e-4,
        steps=10,
        history_every=3,
        progress=False,
    )


@pytest.mark.parametrize("prescribed", [False, True])
def test_ordinary_distributed_hex_result_work_and_partition_restart(
    tmp_path, prescribed
):
    from pathlib import Path

    directory = Path(MPI.COMM_WORLD.bcast(str(tmp_path), root=0))
    reference = ordinary_step(prescribed=prescribed)
    result = reference.solve_result(
        output=directory / "reference.xdmf",
        field_variables=("S", "E", "MISES", "SENER"),
    )
    result.write_manifest(directory / "result.json", comm=MPI.COMM_WORLD)
    assert result.performance["parallel"]["rank_count"] == MPI.COMM_WORLD.size
    assert (
        result.metadata["step"]["performance"]["source"]
        == "SimulationResult.performance"
    )
    expected = 0.5 * 0.08 * (10e-4) ** 2
    last = reference.history_records[-1]
    assert last["kinetic_energy"] == pytest.approx(expected)
    assert last["external_work"] == pytest.approx(expected)
    assert "hourglass_energy" in result.histories
    assert result.fields["S"].location == "cells"
    partial = ordinary_step(prescribed=prescribed)
    partial.run(until_step=5)
    checkpoint = partial.save_checkpoint(directory / "parallel_hex")
    resumed = ordinary_step(prescribed=prescribed)
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    for name in ("u", "v", "a"):
        np.testing.assert_allclose(
            getattr(resumed.state, name).value.x.array,
            getattr(reference.state, name).value.x.array,
            atol=1e-15,
        )
    assert resumed.history_records[-1] == pytest.approx(last)


def test_ordinary_hex_step_tolerates_empty_owned_cell_partition():
    domain, _, _ = problem(MPI.COMM_WORLD, counts=(1, 1, 1))
    step = ordinary_step(domain=domain)
    step.solve_result(field_variables=("S", "E", "MISES", "SENER"))
    assert step.history_records[-1]["kinetic_energy"] == pytest.approx(
        0.5 * 0.08 * (10e-4) ** 2
    )


def test_distributed_material_regions_preserve_owned_mass_and_affine_force():
    from agentfem import constitutive, elements, fields, models, studies
    from agentfem import mesh as mesh_api
    from agentfem.constitutive.elasticity import stress

    domain, _, _ = problem(MPI.COMM_WORLD)
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    regions = mesh_api.partition_cells(
        domain, left=lambda x: x[0] < 0.5, right=lambda x: x[0] >= 0.5
    )
    a = constitutive.isotropic_elastic(young=100, poisson=0, density=2)
    b = constitutive.isotropic_elastic(young=40, poisson=0.2, density=4)
    model.material(a, region=regions.left)
    model.material(b, region=regions.right)
    u.value.interpolate(lambda x: np.vstack((0.01 * x[0], 0.02 * x[1], 0.03 * x[2])))
    policy = elements.uniform_strain_hex8(hourglass_modulus=50, hourglass_scale=0.1)
    step = model.step(
        target=u, element_policy=policy, dt="auto", steps=2, progress=False
    )
    assert domain.comm.allreduce(
        step.integrator.mass.mass_diagonal[::3].sum()
    ) == pytest.approx(0.12)
    v = ufl.TestFunction(u.value.function_space)
    expression = sum(
        ufl.inner(stress(u, law), ufl.sym(ufl.grad(v))) * region.measure
        for law, region in ((a, regions.left), (b, regions.right))
    )
    expected = assembly.assemble_vector(fem.form(expression))
    actual = step.residual.assemble_vector()
    try:
        np.testing.assert_allclose(actual.array, expected.array, atol=1e-13)
    finally:
        actual.destroy()
        expected.destroy()
    step.run()
