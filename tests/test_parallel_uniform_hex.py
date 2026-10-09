# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Owned-cell/ghost-vector gate only; public Hex8 Step remains serial."""

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
            count = cells.num_nodes if hasattr(cells, "num_nodes") else sum(
                array.size // 8 for array in cells
            )
            result = graph.adjacencylist(np.zeros((count, 1), dtype=np.int32))
            return getattr(result, "_cpp_object", result)

    domain = mesh.create_box(
        comm, [[0, 0, 0], [1, 0.2, 0.2]], counts, cell_type=mesh.CellType.hexahedron,
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
        operator.stable_dt(safety=.5 if MPI.COMM_WORLD.rank == 0 else .8)
