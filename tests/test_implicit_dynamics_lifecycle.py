# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Focused correctness and lifecycle tests for implicit dynamics."""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from agentfem import constitutive, fields, models, problems, studies
from agentfem import mesh as agentfem_mesh
from agentfem.constraints import boundary


@pytest.mark.parametrize("dimension,component", ((2, 0), (2, 1), (3, 2)))
def test_zero_kinematic_bc_preserves_component_dof_identity(dimension, component):
    if dimension == 2:
        domain = mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    else:
        domain = mesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2)
    V = fem.functionspace(domain, ("Lagrange", 1, (dimension,)))
    _constant, source = boundary.component_dirichlet_bc(
        V,
        component,
        marker=lambda x: np.isclose(x[0], 0.0),
        value=1.25,
    )

    zero = problems._zero_kinematic_bcs((source,))[0]

    source_dofs, source_owned = source.dof_indices()
    zero_dofs, zero_owned = zero.dof_indices()
    np.testing.assert_array_equal(zero_dofs, source_dofs)
    assert zero_owned == source_owned
    assert zero.function_space == source.function_space
    assert np.all(source_dofs % dimension == component)


def test_zero_kinematic_bc_preserves_function_valued_vector_constraint():
    domain = mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    value = fem.Function(V)
    value.interpolate(lambda x: np.vstack((1.0 + x[1], 2.0 - x[0])))
    facets = mesh.locate_entities_boundary(
        domain,
        1,
        lambda x: np.isclose(x[0], 0.0),
    )
    dofs = fem.locate_dofs_topological(V, 1, facets)
    source = fem.dirichletbc(value, dofs)

    zero = problems._zero_kinematic_bcs((source,))[0]

    source_dofs, source_owned = source.dof_indices()
    zero_dofs, zero_owned = zero.dof_indices()
    np.testing.assert_array_equal(zero_dofs, source_dofs)
    assert zero_owned == source_owned
    assert zero.function_space == source.function_space
    constrained = fem.Function(V)
    constrained.x.array[:] = 3.0
    zero.set(constrained.x.array)
    np.testing.assert_allclose(constrained.x.array[zero_dofs], 0.0)


def _implicit_cantilever(
    operator_policy: str,
    *,
    steps: int = 4,
    comm=MPI.COMM_SELF,
):
    domain = agentfem_mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 1),
        comm=comm,
        cell_type="triangle",
    )
    study = studies.implicit_dynamics(
        physics="solid_mechanics",
        dimension=2,
        assumption="plane_stress",
        method="newmark",
    )
    model = models.create(study=study, mesh=domain)
    displacement = model.field(fields.displacement(domain))
    model.material(
        constitutive.isotropic_elastic(
            young=2.0e5,
            poisson=0.3,
            density=1.0e3,
        )
    )
    left = agentfem_mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 0.0),
        name="left",
        tag=1,
    )
    right = agentfem_mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 1.0),
        name="right",
        tag=2,
    )
    model.fix(displacement, on=left, value=0.0)
    model.traction((10.0, 0.0), on=right)
    step = model.step(
        target=displacement,
        dt=1.0e-3,
        steps=steps,
        operator_policy=operator_policy,
        progress=False,
    )
    return step


def test_auto_reuses_one_operator_across_partial_runs_and_records_result():
    step = _implicit_cantilever("auto")

    step.run(until_step=2)
    partial = step.operator_lifecycle_summary()
    assert partial["selected_policy"] == "reuse"
    assert partial["matrix_assembly_count"] == 1
    assert partial["rhs_assembly_count"] == 2
    assert partial["solve_count"] == 2

    result = step.solve_result()
    final = step.operator_lifecycle_summary()
    assert final["matrix_assembly_count"] == 1
    assert final["rhs_assembly_count"] == 4
    assert final["solve_count"] == 4
    assert result.metadata["step"]["operator_lifecycle"] == final
    assert step._prepared_problem is None


def test_refresh_policy_matches_reuse_and_assembles_each_step():
    reused = _implicit_cantilever("reuse")
    refreshed = _implicit_cantilever("refresh_each_step")

    reused.solve()
    refreshed.solve()

    np.testing.assert_allclose(
        refreshed.state.u.value.x.array,
        reused.state.u.value.x.array,
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        refreshed.state.v.value.x.array,
        reused.state.v.value.x.array,
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    refreshed_lifecycle = refreshed.operator_lifecycle_summary()
    assert refreshed_lifecycle["selected_policy"] == "refresh_each_step"
    assert refreshed_lifecycle["matrix_assembly_count"] == 4
    assert refreshed_lifecycle["rhs_assembly_count"] == 4
    assert refreshed_lifecycle["solve_count"] == 4


def test_reuse_fails_closed_if_step_operator_controls_change():
    step = _implicit_cantilever("reuse", steps=2)
    step.run(until_step=1)
    step.dt *= 2.0

    with pytest.raises(RuntimeError, match="AFM-DYNAMICS-OPERATOR-001"):
        step.run()

    assert step._prepared_problem is None


def test_implicit_dynamics_rejects_unknown_operator_policy():
    with pytest.raises(ValueError, match="operator_policy"):
        _implicit_cantilever("sometimes")


def test_fixed_operator_reuse_is_collective_under_mpi():
    step = _implicit_cantilever("auto", steps=3, comm=MPI.COMM_WORLD)

    step.solve()

    lifecycle = step.operator_lifecycle_summary()
    records = MPI.COMM_WORLD.allgather(lifecycle)
    assert all(item == records[0] for item in records)
    assert lifecycle["matrix_assembly_count"] == 1
    assert lifecycle["rhs_assembly_count"] == 3
    assert lifecycle["solve_count"] == 3
