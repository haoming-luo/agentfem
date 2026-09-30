from __future__ import annotations

import basix.ufl
from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np
import pytest
import ufl

from agentfem import boundary_models, operators, problems, time
from agentfem import mesh as agent_mesh


def _cube(comm, *, cells_x: int = 1):
    return mesh.create_unit_cube(
        comm,
        cells_x,
        1,
        1,
        cell_type=mesh.CellType.tetrahedron,
    )


def _left_region(domain):
    facet_dimension = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain,
        facet_dimension,
        lambda x: np.isclose(x[0], 0.0),
    )
    tags = mesh.meshtags(
        domain,
        facet_dimension,
        facets,
        np.full(facets.size, 27, dtype=np.int32),
    )
    return agent_mesh.tagged_boundary_region(
        domain,
        tags,
        tag=27,
        name="left_contact_slave",
    )


def _vector_space(domain):
    return fem.functionspace(
        domain,
        basix.ufl.element(
            "Lagrange",
            domain.basix_cell(),
            1,
            shape=(3,),
        ),
    )


def _bulk_residual(domain, function_space, value):
    test = ufl.TestFunction(function_space)
    source = fem.Constant(domain, np.asarray(value, dtype=float))
    return operators.OperatorForm(
        name="bulk_residual",
        expression=ufl.inner(source, test) * ufl.dx,
        kind="bulk_residual",
        role="residual",
        family="test",
    )


def _contact(
    comm,
    *,
    cells_x: int = 1,
    displacement_value: float = 0.1,
    bulk_value=(0.0, 0.0, 0.0),
):
    domain = _cube(comm, cells_x=cells_x)
    function_space = _vector_space(domain)
    displacement = fem.Function(function_space, name="Displacement")
    displacement.x.array.reshape((-1, 3))[:, 0] = displacement_value
    displacement.x.scatter_forward()
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        function_space,
    )
    surface = boundary_models.rigid_plane(
        point=(0.05, 0.0, 0.0),
        normal=(-1.0, 0.0, 0.0),
    )
    residual = boundary_models.dolfinx_explicit_contact_residual(
        _bulk_residual(domain, function_space, bulk_value),
        adapter=adapter,
        displacement=displacement,
        projector=surface,
        penalty=200.0,
        maximum_stable_time_increment=1.0e-3,
        surface_reference_point=(0.0, 0.0, 0.0),
    )
    return domain, displacement, residual


def test_explicit_contact_residual_assembles_force_and_accepts_evidence():
    domain, _displacement, residual = _contact(MPI.COMM_SELF)

    vector = residual.assemble_vector()
    try:
        values = vector.array.reshape((-1, 3))
        np.testing.assert_allclose(np.sum(values, axis=0), (10.0, 0.0, 0.0))
    finally:
        vector.destroy()

    assert residual.trial_evidence is not None
    np.testing.assert_allclose(
        residual.trial_evidence.contact_force_on_structure,
        (-10.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(residual.trial_evidence.potential_energy, 0.25)
    residual.commit()
    assert residual.trial_evidence is None
    assert residual.accepted_evidence is not None
    assert residual.accepted_evaluations == 1
    assert residual.lifecycle.state.accepted is not None
    assert residual.summary()["surface_motion"] == "fixed_only"
    assert domain.comm.size == 1


def test_explicit_contact_residual_rolls_back_failed_trial():
    _domain, displacement, residual = _contact(MPI.COMM_SELF)
    vector = residual.assemble_vector()
    vector.destroy()
    residual.rollback()
    assert residual.trial_evidence is None
    assert residual.lifecycle.state.trial is None

    displacement.x.array[:] = np.nan
    with pytest.raises(ValueError, match="rejected collectively"):
        residual.assemble_vector()
    assert residual.trial_evidence is None
    assert residual.lifecycle.state.trial is None


def test_explicit_contact_residual_enforces_declared_stability_limit():
    _domain, displacement, residual = _contact(MPI.COMM_SELF)
    state = problems.second_order_state(displacement)
    diagonal = np.ones(displacement.x.array.shape, dtype=float)
    mass = operators.LumpedMassOperator(mass=diagonal, inv_mass=diagonal)
    integrator = time.explicit.central_difference(state=state, mass=mass)

    with pytest.raises(ValueError, match="exceeds.*stability limit"):
        problems.explicit_dynamics(
            state=state,
            integrator=integrator,
            residual=residual,
            dt=2.0e-3,
            steps=1,
            progress=False,
        )

    step = problems.explicit_dynamics(
        state=state,
        integrator=integrator,
        residual=residual,
        dt=5.0e-4,
        steps=1,
        progress=False,
    )
    assert step.dt == pytest.approx(5.0e-4)


def test_explicit_contact_residual_advances_through_central_difference():
    _domain, displacement, residual = _contact(MPI.COMM_SELF)
    state = problems.second_order_state(displacement)
    diagonal = np.ones(displacement.x.array.shape, dtype=float)
    mass = operators.LumpedMassOperator(mass=diagonal, inv_mass=diagonal)
    integrator = time.explicit.central_difference(state=state, mass=mass)
    step = problems.explicit_dynamics(
        state=state,
        integrator=integrator,
        residual=residual,
        dt=5.0e-4,
        steps=1,
        progress=False,
    )

    step.run()

    acceleration = state.a.value.x.array.reshape((-1, 3)).sum(axis=0)
    np.testing.assert_allclose(acceleration, (-10.0, 0.0, 0.0), atol=1.0e-13)
    assert step.completed_steps == 1
    assert residual.accepted_evaluations == 1
    assert residual.lifecycle.state.accepted is not None


def test_explicit_contact_residual_preserves_global_force_under_mpi():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("Explicit contact residual is reviewed on two ranks.")
    comm = MPI.COMM_WORLD
    _domain, _displacement, residual = _contact(
        comm,
        cells_x=2,
        bulk_value=(1.0, 1.0, 1.0),
    )

    vector = residual.assemble_vector()
    try:
        index_map = residual.adapter.function_space.dofmap.index_map
        owned = int(index_map.size_local) * 3
        local = vector.array[:owned].reshape((-1, 3)).sum(axis=0)
        total = np.empty(3, dtype=float)
        comm.Allreduce(local, total, op=MPI.SUM)
    finally:
        vector.destroy()

    # Unit-cube bulk resultant is (1, 1, 1).  The contact residual contributes
    # (10, 0, 0).  This catches accidental second reverse-scattering of the
    # already assembled bulk ghost entries.
    np.testing.assert_allclose(total, (11.0, 1.0, 1.0), atol=1.0e-13)
    np.testing.assert_allclose(
        residual.trial_evidence.contact_force_on_structure,
        (-10.0, 0.0, 0.0),
        atol=1.0e-13,
    )
    assert residual.trial_evidence.active_point_count == 6
    residual.commit()


def test_explicit_step_rolls_back_if_residual_commit_fails():
    class Residual:
        def __init__(self):
            self.rollbacks = 0

        def commit(self):
            raise RuntimeError("commit failed")

        def rollback(self):
            self.rollbacks += 1

    class Integrator:
        def step(self, *args, **kwargs):
            return None

    from agentfem._transient_problems import ExplicitDynamicsStep

    residual = Residual()
    step = ExplicitDynamicsStep(
        name="commit_failure",
        state=object(),
        integrator=Integrator(),
        residual=residual,
        dt=1.0,
        steps=1,
    )

    with pytest.raises(RuntimeError, match="commit failed"):
        step._advance_one(1.0)
    assert residual.rollbacks == 1
