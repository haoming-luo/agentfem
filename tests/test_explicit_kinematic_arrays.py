"""Central-difference algebra without temporary FE fields, including ghosts."""
import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from agentfem import fields, state, time
from agentfem.kernel import dofs


@pytest.mark.parametrize("dt", [1e-6, 0.125])
def test_array_kinematics_match_field_algebra_with_stale_ghosts(monkeypatch, dt):
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    space = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    history = state.second_order_state(fem.Function(space))
    xyz = space.tabulate_dof_coordinates()
    owned = dofs.owned_size(history.u)
    for index, value in enumerate((history.u, history.v, history.a, history.a_next)):
        values = np.column_stack((xyz[:, 0] + index + .25,
                                  xyz[:, 1] ** 2 - index - .5)).ravel()
        value.x.array[:] = values
        value.x.array[owned:] = 999. + index
    before = [value.x.array.copy() for value in (history.u, history.v, history.a, history.a_next)]
    predicted = history.u + dt * history.v + .5 * dt**2 * history.a
    midpoint = history.v + .5 * dt * history.a
    corrected = midpoint + .5 * dt * history.a_next

    def forbidden(*args, **kwargs):
        raise AssertionError("Kinematic update allocated a temporary finite-element field")

    monkeypatch.setattr(fields, "empty_like", forbidden)
    integrator = time.explicit.central_difference(state=history, mass=np.ones(owned))
    integrator.predict_displacement(dt)
    integrator.update_midstep_velocity(dt)
    integrator.update_velocity(dt)
    for actual, expected in ((history.u_next, predicted), (history.v_mid, midpoint),
                             (history.v_next, corrected)):
        np.testing.assert_array_equal(actual.x.array, expected.x.array)
    for actual, expected in zip((history.u, history.v, history.a, history.a_next), before):
        np.testing.assert_array_equal(actual.x.array, expected)


@pytest.mark.parametrize("method,target,source", [
    ("predict_displacement", "u_next", "v"),
    ("update_midstep_velocity", "v_mid", "a"),
    ("update_velocity", "v_next", "a_next"),
])
def test_space_mismatch_rejected_before_target_mutation(method, target, source):
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    space = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    history = state.second_order_state(fem.Function(space))
    destination = getattr(history, target)
    destination.x.array[:] = 123.
    before = destination.x.array.copy()
    other = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    setattr(history, source, fields.wrap(fem.Function(other)))
    integrator = time.explicit.central_difference(state=history,
                                                 mass=np.ones(dofs.owned_size(history.u)))
    with pytest.raises(ValueError, match="same function space"):
        getattr(integrator, method)(.001)
    np.testing.assert_array_equal(destination.x.array, before)
