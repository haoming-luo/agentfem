"""Independent scalar finite-J2/rigid-stop reference, not another FE backend.

The prescribed lateral strain is zero. For F=diag(exp(e),1,1), nu=0 and
Fp=diag(exp(a),exp(-a/2),exp(-a/2)), trial equivalent Kirchhoff stress is
E*abs(e-3*a/2). Linear hardening gives dgamma=(qtrial-Y-H*p)/(3*E/2+H),
with a signed update of a and a nonnegative update of equivalent strain p.
The first Piola stress is E*(e-a_new)/exp(e). Unilateral equilibrium is then
a one-dimensional root or the prescribed rigid-stop stretch. No AgentFEM
material update, contact operator, tangent or nonlinear solver is used in
this reference. Homogeneous states do not demonstrate spatial convergence.
"""

from math import exp, log

import numpy as np
import pytest

from agentfem import amplitudes
from test_finite_j2_normal_augmentation import _problem, _state


def _response(stretch, old_a, old_p):
    young, yield_stress, hardening = 1000., 2., 100.
    e = log(stretch)
    shifted = e - 1.5 * old_a
    excess = young * abs(shifted) - yield_stress - hardening * old_p
    gamma = max(0., excess) / (1.5 * young + hardening)
    a = old_a + (gamma if shifted >= 0 else -gamma)
    p = old_p + gamma
    nominal = young * (e - a) / stretch
    return nominal, a, p


def _equilibrium(traction, old_a, old_p):
    stop = 1.005
    nominal, a, p = _response(stop, old_a, old_p)
    if traction >= nominal:
        return stop, a, p, traction - nominal
    low, high = .95, stop
    assert _response(low, old_a, old_p)[0] < traction
    for _ in range(60):
        middle = .5 * (low + high)
        if _response(middle, old_a, old_p)[0] < traction:
            low = middle
        else:
            high = middle
    stretch = .5 * (low + high)
    _, a, p = _response(stretch, old_a, old_p)
    return stretch, a, p, 0.


@pytest.mark.parametrize('cells', [1, 2])
@pytest.mark.parametrize('restart', [False, True])
def test_coaxial_plastic_contact_reversal_matches_scalar_reference(cells, restart, tmp_path):
    path = amplitudes.tabular((0., .25, .5, .625, .75, 1.),
                              (0., 1., 0., -.5, 0., 1.))
    def build():
        return _problem(
            cells=cells, penalty=5000., increments=16,
            transverse_boundary_fixed=True, load_amplitude=path,
            penetration_tolerance=1e-9, complementarity_tolerance=1e-8,
            maximum_augmentations=120,
        )[1]

    problem = build()
    a = p = 0.
    observations = []
    reference_coordinates = problem.solution.function_space.tabulate_dof_coordinates()
    for factor in np.linspace(1/16, 1., 16):
        traction = 10. * float(path(factor))
        stretch, a, p, pressure = _equilibrium(traction, a, p)
        problem.solve(until=float(factor))
        expected_u = np.zeros_like(reference_coordinates)
        expected_u[:, 0] = (stretch - 1.) * reference_coordinates[:, 0]
        np.testing.assert_allclose(problem.solution.x.array.reshape(-1, 3), expected_u,
                                   rtol=0., atol=3e-9)
        state = _state(problem)
        fp = np.diag([exp(a), exp(-a/2), exp(-a/2)])
        np.testing.assert_allclose(state[:, :9], np.broadcast_to(fp.ravel(), state[:, :9].shape),
                                   rtol=0., atol=3e-8)
        np.testing.assert_allclose(state[:, 9], p, rtol=0., atol=3e-8)
        np.testing.assert_allclose(state[:, 10], 2.*p, rtol=0., atol=6e-8)
        # Unit reference area: normal pressure is the rigid-stop reaction.
        force = problem.contact_provider.evaluate().contact_force_on_structure
        np.testing.assert_allclose(force, [-pressure, 0., 0.], rtol=0., atol=3e-6)
        observations.append((factor, a, p, pressure))
        if restart and factor == .625:
            checkpoint = problem.save_checkpoint(tmp_path / 'reverse_plastic')
            problem = build()
            problem.load_checkpoint(checkpoint)
    assert observations[3][3] > 0.      # First loading reaches the stop.
    assert observations[7][3] == 0.     # Unloading opens the contact.
    assert observations[9][1] < observations[3][1]  # Reverse plastic flow.
    assert observations[9][2] > observations[3][2]  # PEEQ cannot reverse.
    assert observations[-1][3] > 0.     # Reloading closes it again.
