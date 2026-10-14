# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem._q1_exact_order import convex_closest_parameters
from agentfem import _q1_surface_geometry as geometry


@pytest.mark.parametrize("uv", [(1e-8, 1e-12), (.37, .61), (1-1e-12, 1e-9)])
def test_warped_interior_recovery_has_small_parameter_error(uv):
    x = np.array([[0., 0, 0], [1, 0, 1e-7], [1.03, 1.4, 3e-7], [0, 1.41, -1e-7]])
    u, v = uv
    b, c, d = x[1]-x[0], x[3]-x[0], x[2]-x[1]-x[3]+x[0]
    normal = np.cross(b+d*v, c+d*u)
    normal /= np.linalg.norm(normal)
    point = x[0]+b*u+c*v+d*u*v+normal*1e-5
    result = convex_closest_parameters(x, point)
    assert result is not None
    np.testing.assert_allclose(result, uv, rtol=0, atol=4e-15)


@pytest.mark.parametrize("point,expected", [([-1., .3, .2], [0., .3]),
                                            ([2., 3., .1], [1., 1.]),
                                            ([.2, .4, .5], [.2, .4])])
def test_affine_box_minimum_including_active_constraints(point, expected):
    x = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    np.testing.assert_allclose(convex_closest_parameters(x, point), expected,
                               atol=1e-15, rtol=0)


def test_recovery_does_not_admit_unproved_nonconvex_patch():
    x = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 2], [0, 1, 0]])
    assert convex_closest_parameters(x, [.5, .5, 10]) is None


def test_missing_polynomial_candidate_recovers_only_with_certificate(monkeypatch):
    x = np.array([[0., 0, 0], [1, 0, 0], [1, 1, .02], [0, 1, 0]])
    point = np.array([.3, .4, .1])
    reference = geometry._q1_closest_point_polynomial(x, point)
    monkeypatch.setattr(geometry, "_projection_roots", lambda _: [])
    actual = geometry._q1_closest_point_polynomial(x, point)
    np.testing.assert_allclose(actual[0], reference[0], atol=1e-13, rtol=0)
    np.testing.assert_allclose(actual[2], reference[2], atol=1e-13, rtol=0)


def test_skew_anisotropic_comparison_bound_is_positive_and_conservative():
    from agentfem._q1_exact_order import _coefficients, _strong_convexity_modulus

    b, c = np.array([.01, 0., 0.]), np.array([.1, 1., 0.])
    x = np.array([np.zeros(3), b, b+c, c])
    point = .37*b+.61*c+np.array([0., 0., .2])
    hessian = np.array([[b@b, b@c], [b@c, c@c]])
    assert min(hessian.diagonal())-abs(hessian[0, 1]) < 0
    bound = float(_strong_convexity_modulus(_coefficients(x, point)))
    assert 0 < bound <= np.linalg.eigvalsh(hessian)[0]
    np.testing.assert_allclose(convex_closest_parameters(x, point), [.37, .61],
                               atol=4e-15, rtol=0)


def test_comparison_bound_below_sampled_warped_hessian_eigenvalues():
    from agentfem._q1_exact_order import _coefficients, _strong_convexity_modulus

    rng = np.random.default_rng(703)
    for _ in range(25):
        b = np.array([rng.uniform(.01, .1), 0., 0.])
        c = np.array([rng.uniform(.02, .2), 1., 0.])
        d = rng.uniform(-1e-4, 1e-4, 3)
        x = np.array([np.zeros(3), b, b+c+d, c])
        point = np.array([.04, .3, .05])
        bound = float(_strong_convexity_modulus(_coefficients(x, point)))
        for u, v in rng.uniform(0, 1, (20, 2)):
            du, dv = b+d*v, c+d*u
            delta = b*u+c*v+d*u*v-point
            off = du@dv+delta@d
            hessian = [[du@du, off], [off, dv@dv]]
            assert bound <= np.linalg.eigvalsh(hessian)[0]+1e-16


@pytest.mark.parametrize("exponent", [-100, 0, 100])
def test_correlated_hessian_certificate_recovers_warped_patch(exponent):
    from agentfem._q1_exact_order import _coefficients, _bernstein_hessian_modulus

    scale = 2.0**exponent
    x = scale*np.array([[0., .5, -.375], [.875, .25, -.375],
                        [1.125, .875, 0.], [0., .75, .25]])
    p = scale*np.array([.875, -.75, -.875])
    bound = float(_bernstein_hessian_modulus(_coefficients(x, p)))
    assert bound > 0
    np.testing.assert_array_equal(convex_closest_parameters(x, p), [1., 0.])
    a, b, c, d = x[0]-p, x[1]-x[0], x[3]-x[0], x[2]-x[1]-x[3]+x[0]
    for u in np.linspace(0, 1, 11):
        for v in np.linspace(0, 1, 11):
            du, dv = b+d*v, c+d*u
            delta = a+b*u+c*v+d*u*v
            off = du@dv+delta@d
            assert bound <= np.linalg.eigvalsh([[du@du, off], [off, dv@dv]])[0]


def test_bernstein_certificate_rejects_flat_and_indefinite_hessians():
    from agentfem._q1_exact_order import _coefficients, _bernstein_hessian_modulus

    assert _bernstein_hessian_modulus(_coefficients(np.zeros((4, 3)), [1., 2., 3.])) == 0
    x = np.array([[0., 0, 0], [1., 0, 0], [1., 1, 2], [0., 1, 0]])
    assert _bernstein_hessian_modulus(_coefficients(x, [.5, .5, 10])) == 0
