import numpy as np
import pytest
from numpy.polynomial import Polynomial

from agentfem._q1_surface_geometry import _projection_roots, q1_closest_point


@pytest.mark.parametrize('small', [1e-14, 1e-11, -1e-11, .2])
@pytest.mark.parametrize('large', [-7e12, 7e12])
@pytest.mark.parametrize('scale', [1e-100, 1., 1e100])
def test_quadratic_retains_small_root(small, large, scale):
    equation = Polynomial(np.array([small*large, -(small+large), 1.])*scale)
    roots = sorted(_projection_roots(equation), key=abs)
    np.testing.assert_allclose(roots, [small, large], rtol=5e-15, atol=0)


def test_quadratic_zero_repeated_and_complex():
    np.testing.assert_array_equal(sorted(_projection_roots(Polynomial([0., 2., 1.]))), [-2., 0.])
    np.testing.assert_array_equal(_projection_roots(Polynomial([0., 0., 1.])), [0., 0.])
    np.testing.assert_allclose(_projection_roots(Polynomial([1., 0., 1.])), [-1j, 1j])
    np.testing.assert_allclose(_projection_roots(Polynomial([1., -2., 1.])), [1., 1.])


@pytest.mark.parametrize('warpage', [1e-12, 1e-10, 1e-8])
@pytest.mark.parametrize('v', [1e-11, 1e-9, 1-1e-11])
def test_nearly_affine_patch_retains_near_edge_projection(warpage, v):
    vertices = np.array([[0., 0., 0.], [1., 0., 0.],
                         [1., 1., warpage], [0., 1., 0.]])
    u = .17
    normal = np.array([-warpage*v, -warpage*u, 1.])
    normal /= np.linalg.norm(normal)
    position = np.array([u, v, warpage*u*v])
    answer = q1_closest_point(vertices, position + .03*normal)
    np.testing.assert_allclose(answer[0], [u, v], atol=5e-15)
    np.testing.assert_allclose(answer[2], position, atol=5e-15)
