# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""A fast orientation filter never replaces whole-cell validity admission."""
import basix
import numpy as np
import pytest

from agentfem.elements._hex_validity import (
    _coordinate_edges, _affine_reference, reference_perturbation_radius,
    require_positive_hex_jacobian,
)


def native_flags(x, u, f):
    native = pytest.importorskip('agentfem._hex_native')
    return np.frombuffer(native.affine_screen(
        np.ascontiguousarray(x), np.ascontiguousarray(u), np.ascontiguousarray(f),
        reference_perturbation_radius(x),
        np.ascontiguousarray(_coordinate_edges(), dtype=np.int32)), dtype=np.uint8)


@pytest.mark.parametrize('angle', [.6, 1.57, 2.5, np.pi])
@pytest.mark.parametrize('scale', [1e-8, 1., 1e8])
def test_native_large_rotation_is_admitted_and_reflection_rejected(angle, scale):
    x = (basix.cell.geometry(basix.CellType.hexahedron) * [2., 1., .05])[None] * scale
    c, s = np.cos(angle), np.sin(angle)
    f = np.array([[[c, 0, -s], [0, 1, 0], [s, 0, c]]])
    u = x @ (f - np.eye(3)).swapaxes(1, 2) + np.array([2., -3., 4.]) * scale
    assert native_flags(x, u, f)[0] == 1
    assert require_positive_hex_jacobian(x + u)[0] > 0
    reflected = f.copy()
    reflected[:, :, 0] *= -1
    reflected_u = x @ (reflected - np.eye(3)).swapaxes(1, 2)
    assert native_flags(x, reflected_u, reflected)[0] == 0


def test_rotated_nonaffine_admission_is_a_subset_of_reference():
    rng = np.random.default_rng(407)
    n = 600
    x = basix.cell.geometry(basix.CellType.hexahedron) + rng.normal(0, .005, (n, 8, 3))
    rotations, _ = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    rotations[:, :, 0] *= np.linalg.det(rotations)[:, None]
    f = rotations @ (np.eye(3) + rng.normal(0, .03, (n, 3, 3)))
    u = x @ (f - np.eye(3)).swapaxes(1, 2)
    u += rng.normal(size=u.shape) * np.geomspace(1e-8, 1., n)[:, None, None]
    accepted = native_flags(x, u, f).astype(bool)
    reference = _affine_reference(x, u, f, reference_perturbation_radius(x))
    assert accepted.any() and not accepted.all()
    assert np.all(reference[accepted])
    assert np.all(require_positive_hex_jacobian((x + u)[accepted]) > 0)


@pytest.mark.parametrize('bad', [
    np.diag([1., 1., 1e-310]),
    np.array([[1., 1., 0.], [1., np.nextafter(1., 2.), 0.], [0., 0., 1.]]),
    np.diag([1e200, 1e200, 1e200]),
    np.diag([np.nan, 1., 1.]),
])
def test_uncertain_orientation_or_overflow_keeps_fallback(bad):
    x = basix.cell.geometry(basix.CellType.hexahedron)[None]
    f = bad[None]
    u = x @ (f - np.eye(3)).swapaxes(1, 2)
    assert native_flags(x, u, f)[0] == 0


def test_positive_mean_gradient_does_not_admit_inverted_corner():
    x = basix.cell.geometry(basix.CellType.hexahedron)[None]
    f = np.array([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    u = x @ (f - np.eye(3)).swapaxes(1, 2)
    u[0, 7, 2] -= 2.
    assert np.linalg.det(f)[0] > 0
    assert native_flags(x, u, f)[0] == 0
    with pytest.raises(ValueError):
        require_positive_hex_jacobian(x + u)


@pytest.mark.parametrize('stretch', [[2., 1., .5], [.2, .3, .4], [4., .7, 1.3]])
def test_rotated_affine_stretches_do_not_require_near_identity_gram(stretch):
    x = basix.cell.geometry(basix.CellType.hexahedron)[None]
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    f = (rotation @ np.diag(stretch))[None]
    u = x @ (f - np.eye(3)).swapaxes(1, 2)
    assert native_flags(x, u, f)[0] == 1
    assert require_positive_hex_jacobian(x + u)[0] > 0
