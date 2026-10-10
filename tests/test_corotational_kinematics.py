import numpy as np
import pytest

from agentfem.constitutive._corotational import (
    polar_strain_increment,
    push_corotational_stress,
)


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_rigid_rotation_of_prestrained_material():
    old = np.array([[[1.3, 0.2, 0], [0.2, 0.9, 0.1], [0, 0.1, 1.1]]])
    new = rotation(2.9) @ old
    result = polar_strain_increment(old, new)
    np.testing.assert_allclose(result.strain_increment, 0, atol=2.0e-15)
    stress = np.array([[[3.0, 1.0, 0], [1.0, 2.0, 0], [0, 0, 1.0]]])
    np.testing.assert_allclose(
        push_corotational_stress(stress, result.frame_new),
        rotation(2.9) @ stress @ rotation(2.9).T,
        atol=1.0e-14,
    )


def test_coaxial_log_stretch_and_material_axes():
    old = np.array([np.diag([0.8, 1.2, 2.0])])
    new = np.array([rotation(0.9) @ np.diag([1.3, 1.5, 1.1])])
    q = rotation(0.43)[None]
    result = polar_strain_increment(old, new, reference_frame=q)
    expected = (
        q.transpose(0, 2, 1) @ np.diag(np.log([1.3 / 0.8, 1.5 / 1.2, 1.1 / 2.0])) @ q
    )
    np.testing.assert_allclose(result.strain_increment, expected, atol=1.0e-14)


def test_noncommuting_stretches_against_independent_quadrature():
    u0 = np.diag([0.7, 1.4, 1.1])
    u1 = rotation(0.6) @ np.diag([1.6, 0.9, 1.2]) @ rotation(0.6).T
    points, weights = np.polynomial.legendre.leggauss(64)
    expected = np.zeros((3, 3))
    for s, w in zip((points + 1) / 2, weights / 2):
        rate = (u1 - u0) @ np.linalg.inv(u0 + s * (u1 - u0))
        expected += w * (rate + rate.T) / 2
    result = polar_strain_increment(u0[None], u1[None])
    np.testing.assert_allclose(result.strain_increment[0], expected, atol=2.0e-14)


def test_subdivision_and_time_dependent_superposed_rotation():
    u0 = np.diag([0.7, 1.4, 1.1])[None]
    u1 = (rotation(0.6) @ np.diag([1.6, 0.9, 1.2]) @ rotation(0.6).T)[None]
    middle = (u0 + u1) / 2
    full = polar_strain_increment(u0, u1)
    a = polar_strain_increment(u0, middle)
    b = polar_strain_increment(middle, u1)
    np.testing.assert_allclose(
        a.strain_increment + b.strain_increment, full.strain_increment, atol=3.0e-14
    )
    rotated = polar_strain_increment(rotation(-0.9) @ u0, rotation(1.7) @ u1)
    np.testing.assert_allclose(
        rotated.strain_increment, full.strain_increment, atol=3.0e-14
    )


def test_batch_empty_immutable_and_small_increment():
    old = np.broadcast_to(np.eye(3), (9, 3, 3)).copy()
    new = old.copy()
    new[:, 0, 0] += 1.0e-9
    result = polar_strain_increment(old, new)
    np.testing.assert_allclose(
        result.strain_increment[:, 0, 0], np.log1p(new[:, 0, 0] - 1), atol=1.0e-24
    )
    for i in range(9):
        single = polar_strain_increment(old[i : i + 1], new[i : i + 1])
        np.testing.assert_array_equal(
            single.strain_increment[0], result.strain_increment[i]
        )
    assert not result.strain_increment.flags.writeable
    assert polar_strain_increment(old[:0], new[:0]).strain_increment.shape == (0, 3, 3)


@pytest.mark.parametrize(
    "bad",
    [
        np.diag([-1.0, 1, 1]),
        np.diag([0.0, 1, 1]),
        np.diag([1.0e-14, 1, 1]),
        np.full((3, 3), np.nan),
    ],
)
def test_invalid_gradient_rejected(bad):
    with pytest.raises(ValueError):
        polar_strain_increment(np.eye(3)[None], bad[None])


def test_invalid_frames_and_stress_rejected():
    identity = np.eye(3)[None]
    with pytest.raises(ValueError):
        polar_strain_increment(identity, identity, reference_frame=identity * 2)
    with pytest.raises(ValueError):
        push_corotational_stress(identity, identity * 2)
    stress = identity.copy()
    stress[0, 0, 1] = 1
    with pytest.raises(ValueError):
        push_corotational_stress(stress, identity)


def test_random_three_dimensional_frames_and_virtual_power():
    rng = np.random.default_rng(482)
    count = 32

    def rotations():
        q, _ = np.linalg.qr(rng.normal(size=(count, 3, 3)))
        q[:, :, 0] *= np.linalg.det(q)[:, None]
        return q

    a, b, q0, q1, axes = (rotations() for _ in range(5))
    u0 = (a * rng.uniform(0.8, 1.2, size=(count, 1, 3))) @ a.transpose(0, 2, 1)
    u1 = (b * rng.uniform(0.8, 1.2, size=(count, 1, 3))) @ b.transpose(0, 2, 1)
    base = polar_strain_increment(u0, u1, reference_frame=axes)
    rotated = polar_strain_increment(q0 @ u0, q1 @ u1, reference_frame=axes)
    np.testing.assert_allclose(
        rotated.strain_increment, base.strain_increment, atol=4.0e-15
    )
    stress = rng.normal(size=(count, 3, 3))
    stress += stress.transpose(0, 2, 1)
    spatial = push_corotational_stress(stress, rotated.frame_new)
    spatial_increment = (
        rotated.frame_new
        @ rotated.strain_increment
        @ rotated.frame_new.transpose(0, 2, 1)
    )
    np.testing.assert_allclose(
        np.einsum("nij,nij->n", spatial, spatial_increment),
        np.einsum("nij,nij->n", stress, rotated.strain_increment),
        atol=2.0e-14,
    )


@pytest.mark.parametrize("shape", [(3, 3), (2, 2, 2), (2, 3, 4)])
def test_wrong_batch_shape_rejected(shape):
    with pytest.raises(ValueError):
        polar_strain_increment(np.zeros(shape), np.zeros(shape))


def test_complex_and_mismatched_batches_rejected():
    identity = np.eye(3)[None]
    with pytest.raises(ValueError):
        polar_strain_increment(identity.astype(complex), identity)
    with pytest.raises(ValueError):
        polar_strain_increment(identity, np.repeat(identity, 2, axis=0))


def test_stationary_batch_preserves_frames_without_spectral_work(monkeypatch):
    original = np.linalg.svd
    counts = []

    def tracked(value, *args, **kwargs):
        counts.append(len(value))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", tracked)
    old = np.broadcast_to(np.eye(3), (100, 3, 3)).copy()
    old[1] = rotation(0.7) @ np.diag([1.2, 0.9, 1.1])
    new = old.copy()
    new[2, 0, 0] = 1.01
    result = polar_strain_increment(old, new)
    assert sum(counts) == 2  # One stationary nonidentity and one changed endpoint.
    np.testing.assert_array_equal(result.strain_increment[:2], 0)
    np.testing.assert_allclose(result.frame_old[1], rotation(0.7), atol=1e-14)
    assert result.strain_increment[2, 0, 0] == pytest.approx(np.log(1.01))


def test_stationary_invalid_gradient_is_not_skipped():
    for bad in (np.diag([-1.0, 1, 1]), np.diag([1e-14, 1, 1])):
        with pytest.raises(ValueError):
            polar_strain_increment(bad[None], bad[None])


def test_moving_increment_reuses_svd_stretch_spectrum(monkeypatch):
    original = np.linalg.eigh
    calls = []

    def tracked(value, *args, **kwargs):
        calls.append(len(value))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigh", tracked)
    old = np.array([rotation(0.4) @ np.diag([0.8, 1.2, 2.0])])
    new = np.array([rotation(-0.7) @ np.diag([1.3, 1.5, 1.1])])
    result = polar_strain_increment(old, new)
    assert calls == [1]  # Only the relative stretch logarithm needs an eigensolve.
    np.testing.assert_allclose(
        result.strain_increment[0],
        np.diag(np.log([1.3 / 0.8, 1.5 / 1.2, 1.1 / 2.0])),
        atol=1e-14,
    )
