# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem import boundary_models as bm
from agentfem.boundary_models._radial_contact_stability import (
    radial_contact_spectral_envelope,
    validate_radial_envelope_projection,
)


def trace():
    return bm.ContactTrace(
        point_ids=[1, 2],
        node_ids=[[0, 1], [0, 1]],
        shape_values=[[1.2, -0.2], [0.3, 0.7]],
        weights=[0.4, 0.8],
    )


def surface(kind, side):
    if kind == "sphere":
        return bm.rigid_sphere(center=(0, 0, 0), radius=2, admissible_side=side)
    return bm.rigid_cylinder(
        axis_point=(0, 0, 0),
        axis_direction=(0, 0, 1),
        radius=2,
        admissible_side=side,
        axial_interval=(-3, 3),
    )


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
@pytest.mark.parametrize(
    "side,r", [("exterior", 0.7), ("exterior", 1.9), ("interior", 2.2)]
)
def test_envelope_encloses_independent_force_derivative(kind, side, r):
    t, s = trace(), surface(kind, side)
    mass = np.array([[1.0, 3.0, 2.0], [4.0, 2.0, 5.0]])
    penalty = np.array([7.0, 11.0])
    e = radial_contact_spectral_envelope(
        t, mass, surface=s, normal_penalty=penalty, minimum_radial_distance=0.5
    )
    points = np.array([[r, 0, 0], [0, r, 0.0]])
    # B maps independent nodal perturbations to the two query locations.
    b = np.kron(t.shape_values, np.eye(3))

    def force(u):
        p = s.project(points + (b @ u).reshape(2, 3))
        local = penalty[:, None] * np.minimum(p.signed_gaps, 0)[:, None] * p.normals
        return b.T @ (t.weights[:, None] * local).ravel()

    h = 1e-6
    tangent = np.column_stack(
        [(force(h * v) - force(-h * v)) / (2 * h) for v in np.eye(6)]
    )
    scaled = tangent / np.sqrt(mass.ravel()[:, None] * mass.ravel()[None, :])
    np.testing.assert_allclose(scaled, scaled.T, atol=1e-8)
    eig = np.linalg.eigvalsh(scaled)
    assert eig[-1] <= e.positive_rows.max() + 1e-8
    assert eig[0] >= -e.negative_rows.max() - 1e-8
    if side == "exterior":
        assert eig[0] < 0 and e.negative_rows.max() > 0
    else:
        assert eig[0] > -1e-8 and e.negative_rows.max() == 0
    validate_radial_envelope_projection(e, s, s.project(points))
    assert not e.positive_rows.flags.writeable


def test_point_derivative_exposes_curvature_not_only_normal_stiffness():
    s = surface("cylinder", "exterior")
    p = np.array([[0.5, 0, 0.0]])
    h = 1e-6

    def f(x):
        q = s.project(x)
        return (q.signed_gaps[:, None] * q.normals).ravel()

    a = np.column_stack([(f(p + h * v) - f(p - h * v)) / (2 * h) for v in np.eye(3)])
    np.testing.assert_allclose(a, np.diag([1.0, -3.0, 0.0]), atol=2e-9)


@pytest.mark.parametrize("floor", [0, -1, 3, True, np.nan, np.inf, 1j, [1]])
def test_invalid_domain_rejected(floor):
    with pytest.raises(ValueError):
        radial_contact_spectral_envelope(
            trace(),
            np.ones((2, 3)),
            surface=surface("sphere", "exterior"),
            normal_penalty=1,
            minimum_radial_distance=floor,
        )


def test_geometry_domain_and_rim_rejected():
    s = surface("cylinder", "exterior")
    e = radial_contact_spectral_envelope(
        trace(),
        np.ones((2, 3)),
        surface=s,
        normal_penalty=1,
        minimum_radial_distance=0.5,
    )
    for points in ([[0.4, 0, 0]], [[0, 0, 0]], [[1, 0, 3]]):
        with pytest.raises(ValueError):
            validate_radial_envelope_projection(e, s, s.project(points))
    other = surface("sphere", "exterior")
    with pytest.raises(ValueError):
        validate_radial_envelope_projection(e, other, other.project([[1, 0, 0]]))
    current = bm.ContactTrace(
        point_ids=[1],
        node_ids=[[0]],
        shape_values=[[1]],
        weights=[1],
        measure_configuration="current",
    )
    with pytest.raises(ValueError):
        radial_contact_spectral_envelope(
            current,
            np.ones((1, 3)),
            surface=s,
            normal_penalty=1,
            minimum_radial_distance=0.5,
        )


def test_rigid_motion_and_scaled_inertia_preserve_domain_and_sign():
    s = surface("cylinder", "exterior")
    t = trace()
    mass = np.ones((2, 3))
    e = radial_contact_spectral_envelope(
        t, mass, surface=s, normal_penalty=2, minimum_radial_distance=0.5
    )
    scaled = radial_contact_spectral_envelope(
        t, 9 * mass, surface=s, normal_penalty=2, minimum_radial_distance=0.5
    )
    np.testing.assert_allclose(scaled.positive_rows, e.positive_rows / 9)
    np.testing.assert_allclose(scaled.negative_rows, e.negative_rows / 9)
    assert scaled.negative_rows.max() > 0
    motion = bm.prescribed_rigid_motion(translation=(2.0, 3.0, 0.0))
    p = s.project([[3.0, 3.0, 0.0]], motion=motion, factor=1.0)
    validate_radial_envelope_projection(e, s, p)
    assert e.summary()["whole_system_stability_certificate"] is False


@pytest.mark.parametrize("penalty", [True, -1, 0, np.nan, [1, 2, 3], 1j])
def test_invalid_penalty_rejected(penalty):
    with pytest.raises(ValueError):
        radial_contact_spectral_envelope(
            trace(),
            np.ones((2, 3)),
            surface=surface("sphere", "exterior"),
            normal_penalty=penalty,
            minimum_radial_distance=0.5,
        )
