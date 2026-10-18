# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Analytical circular-patch checks for the private radial integration rule."""

import numpy as np
import pytest

from agentfem.boundary_models.contact_radial_partition import (
    _minimum_radial_distance,
    _radial_local_response,
    partition_radial_p1_trace,
    radial_active_triangle_quadrature,
)
from agentfem.boundary_models.contact_trace import ContactTrace
from agentfem.boundary_models.initial_gap_bound import (
    certify_initial_radial_triangle_gap,
)
from agentfem.boundary_models.rigid import rigid_cylinder, rigid_sphere


def _equilateral_face(height):
    angles = 2.0 * np.pi * np.arange(3) / 3.0
    return np.column_stack(
        (1.2 * np.cos(angles), 1.2 * np.sin(angles), np.full(3, height))
    )


def _sphere_response(height, *, order=24, penalty=7.0):
    face = _equilateral_face(height)
    shapes, reference_weights = radial_active_triangle_quadrature(
        face,
        center=np.zeros(3),
        radius=1.0,
        order=order,
    )
    physical_area = 0.5 * np.linalg.norm(np.cross(face[1] - face[0], face[2] - face[0]))
    weights = 2.0 * physical_area * reference_weights
    query = shapes @ face
    distances = np.linalg.norm(query, axis=1)
    gaps = distances - 1.0
    normals_z = query[:, 2] / distances
    hessian_zz = (1.0 - normals_z**2) / distances
    potential = 0.5 * penalty * np.dot(weights, gaps**2)
    force = penalty * np.dot(weights, gaps * normals_z)
    tangent = penalty * np.dot(weights, normals_z**2 + gaps * hessian_zz)
    return float(potential), float(force), float(tangent), float(weights.sum())


def test_small_circular_patch_is_not_present_at_any_default_gauss_point():
    face = _equilateral_face(0.95)
    default_shapes = np.array(
        ((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3))
    )
    assert np.all(np.linalg.norm(default_shapes @ face, axis=1) > 1.0)
    potential, force, tangent, area = _sphere_response(0.95)
    assert potential > 0.0 and force < 0.0 and tangent > 0.0 and area > 0.0


def test_quarter_circular_patch_at_slave_vertex_matches_exact_area():
    height = 0.95
    face = np.asarray(((0.0, 0.0, height), (1.0, 0.0, height), (0.0, 1.0, height)))
    shapes, weights = radial_active_triangle_quadrature(
        face, center=(0.0, 0.0, 0.0), radius=1.0, order=24
    )
    radius_squared = 1.0 - height**2
    assert 2.0 * 0.5 * weights.sum() == pytest.approx(
        0.25 * np.pi * radius_squared, rel=5.0e-8
    )
    assert np.all(shapes >= 0.0)
    assert np.all(weights > 0.0)


def test_circle_clipped_by_slave_edge_matches_exact_segment_area():
    height = 0.6
    face = np.asarray(((0.0, 0.0, height), (1.0, 0.0, height), (0.0, 1.0, height)))
    _shapes, weights = radial_active_triangle_quadrature(
        face, center=(0.0, 0.0, 0.0), radius=1.0, order=24
    )
    disk_radius = 0.8
    diagonal_distance = 1.0 / np.sqrt(2.0)
    circular_cap = disk_radius**2 * np.arccos(diagonal_distance / disk_radius)
    circular_cap -= diagonal_distance * np.sqrt(disk_radius**2 - diagonal_distance**2)
    expected_area = 0.25 * np.pi * disk_radius**2 - circular_cap
    assert weights.sum() == pytest.approx(expected_area, rel=5.0e-8)


@pytest.mark.parametrize("height", [0.8, 0.95, 0.99])
def test_circular_patch_area_potential_force_and_tangent(height):
    penalty = 7.0
    area_expected = np.pi * (1.0 - height**2)
    force_expected = -np.pi * penalty * height * (1.0 - height) ** 2
    tangent_expected = -np.pi * penalty * (1.0 - height) * (1.0 - 3.0 * height)
    primitive = lambda value: value**4 / 4.0 - 2.0 * value**3 / 3.0 + value**2 / 2.0
    potential_expected = np.pi * penalty * (primitive(1.0) - primitive(height))
    potential, force, tangent, area = _sphere_response(height)
    assert area == pytest.approx(area_expected, rel=5e-8)
    assert potential == pytest.approx(potential_expected, rel=5e-7)
    assert force == pytest.approx(force_expected, rel=5e-7)
    assert tangent == pytest.approx(tangent_expected, rel=5e-6)


def test_radial_partition_rejects_bad_contract_and_keeps_positive_weights():
    face = _equilateral_face(0.95)
    for axis in (None, np.array((0.0, 0.0, 1.0))):
        shapes, weights = radial_active_triangle_quadrature(
            face,
            center=np.zeros(3),
            radius=1.0,
            axis=axis,
            order=12,
        )
        assert len(shapes) > 0
        assert np.all(weights > 0.0)
        assert np.all(weights < 0.5)
        assert np.all(shapes >= -1.0e-12)
        assert np.allclose(shapes.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="unit cylinder axis"):
        radial_active_triangle_quadrature(
            face,
            center=(0.0, 0.0, 0.0),
            radius=1.0,
            axis=(0.0, 0.0, 2.0),
        )
    with pytest.raises(ValueError, match="Gauss order"):
        radial_active_triangle_quadrature(
            face,
            center=(0.0, 0.0, 0.0),
            radius=1.0,
            order=1,
        )
    with pytest.raises(NotImplementedError, match="3D surface"):
        partition_radial_p1_trace(
            _default_trace(face),
            face,
            rigid_sphere(center=(0.0, 0.0), radius=1.0),
            penalty=7.0,
        )


def test_full_nodal_sphere_contact_tangent_at_nonzero_trial():
    reference = _equilateral_face(0.95)
    trial = reference + np.array(
        ((0.02, 0.01, 0.00), (0.00, -0.02, 0.01), (-0.01, 0.00, -0.01))
    )
    reference_area = 0.5 * np.linalg.norm(
        np.cross(
            reference[1] - reference[0],
            reference[2] - reference[0],
        )
    )

    def response(positions):
        shapes, unit_weights = radial_active_triangle_quadrature(
            positions,
            center=(0.0, 0.0, 0.0),
            radius=1.0,
            order=24,
        )
        weights = 2.0 * reference_area * unit_weights
        points = shapes @ positions
        lengths = np.linalg.norm(points, axis=1)
        gaps = lengths - 1.0
        normals = points / lengths[:, None]
        hessian = (
            np.eye(3)[None, :, :] - normals[:, :, None] * normals[:, None, :]
        ) / lengths[:, None, None]
        residual = np.einsum(
            "q,qa,qi->ai",
            7.0 * weights * gaps,
            shapes,
            normals,
        ).reshape(-1)
        point_tangent = (
            normals[:, :, None] * normals[:, None, :] + gaps[:, None, None] * hessian
        )
        tangent = np.einsum(
            "q,qa,qb,qij->aibj",
            7.0 * weights,
            shapes,
            shapes,
            point_tangent,
        ).reshape(9, 9)
        return residual, tangent

    _, tangent = response(trial)
    numerical = np.empty((9, 9))
    step = 1.0e-5
    for column in range(9):
        direction = np.zeros((3, 3))
        direction.reshape(-1)[column] = step
        positive, _ = response(trial + direction)
        negative, _ = response(trial - direction)
        numerical[:, column] = (positive - negative) / (2.0 * step)
    np.testing.assert_allclose(tangent, numerical, rtol=2e-5, atol=2e-5)


def _default_trace(face):
    area = 0.5 * np.linalg.norm(np.cross(face[1] - face[0], face[2] - face[0]))
    return ContactTrace(
        point_ids=(0, 1, 2),
        node_ids=np.tile(np.arange(3), (3, 1)),
        shape_values=(
            (2 / 3, 1 / 6, 1 / 6),
            (1 / 6, 2 / 3, 1 / 6),
            (1 / 6, 1 / 6, 2 / 3),
        ),
        weights=(area / 3,) * 3,
    )


def test_bounded_trace_finds_missed_sphere_patch_and_rejects_axis(monkeypatch):
    face = _equilateral_face(0.95)
    trace = _default_trace(face)
    sphere = rigid_sphere(center=(0.0, 0.0, 0.0), radius=1.0)
    selected = partition_radial_p1_trace(trace, face, sphere, penalty=7.0)
    assert selected.point_count > trace.point_count
    assert np.all(
        sphere.project(selected.evaluate(face).query_points).signed_gaps < 0.0
    )
    from agentfem.boundary_models import contact_radial_partition as radial_module

    with monkeypatch.context() as disabled:
        disabled.setattr(
            radial_module,
            "radial_active_triangle_quadrature",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("An exactly open face must not build trial quadrature.")
            ),
        )
        assert trace is partition_radial_p1_trace(
            trace,
            face + (0.0, 0.0, 1.0),
            sphere,
            penalty=7.0,
        )
    cylinder = rigid_cylinder(
        axis_point=(0.0, 0.0, 0.0),
        axis_direction=(0.0, 0.0, 1.0),
        radius=1.0,
    )
    with pytest.raises(
        NotImplementedError, match="undefined sphere center or cylinder axis"
    ):
        partition_radial_p1_trace(trace, face, cylinder, penalty=7.0)


def test_full_nodal_cylinder_strip_tangent_at_nonzero_trial():
    reference = _equilateral_face(0.95)
    trial = reference + np.array(
        ((0.01, 0.00, 0.00), (0.00, -0.01, 0.01), (-0.01, 0.00, -0.01))
    )
    reference_area = 0.5 * np.linalg.norm(
        np.cross(
            reference[1] - reference[0],
            reference[2] - reference[0],
        )
    )
    axis = np.asarray((0.0, 1.0, 0.0))

    def response(positions):
        shapes, weights = radial_active_triangle_quadrature(
            positions,
            center=(0.0, 0.0, 0.0),
            radius=1.0,
            axis=axis,
            order=24,
        )
        return _radial_local_response(
            shapes,
            weights,
            positions,
            reference_area,
            center=np.zeros(3),
            axis=axis,
            radius=1.0,
            side=1.0,
            penalty=7.0,
        )

    baseline = response(trial)
    tangent = baseline[10:].reshape((9, 9))
    numerical = np.empty((9, 9))
    step = 1.0e-5
    for column in range(9):
        direction = np.zeros((3, 3))
        direction.reshape(-1)[column] = step
        numerical[:, column] = (
            response(trial + direction)[1:10] - response(trial - direction)[1:10]
        ) / (2.0 * step)
    np.testing.assert_allclose(tangent, numerical, rtol=2e-5, atol=2e-5)


def test_exact_initial_radial_extrema_catch_unsampled_overlap():
    face = _equilateral_face(0.95)
    sphere = rigid_sphere(center=(0.0, 0.0, 0.0), radius=1.0)
    bounded = certify_initial_radial_triangle_gap(
        face[None, :, :],
        sphere,
        penetration_limit=0.051,
    )
    assert bounded.maximum_penetration_bound == pytest.approx(0.05, abs=1e-12)
    with pytest.raises(ValueError, match="AFM-CONTACT-INITIAL-OVERLAP"):
        certify_initial_radial_triangle_gap(
            face[None, :, :],
            sphere,
            penetration_limit=0.049,
        )
    cylinder = rigid_cylinder(
        axis_point=(0.0, 0.0, 0.0),
        axis_direction=(0.0, 1.0, 0.0),
        radius=1.0,
        axial_interval=(-2.0, 2.0),
    )
    assert certify_initial_radial_triangle_gap(
        face[None, :, :],
        cylinder,
        penetration_limit=0.051,
    ).maximum_penetration_bound == pytest.approx(0.05, abs=1e-12)
    rim = rigid_cylinder(
        axis_point=(0.0, 0.0, 0.0),
        axis_direction=(0.0, 1.0, 0.0),
        radius=1.0,
        axial_interval=(-0.5, 0.5),
    )
    with pytest.raises(NotImplementedError, match="unsupported cylinder rim"):
        certify_initial_radial_triangle_gap(
            face[None, :, :],
            rim,
            penetration_limit=0.051,
        )


def test_radial_face_minimum_never_exceeds_dense_independent_samples():
    random = np.random.default_rng(20261019)
    grid = np.asarray(
        [
            (1.0 - a / 24.0 - b / 24.0, a / 24.0, b / 24.0)
            for a in range(25)
            for b in range(25 - a)
        ]
    )
    for _ in range(100):
        vertices = random.normal(size=(3, 3))
        for axis in (None, np.asarray((0.0, 0.0, 1.0))):
            projector = np.eye(3) if axis is None else np.eye(3) - np.outer(axis, axis)
            relative = vertices @ projector
            minimum = _minimum_radial_distance(relative)
            sampled = np.linalg.norm(grid @ relative, axis=1)
            assert minimum <= np.min(sampled) + 5.0e-13
            assert np.max(np.linalg.norm(relative, axis=1)) >= np.max(sampled) - 5.0e-13


def test_interior_sphere_uses_complement_of_circular_patch():
    face = _equilateral_face(0.95)
    shape, weights = radial_active_triangle_quadrature(
        face,
        center=(0.0, 0.0, 0.0),
        radius=1.0,
        admissible_side="interior",
        order=24,
    )
    physical_area = 0.5 * np.linalg.norm(
        np.cross(
            face[1] - face[0],
            face[2] - face[0],
        )
    )
    outside_area = 2.0 * physical_area * weights.sum()
    assert outside_area == pytest.approx(
        physical_area - np.pi * (1.0 - 0.95**2),
        rel=5e-8,
    )
    assert np.all(np.linalg.norm(shape @ face, axis=1) > 1.0)
    selected = partition_radial_p1_trace(
        _default_trace(face),
        face,
        rigid_sphere(center=(0.0, 0.0, 0.0), radius=1.0, admissible_side="interior"),
        penalty=7.0,
    )
    assert selected.point_count > 3


def test_quadratic_active_region_is_scale_and_rigid_frame_invariant():
    face = _equilateral_face(0.95)
    baseline_shapes, baseline_weights = radial_active_triangle_quadrature(
        face, center=(0.0, 0.0, 0.0), radius=1.0, order=24
    )
    for scale in (1.0e-6, 1.0e-3, 1.0e3):
        shapes, weights = radial_active_triangle_quadrature(
            scale * face, center=(0.0, 0.0, 0.0), radius=scale, order=24
        )
        np.testing.assert_allclose(shapes, baseline_shapes, atol=2.0e-12)
        np.testing.assert_allclose(weights, baseline_weights, atol=2.0e-12)
    angle = 0.39
    rotation = np.asarray(
        (
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        )
    )
    center = np.asarray((2.0, -0.3, 0.5))
    shapes, weights = radial_active_triangle_quadrature(
        face @ rotation.T + center, center=center, radius=1.0, order=24
    )
    np.testing.assert_allclose(shapes, baseline_shapes, atol=2.0e-12)
    np.testing.assert_allclose(weights, baseline_weights, atol=2.0e-12)
