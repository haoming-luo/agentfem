"""Research-only certified-face aggregation and virtual-work regressions."""

from __future__ import annotations

import numpy as np
import pytest

from agentfem.boundary_models.closed_polyhedron import closed_triangulated_rigid_solid
from agentfem.boundary_models.contact_face_operator_research import (
    assemble_certified_contact_faces,
)
from agentfem.boundary_models.contact_feature_pair_geometry_research import (
    integrate_certified_closed_two_feature_penalty,
    integrate_discovered_partial_closed_two_feature_penalty,
)
from agentfem.boundary_models.contact_feature_pair_research import (
    AffineDistanceFeature,
    integrate_two_affine_feature_distances,
)


def _tool():
    polygon = np.asarray(((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)), dtype=float)
    vertices = np.vstack(
        (
            np.column_stack((polygon, np.zeros(6))),
            np.column_stack((polygon, np.ones(6))),
        )
    )
    top = ((0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5))
    faces = [tuple(reversed(face)) for face in top]
    faces += [tuple(6 + index for index in face) for face in top]
    for first in range(6):
        second = (first + 1) % 6
        faces.extend(((first, second, second + 6), (first, second + 6, first + 6)))
    return closed_triangulated_rigid_solid(
        vertices=vertices,
        triangles=faces,
        reviewed_embedded=True,
    )


_POSITIONS = np.asarray(
    (
        (0.895, 0.875, -0.02),
        (0.905, 0.875, 0.175),
        (0.895, 0.885, 0.16),
        (0.905, 0.885, 0.165),
    )
)
_FACES = np.asarray(((0, 1, 2), (1, 3, 2)))


def _assemble(reference, current, tool):
    first = integrate_discovered_partial_closed_two_feature_penalty(
        reference[_FACES[0]], current[_FACES[0]], tool, penalty=500.0
    )
    second = integrate_certified_closed_two_feature_penalty(
        reference[_FACES[1]], current[_FACES[1]], tool, (0, 12), penalty=500.0
    )
    return assemble_certified_contact_faces(
        (100, 101),
        _FACES,
        (first, second),
        reference_nodal_positions=reference,
        current_nodal_positions=current,
        number_of_nodes=4,
    )


def test_mixed_partial_and_whole_face_operator_has_global_energy_gradient_and_tangent():
    tool = _tool()
    operator = _assemble(_POSITIONS, _POSITIONS, tool)
    assert tuple(operator.face_ids) == (100, 101)
    assert operator.face_activity_kinds == ("partial", "whole")
    assert operator.face_feature_kinds == (("face", "edge"), ("face", "edge"))
    assert operator.active_reference_area > 0.0
    assert operator.potential_energy > 0.0
    assert sum(operator.face_potential_energies) == pytest.approx(
        operator.potential_energy
    )
    assert sum(operator.face_active_reference_areas) == pytest.approx(
        operator.active_reference_area
    )
    np.testing.assert_allclose(
        operator.force_on_structure, -operator.nodal_residual.sum(axis=0)
    )
    np.testing.assert_allclose(operator.force_on_structure, -operator.force_on_tool)
    reference_point = np.asarray((0.5, 0.5, 0.0))
    np.testing.assert_allclose(
        operator.moment_on_structure_about(reference_point),
        operator.moment_on_structure_about_origin
        - np.cross(reference_point, operator.force_on_structure),
    )
    step = 1e-6
    for node in range(4):
        for component in range(3):
            direction = np.zeros((4, 3))
            direction[node, component] = step
            plus = _assemble(_POSITIONS, _POSITIONS + direction, tool)
            minus = _assemble(_POSITIONS, _POSITIONS - direction, tool)
            assert operator.nodal_residual[node, component] == pytest.approx(
                (plus.potential_energy - minus.potential_energy) / (2 * step),
                rel=3e-5,
                abs=3e-7,
            )
            np.testing.assert_allclose(
                operator.tangent_action(direction / step),
                (plus.nodal_residual - minus.nodal_residual) / (2 * step),
                rtol=3e-5,
                atol=3e-7,
            )


def test_face_operator_refuses_uncertified_distance_and_duplicate_face_id():
    tool = _tool()
    first = integrate_discovered_partial_closed_two_feature_penalty(
        _POSITIONS[_FACES[0]], _POSITIONS[_FACES[0]], tool, penalty=500.0
    )
    plane = AffineDistanceFeature(
        np.zeros(3), np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)))
    )
    point = AffineDistanceFeature(np.ones(3), np.empty((3, 0)))
    raw = integrate_two_affine_feature_distances(
        _POSITIONS[_FACES[0]], _POSITIONS[_FACES[0]], plane, point
    )
    with pytest.raises(ValueError, match="certified P1 faces"):
        assemble_certified_contact_faces(
            (1, 2),
            _FACES,
            (first, raw),
            reference_nodal_positions=_POSITIONS,
            current_nodal_positions=_POSITIONS,
            number_of_nodes=4,
        )
    with pytest.raises(ValueError, match="certified P1 faces"):
        assemble_certified_contact_faces(
            (1, 1),
            _FACES,
            (first, first),
            reference_nodal_positions=_POSITIONS,
            current_nodal_positions=_POSITIONS,
            number_of_nodes=4,
        )
    with pytest.raises(ValueError, match="exact integers"):
        assemble_certified_contact_faces(
            (1.1, 2.0),
            _FACES,
            (first, first),
            reference_nodal_positions=_POSITIONS,
            current_nodal_positions=_POSITIONS,
            number_of_nodes=4,
        )
    with pytest.raises(ValueError, match="exact integers"):
        assemble_certified_contact_faces(
            (1, 2),
            _FACES.astype(float),
            (first, first),
            reference_nodal_positions=_POSITIONS,
            current_nodal_positions=_POSITIONS,
            number_of_nodes=4,
        )
    with pytest.raises(ValueError, match="certified P1 faces"):
        assemble_certified_contact_faces(
            (1, 2),
            ((0, 1, 2), (2, 0, 1)),
            (first, first),
            reference_nodal_positions=_POSITIONS,
            current_nodal_positions=_POSITIONS,
            number_of_nodes=4,
        )


def test_face_operator_refuses_mixed_tool_identities():
    tool = _tool()
    translated = closed_triangulated_rigid_solid(
        vertices=tool.facets.vertices + np.asarray((0.0, 0.0, 1e-6)),
        triangles=tool.facets.triangles,
        reviewed_embedded=True,
    )
    first = integrate_discovered_partial_closed_two_feature_penalty(
        _POSITIONS[_FACES[0]], _POSITIONS[_FACES[0]], tool, penalty=500.0
    )
    second = integrate_certified_closed_two_feature_penalty(
        _POSITIONS[_FACES[1]],
        _POSITIONS[_FACES[1]],
        translated,
        (0, 12),
        penalty=500.0,
    )
    with pytest.raises(ValueError, match="different rigid tools"):
        assemble_certified_contact_faces(
            (1, 2),
            _FACES,
            (first, second),
            reference_nodal_positions=_POSITIONS,
            current_nodal_positions=_POSITIONS,
            number_of_nodes=4,
        )


def test_face_operator_refuses_stale_trial_or_mispaired_face_geometry():
    tool = _tool()
    first = integrate_discovered_partial_closed_two_feature_penalty(
        _POSITIONS[_FACES[0]], _POSITIONS[_FACES[0]], tool, penalty=500.0
    )
    second = integrate_certified_closed_two_feature_penalty(
        _POSITIONS[_FACES[1]],
        _POSITIONS[_FACES[1]],
        tool,
        (0, 12),
        penalty=500.0,
    )
    changed = _POSITIONS.copy()
    changed[0, 2] += 1e-5
    with pytest.raises(ValueError, match="paired with different geometry"):
        assemble_certified_contact_faces(
            (1, 2),
            _FACES,
            (first, second),
            reference_nodal_positions=_POSITIONS,
            current_nodal_positions=changed,
            number_of_nodes=4,
        )
    with pytest.raises(ValueError, match="paired with different geometry"):
        assemble_certified_contact_faces(
            (1, 2),
            _FACES[[1, 0]],
            (first, second),
            reference_nodal_positions=_POSITIONS,
            current_nodal_positions=_POSITIONS,
            number_of_nodes=4,
        )


def test_face_operator_tangent_commutes_with_a_bounded_mpc_reduction():
    tool = _tool()
    operator = _assemble(_POSITIONS, _POSITIONS, tool)
    full_tangent = np.zeros((12, 12))
    np.add.at(
        full_tangent,
        (operator.tangent_rows, operator.tangent_columns),
        operator.tangent_values,
    )
    transform = np.eye(12)[:, np.arange(12) != 9]
    transform[9] = 0.0
    transform[9, 6] = 1.0  # tied x displacement of node 3 to node 2
    reduced = transform.T @ full_tangent @ transform
    step = 1e-6
    for reduced_column in (0, 6, 10):
        direction = (transform[:, reduced_column] * step).reshape(4, 3)
        plus = _assemble(_POSITIONS, _POSITIONS + direction, tool)
        minus = _assemble(_POSITIONS, _POSITIONS - direction, tool)
        numerical = transform.T @ (
            (plus.nodal_residual - minus.nodal_residual) / (2 * step)
        ).reshape(-1)
        np.testing.assert_allclose(
            reduced[:, reduced_column],
            numerical,
            rtol=3e-5,
            atol=3e-7,
        )
