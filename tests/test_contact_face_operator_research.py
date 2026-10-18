"""Research-only certified-face aggregation and virtual-work regressions."""

from __future__ import annotations

import json

import numpy as np
import pytest

from agentfem.boundary_models.closed_polyhedron import closed_triangulated_rigid_solid
from agentfem.boundary_models.contact_face_operator_research import (
    assemble_certified_contact_faces,
    evaluate_certified_closed_p1_contact_operator,
    evaluate_moving_certified_closed_p1_contact_operator,
    extract_complete_p1_face_layout,
)
from agentfem.boundary_models.contact_feature_pair_geometry_research import (
    certify_separated_closed_contact_face,
    integrate_certified_closed_two_feature_penalty,
    integrate_discovered_closed_contact_face,
    integrate_discovered_partial_closed_two_feature_penalty,
)
from agentfem.boundary_models.contact_feature_pair_research import (
    AffineDistanceFeature,
    integrate_two_affine_feature_distances,
)
from agentfem.boundary_models.contact_trace import ContactTrace
from agentfem.boundary_models.rigid import prescribed_rigid_motion


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


def _positive_face_trace(positions, faces, identities):
    vertices = positions[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
        axis=1,
    )
    shapes = np.asarray(
        ((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3))
    )
    return ContactTrace(
        point_ids=(3 * np.asarray(identities)[:, None] + np.arange(3)).reshape(-1),
        node_ids=np.repeat(faces, 3, axis=0),
        shape_values=np.tile(shapes, (len(faces), 1)),
        weights=np.repeat(area / 3, 3),
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
    np.testing.assert_allclose(
        np.sum(operator.face_forces_on_structure, axis=0),
        operator.force_on_structure,
    )
    np.testing.assert_allclose(
        np.sum(operator.face_moments_on_structure_about_origin, axis=0),
        operator.moment_on_structure_about_origin,
    )
    assert not operator.face_forces_on_structure.flags.writeable
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
    np.testing.assert_allclose(full_tangent, full_tangent.T, rtol=1e-12, atol=1e-10)
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


def test_face_operator_is_covariant_under_a_common_rigid_frame_change():
    tool = _tool()
    baseline = _assemble(_POSITIONS, _POSITIONS, tool)
    axis = np.asarray((0.3, -0.4, 0.5))
    axis /= np.linalg.norm(axis)
    angle = 0.37
    skew = np.asarray(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0))
    )
    rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew
    shift = np.asarray((1.2, -0.4, 0.8))

    def moved(points):
        return np.asarray(points) @ rotation.T + shift

    transformed_tool = closed_triangulated_rigid_solid(
        vertices=moved(tool.facets.vertices),
        triangles=tool.facets.triangles,
        reviewed_embedded=True,
    )
    transformed = _assemble(moved(_POSITIONS), moved(_POSITIONS), transformed_tool)
    assert transformed.potential_energy == pytest.approx(baseline.potential_energy)
    np.testing.assert_allclose(
        transformed.nodal_residual,
        baseline.nodal_residual @ rotation.T,
        rtol=1e-10,
        atol=1e-10,
    )
    direction = np.arange(12, dtype=float).reshape(4, 3) / 12
    np.testing.assert_allclose(
        transformed.tangent_action(direction @ rotation.T),
        baseline.tangent_action(direction) @ rotation.T,
        rtol=1e-9,
        atol=1e-8,
    )
    reference = np.asarray((0.5, 0.7, -0.2))
    np.testing.assert_allclose(
        transformed.moment_on_structure_about(moved(reference)),
        baseline.moment_on_structure_about(reference) @ rotation.T,
        rtol=1e-9,
        atol=1e-9,
    )


def test_complete_face_batch_keeps_proven_zero_face_without_fake_point_samples():
    tool = _tool()
    distant = np.asarray(((3.0, 3.0, 3.0), (3.1, 3.0, 3.0), (3.0, 3.1, 3.0)))
    positions = np.vstack((_POSITIONS, distant))
    faces = np.vstack((_FACES, np.asarray((4, 5, 6))))
    partial = integrate_discovered_partial_closed_two_feature_penalty(
        positions[faces[0]], positions[faces[0]], tool, penalty=500.0
    )
    whole = integrate_certified_closed_two_feature_penalty(
        positions[faces[1]], positions[faces[1]], tool, (0, 12), penalty=500.0
    )
    separated = certify_separated_closed_contact_face(
        positions[faces[2]], positions[faces[2]], tool
    )
    operator = assemble_certified_contact_faces(
        (100, 101, 102),
        faces,
        (partial, whole, separated),
        reference_nodal_positions=positions,
        current_nodal_positions=positions,
        number_of_nodes=7,
    )
    assert operator.face_activity_kinds == ("partial", "whole", "separated")
    assert operator.face_feature_kinds[-1] is None
    assert operator.face_separation_lower_bounds[:2] == (None, None)
    assert operator.face_separation_lower_bounds[-1] > 0.0
    assert operator.face_potential_energies[-1] == 0.0
    assert operator.face_active_reference_areas[-1] == 0.0
    np.testing.assert_array_equal(operator.face_forces_on_structure[-1], 0.0)
    np.testing.assert_array_equal(
        operator.face_moments_on_structure_about_origin[-1], 0.0
    )
    assert len(operator.tangent_values) == 162
    np.testing.assert_array_equal(operator.nodal_residual[4:], 0.0)
    assert separated.separation.minimum_gap_lower_bound > 0.0
    moved = positions.copy()
    moved[4:, 0] -= 2.5
    with pytest.raises(ValueError, match="paired with different geometry"):
        assemble_certified_contact_faces(
            (100, 101, 102),
            faces,
            (partial, whole, separated),
            reference_nodal_positions=positions,
            current_nodal_positions=moved,
            number_of_nodes=7,
        )


def test_unproven_zero_face_is_rejected_before_operator_assembly():
    tool = _tool()
    with pytest.raises(NotImplementedError, match="not proved separated"):
        certify_separated_closed_contact_face(
            _POSITIONS[_FACES[0]], _POSITIONS[_FACES[0]], tool
        )


def test_all_separated_faces_produce_an_exact_zero_operator():
    tool = _tool()
    positions = np.asarray(((3.0, 3.0, 3.0), (3.1, 3.0, 3.0), (3.0, 3.1, 3.0)))
    separated = certify_separated_closed_contact_face(positions, positions, tool)
    operator = assemble_certified_contact_faces(
        (17,),
        ((0, 1, 2),),
        (separated,),
        reference_nodal_positions=positions,
        current_nodal_positions=positions,
        number_of_nodes=3,
    )
    assert operator.potential_energy == 0.0
    assert operator.active_reference_area == 0.0
    assert len(operator.tangent_values) == 0
    np.testing.assert_array_equal(operator.nodal_residual, 0.0)
    np.testing.assert_array_equal(operator.tangent_action(np.ones((3, 3))), 0.0)


def test_one_generic_discovery_route_certifies_partial_whole_and_separated_faces():
    tool = _tool()
    distant = np.asarray(((3.0, 3.0, 3.0), (3.1, 3.0, 3.0), (3.0, 3.1, 3.0)))
    positions = np.vstack((_POSITIONS, distant))
    faces = np.vstack((_FACES, np.asarray((4, 5, 6))))
    responses = tuple(
        integrate_discovered_closed_contact_face(
            positions[nodes], positions[nodes], tool, penalty=500.0
        )
        for nodes in faces
    )
    operator = assemble_certified_contact_faces(
        (100, 101, 102),
        faces,
        responses,
        reference_nodal_positions=positions,
        current_nodal_positions=positions,
        number_of_nodes=7,
    )
    assert operator.face_activity_kinds == ("partial", "whole", "separated")
    reference = np.vstack((_POSITIONS, distant))
    changed = positions.copy()
    changed[0, 2] += 1e-6
    reevaluated = tuple(
        integrate_discovered_closed_contact_face(
            reference[nodes], changed[nodes], tool, penalty=500.0
        )
        for nodes in faces
    )
    next_operator = assemble_certified_contact_faces(
        (100, 101, 102),
        faces,
        reevaluated,
        reference_nodal_positions=reference,
        current_nodal_positions=changed,
        number_of_nodes=7,
    )
    assert next_operator.potential_energy != operator.potential_energy
    complete = evaluate_certified_closed_p1_contact_operator(
        _positive_face_trace(reference, faces, (100, 101, 102)),
        reference,
        changed,
        tool,
        penalty=500.0,
    )
    np.testing.assert_allclose(complete.nodal_residual, next_operator.nodal_residual)
    np.testing.assert_allclose(complete.tangent_values, next_operator.tangent_values)
    assert complete.potential_energy == pytest.approx(next_operator.potential_energy)


def test_discovery_refuses_a_face_crossing_two_tool_activation_planes():
    tool = _tool()
    face = _POSITIONS[_FACES[0]].copy()
    face[1, 2] = 1.2
    with pytest.raises(NotImplementedError):
        integrate_discovered_closed_contact_face(face, face, tool, penalty=500.0)
    with pytest.raises(NotImplementedError, match="slave_face_id=42"):
        evaluate_certified_closed_p1_contact_operator(
            _positive_face_trace(face, np.asarray(((0, 1, 2),)), (42,)),
            face,
            face,
            tool,
            penalty=500.0,
        )


def test_complete_trial_rejects_collapsed_current_face_before_contact_search():
    tool = _tool()
    distant = np.asarray(((3.0, 3.0, 3.0), (3.1, 3.0, 3.0), (3.0, 3.1, 3.0)))
    current = distant.copy()
    current[2] = current[1]
    with pytest.raises(ValueError, match="collapsed current P1 slave_face_id=42"):
        evaluate_certified_closed_p1_contact_operator(
            _positive_face_trace(distant, np.asarray(((0, 1, 2),)), (42,)),
            distant,
            current,
            tool,
            penalty=500.0,
        )


def test_face_layout_requires_all_three_positive_reference_points_per_face():
    points = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    shapes = np.asarray(
        ((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3))
    )
    trace = ContactTrace(
        point_ids=(51, 52, 53),
        node_ids=np.tile(np.arange(3), (3, 1)),
        shape_values=shapes,
        weights=(1 / 6, 1 / 6, 1 / 6),
    )
    layout = extract_complete_p1_face_layout(trace, points)
    assert tuple(layout.face_ids) == (17,)
    np.testing.assert_array_equal(layout.face_node_ids, ((0, 1, 2),))
    with pytest.raises(ValueError, match="complete P1 faces"):
        extract_complete_p1_face_layout(
            ContactTrace(
                point_ids=(51, 52, 54),
                node_ids=trace.node_ids,
                shape_values=trace.shape_values,
                weights=trace.weights,
            ),
            points,
        )
    with pytest.raises(ValueError, match="positive reference area"):
        extract_complete_p1_face_layout(
            ContactTrace(
                point_ids=trace.point_ids,
                node_ids=trace.node_ids,
                shape_values=trace.shape_values,
                weights=(1 / 12, 1 / 12, 1 / 12),
            ),
            points,
        )


def test_real_dolfinx_owned_tetra_faces_preserve_complete_face_layout():
    import basix.ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI

    from agentfem import boundary_models, mesh as agent_mesh

    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    space = fem.functionspace(
        domain,
        basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,)),
    )
    dim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, dim, lambda x: np.isclose(x[0], 0.0))
    tags = mesh.meshtags(domain, dim, facets, np.full(len(facets), 17, dtype=np.int32))
    region = agent_mesh.tagged_boundary_region(domain, tags, tag=17, name="left_slave")
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(region, space)
    layout = extract_complete_p1_face_layout(
        adapter.trace, adapter.reference_nodal_positions
    )
    assert len(layout.face_ids) == 2
    assert layout.number_of_nodes == len(adapter.reference_nodal_positions)
    assert sum(
        0.5
        * np.linalg.norm(
            np.cross(
                adapter.reference_nodal_positions[nodes[1]]
                - adapter.reference_nodal_positions[nodes[0]],
                adapter.reference_nodal_positions[nodes[2]]
                - adapter.reference_nodal_positions[nodes[0]],
            )
        )
        for nodes in layout.face_node_ids
    ) == pytest.approx(1.0)
    source_tool = _tool()
    distant_tool = closed_triangulated_rigid_solid(
        vertices=source_tool.facets.vertices + 10.0,
        triangles=source_tool.facets.triangles,
        reviewed_embedded=True,
    )
    reference = adapter.reference_nodal_positions
    displacement = fem.Function(space)
    current = adapter.current_nodal_positions(displacement)
    responses = tuple(
        certify_separated_closed_contact_face(
            reference[nodes], current[nodes], distant_tool
        )
        for nodes in layout.face_node_ids
    )
    operator = assemble_certified_contact_faces(
        layout.face_ids,
        layout.face_node_ids,
        responses,
        reference_nodal_positions=reference,
        current_nodal_positions=current,
        number_of_nodes=layout.number_of_nodes,
    )
    assert operator.face_activity_kinds == ("separated", "separated")
    np.testing.assert_array_equal(operator.nodal_residual, 0.0)
    direct = evaluate_certified_closed_p1_contact_operator(
        adapter.trace,
        reference,
        current,
        distant_tool,
        penalty=500.0,
    )
    np.testing.assert_array_equal(direct.nodal_residual, operator.nodal_residual)
    np.testing.assert_array_equal(direct.face_ids, operator.face_ids)


def _dolfinx_mixed_active_patch():
    import basix.ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI
    import ufl

    from agentfem import boundary_models, mesh as agent_mesh

    coordinate_element = ufl.Mesh(
        basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    )
    domain = agent_mesh.from_arrays(
        cells=((0, 2, 1, 3),),
        coordinates=_POSITIONS,
        coordinate_element=coordinate_element,
        comm=MPI.COMM_SELF,
    )
    space = fem.functionspace(
        domain,
        basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,)),
    )
    dim = domain.topology.dim - 1
    domain.topology.create_entities(dim)
    domain.topology.create_connectivity(dim, domain.topology.dim)
    domain.topology.create_entity_permutations()
    facet_ids = np.arange(domain.topology.index_map(dim).size_local, dtype=np.int32)
    geometry = mesh.entities_to_geometry(domain, dim, facet_ids, permute=True)
    selected = []
    for facet_id, node_ids in zip(facet_ids, geometry, strict=True):
        vertices = domain.geometry.x[node_ids]
        if any(
            all(
                np.min(np.linalg.norm(vertices - vertex, axis=1)) < 1e-12
                for vertex in _POSITIONS[face]
            )
            for face in _FACES
        ):
            selected.append(int(facet_id))
    assert len(selected) == 2
    tags = mesh.meshtags(
        domain,
        dim,
        np.asarray(selected, dtype=np.int32),
        np.full(2, 17, dtype=np.int32),
    )
    region = agent_mesh.tagged_boundary_region(
        domain, tags, tag=17, name="mixed_reentrant_patch"
    )
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(region, space)
    return domain, space, adapter, region


def test_real_dolfinx_mixed_active_patch_lowers_to_the_certified_face_operator():
    from dolfinx import fem

    _, space, adapter, _ = _dolfinx_mixed_active_patch()
    reference = adapter.reference_nodal_positions
    current = adapter.current_nodal_positions(fem.Function(space))
    operator = evaluate_certified_closed_p1_contact_operator(
        adapter.trace, reference, current, _tool(), penalty=500.0
    )
    assert set(operator.face_activity_kinds) == {"partial", "whole"}
    assert operator.potential_energy == pytest.approx(
        _assemble(_POSITIONS, _POSITIONS, _tool()).potential_energy,
        rel=1e-10,
    )


def test_real_elastic_tetra_matrix_and_certified_face_operator_share_newton_rows():
    from dolfinx import fem
    from petsc4py import PETSc
    import ufl

    from agentfem.boundary_models._petsc_contact_rows import (
        add_contact_coo_tangent,
        add_contact_nodal_residual,
    )

    domain, space, adapter, _ = _dolfinx_mixed_active_patch()
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    young, poisson = 1.0e5, 0.3
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    strain = lambda field: ufl.sym(ufl.grad(field))
    stress = 2 * mu * strain(trial) + lam * ufl.tr(strain(trial)) * ufl.Identity(3)
    matrix = fem.petsc.assemble_matrix(
        fem.form(ufl.inner(stress, strain(test)) * ufl.dx(domain=domain))
    )
    matrix.assemble()
    size = 3 * len(adapter.reference_nodal_positions)
    dofs = np.arange(size, dtype=np.int32)
    bulk = np.asarray(matrix.getValues(dofs, dofs))
    tool = _tool()
    trace = adapter.trace
    reference = adapter.reference_nodal_positions

    def contact(values):
        return evaluate_certified_closed_p1_contact_operator(
            trace,
            reference,
            reference + values.reshape(-1, 3),
            tool,
            penalty=500.0,
        )

    fixed = np.asarray((0, 1, 2, 4, 9, 10, 11), dtype=np.int64)
    free = np.asarray([index for index in range(size) if index not in fixed])
    assert np.linalg.eigvalsh(bulk[np.ix_(free, free)])[0] > 0.0
    target = np.zeros(size)
    target[free] = np.asarray((1.0, -1.0, 0.5, 0.8, -0.6)) * 1e-6
    exact = contact(target)
    load = bulk @ target + exact.nodal_residual.reshape(-1)
    constrained = matrix.copy()
    constrained.zeroRowsColumns(fixed.astype(PETSc.IntType), diag=1.0)
    add_contact_coo_tangent(
        constrained,
        exact.tangent_rows,
        exact.tangent_columns,
        exact.tangent_values,
        constrained_dofs=fixed,
    )
    constrained.assemble()
    inserted = np.asarray(constrained.getValues(dofs, dofs))
    expected = bulk.copy()
    np.add.at(
        expected,
        (exact.tangent_rows, exact.tangent_columns),
        exact.tangent_values,
    )
    np.testing.assert_allclose(
        inserted[np.ix_(free, free)], expected[np.ix_(free, free)]
    )
    np.testing.assert_array_equal(inserted[np.ix_(fixed, free)], 0.0)
    np.testing.assert_array_equal(inserted[np.ix_(free, fixed)], 0.0)
    vector = matrix.createVecRight()
    vector.set(0.0)
    add_contact_nodal_residual(vector, exact.nodal_residual)
    np.testing.assert_allclose(vector.array, exact.nodal_residual.reshape(-1))
    iterate = np.zeros(size)
    for _ in range(8):
        response = contact(iterate)
        residual = bulk @ iterate + response.nodal_residual.reshape(-1) - load
        if np.linalg.norm(residual[free]) < 1e-11:
            break
        tangent = bulk.copy()
        np.add.at(
            tangent,
            (response.tangent_rows, response.tangent_columns),
            response.tangent_values,
        )
        step = np.linalg.solve(tangent[np.ix_(free, free)], -residual[free])
        iterate[free] += step
    np.testing.assert_allclose(iterate[free], target[free], rtol=1e-6, atol=1e-11)
    assert (
        np.linalg.norm(
            (bulk @ iterate + contact(iterate).nodal_residual.reshape(-1) - load)[free]
        )
        < 1e-11
    )


def test_contact_petsc_rows_reject_invalid_input_before_matrix_mutation():
    from mpi4py import MPI
    from petsc4py import PETSc

    from agentfem.boundary_models._petsc_contact_rows import (
        add_contact_coo_tangent,
        add_contact_nodal_residual,
    )

    matrix = PETSc.Mat().createAIJ((3, 3), comm=MPI.COMM_SELF)
    matrix.setUp()
    matrix.setDiagonal(PETSc.Vec().createWithArray(np.ones(3), comm=MPI.COMM_SELF))
    matrix.assemble()
    dofs = np.arange(3, dtype=PETSc.IntType)
    initial = np.asarray(matrix.getValues(dofs, dofs)).copy()
    with pytest.raises(ValueError, match="finite serial COO"):
        add_contact_coo_tangent(matrix, (0,), (1,), (float("nan"),))
    with pytest.raises(ValueError, match="Constrained contact DOFs"):
        add_contact_coo_tangent(matrix, (0,), (1,), (1.0,), constrained_dofs=(3,))
    with pytest.raises(ValueError, match="identities must be integers"):
        add_contact_coo_tangent(matrix, (0.5,), (1,), (1.0,))
    np.testing.assert_array_equal(matrix.getValues(dofs, dofs), initial)
    vector = matrix.createVecRight()
    vector.set(0.0)
    with pytest.raises(ValueError, match="finite PETSc local rows"):
        add_contact_nodal_residual(vector, ((float("nan"), 0.0, 0.0),))
    np.testing.assert_array_equal(vector.array, 0.0)


def test_real_dolfinx_certified_face_trial_provider_rejects_stale_displacement():
    from types import SimpleNamespace

    from dolfinx import fem
    from petsc4py import PETSc
    import ufl

    from agentfem.boundary_models._certified_face_trial_provider import (
        DolfinxCertifiedFaceContactTrialProvider,
    )
    from agentfem.boundary_models.contact_pair import rigid_contact_pair
    from agentfem.boundary_models.rigid_body import rigid_body

    domain, space, adapter, region = _dolfinx_mixed_active_patch()
    displacement = fem.Function(space)
    pair = rigid_contact_pair(
        region, rigid_body(_tool()), penalty=500.0, name="reviewed_tool"
    )
    provider = DolfinxCertifiedFaceContactTrialProvider(pair, adapter, displacement)
    trial = provider.evaluate()
    assert trial.operator.face_activity_kinds == ("partial", "whole")
    size = 3 * len(adapter.reference_nodal_positions)
    trial_function, test_function = ufl.TrialFunction(space), ufl.TestFunction(space)
    matrix = fem.petsc.assemble_matrix(
        fem.form(
            ufl.inner(
                ufl.sym(ufl.grad(trial_function)),
                ufl.sym(ufl.grad(test_function)),
            )
            * ufl.dx(domain=domain)
        )
    )
    matrix.assemble()
    dofs = np.arange(size, dtype=PETSc.IntType)
    initial_matrix = np.asarray(matrix.getValues(dofs, dofs)).copy()
    vector = matrix.createVecRight()
    vector.set(0.0)
    provider.add_residual(vector, trial)
    np.testing.assert_array_equal(
        vector.array, trial.operator.nodal_residual.reshape(-1)
    )
    dual = provider.dual_evidence(
        SimpleNamespace(solution=displacement, last_solve_info=object())
    )
    np.testing.assert_allclose(dual.resultant, trial.operator.force_on_structure)
    assert provider.current_face_diagnostics()["location"] == "boundary_faces"
    provider.add_tangent(matrix, trial)
    matrix.assemble()
    expected = initial_matrix.copy()
    np.add.at(
        expected,
        (trial.operator.tangent_rows, trial.operator.tangent_columns),
        trial.operator.tangent_values,
    )
    np.testing.assert_allclose(matrix.getValues(dofs, dofs), expected)
    displacement.x.array[0] += 1e-5
    before = vector.array.copy()
    with pytest.raises(ValueError, match="another Newton trial"):
        provider.add_residual(vector, trial)
    np.testing.assert_array_equal(vector.array, before)
    with pytest.raises(ValueError, match="another Newton trial"):
        provider.add_tangent(matrix, trial)
    np.testing.assert_allclose(matrix.getValues(dofs, dofs), expected)


def test_certified_face_trial_provider_proves_declared_initial_overlap():
    from dolfinx import fem

    from agentfem.boundary_models._certified_face_trial_provider import (
        DolfinxCertifiedFaceContactTrialProvider,
    )
    from agentfem.boundary_models.contact_pair import rigid_contact_pair
    from agentfem.boundary_models.rigid_body import rigid_body

    _, space, adapter, region = _dolfinx_mixed_active_patch()
    pair = rigid_contact_pair(
        region,
        rigid_body(_tool()),
        penalty=500.0,
        initial_penetration_limit=0.1,
    )
    with pytest.raises(ValueError, match="AFM-CONTACT-INITIAL-OVERLAP"):
        DolfinxCertifiedFaceContactTrialProvider(pair, adapter, fem.Function(space))
    admitted = rigid_contact_pair(
        region,
        rigid_body(_tool()),
        penalty=500.0,
        initial_penetration_limit=0.5,
    )
    provider = DolfinxCertifiedFaceContactTrialProvider(
        admitted, adapter, fem.Function(space)
    )
    bound = provider.initial_gap_bound
    assert bound is not None
    assert 0.1 < bound.maximum_penetration_bound <= 0.5
    diagnostics = provider.current_face_diagnostics()
    assert diagnostics["initial_contact"]["maximum_penetration_bound"] == (
        bound.maximum_penetration_bound
    )


def test_real_dolfinx_certified_face_trial_provider_rejects_stale_tool_pose():
    from types import SimpleNamespace

    from dolfinx import fem
    from petsc4py import PETSc

    from agentfem.boundary_models._certified_face_trial_provider import (
        DolfinxCertifiedFaceContactTrialProvider,
    )
    from agentfem.boundary_models.contact_pair import rigid_contact_pair
    from agentfem.boundary_models.contact_work import (
        prescribed_rigid_motion_schedule,
    )
    from agentfem.boundary_models.rigid_body import rigid_body

    _, space, adapter, region = _dolfinx_mixed_active_patch()
    motion = prescribed_rigid_motion(
        translation=(2e-5, -3e-5, 1e-5),
        rotation=(1e-4, 0.0, 0.0),
        reference_point=(0.5, 0.5, 0.5),
    )
    body = rigid_body(
        _tool(),
        motion_schedule=prescribed_rigid_motion_schedule(motion, end_time=1.0),
    )
    pair = rigid_contact_pair(region, body, penalty=500.0)
    provider = DolfinxCertifiedFaceContactTrialProvider(
        pair, adapter, fem.Function(space)
    )
    provider.update_motion(0.25)
    trial = provider.evaluate()
    assert trial.motion_factor == 0.25
    assert trial.operator.motion_factor == 0.25
    dual = provider.dual_evidence(
        SimpleNamespace(solution=provider.displacement, last_solve_info=object())
    )
    np.testing.assert_allclose(dual.force[:3], trial.operator.force_on_structure)
    reference = motion.state(0.25)["reference_point"]
    np.testing.assert_allclose(
        dual.force[3:], trial.operator.moment_on_structure_about(reference)
    )
    np.testing.assert_allclose(dual.coordinate, motion.generalized_coordinate(0.25))
    step = 1e-5
    provider.update_motion(0.25 + step)
    plus = provider.evaluate().potential_energy
    provider.update_motion(0.25 - step)
    minus = provider.evaluate().potential_energy
    coordinate_direction = (
        motion.generalized_coordinate(0.25 + step)
        - motion.generalized_coordinate(0.25 - step)
    ) / (2 * step)
    assert (plus - minus) / (2 * step) == pytest.approx(
        float(dual.force @ coordinate_direction), rel=1e-5, abs=1e-8
    )
    provider.update_motion(0.25)
    saved = provider.snapshot_motion()
    vector = PETSc.Vec().createSeq(3 * len(adapter.reference_nodal_positions))
    vector.set(0.0)
    provider.update_motion(0.5)
    with pytest.raises(ValueError, match="another Newton trial"):
        provider.add_residual(vector, trial)
    np.testing.assert_array_equal(vector.array, 0.0)
    provider.restore_motion(saved)
    provider.add_residual(vector, trial)
    np.testing.assert_array_equal(
        vector.array, trial.operator.nodal_residual.reshape(-1)
    )


def _research_two_feature_j2_step(
    *, moving=False, yield_stress=1e9, contact_penalty=1.0,
    second_pair=False, second_surface=None, second_friction=None,
    second_tangential_penalty=1.0,
    contact_integration="certified_closed_two_feature_research",
):
    from dolfinx import mesh as dolfinx_mesh

    from agentfem import (
        constitutive,
        fields,
        mesh as agent_mesh,
        models,
        solvers,
        steps,
        studies,
    )
    from agentfem.boundary_models.contact_pair import rigid_contact_pair
    from agentfem.boundary_models.contact_work import prescribed_rigid_motion_schedule
    from agentfem.boundary_models.rigid_body import rigid_body

    domain, _, _, contact = _dolfinx_mixed_active_patch()
    dim = domain.topology.dim - 1
    facet_ids = np.arange(domain.topology.index_map(dim).size_local, dtype=np.int32)
    geometry = dolfinx_mesh.entities_to_geometry(
        domain, dim, facet_ids, permute=True
    )
    support_nodes = _POSITIONS[[0, 1, 3]]
    support_id = next(
        int(facet_id)
        for facet_id, nodes in zip(facet_ids, geometry, strict=True)
        if all(
            np.min(np.linalg.norm(domain.geometry.x[nodes] - vertex, axis=1))
            < 1e-12
            for vertex in support_nodes
        )
    )
    tags = dolfinx_mesh.meshtags(
        domain,
        dim,
        np.asarray((support_id,), dtype=np.int32),
        np.asarray((19,), dtype=np.int32),
    )
    support = agent_mesh.tagged_boundary_region(
        domain, tags, tag=19, name="support"
    )
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain
    )
    displacement = model.field(fields.displacement(domain))
    model.clamp(displacement, on=support)
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=1000.0,
            poisson=0.0,
            yield_stress=yield_stress,
            hardening_modulus=100.0,
        )
    )
    motion_schedule = None
    if moving:
        motion_schedule = prescribed_rigid_motion_schedule(
            prescribed_rigid_motion(
                translation=(0.0, 0.0, -0.002),
                reference_point=(0.9, 0.88, 0.0),
            ),
            end_time=1.0,
        )
    model.add_boundary_model(
        rigid_contact_pair(
            contact,
            rigid_body(_tool(), motion_schedule=motion_schedule),
            penalty=contact_penalty,
            initial_penetration_limit=0.5,
            name="reentrant_tool",
        )
    )
    if second_pair:
        model.add_boundary_model(
            rigid_contact_pair(
                contact, rigid_body(_tool() if second_surface is None else second_surface),
                penalty=contact_penalty,
                friction_coefficient=second_friction,
                tangential_penalty=(
                    second_tangential_penalty if second_friction is not None else None
                ),
                name="second_tool",
            )
        )
    step = model.step(
        target=displacement,
        material=material,
        contact_integration=contact_integration,
        incrementation=steps.fixed(2),
        solver_options=solvers.newton(
            relative_tolerance=1e-8,
            absolute_tolerance=1e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        progress=False,
    )
    return step, displacement


def test_research_two_feature_multiple_pairs_share_one_j2_step(tmp_path):
    step, _ = _research_two_feature_j2_step(second_pair=True, moving=True)
    result = step.solve_result()
    assert result.status == "completed"
    assert len(step.accepted_increments) == 2
    faces = result.metadata["rigid_contact_face_diagnostics"]
    assert len(faces) == 2
    assert {item["constraint"] for item in faces} == {
        "reentrant_tool", "second_tool",
    }
    path = result.metadata["rigid_contact_face_path"]
    assert path["complete"] is True
    assert len(path["pairs"]) == 2
    assert all(item["accepted_station_count"] == 3 for item in path["pairs"])
    trial = step.contact_provider.evaluate()
    np.testing.assert_allclose(
        trial.nodal_structural_residual,
        sum(selected.nodal_structural_residual for _, selected in trial.pairs),
        rtol=0.0, atol=1e-14,
    )
    assert trial.potential_energy == pytest.approx(
        sum(selected.potential_energy for _, selected in trial.pairs),
        rel=1e-13,
    )
    partial, _ = _research_two_feature_j2_step(second_pair=True, moving=True)
    partial.solve(until=0.5)
    saved = partial.save_checkpoint(tmp_path / "two_certified_tools")
    resumed, _ = _research_two_feature_j2_step(second_pair=True, moving=True)
    resumed.load_checkpoint(saved)
    assert resumed.solve_result().status == "completed"
    np.testing.assert_allclose(
        resumed.solution.x.array, step.solution.x.array, rtol=1e-9, atol=1e-12
    )
    assert resumed.contact_face_path == step.contact_face_path
    assert resumed.energy_history[-1].tool_work == pytest.approx(
        step.energy_history[-1].tool_work, rel=1e-9, abs=1e-12
    )


def test_research_face_and_sampled_point_modes_share_one_j2_step(tmp_path):
    from agentfem import boundary_models

    settings = dict(
        moving=True,
        second_pair=True,
        second_surface=boundary_models.rigid_plane(
            point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
        ),
        contact_integration={
            "reentrant_tool": "certified_closed_two_feature_research",
            "second_tool": "planar_active_partition",
        },
    )
    step, _ = _research_two_feature_j2_step(**settings)
    trial = step.contact_provider.evaluate()
    assert tuple(name for name, _ in trial.pairs) == (
        "reentrant_tool", "second_tool",
    )
    np.testing.assert_allclose(
        trial.nodal_structural_residual,
        sum(item.nodal_structural_residual for _, item in trial.pairs),
        rtol=0.0, atol=1e-14,
    )
    result = step.solve_result()
    assert result.status == "completed"
    assert len(step.accepted_increments) == 2
    assert result.metadata["rigid_contact_face_path"]["pairs"][0][
        "constraint"
    ] == "reentrant_tool"
    assert result.metadata["rigid_contact_point_diagnostics"][
        "constraint"
    ] == "second_tool"
    assert len(result.metadata["constraint_duals"]) == 2
    assert len(result.metadata["constraint_work_evidence"]) == 2
    assert step.energy_history[-1].tool_work > 0.0
    partial, _ = _research_two_feature_j2_step(**settings)
    partial.solve(until=0.5)
    saved = partial.save_checkpoint(tmp_path / "mixed_face_point")
    resumed, _ = _research_two_feature_j2_step(**settings)
    resumed.load_checkpoint(saved)
    assert resumed.solve_result().status == "completed"
    np.testing.assert_allclose(
        resumed.solution.x.array, step.solution.x.array, rtol=1e-9, atol=1e-12,
    )
    assert resumed.energy_history[-1].tool_work == pytest.approx(
        step.energy_history[-1].tool_work, rel=1e-9, abs=1e-12,
    )


def test_research_mixed_face_point_trial_rejects_stale_pose_before_petsc_write():
    from petsc4py import PETSc

    from agentfem import boundary_models

    step, _ = _research_two_feature_j2_step(
        second_pair=True,
        second_surface=boundary_models.rigid_plane(
            point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
        ),
        contact_integration={
            "reentrant_tool": "certified_closed_two_feature_research",
            "second_tool": "planar_active_partition",
        },
    )
    provider = step.contact_provider
    trial = provider.evaluate()
    vector = PETSc.Vec().createSeq(len(step.solution.x.array))
    vector.set(0.0)
    try:
        provider.providers[0].update_motion(0.5)
        with pytest.raises(ValueError, match="differs from current"):
            provider.add_residual(vector, trial)
        np.testing.assert_array_equal(vector.array, 0.0)
    finally:
        vector.destroy()


def test_research_mixed_face_point_tangent_matches_total_directional_difference():
    from agentfem import boundary_models

    step, _ = _research_two_feature_j2_step(
        second_pair=True,
        second_surface=boundary_models.rigid_plane(
            point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
        ),
        contact_integration={
            "reentrant_tool": "certified_closed_two_feature_research",
            "second_tool": "planar_active_partition",
        },
    )
    provider = step.contact_provider
    original = step.solution.x.array.copy()
    direction = np.linspace(-0.7, 0.9, len(original)).reshape(-1, 3)
    expected = provider.evaluate().tangent_action(direction)
    samples = []
    h = 1e-7
    try:
        for sign in (1.0, -1.0):
            step.solution.x.array[:] = original + sign * h * direction.reshape(-1)
            step.solution.x.scatter_forward()
            samples.append(provider.evaluate().nodal_structural_residual)
    finally:
        step.solution.x.array[:] = original
        step.solution.x.scatter_forward()
    np.testing.assert_allclose(
        expected, (samples[0] - samples[1]) / (2.0 * h),
        rtol=3e-5, atol=3e-7,
    )


def test_research_mixed_face_and_reviewed_point_friction_restart(tmp_path):
    from agentfem import boundary_models

    settings = dict(
        moving=True,
        contact_penalty=100.0,
        second_pair=True,
        second_surface=boundary_models.rigid_plane(
            point=(0.91, 0.0, 0.0), normal=(1.0, 0.0, 0.0),
        ),
        second_friction=0.001,
        second_tangential_penalty=1000.0,
        contact_integration={
            "reentrant_tool": "certified_closed_two_feature_research",
            "second_tool": "gauss",
        },
    )
    step, _ = _research_two_feature_j2_step(**settings)
    result = step.solve_result()
    assert result.status == "completed"
    assert result.histories["friction_dissipation"].values[-1] > 0.0
    assert step.contact_provider.providers[1].current_point_diagnostics()[
        "active_point_count"
    ] > 0
    assert step.contact_provider.providers[1].friction_state is not None
    assert result.metadata["rigid_contact_face_path"]["complete"] is True
    partial, _ = _research_two_feature_j2_step(**settings)
    partial.solve(until=0.5)
    saved = partial.save_checkpoint(tmp_path / "mixed_face_point_friction")
    resumed, _ = _research_two_feature_j2_step(**settings)
    resumed.load_checkpoint(saved)
    assert resumed.solve_result().status == "completed"
    np.testing.assert_allclose(
        resumed.solution.x.array, step.solution.x.array,
        rtol=1e-9, atol=1e-12,
    )
    assert resumed.energy_history[-1].friction_dissipation == pytest.approx(
        step.energy_history[-1].friction_dissipation,
        rel=1e-9, abs=1e-12,
    )


def test_research_mixed_face_point_failed_acceptance_rolls_back_both_pairs(
    monkeypatch,
):
    from agentfem import boundary_models

    step, _ = _research_two_feature_j2_step(
        moving=True, yield_stress=1e-4, contact_penalty=100.0,
        second_pair=True,
        second_surface=boundary_models.rigid_plane(
            point=(0.91, 0.0, 0.0), normal=(1.0, 0.0, 0.0),
        ),
        second_friction=0.001,
        second_tangential_penalty=1000.0,
        contact_integration={
            "reentrant_tool": "certified_closed_two_feature_research",
            "second_tool": "gauss",
        },
    )
    step.solve(until=0.5)
    accepted_u = step.solution.x.array.copy()
    accepted_material = step.response.state.committed_state_vectors().copy()
    assert np.max(accepted_material[:, -2]) > 0.0
    accepted_pose = step.contact_provider.snapshot_motion()
    accepted_contact_state = json.dumps(
        step.contact_provider.snapshot_accepted_state(), sort_keys=True,
    )
    accepted_energy = tuple(step.energy_history)
    accepted_face_path = dict(step.contact_face_path)
    accepted_point_path = dict(step.contact_feature_path)
    original = step._record_energy

    def fail_after_energy(*args, **kwargs):
        original(*args, **kwargs)
        raise ValueError("intentional mixed-contact acceptance failure")

    monkeypatch.setattr(step, "_record_energy", fail_after_energy)
    with pytest.raises(RuntimeError, match="intentional mixed-contact"):
        step.solve()
    assert step.accepted_load_factor == pytest.approx(0.5)
    assert step.contact_provider.snapshot_motion() == accepted_pose
    assert json.dumps(
        step.contact_provider.snapshot_accepted_state(), sort_keys=True,
    ) == accepted_contact_state
    np.testing.assert_array_equal(step.solution.x.array, accepted_u)
    np.testing.assert_array_equal(
        step.response.state.committed_state_vectors(), accepted_material,
    )
    assert tuple(step.energy_history) == accepted_energy
    assert step.contact_face_path == accepted_face_path
    assert step.contact_feature_path == accepted_point_path


def test_research_two_tool_trial_refuses_one_stale_pose_before_petsc_mutation():
    from petsc4py import PETSc

    step, _ = _research_two_feature_j2_step(second_pair=True)
    provider = step.contact_provider
    trial = provider.evaluate()
    vector = PETSc.Vec().createSeq(len(step.solution.x.array))
    vector.set(0.0)
    try:
        provider.providers[1].update_motion(0.5)
        with pytest.raises(ValueError, match="another Newton trial"):
            provider.add_residual(vector, trial)
        np.testing.assert_array_equal(vector.array, 0.0)
    finally:
        vector.destroy()


def test_research_two_tool_pose_restore_is_atomic():
    step, _ = _research_two_feature_j2_step(second_pair=True)
    provider = step.contact_provider
    provider.update_motion(0.5)
    before = provider.snapshot_motion()
    with pytest.raises(ValueError, match="factor must lie"):
        provider.restore_motion((("reentrant_tool", 0.25), ("second_tool", 2.0)))
    assert provider.snapshot_motion() == before


def test_research_two_feature_faces_enter_existing_serial_j2_step(tmp_path):
    step, displacement = _research_two_feature_j2_step()
    result = step.solve_result()
    assert result.status == "completed"
    assert len(step.accepted_increments) == 2
    assert np.linalg.norm(displacement.value.x.array) > 0.0
    face = result.metadata["rigid_contact_face_diagnostics"]
    assert face["location"] == "boundary_faces"
    assert face["all_faces_certified_at_current_trial"] is True
    assert len(face["face_ids"]) == 2
    path = result.metadata["rigid_contact_face_path"]
    assert path["complete"] is True
    assert path["accepted_station_count"] == 3
    assert path["pairs"][0]["accepted_station_count"] == 3
    assert path["pairs"][0]["all_observed_faces_certified"] is True
    assert path["pairs"][0]["maximum_mixed_feature_face_count"] >= 1
    assert "rigid_contact_point_diagnostics" not in result.metadata
    from agentfem import results

    with pytest.raises(NotImplementedError, match="AFM-CONTACT-EXPORT-FACE"):
        results.export_accepted_contact_points(
            step, tmp_path / "point_export", result=result
        )


def test_research_two_feature_j2_restart_matches_uninterrupted_path(tmp_path):
    full, _ = _research_two_feature_j2_step()
    assert full.solve_result().status == "completed"
    partial, _ = _research_two_feature_j2_step()
    partial.solve(until=0.5)
    assert len(partial.accepted_increments) == 1
    saved = partial.save_checkpoint(tmp_path / "certified_face")
    restored, _ = _research_two_feature_j2_step()
    restored.load_checkpoint(saved)
    resumed_result = restored.solve_result()
    assert resumed_result.status == "completed"
    np.testing.assert_allclose(
        restored.solution.x.array, full.solution.x.array, rtol=1e-9, atol=1e-12
    )
    assert restored.energy_history[-1].contact_potential == pytest.approx(
        full.energy_history[-1].contact_potential, rel=1e-9, abs=1e-12
    )
    assert restored.contact_face_path == full.contact_face_path
    assert resumed_result.metadata["rigid_contact_face_path"]["complete"] is True


def test_research_two_feature_moving_tool_has_accepted_work_and_restart(tmp_path):
    full, _ = _research_two_feature_j2_step(moving=True)
    result = full.solve_result()
    assert result.status == "completed"
    assert full.energy_history[-1].tool_work > 0.0
    assert len(result.metadata["constraint_work_evidence"]) == 1
    partial, _ = _research_two_feature_j2_step(moving=True)
    partial.solve(until=0.5)
    saved = partial.save_checkpoint(tmp_path / "moving_certified_face")
    resumed, _ = _research_two_feature_j2_step(moving=True)
    resumed.load_checkpoint(saved)
    assert resumed.solve_result().status == "completed"
    assert resumed.energy_history[-1].tool_work == pytest.approx(
        full.energy_history[-1].tool_work, rel=1e-9, abs=1e-12
    )


def test_research_two_feature_plastic_history_survives_restart(tmp_path):
    settings = {"moving": True, "yield_stress": 1e-4, "contact_penalty": 100.0}
    full, _ = _research_two_feature_j2_step(**settings)
    assert full.solve_result().status == "completed"
    full_state = full.response.state.committed_state_vectors()
    assert np.max(full_state[:, -2]) > 0.0
    partial, _ = _research_two_feature_j2_step(**settings)
    partial.solve(until=0.5)
    saved = partial.save_checkpoint(tmp_path / "plastic_certified_face")
    resumed, _ = _research_two_feature_j2_step(**settings)
    resumed.load_checkpoint(saved)
    assert resumed.solve_result().status == "completed"
    np.testing.assert_allclose(
        resumed.response.state.committed_state_vectors(), full_state,
        rtol=1e-9, atol=1e-12,
    )
    np.testing.assert_allclose(
        resumed.solution.x.array, full.solution.x.array,
        rtol=1e-9, atol=1e-12,
    )


@pytest.mark.parametrize(
    ("yield_stress", "contact_penalty", "second_pair"),
    ((1e9, 1.0, False), (1e-4, 100.0, False), (1e9, 1.0, True)),
)
def test_research_two_feature_total_j2_tangent_matches_directional_difference(
    yield_stress, contact_penalty, second_pair
):
    step, _ = _research_two_feature_j2_step(
        moving=True, yield_stress=yield_stress,
        contact_penalty=contact_penalty, second_pair=second_pair,
    )
    step.solve(until=0.5)
    reference = step.solution.x.array.copy()
    direction = np.linspace(-0.25, 0.3, len(reference))
    direction[np.asarray(step.contact_constrained_dofs, dtype=int)] = 0.0
    free = np.setdiff1d(
        np.arange(len(reference)), step.contact_constrained_dofs
    )

    def correction_rhs(scale):
        step.solution.x.array[:] = reference + scale * direction
        step.solution.x.scatter_forward()
        step._update_response(start_factor=0.5, target_factor=0.5)
        rhs, _ = step._correction_rhs()
        values = rhs.array.copy()
        rhs.destroy()
        return values

    try:
        # At the accepted plastic return, positive and negative perturbations
        # may straddle the elastic-unloading/plastic-loading kink. Probe a
        # fixed old State on either side of the kink instead.
        for base_scale in ((0.0,) if yield_stress >= 1.0 else (-1e-3, 1e-3)):
            correction_rhs(base_scale)
            matrix = step._assemble_tangent()
            input_vector = matrix.createVecRight()
            output_vector = matrix.createVecLeft()
            try:
                input_vector.array[:] = direction
                matrix.mult(input_vector, output_vector)
                expected = -output_vector.array.copy()
            finally:
                output_vector.destroy()
                input_vector.destroy()
            size = 1e-6
            measured = (
                correction_rhs(base_scale + size)
                - correction_rhs(base_scale - size)
            ) / (2 * size)
            np.testing.assert_allclose(
                measured[free], expected[free], rtol=2e-4, atol=2e-6
            )
    finally:
        step.solution.x.array[:] = reference
        step.solution.x.scatter_forward()
        step.state_transaction.prepare_resume()


@pytest.mark.parametrize(
    ("yield_stress", "contact_penalty"),
    ((1e9, 1.0), (1e-4, 100.0)),
)
def test_research_two_feature_failure_rolls_back_material_pose_and_work(
    monkeypatch, yield_stress, contact_penalty
):
    step, _ = _research_two_feature_j2_step(
        moving=True, yield_stress=yield_stress, contact_penalty=contact_penalty
    )
    step.solve(until=0.5)
    accepted_u = step.solution.x.array.copy()
    accepted_material = step.response.state.committed_state_vectors().copy()
    if yield_stress < 1.0:
        assert np.max(accepted_material[:, -2]) > 0.0
    accepted_energy = tuple(step.energy_history)
    accepted_face_path = dict(step.contact_face_path)
    accepted_pose = step.contact_provider.current_factor
    original = step._record_energy

    def fail_after_energy(*args, **kwargs):
        original(*args, **kwargs)
        raise ValueError("intentional certified-face acceptance failure")

    monkeypatch.setattr(step, "_record_energy", fail_after_energy)
    with pytest.raises(RuntimeError, match="intentional certified-face"):
        step.solve()
    assert step.accepted_load_factor == pytest.approx(0.5)
    assert step.contact_provider.current_factor == accepted_pose
    np.testing.assert_array_equal(step.solution.x.array, accepted_u)
    np.testing.assert_array_equal(
        step.response.state.committed_state_vectors(), accepted_material
    )
    assert tuple(step.energy_history) == accepted_energy
    assert step.contact_face_path == accepted_face_path


def test_research_two_feature_corrupt_face_path_checkpoint_is_atomic(tmp_path):
    source, _ = _research_two_feature_j2_step()
    source.solve(until=0.5)
    saved = source.save_checkpoint(tmp_path / "good_face_path")
    with np.load(saved, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    face_path = json.loads(str(payload["contact_face_path"]))
    face_path["reentrant_tool"]["accepted_station_count"] = 100
    payload["contact_face_path"] = json.dumps(face_path, sort_keys=True)
    corrupt = tmp_path / "corrupt_face_path.npz"
    np.savez(corrupt, **payload)
    receiver, _ = _research_two_feature_j2_step()
    before_u = receiver.solution.x.array.copy()
    with pytest.raises(ValueError, match="Contact face path"):
        receiver.load_checkpoint(corrupt)
    assert receiver.accepted_load_factor == 0.0
    assert not receiver.contact_face_path
    np.testing.assert_array_equal(receiver.solution.x.array, before_u)


def test_random_accepted_local_proofs_agree_with_independent_tool_projection():
    tool = _tool()
    generator = np.random.default_rng(20261019)
    barycentric = np.asarray(
        [(i / 12, j / 12, 1 - (i + j) / 12) for i in range(13) for j in range(13 - i)]
    )
    accepted = 0
    for _ in range(32):
        current = _POSITIONS + generator.normal(scale=1e-5, size=_POSITIONS.shape)
        for nodes in _FACES:
            face = current[nodes]
            try:
                certificate = integrate_discovered_closed_contact_face(
                    _POSITIONS[nodes], face, tool, penalty=500.0
                )
            except NotImplementedError:
                continue
            accepted += 1
            points = barycentric @ face
            projection = tool.project(points)
            assert projection.all_valid
            if not hasattr(certificate, "geometry"):
                assert np.min(projection.signed_gaps) > 0.0
                continue
            first = certificate.geometry.first
            second = certificate.geometry.second
            first_offsets = (points - first.anchor) @ first.normal_projector
            second_offsets = (points - second.anchor) @ second.normal_projector
            estimated_squared = np.minimum(
                np.einsum("ij,ij->i", first_offsets, first_offsets),
                np.einsum("ij,ij->i", second_offsets, second_offsets),
            )
            if certificate.response.inactive_reference_area:
                facet_id = certificate.geometry.facet_indices[0]
                normal = tool.facets.facet_normals[facet_id]
                active = (points - tool._triangles[facet_id][0]) @ normal < -1e-10
            else:
                active = np.ones(len(points), dtype=bool)
            np.testing.assert_allclose(
                estimated_squared[active],
                projection.signed_gaps[active] ** 2,
                rtol=1e-9,
                atol=1e-12,
            )
            assert np.all(projection.signed_gaps[active] < 0.0)
            if np.any(~active):
                assert np.all(projection.signed_gaps[~active] > 0.0)
    assert accepted >= 32


def test_prescribed_tool_translation_work_matches_fixed_body_potential_change():
    base_tool = _tool()
    trace = _positive_face_trace(_POSITIONS, _FACES, (100, 101))
    travel = 1e-4
    direction = np.asarray((0.0, 0.0, 1.0))

    def state(factor):
        moved = closed_triangulated_rigid_solid(
            vertices=base_tool.facets.vertices + factor * travel * direction,
            triangles=base_tool.facets.triangles,
            reviewed_embedded=True,
        )
        return evaluate_certified_closed_p1_contact_operator(
            trace, _POSITIONS, _POSITIONS, moved, penalty=500.0
        )

    start = state(0.0)
    finish = state(1.0)
    abscissae, weights = np.polynomial.legendre.leggauss(8)
    path_work = sum(
        weight
        * np.dot(state((abscissa + 1) / 2).force_on_structure, travel * direction)
        / 2
        for abscissa, weight in zip(abscissae, weights, strict=True)
    )
    assert path_work == pytest.approx(
        finish.potential_energy - start.potential_energy,
        rel=1e-7,
        abs=1e-9,
    )


def test_prescribed_tool_rotation_work_matches_fixed_body_potential_change():
    base_tool = _tool()
    trace = _positive_face_trace(_POSITIONS, _FACES, (100, 101))
    angle = 1e-4
    axis = np.asarray((1.0, 0.0, 0.0))
    skew = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))

    def state(factor):
        theta = factor * angle
        rotation = np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * skew @ skew
        moved = closed_triangulated_rigid_solid(
            vertices=base_tool.facets.vertices @ rotation.T,
            triangles=base_tool.facets.triangles,
            reviewed_embedded=True,
        )
        return evaluate_certified_closed_p1_contact_operator(
            trace, _POSITIONS, _POSITIONS, moved, penalty=500.0
        )

    start = state(0.0)
    finish = state(1.0)
    abscissae, weights = np.polynomial.legendre.leggauss(8)
    path_work = sum(
        weight
        * np.dot(
            state((abscissa + 1) / 2).moment_on_structure_about_origin, angle * axis
        )
        / 2
        for abscissa, weight in zip(abscissae, weights, strict=True)
    )
    assert path_work == pytest.approx(
        finish.potential_energy - start.potential_energy,
        rel=1e-7,
        abs=1e-9,
    )


def test_combined_tool_translation_rotation_uses_moment_at_moving_reference():
    base_tool = _tool()
    trace = _positive_face_trace(_POSITIONS, _FACES, (100, 101))
    travel = np.asarray((2e-5, -3e-5, 1e-5))
    angle = 1e-4
    axis = np.asarray((1.0, 0.0, 0.0))
    skew = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))

    def state(factor):
        theta = factor * angle
        rotation = np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * skew @ skew
        moved = closed_triangulated_rigid_solid(
            vertices=base_tool.facets.vertices @ rotation.T + factor * travel,
            triangles=base_tool.facets.triangles,
            reviewed_embedded=True,
        )
        result = evaluate_certified_closed_p1_contact_operator(
            trace, _POSITIONS, _POSITIONS, moved, penalty=500.0
        )
        return result, factor * travel

    start, _ = state(0.0)
    finish, _ = state(1.0)
    abscissae, weights = np.polynomial.legendre.leggauss(8)
    path_work = 0.0
    for abscissa, weight in zip(abscissae, weights, strict=True):
        result, reference = state((abscissa + 1) / 2)
        power = np.dot(result.force_on_structure, travel) + np.dot(
            result.moment_on_structure_about(reference), angle * axis
        )
        path_work += weight * power / 2
    assert path_work == pytest.approx(
        finish.potential_energy - start.potential_energy,
        rel=1e-7,
        abs=1e-9,
    )


def test_inverse_rigid_pose_certification_matches_independently_moved_tool():
    tool = _tool()
    trace = _positive_face_trace(_POSITIONS, _FACES, (100, 101))
    motion = prescribed_rigid_motion(
        translation=(2e-5, -3e-5, 1e-5),
        rotation=(1e-4, 0.0, 0.0),
        reference_point=(0.5, 0.5, 0.5),
    )
    current = _POSITIONS.copy()
    current[1, 0] += 1e-6
    for factor in (0.0, 0.5, 1.0):
        pose = motion.state(factor)
        moved_vertices = (
            np.asarray(pose["reference_point"])
            + (tool.facets.vertices - motion.reference_point)
            @ np.asarray(pose["rotation_matrix"]).T
        )
        independently_moved_tool = closed_triangulated_rigid_solid(
            vertices=moved_vertices,
            triangles=tool.facets.triangles,
            reviewed_embedded=True,
        )
        frame = evaluate_moving_certified_closed_p1_contact_operator(
            trace,
            _POSITIONS,
            current,
            tool,
            motion=motion,
            factor=factor,
            penalty=500.0,
        )
        assert frame.motion_factor == factor
        np.testing.assert_array_equal(
            frame.moving_reference_point, pose["reference_point"]
        )
        assert not frame.moving_reference_point.flags.writeable
        rebuilt = evaluate_certified_closed_p1_contact_operator(
            trace, _POSITIONS, current, independently_moved_tool, penalty=500.0
        )
        assert frame.face_activity_kinds == rebuilt.face_activity_kinds
        assert frame.potential_energy == pytest.approx(
            rebuilt.potential_energy, rel=1e-9
        )
        np.testing.assert_allclose(
            frame.nodal_residual, rebuilt.nodal_residual, rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(
            frame.tangent_values, rebuilt.tangent_values, rtol=1e-7, atol=1e-7
        )
        np.testing.assert_allclose(
            frame.face_forces_on_structure,
            rebuilt.face_forces_on_structure,
            rtol=1e-8,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            frame.face_moments_on_structure_about_origin,
            rebuilt.face_moments_on_structure_about_origin,
            rtol=1e-7,
            atol=1e-10,
        )


def test_inverse_rigid_pose_work_is_the_prescribed_potential_change():
    tool = _tool()
    trace = _positive_face_trace(_POSITIONS, _FACES, (100, 101))
    motion = prescribed_rigid_motion(
        translation=(2e-5, -3e-5, 1e-5),
        rotation=(1e-4, 0.0, 0.0),
        reference_point=(0.5, 0.5, 0.5),
    )

    def state(factor):
        return evaluate_moving_certified_closed_p1_contact_operator(
            trace,
            _POSITIONS,
            _POSITIONS,
            tool,
            motion=motion,
            factor=factor,
            penalty=500.0,
        )

    start, finish = state(0.0), state(1.0)
    abscissae, weights = np.polynomial.legendre.leggauss(8)
    work = 0.0
    for abscissa, weight in zip(abscissae, weights, strict=True):
        factor = (abscissa + 1.0) / 2.0
        response = state(factor)
        moving_reference = motion.state(factor)["reference_point"]
        power = np.dot(response.force_on_structure, motion.translation) + np.dot(
            response.moment_on_structure_about(moving_reference), motion.rotation
        )
        work += weight * power / 2.0
    assert work == pytest.approx(
        finish.potential_energy - start.potential_energy, rel=1e-7, abs=1e-9
    )


def test_inverse_rigid_pose_tangent_matches_recertified_displacement_difference():
    tool = _tool()
    trace = _positive_face_trace(_POSITIONS, _FACES, (100, 101))
    motion = prescribed_rigid_motion(
        translation=(2e-5, -3e-5, 1e-5),
        rotation=(1e-4, 0.0, 0.0),
        reference_point=(0.5, 0.5, 0.5),
    )
    direction = np.asarray(
        ((0.4, -0.3, 0.2), (0.1, 0.5, -0.4), (-0.2, 0.3, 0.1), (0.3, -0.1, 0.2))
    )

    def state(current):
        return evaluate_moving_certified_closed_p1_contact_operator(
            trace,
            _POSITIONS,
            current,
            tool,
            motion=motion,
            factor=0.6,
            penalty=500.0,
        )

    center = state(_POSITIONS)
    distance = 1e-7
    plus = state(_POSITIONS + distance * direction)
    minus = state(_POSITIONS - distance * direction)
    assert (plus.potential_energy - minus.potential_energy) / (2 * distance) == (
        pytest.approx(np.sum(center.nodal_residual * direction), rel=2e-5)
    )
    np.testing.assert_allclose(
        (plus.nodal_residual - minus.nodal_residual) / (2 * distance),
        center.tangent_action(direction),
        rtol=5e-5,
        atol=1e-6,
    )


def test_large_prescribed_rigid_frame_change_preserves_certified_face_energy():
    tool = _tool()
    trace = _positive_face_trace(_POSITIONS, _FACES, (100, 101))
    motion = prescribed_rigid_motion(
        translation=(0.3, -0.2, 0.5),
        rotation=(0.4, -0.3, 0.2),
        reference_point=(0.5, 0.5, 0.5),
    )
    pose = motion.state(1.0)
    rotation = np.asarray(pose["rotation_matrix"])
    moved_slave = (
        np.asarray(pose["reference_point"])
        + (_POSITIONS - motion.reference_point) @ rotation.T
    )
    stationary = evaluate_certified_closed_p1_contact_operator(
        trace, _POSITIONS, _POSITIONS, tool, penalty=500.0
    )
    moving = evaluate_moving_certified_closed_p1_contact_operator(
        trace,
        _POSITIONS,
        moved_slave,
        tool,
        motion=motion,
        factor=1.0,
        penalty=500.0,
    )
    assert moving.potential_energy == pytest.approx(
        stationary.potential_energy, rel=1e-10
    )
    np.testing.assert_allclose(
        moving.nodal_residual, stationary.nodal_residual @ rotation.T, rtol=1e-9
    )
    np.testing.assert_allclose(
        moving.force_on_structure,
        stationary.force_on_structure @ rotation.T,
        rtol=1e-9,
    )
