"""Research-only certified-face aggregation and virtual-work regressions."""

from __future__ import annotations

import numpy as np
import pytest

from agentfem.boundary_models.closed_polyhedron import closed_triangulated_rigid_solid
from agentfem.boundary_models.contact_face_operator_research import (
    assemble_certified_contact_faces,
    evaluate_certified_closed_p1_contact_operator,
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


def test_real_dolfinx_mixed_active_patch_lowers_to_the_certified_face_operator():
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
