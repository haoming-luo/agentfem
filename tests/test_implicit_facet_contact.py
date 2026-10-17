from __future__ import annotations

import numpy as np
import pytest

from agentfem import boundary_models
from agentfem.boundary_models.implicit_facet_contact import (
    linearize_piecewise_planar_contact,
)
from agentfem.boundary_models.implicit_contact import (
    CombinedImplicitRigidContactContribution,
)


def _flat_tool():
    return boundary_models.triangulated_rigid_surface(
        vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)),
        triangles=((0, 1, 2), (1, 3, 2)),
    )


def test_reviewed_edge_adjacency_distinguishes_seam_from_free_edges():
    surface = _flat_tool()
    assert surface.edge_neighbors.shape == surface.triangles.shape
    # Local edge 0 of the first triangle and local edge 1 of the second
    # share (1, 2).  Every other edge belongs to the open boundary.
    np.testing.assert_array_equal(surface.edge_neighbors[0], (1, -1, -1))
    np.testing.assert_array_equal(surface.edge_neighbors[1], (-1, 0, -1))
    assert not surface.edge_neighbors.flags.writeable


def test_independent_pair_contributions_add_without_merging_point_identity():
    upper = _flat_tool()
    lower = boundary_models.triangulated_rigid_surface(
        vertices=((0, 0, -0.2), (1, 0, -0.2), (0, 1, -0.2), (1, 1, -0.2)),
        triangles=((0, 2, 1), (1, 2, 3)),
    )
    nodes = np.asarray(((0.2, 0.3, -0.1),) * 3)
    first = _evaluate(upper, nodes)
    second = _evaluate(lower, nodes)
    combined = CombinedImplicitRigidContactContribution(
        (("upper", first), ("lower", second)),
    )
    assert first.active_points == second.active_points == 1
    # Both pairs legitimately use point ID 71; their separate pair identities
    # avoid collisions in evidence and future accepted-work accounting.
    np.testing.assert_array_equal(first.assembly.point_ids, second.assembly.point_ids)
    assert tuple(name for name, _ in combined.pairs) == ("upper", "lower")
    assert combined.active_points == 2
    np.testing.assert_allclose(
        combined.nodal_structural_residual,
        first.assembly.nodal_structural_residual
        + second.assembly.nodal_structural_residual,
    )
    assert combined.potential_energy == pytest.approx(
        first.assembly.potential_energy + second.assembly.potential_energy,
    )
    direction = np.asarray(((0.1, 0.2, 0.3),) * 3)
    np.testing.assert_allclose(
        combined.tangent_action(direction),
        first.tangent_action(direction) + second.tangent_action(direction),
    )
    with pytest.raises(ValueError, match="unique"):
        CombinedImplicitRigidContactContribution(
            (("same", first), ("same", second)),
        )


def _evaluate(surface, nodes, *, shape=(0.2, 0.3, 0.5), weight=0.75):
    trace = boundary_models.ContactTrace(
        point_ids=(71,),
        node_ids=((0, 1, 2),),
        shape_values=(shape,),
        weights=(weight,),
        source="three-node_reference_trace",
    )
    evaluation = trace.evaluate(nodes)
    record = boundary_models.ContactProjectionRecord(
        trace.point_ids,
        surface.project(evaluation.query_points),
    )
    response = boundary_models.frictionless_penalty_contact_law(125.0).evaluate(
        record.projection
    )
    return linearize_piecewise_planar_contact(
        evaluation,
        record,
        response,
        surface,
        surface_reference_point=(0, 0, 0),
    )


@pytest.mark.parametrize("xy", [(0.3, 0.3), (0.7, 0.7), (0.5, 0.5)])
def test_piecewise_planar_tangent_matches_frozen_branch_residual_and_energy(xy):
    surface = _flat_tool()
    base = np.asarray(
        ((xy[0] - 0.1, xy[1], -0.1),
         (xy[0] + 0.2, xy[1], -0.1),
         (xy[0], xy[1], -0.1)),
        dtype=float,
    )
    # The quadrature point sits on an interior facet or the coplanar seam.
    base[:, 0] += xy[0] - np.dot((0.2, 0.3, 0.5), base[:, 0])
    direction = np.asarray(((0.2, -0.1, 0.3), (-0.1, 0.05, -0.2), (0.1, 0.2, 0.4)))
    contribution = _evaluate(surface, base)
    assert contribution.active_points == 1
    assert contribution.assembly.potential_energy > 0.0
    np.testing.assert_allclose(
        contribution.assembly.contact_force_on_structure,
        -np.sum(contribution.assembly.nodal_structural_residual, axis=0),
    )
    step = 1.0e-7
    plus = _evaluate(surface, base + step * direction)
    minus = _evaluate(surface, base - step * direction)
    difference = (
        plus.assembly.nodal_structural_residual
        - minus.assembly.nodal_structural_residual
    ) / (2 * step)
    np.testing.assert_allclose(contribution.tangent_action(direction), difference, rtol=1e-8, atol=1e-8)
    energy_difference = (
        plus.assembly.potential_energy - minus.assembly.potential_energy
    ) / (2 * step)
    np.testing.assert_allclose(
        energy_difference,
        np.sum(contribution.assembly.nodal_structural_residual * direction),
        rtol=1e-8,
        atol=1e-8,
    )
    matrix = np.zeros((9, 9))
    np.add.at(matrix, (contribution.tangent_rows, contribution.tangent_columns), contribution.tangent_values)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-13)
    assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-12


def test_open_patch_boundary_is_rejected_when_active_but_inactive_is_zero():
    surface = _flat_tool()
    boundary = np.asarray(((0.0, 0.5, -0.1),) * 3)
    with pytest.raises(ValueError, match="AFM-IMPLICIT-CONTACT-BRANCH"):
        _evaluate(surface, boundary)
    inactive = _evaluate(surface, np.asarray(((0.0, 0.5, 0.1),) * 3))
    assert inactive.active_points == 0
    assert inactive.tangent_values.size == 0
    np.testing.assert_array_equal(inactive.tangent_action(np.ones((3, 3))), 0.0)


def test_boundary_vertex_is_rejected_even_when_selected_facet_has_two_neighbors():
    # All three triangles are coplanar. The middle facet has a neighbor on
    # both edges incident on vertex 0, but the *whole* vertex fan is open.
    # Facet IDs force the middle facet to win a same-point/same-normal tie.
    surface = boundary_models.triangulated_rigid_surface(
        vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0),
                  (-1, 0, 0), (0, -1, 0)),
        triangles=((0, 1, 2), (0, 2, 3), (0, 3, 4)),
        facet_ids=(10, 0, 20),
    )
    nodes = np.asarray(((0.0, 0.0, -0.1),) * 3)
    projection = surface.project(nodes[:1])
    assert projection.all_valid
    assert int(projection.entity_ids[0]) == 0
    with pytest.raises(ValueError, match="AFM-IMPLICIT-CONTACT-BRANCH"):
        _evaluate(surface, nodes)


def test_closed_coplanar_vertex_fan_retains_the_planar_tangent():
    surface = boundary_models.triangulated_rigid_surface(
        vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0),
                  (-1, 0, 0), (0, -1, 0)),
        triangles=((0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1)),
        facet_ids=(10, 0, 20, 30),
    )
    nodes = np.asarray(((0.0, 0.0, -0.1),) * 3)
    contribution = _evaluate(surface, nodes)
    assert contribution.active_points == 1
    assert contribution.assembly.potential_energy > 0.0


def test_noncoplanar_edge_never_gets_a_fabricated_tangent():
    surface = boundary_models.triangulated_rigid_surface(
        vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
        triangles=((0, 1, 2), (1, 0, 3)),
    )
    positions = np.asarray(((0.5, -0.01, -0.01),) * 3)
    with pytest.raises(ValueError, match="invalid projection|AFM-IMPLICIT-CONTACT-BRANCH"):
        _evaluate(surface, positions)


def test_current_area_and_stale_projection_are_rejected():
    surface = _flat_tool()
    nodes = np.asarray(((0.2, 0.3, -0.1),) * 3)
    trace = boundary_models.ContactTrace(
        (17,), ((0, 1, 2),), ((0.2, 0.3, 0.5),), (1.0,),
        measure_configuration="current",
    )
    evaluation = trace.evaluate(nodes)
    record = boundary_models.ContactProjectionRecord(trace.point_ids, surface.project(evaluation.query_points))
    response = boundary_models.frictionless_penalty_contact_law(10).evaluate(record.projection)
    with pytest.raises(NotImplementedError, match="current-area"):
        linearize_piecewise_planar_contact(evaluation, record, response, surface)
    reference_trace = boundary_models.ContactTrace(
        (17,), ((0, 1, 2),), ((0.2, 0.3, 0.5),), (1.0,),
    )
    moved = reference_trace.evaluate(nodes + (0, 0, 0.01))
    with pytest.raises(ValueError, match="trace evaluation"):
        linearize_piecewise_planar_contact(moved, record, response, surface)
