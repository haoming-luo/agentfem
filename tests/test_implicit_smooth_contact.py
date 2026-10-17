"""Curved rigid contact: exact gap derivative and independent finite differences."""

from __future__ import annotations

import numpy as np
import pytest

from agentfem import boundary_models
from agentfem.boundary_models.implicit_smooth_contact import (
    linearize_smooth_rigid_contact,
    smooth_gap_hessian,
)


def _contribution(surface, nodes, *, motion=None, factor=0.0):
    trace = boundary_models.ContactTrace(
        point_ids=(71,),
        node_ids=((0, 1, 2),),
        shape_values=((0.2, 0.3, 0.5),),
        weights=(0.75,),
        source="three_node_reference_trace",
    )
    evaluation = trace.evaluate(nodes)
    projection = surface.project(evaluation.query_points, motion=motion, factor=factor)
    record = boundary_models.ContactProjectionRecord(trace.point_ids, projection)
    response = boundary_models.frictionless_penalty_contact_law(125.0).evaluate(
        projection
    )
    return linearize_smooth_rigid_contact(
        evaluation,
        record,
        response,
        surface,
        motion=motion,
        factor=factor,
        surface_reference_point=(0, 0, 0),
    )


@pytest.mark.parametrize(
    "surface,center",
    [
        (boundary_models.rigid_sphere(center=(0, 0, 0), radius=1), (0.9, 0.1, 0.1)),
        (
            boundary_models.rigid_sphere(
                center=(0, 0, 0), radius=1, admissible_side="interior"
            ),
            (1.1, 0.1, 0.1),
        ),
        (
            boundary_models.rigid_cylinder(
                axis_point=(0, 0, 0), axis_direction=(0, 0, 1), radius=1
            ),
            (0.9, 0.1, 0.1),
        ),
        (
            boundary_models.rigid_cylinder(
                axis_point=(0, 0, 0),
                axis_direction=(0, 0, 1),
                radius=1,
                admissible_side="interior",
            ),
            (1.1, 0.1, 0.1),
        ),
    ],
)
def test_curved_tangent_and_potential_match_independent_finite_difference(
    surface, center
):
    center = np.asarray(center)
    nodes = np.stack(
        (center + (0.02, 0, 0), center + (0, 0.03, 0), center - (0.008, 0.018, 0))
    )
    direction = np.asarray(((0.2, -0.1, 0.3), (-0.1, 0.05, -0.2), (0.1, 0.2, 0.4)))
    contribution = _contribution(surface, nodes)
    assert contribution.active_points == 1
    step = 1e-7
    plus = _contribution(surface, nodes + step * direction)
    minus = _contribution(surface, nodes - step * direction)
    tangent_difference = (
        plus.assembly.nodal_structural_residual
        - minus.assembly.nodal_structural_residual
    ) / (2 * step)
    np.testing.assert_allclose(
        contribution.tangent_action(direction),
        tangent_difference,
        rtol=2e-8,
        atol=2e-8,
    )
    potential_difference = (
        plus.assembly.potential_energy - minus.assembly.potential_energy
    ) / (2 * step)
    np.testing.assert_allclose(
        potential_difference,
        np.sum(contribution.assembly.nodal_structural_residual * direction),
        rtol=2e-8,
        atol=2e-8,
    )
    matrix = np.zeros((9, 9))
    np.add.at(
        matrix,
        (contribution.tangent_rows, contribution.tangent_columns),
        contribution.tangent_values,
    )
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-13)


def test_rotating_cylinder_hessian_tracks_current_axis():
    surface = boundary_models.rigid_cylinder(
        axis_point=(0, 0, 0),
        axis_direction=(0, 0, 1),
        radius=1,
    )
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.1, -0.2, 0),
        rotation=(0, 0.4, 0),
        reference_point=(0, 0, 0),
    )
    points = np.asarray(((0.9, 0.1, 0.2),) * 3)
    factor = 0.75
    contribution = _contribution(surface, points, motion=motion, factor=factor)
    direction = np.asarray(((0.1, 0.3, -0.2), (0.2, -0.1, 0.4), (-0.3, 0.1, 0.2)))
    step = 1e-7
    plus = _contribution(
        surface, points + step * direction, motion=motion, factor=factor
    )
    minus = _contribution(
        surface, points - step * direction, motion=motion, factor=factor
    )
    np.testing.assert_allclose(
        contribution.tangent_action(direction),
        (
            plus.assembly.nodal_structural_residual
            - minus.assembly.nodal_structural_residual
        )
        / (2 * step),
        rtol=2e-8,
        atol=2e-8,
    )


def test_singular_projection_is_rejected_before_curvature_is_invented():
    surface = boundary_models.rigid_sphere(center=(0, 0, 0), radius=1)
    projection = surface.project(((0, 0, 0),))
    with pytest.raises(ValueError, match="valid matching"):
        smooth_gap_hessian(surface, projection)
