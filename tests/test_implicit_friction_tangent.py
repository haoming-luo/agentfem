"""Bounded fixed-old-state derivative of planar Coulomb return mapping."""

from __future__ import annotations

import numpy as np
import pytest

from agentfem import boundary_models
from agentfem.boundary_models.implicit_friction import (
    linearize_planar_penalty_friction,
)


@pytest.mark.parametrize("increment,expected_branch", [
    (0.01, "sticking"),
    (0.10, "sliding"),
])
@pytest.mark.parametrize("normal", [
    (0.0, 0.0, 1.0),
    (0.0, 2**-0.5, 2**-0.5),
])
def test_planar_friction_tangent_matches_fixed_old_state_directional_difference(
    increment, expected_branch, normal,
):
    surface = boundary_models.rigid_plane(point=(0, 0, 0), normal=normal)
    trace = boundary_models.ContactTrace(
        point_ids=(7,), node_ids=((0, 1, 2),),
        shape_values=((0.2, 0.3, 0.5),), weights=(0.75,),
    )
    normal_law = boundary_models.frictionless_penalty_contact_law(100.0)
    friction_law = boundary_models.penalty_coulomb_friction_law(
        0.2, 50.0,
    )
    base_position = -0.1 * np.asarray(normal)
    nodes = np.repeat(base_position[None, :], 3, axis=0)
    accepted_position = base_position - np.asarray((increment, 0.0, 0.0))

    def trial(nodal_positions):
        evaluation = trace.evaluate(nodal_positions)
        record = boundary_models.ContactProjectionRecord(
            trace.point_ids, surface.project(evaluation.query_points),
        )
        normal = normal_law.evaluate(record.projection)
        friction = friction_law.evaluate(
            point_ids=trace.point_ids,
            normal_response=normal,
            relative_displacement_increment=(
                evaluation.query_points - accepted_position
            ),
        )
        assembled = evaluation.assemble_friction(record, friction).mechanical
        tangent = linearize_planar_penalty_friction(
            evaluation, record, normal, friction, friction_law, surface,
        )
        return friction, assembled.nodal_structural_residual, tangent

    friction, base_residual, tangent = trial(nodes)
    assert bool(getattr(friction, expected_branch)[0])
    assert tangent.sticking_points + tangent.sliding_points == 1
    assert not tangent.values.flags.writeable
    if expected_branch == "sliding":
        matrix = np.zeros((9, 9))
        np.add.at(matrix, (tangent.rows, tangent.columns), tangent.values)
        assert np.linalg.norm(matrix - matrix.T) > 1.0e-6
    direction = np.asarray(((0.2, 0.3, 0.1), (-0.4, 0.5, -0.2), (0.1, -0.1, 0.3)))
    perturbation = 1.0e-7
    _, perturbed_residual, _ = trial(nodes + perturbation * direction)
    np.testing.assert_allclose(
        (perturbed_residual - base_residual) / perturbation,
        tangent.tangent_action(direction), rtol=2.0e-6, atol=2.0e-7,
    )


def test_planar_friction_tangent_rejects_nonsmooth_cap_and_opening():
    surface = boundary_models.rigid_plane(point=(0, 0, 0), normal=(0, 0, 1))
    trace = boundary_models.ContactTrace(
        point_ids=(7,), node_ids=((0, 1, 2),),
        shape_values=((0.2, 0.3, 0.5),), weights=(1.0,),
    )
    normal_law = boundary_models.frictionless_penalty_contact_law(100.0)
    law = boundary_models.penalty_coulomb_friction_law(0.2, 50.0)

    def tangent_at(z, slip):
        evaluation = trace.evaluate(np.asarray(((0.0, 0.0, z),) * 3))
        record = boundary_models.ContactProjectionRecord(
            trace.point_ids, surface.project(evaluation.query_points),
        )
        normal = normal_law.evaluate(record.projection)
        friction = law.evaluate(
            point_ids=trace.point_ids, normal_response=normal,
            relative_displacement_increment=((slip, 0.0, 0.0),),
        )
        return linearize_planar_penalty_friction(
            evaluation, record, normal, friction, law, surface,
        )

    # At z=-0.1 the pressure cap is mu*k_n*0.1 = 2; k_t*0.04 = 2.
    with pytest.raises(ValueError, match="AFM-IMPLICIT-FRICTION-BRANCH"):
        tangent_at(-0.1, 0.04)
    with pytest.raises(ValueError, match="AFM-IMPLICIT-FRICTION-BRANCH"):
        tangent_at(0.0, 0.01)
    opened = tangent_at(0.1, 0.01)
    assert opened.values.size == 0


def test_planar_friction_tangent_rejects_a_stale_projection():
    surface = boundary_models.rigid_plane(point=(0, 0, 0), normal=(0, 0, 1))
    trace = boundary_models.ContactTrace(
        point_ids=(7,), node_ids=((0,),), shape_values=((1.0,),),
        weights=(1.0,),
    )
    evaluation = trace.evaluate(np.asarray(((0.0, 0.0, -0.1),)))
    record = boundary_models.ContactProjectionRecord(
        trace.point_ids,
        surface.project(evaluation.query_points + ((0.01, 0.0, 0.0),)),
    )
    normal = boundary_models.frictionless_penalty_contact_law(100.0).evaluate(
        record.projection,
    )
    law = boundary_models.penalty_coulomb_friction_law(0.2, 50.0)
    friction = law.evaluate(
        point_ids=trace.point_ids, normal_response=normal,
        relative_displacement_increment=((0.01, 0.0, 0.0),),
    )
    with pytest.raises(ValueError, match="stale"):
        linearize_planar_penalty_friction(
            evaluation, record, normal, friction, law, surface,
        )
