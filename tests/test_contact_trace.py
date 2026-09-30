from __future__ import annotations

import numpy as np
import pytest

from agentfem import boundary_models


def _line_trace():
    offset = 1.0 / (2.0 * np.sqrt(3.0))
    return boundary_models.ContactTrace(
        point_ids=(1001, 1002),
        node_ids=((0, 1), (0, 1)),
        shape_values=((0.5 + offset, 0.5 - offset), (0.5 - offset, 0.5 + offset)),
        weights=(0.5, 0.5),
        measure_configuration="reference",
        source="two_point_line_quadrature",
    )


def _record(trace, projection):
    return boundary_models.ContactProjectionRecord(trace.point_ids, projection)


def test_contact_trace_interpolates_and_assembles_plane_response():
    trace = _line_trace()
    evaluation = trace.evaluate(((0.0, -0.1), (1.0, -0.1)))
    surface = boundary_models.rigid_plane(
        point=(0.0, 0.0),
        normal=(0.0, 1.0),
    )
    projection = surface.project(evaluation.query_points)
    record = _record(trace, projection)
    response = boundary_models.frictionless_penalty_contact_law(100.0).evaluate(
        record.projection
    )

    assembly = evaluation.assemble(
        record,
        response,
        surface_reference_point=(0.0, 0.0),
    )

    np.testing.assert_allclose(
        evaluation.query_points[:, 0],
        (0.5 - 1.0 / (2.0 * np.sqrt(3.0)), 0.5 + 1.0 / (2.0 * np.sqrt(3.0))),
    )
    np.testing.assert_allclose(evaluation.query_points[:, 1], -0.1)
    np.testing.assert_allclose(
        assembly.nodal_structural_residual,
        ((0.0, -5.0), (0.0, -5.0)),
    )
    np.testing.assert_allclose(assembly.contact_force_on_structure, (0.0, 10.0))
    np.testing.assert_allclose(assembly.contact_force_on_surface, (0.0, -10.0))
    np.testing.assert_allclose(assembly.surface_generalized_force, (0.0, 10.0))
    np.testing.assert_allclose(assembly.surface_generalized_moment, (5.0,))
    np.testing.assert_allclose(assembly.potential_energy, 0.5)
    np.testing.assert_allclose(assembly.point_potential_contributions, (0.25, 0.25))
    assert assembly.point_ids.tolist() == [1001, 1002]
    assert assembly.summary()["linearization"] == "not_provided"


def test_contact_trace_assembles_three_dimensional_resultant_and_moment():
    trace = boundary_models.ContactTrace(
        point_ids=(71,),
        node_ids=((0, 1, 2),),
        shape_values=((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),),
        weights=(0.5,),
        source="triangle_centroid_reference_rule",
    )
    evaluation = trace.evaluate(((0.0, 0.0, -0.2), (1.0, 0.0, -0.2), (0.0, 1.0, -0.2)))
    surface = boundary_models.rigid_plane(
        point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
    )
    projection = surface.project(evaluation.query_points)
    record = _record(trace, projection)
    response = boundary_models.frictionless_penalty_contact_law(50.0).evaluate(
        record.projection
    )

    assembly = evaluation.assemble(
        record,
        response,
        surface_reference_point=(0.0, 0.0, 0.0),
    )

    np.testing.assert_allclose(
        assembly.nodal_structural_residual,
        ((0.0, 0.0, -5.0 / 3.0),) * 3,
    )
    np.testing.assert_allclose(assembly.contact_force_on_structure, (0.0, 0.0, 5.0))
    np.testing.assert_allclose(
        assembly.surface_generalized_moment,
        (5.0 / 3.0, -5.0 / 3.0, 0.0),
    )
    np.testing.assert_allclose(assembly.potential_energy, 0.5)


def test_contact_trace_assembles_friction_force_and_distinct_energy_channels():
    trace = _line_trace()
    evaluation = trace.evaluate(((0.0, -0.1), (1.0, -0.1)))
    surface = boundary_models.rigid_plane(
        point=(0.0, 0.0),
        normal=(0.0, 1.0),
    )
    record = _record(trace, surface.project(evaluation.query_points))
    normal = boundary_models.frictionless_penalty_contact_law(100.0).evaluate(
        record.projection
    )
    friction = boundary_models.penalty_coulomb_friction_law(0.5, 100.0).evaluate(
        point_ids=trace.point_ids,
        normal_response=normal,
        relative_displacement_increment=((0.02, 0.0), (0.1, 0.0)),
    )

    assembly = evaluation.assemble_friction(
        record,
        friction,
        surface_reference_point=(0.0, -1.0),
    )

    np.testing.assert_allclose(
        np.sum(assembly.nodal_structural_residual, axis=0),
        (3.5, 0.0),
    )
    np.testing.assert_allclose(assembly.contact_force_on_structure, (-3.5, 0.0))
    np.testing.assert_allclose(assembly.contact_force_on_surface, (3.5, 0.0))
    np.testing.assert_allclose(assembly.surface_generalized_moment, (3.5,))
    assert assembly.recoverable_penalty_energy == pytest.approx(0.0725)
    assert assembly.dissipation_increment == pytest.approx(0.125)
    assert assembly.cumulative_dissipation == pytest.approx(0.125)
    assert assembly.separation_release_increment == pytest.approx(0.0)
    assert assembly.summary()["energy_semantics"] == (
        "recoverable_dissipated_release_separate"
    )


def test_contact_trace_integrates_separation_release_without_calling_it_friction():
    trace = _line_trace()
    closed_evaluation = trace.evaluate(((0.0, -0.1), (1.0, -0.1)))
    surface = boundary_models.rigid_plane(
        point=(0.0, 0.0),
        normal=(0.0, 1.0),
    )
    closed_record = _record(trace, surface.project(closed_evaluation.query_points))
    closed_normal = boundary_models.frictionless_penalty_contact_law(100.0).evaluate(
        closed_record.projection
    )
    law = boundary_models.penalty_coulomb_friction_law(0.5, 100.0)
    closed_friction = law.evaluate(
        point_ids=trace.point_ids,
        normal_response=closed_normal,
        relative_displacement_increment=((0.02, 0.0), (0.1, 0.0)),
    )
    opened_evaluation = trace.evaluate(((0.0, 0.1), (1.0, 0.1)))
    opened_record = _record(trace, surface.project(opened_evaluation.query_points))
    opened_normal = boundary_models.frictionless_penalty_contact_law(100.0).evaluate(
        opened_record.projection
    )
    opened_friction = law.evaluate(
        point_ids=trace.point_ids,
        normal_response=opened_normal,
        relative_displacement_increment=((0.0, 0.0), (0.0, 0.0)),
        accepted=closed_friction.record,
    )

    assembly = opened_evaluation.assemble_friction(
        opened_record,
        opened_friction,
    )

    assert assembly.recoverable_penalty_energy == pytest.approx(0.0)
    assert assembly.dissipation_increment == pytest.approx(0.0)
    assert assembly.cumulative_dissipation == pytest.approx(0.125)
    assert assembly.separation_release_increment == pytest.approx(0.0725)
    assert assembly.cumulative_separation_release == pytest.approx(0.0725)


def test_contact_trace_rejects_friction_from_another_identity_or_normal_frame():
    trace = _line_trace()
    evaluation = trace.evaluate(((0.0, -0.1), (1.0, -0.1)))
    surface = boundary_models.rigid_plane(
        point=(0.0, 0.0),
        normal=(0.0, 1.0),
    )
    record = _record(trace, surface.project(evaluation.query_points))
    normal = boundary_models.frictionless_penalty_contact_law(100.0).evaluate(
        record.projection
    )
    law = boundary_models.penalty_coulomb_friction_law(0.5, 100.0)
    wrong_identity = law.evaluate(
        point_ids=(2001, 2002),
        normal_response=normal,
        relative_displacement_increment=((0.0, 0.0), (0.0, 0.0)),
    )
    with pytest.raises(ValueError, match="identity differs"):
        evaluation.assemble_friction(record, wrong_identity)

    rotated_surface = boundary_models.rigid_plane(
        point=(0.0, 0.0),
        normal=(1.0, 0.0),
    )
    rotated_normal = boundary_models.frictionless_penalty_contact_law(100.0).evaluate(
        rotated_surface.project(evaluation.query_points)
    )
    wrong_frame = law.evaluate(
        point_ids=trace.point_ids,
        normal_response=rotated_normal,
        relative_displacement_increment=((0.0, 0.0), (0.0, 0.0)),
    )
    with pytest.raises(ValueError, match="normals differ"):
        evaluation.assemble_friction(record, wrong_frame)


def test_contact_trace_rejects_response_from_different_evaluation():
    trace = _line_trace()
    evaluation = trace.evaluate(((0.0, -0.1), (1.0, -0.1)))
    other = trace.evaluate(((0.0, -0.2), (1.0, -0.2)))
    surface = boundary_models.rigid_plane(
        point=(0.0, 0.0),
        normal=(0.0, 1.0),
    )
    projection = surface.project(other.query_points)
    record = _record(trace, projection)
    response = boundary_models.frictionless_penalty_contact_law(100.0).evaluate(
        record.projection
    )

    with pytest.raises(ValueError, match="not evaluated at this trace"):
        evaluation.assemble(record, response)

    wrong_identity = boundary_models.ContactProjectionRecord(
        (2001, 2002),
        surface.project(evaluation.query_points),
    )
    wrong_response = boundary_models.frictionless_penalty_contact_law(100.0).evaluate(
        wrong_identity.projection
    )
    with pytest.raises(ValueError, match="identity differs"):
        evaluation.assemble(wrong_identity, wrong_response)


def test_contact_trace_canonicalizes_all_rows_by_stable_point_identity():
    trace = boundary_models.ContactTrace(
        point_ids=(9, 3),
        node_ids=((1,), (0,)),
        shape_values=((1.0,), (1.0,)),
        weights=(0.9, 0.3),
    )

    assert trace.point_ids.tolist() == [3, 9]
    assert trace.node_ids.tolist() == [[0], [1]]
    np.testing.assert_allclose(trace.weights, (0.3, 0.9))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"point_ids": (1, 1)}, "unique"),
        ({"node_ids": ((0, 0), (0, 1))}, "unique within"),
        ({"shape_values": ((0.25, 0.25), (0.5, 0.5))}, "partition"),
        ({"weights": (0.5, 0.0)}, "positive"),
        ({"measure_configuration": "mystery"}, "reference.*current"),
    ],
)
def test_contact_trace_fails_closed_on_invalid_contract(changes, message):
    values = {
        "point_ids": (1, 2),
        "node_ids": ((0, 1), (0, 1)),
        "shape_values": ((0.5, 0.5), (0.5, 0.5)),
        "weights": (0.5, 0.5),
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        boundary_models.ContactTrace(**values)


def test_contact_trace_rejects_invalid_nodal_positions_and_reference_point():
    trace = _line_trace()
    with pytest.raises(ValueError, match="outside nodal positions"):
        trace.evaluate(((0.0, 0.0),))
    with pytest.raises(ValueError, match="finite"):
        trace.evaluate(((0.0, 0.0), (np.nan, 0.0)))

    evaluation = trace.evaluate(((0.0, -0.1), (1.0, -0.1)))
    surface = boundary_models.rigid_plane(
        point=(0.0, 0.0),
        normal=(0.0, 1.0),
    )
    projection = surface.project(evaluation.query_points)
    record = _record(trace, projection)
    response = boundary_models.frictionless_penalty_contact_law(100.0).evaluate(
        record.projection
    )
    with pytest.raises(ValueError, match="spatial dimension"):
        evaluation.assemble(
            record,
            response,
            surface_reference_point=(0.0, 0.0, 0.0),
        )


def test_contact_trace_supports_empty_parallel_shards_and_immutable_results():
    trace = boundary_models.ContactTrace(
        point_ids=np.empty((0,), dtype=np.int64),
        node_ids=np.empty((0, 3), dtype=np.int64),
        shape_values=np.empty((0, 3), dtype=float),
        weights=np.empty((0,), dtype=float),
        source="empty_owned_shard",
    )
    evaluation = trace.evaluate(np.empty((0, 3), dtype=float))
    surface = boundary_models.rigid_plane(
        point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
    )
    projection = surface.project(evaluation.query_points)
    record = _record(trace, projection)
    response = boundary_models.frictionless_penalty_contact_law(1.0).evaluate(
        record.projection
    )

    assembly = evaluation.assemble(
        record,
        response,
        surface_reference_point=(0.0, 0.0, 0.0),
    )

    assert assembly.nodal_structural_residual.shape == (0, 3)
    np.testing.assert_allclose(assembly.contact_force_on_structure, 0.0)
    np.testing.assert_allclose(assembly.surface_generalized_moment, 0.0)
    with pytest.raises(ValueError):
        assembly.point_ids[0:0] = ()

    friction = boundary_models.penalty_coulomb_friction_law(0.3, 1.0).evaluate(
        point_ids=trace.point_ids,
        normal_response=response,
        relative_displacement_increment=np.empty((0, 3), dtype=float),
    )
    friction_assembly = evaluation.assemble_friction(
        record,
        friction,
        surface_reference_point=(0.0, 0.0, 0.0),
    )
    assert friction_assembly.nodal_structural_residual.shape == (0, 3)
    assert friction_assembly.cumulative_dissipation == pytest.approx(0.0)
    assert friction_assembly.cumulative_separation_release == pytest.approx(0.0)
    with pytest.raises(ValueError):
        friction_assembly.point_dissipation_increment_contributions[0:0] = ()
