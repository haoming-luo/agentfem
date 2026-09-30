from __future__ import annotations

import copy

import numpy as np
import pytest

from agentfem import boundary_models


def _normal_response(points, *, penalty=100.0, normal=(0.0, 1.0)):
    surface = boundary_models.rigid_plane(
        point=tuple(0.0 for _ in normal),
        normal=normal,
    )
    projection = surface.project(points)
    return boundary_models.frictionless_penalty_contact_law(penalty).evaluate(
        projection
    )


def test_penalty_coulomb_stick_and_slide_have_distinct_energy_semantics():
    normal = _normal_response(((0.0, -0.2), (1.0, -0.2)))
    law = boundary_models.penalty_coulomb_friction_law(
        coefficient=0.5,
        tangential_penalty=100.0,
    )

    response = law.evaluate(
        point_ids=(8, 3),
        normal_response=normal,
        relative_displacement_increment=((0.05, 9.0), (0.2, -4.0)),
    )

    # Canonical point order is 3, 8. Normal components of the supplied motion
    # do not enter the tangential update.
    np.testing.assert_array_equal(response.record.point_ids, (3, 8))
    np.testing.assert_array_equal(response.sticking, (False, True))
    np.testing.assert_array_equal(response.sliding, (True, False))
    np.testing.assert_allclose(
        response.record.elastic_slips,
        ((0.1, 0.0), (0.05, 0.0)),
    )
    np.testing.assert_allclose(
        response.contact_tractions_on_structure,
        ((-10.0, 0.0), (-5.0, 0.0)),
    )
    np.testing.assert_allclose(
        response.recoverable_penalty_energy_densities,
        (0.5, 0.125),
    )
    np.testing.assert_allclose(response.dissipation_increment_densities, (1.0, 0.0))
    np.testing.assert_allclose(
        response.record.cumulative_dissipation_densities,
        response.dissipation_increment_densities,
    )


def test_friction_state_is_transactional_and_restartable():
    normal = _normal_response(((0.0, -0.1),))
    law = boundary_models.penalty_coulomb_friction_law(0.3, 50.0)
    first = law.evaluate(
        point_ids=(42,),
        normal_response=normal,
        relative_displacement_increment=((0.02, 0.0),),
    )
    state = boundary_models.tangential_contact_state()
    state.begin(first)
    state.rollback()
    assert state.accepted is None

    state.begin(first)
    state.commit()
    snapshot = state.snapshot()
    second = law.evaluate(
        point_ids=(42,),
        normal_response=normal,
        relative_displacement_increment=((0.01, 0.0),),
        accepted=state.accepted,
    )
    state.begin(second)
    with pytest.raises(RuntimeError, match="accepted boundary"):
        state.snapshot()
    state.rollback()

    restored = boundary_models.tangential_contact_state()
    restored.restore(snapshot)
    np.testing.assert_allclose(restored.accepted.elastic_slips, ((0.02, 0.0),))

    corrupted = copy.deepcopy(snapshot)
    corrupted["accepted"]["cumulative_dissipation_densities"][0] = -1.0
    with pytest.raises(ValueError, match="cannot be negative"):
        restored.restore(corrupted)


def test_tangential_history_follows_rotating_surface_normal_without_energy_loss():
    first_normal = _normal_response(
        ((-0.1, 0.0),),
        normal=(1.0, 0.0),
    )
    law = boundary_models.penalty_coulomb_friction_law(10.0, 25.0)
    first = law.evaluate(
        point_ids=(1,),
        normal_response=first_normal,
        relative_displacement_increment=((0.0, 0.2),),
    )
    rotated_normal = _normal_response(
        ((0.0, -0.1),),
        normal=(0.0, 1.0),
    )
    rotated = law.evaluate(
        point_ids=(1,),
        normal_response=rotated_normal,
        relative_displacement_increment=((0.0, 0.0),),
        accepted=first.record,
    )

    np.testing.assert_allclose(rotated.record.elastic_slips, ((-0.2, 0.0),))
    assert np.linalg.norm(rotated.record.elastic_slips[0]) == pytest.approx(0.2)
    assert rotated.recoverable_penalty_energy_densities[0] == pytest.approx(0.5)
    assert rotated.dissipation_increment_densities[0] == pytest.approx(0.0)


def test_three_dimensional_history_uses_minimal_normal_rotation():
    first_normal = _normal_response(
        ((-0.1, 0.0, 0.0),),
        normal=(1.0, 0.0, 0.0),
    )
    law = boundary_models.penalty_coulomb_friction_law(10.0, 25.0)
    first = law.evaluate(
        point_ids=(1,),
        normal_response=first_normal,
        relative_displacement_increment=((0.0, 0.2, 0.3),),
    )
    rotated_normal = _normal_response(
        ((0.0, -0.1, 0.0),),
        normal=(0.0, 1.0, 0.0),
    )
    rotated = law.evaluate(
        point_ids=(1,),
        normal_response=rotated_normal,
        relative_displacement_increment=((0.0, 0.0, 0.0),),
        accepted=first.record,
    )

    np.testing.assert_allclose(rotated.record.elastic_slips, ((-0.2, 0.0, 0.3),))
    assert np.linalg.norm(rotated.record.elastic_slips[0]) == pytest.approx(
        np.sqrt(0.13)
    )


def test_contact_opening_reports_penalty_release_without_false_dissipation():
    closed = _normal_response(((0.0, -0.1),))
    law = boundary_models.penalty_coulomb_friction_law(1.0, 80.0)
    first = law.evaluate(
        point_ids=(4,),
        normal_response=closed,
        relative_displacement_increment=((0.025, 0.0),),
    )
    opened = _normal_response(((0.0, 0.2),))
    released = law.evaluate(
        point_ids=(4,),
        normal_response=opened,
        relative_displacement_increment=((1.0, 0.0),),
        accepted=first.record,
    )

    np.testing.assert_allclose(released.record.elastic_slips, 0.0)
    assert released.separation_release_densities[0] == pytest.approx(0.025)
    assert released.dissipation_increment_densities[0] == pytest.approx(0.0)
    assert released.record.cumulative_dissipation_densities[0] == pytest.approx(0.0)
    assert released.record.cumulative_separation_release_densities[0] == pytest.approx(
        0.025
    )


def test_friction_rejects_ambiguous_history_transport_and_identity_changes():
    normal = _normal_response(((-0.1, 0.0),), normal=(1.0, 0.0))
    law = boundary_models.penalty_coulomb_friction_law(0.2, 10.0)
    first = law.evaluate(
        point_ids=(7,),
        normal_response=normal,
        relative_displacement_increment=((0.0, 0.1),),
    )
    reversed_normal = _normal_response(((0.1, 0.0),), normal=(-1.0, 0.0))
    with pytest.raises(ValueError, match="ambiguous"):
        law.evaluate(
            point_ids=(7,),
            normal_response=reversed_normal,
            relative_displacement_increment=((0.0, 0.0),),
            accepted=first.record,
        )
    with pytest.raises(ValueError, match="identity differs"):
        law.evaluate(
            point_ids=(8,),
            normal_response=normal,
            relative_displacement_increment=((0.0, 0.0),),
            accepted=first.record,
        )


def test_friction_contract_refuses_invalid_parameters_and_claims_no_implicit_tangent():
    with pytest.raises(ValueError, match="non-negative"):
        boundary_models.penalty_coulomb_friction_law(-0.1, 1.0)
    with pytest.raises(ValueError, match="positive"):
        boundary_models.penalty_coulomb_friction_law(0.2, 0.0)

    law = boundary_models.penalty_coulomb_friction_law(0.2, 100.0)
    assert law.summary()["linearization"] == "not_provided"
    assert law.summary()["intended_procedure"] == "explicit"
