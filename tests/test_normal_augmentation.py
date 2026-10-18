"""Fixed-reference-trace normal augmentation, without a global Step claim."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from agentfem import boundary_models
from agentfem.boundary_models.normal_augmentation import (
    FixedTraceNormalMultiplierState,
    integrate_fixed_trace_augmented_normal,
)


def _single_trace(*, weight=1.0):
    return boundary_models.ContactTrace(
        point_ids=(17,), node_ids=((0,),), shape_values=((1.0,),),
        weights=(weight,), source="fixed_single_point_reference_trace",
    )


def _record(surface, point, trace):
    evaluation = trace.evaluate(np.asarray((point,), dtype=float))
    record = boundary_models.ContactProjectionRecord(
        trace.point_ids, surface.project(evaluation.query_points),
    )
    return evaluation, record


def _state(trace, surface, *, penalty, reference_nodes=None, initial_multiplier=None):
    if reference_nodes is None:
        reference_nodes = ((0.0, 0.0, 0.0),)
    return FixedTraceNormalMultiplierState(
        trace,
        reference_evaluation=trace.evaluate(reference_nodes),
        surface_fingerprint=surface.geometry_fingerprint,
        penalty=penalty,
        initial_multiplier=initial_multiplier,
    )


def test_independent_spring_plane_augmentation_reaches_unilateral_solution():
    """A spring driven through a wall has exact u=0 and reaction K*travel."""

    plane = boundary_models.rigid_plane(point=(0, 0, 0), normal=(1, 0, 0))
    trace = _single_trace()
    state = _state(trace, plane, penalty=10.0)
    spring, target = 10.0, -0.1
    for _ in range(40):
        frozen_multiplier = state.iterate_multiplier[0]
        displacement = (spring * target + frozen_multiplier) / (spring + state.penalty)
        evaluation, record = _record(plane, (displacement, 0, 0), trace)
        point = state.evaluate(evaluation, record)
        assert spring * (displacement - target) - point.pressure[0] == pytest.approx(0, abs=1e-14)
        state.mark_equilibrium_converged(
            evaluation, record, total_residual_norm=0.0, total_residual_tolerance=1e-12,
        )
        if state.acceptance_diagnostics(evaluation, record)["maximum_penetration"] <= 1e-9:
            state.commit(
                evaluation, record,
                penetration_tolerance=1e-9,
                complementarity_tolerance=1e-9,
            )
            break
        state.augment_after_converged(evaluation, record)
    else:
        pytest.fail("The exact spring/plane augmented solution did not converge.")
    assert abs(displacement) <= 1e-9
    # The stopped iterate has |u| <= 1e-9, hence |lambda-lambda*| <= 20e-9.
    assert state.accepted_multiplier[0] == pytest.approx(1.0, abs=2e-8)
    with pytest.raises(ValueError, match="Total equilibrium"):
        state.commit(
            evaluation, record,
            penetration_tolerance=1,
            complementarity_tolerance=1,
        )


def test_multiplier_trial_rollback_and_atomic_restore():
    plane = boundary_models.rigid_plane(point=(0, 0, 0), normal=(1, 0, 0))
    trace = _single_trace()
    state = _state(trace, plane, penalty=10.0)
    evaluation, record = _record(plane, (-0.1, 0, 0), trace)
    with pytest.raises(RuntimeError, match="current converged projection"):
        state.augment_after_converged(evaluation, record)
    with pytest.raises(ValueError, match="not converged"):
        state.mark_equilibrium_converged(
            evaluation, record, total_residual_norm=1, total_residual_tolerance=1e-3,
        )
    state.mark_equilibrium_converged(
        evaluation, record, total_residual_norm=0, total_residual_tolerance=1e-12,
    )
    state.augment_after_converged(evaluation, record)
    assert state.iterate_multiplier[0] == pytest.approx(1.0)
    assert state.accepted_multiplier[0] == 0
    with pytest.raises(ValueError, match="Total equilibrium"):
        state.commit(
            evaluation, record,
            penetration_tolerance=0.01,
            complementarity_tolerance=0.01,
        )
    state.mark_equilibrium_converged(
        evaluation, record, total_residual_norm=0, total_residual_tolerance=1e-12,
    )
    with pytest.raises(ValueError, match="penetration/complementarity"):
        state.commit(
            evaluation, record, penetration_tolerance=0.01, complementarity_tolerance=0.01,
        )
    assert state.accepted_multiplier[0] == 0
    state.rollback()
    assert state.iterate_multiplier[0] == 0
    assert state.augmentation_count == 0

    open_evaluation, open_record = _record(plane, (0.0, 0, 0), trace)
    state.mark_equilibrium_converged(
        open_evaluation, open_record, total_residual_norm=0, total_residual_tolerance=1e-12,
    )
    state.commit(
        open_evaluation, open_record,
        penetration_tolerance=0,
        complementarity_tolerance=0,
    )
    good = state.accepted_snapshot()
    candidate = _state(_single_trace(), plane, penalty=10.0)
    candidate.restore_accepted(good)
    assert candidate.accepted_snapshot() == good
    for key, value in (
        ("trace_identity", "wrong"), ("surface_fingerprint", "wrong"),
        ("point_ids", [18]), ("penalty", 20.0), ("multiplier", [-1]),
    ):
        damaged = copy.deepcopy(good)
        damaged[key] = value
        with pytest.raises(ValueError):
            candidate.restore_accepted(damaged)
        assert candidate.accepted_snapshot() == good


def test_inactive_augmented_functional_is_not_physical_stored_energy():
    plane = boundary_models.rigid_plane(point=(0, 0, 0), normal=(1, 0, 0))
    trace = _single_trace()
    state = _state(trace, plane, penalty=10.0, initial_multiplier=(1.0,))
    evaluation, record = _record(plane, (0.2, 0, 0), trace)
    response = state.evaluate(evaluation, record)
    assert response.pressure[0] == 0
    assert response.algorithmic_density[0] == pytest.approx(-0.05)
    assert not hasattr(response, "potential_energy")


@pytest.mark.parametrize(
    "surface,center",
    [
        (
            boundary_models.rigid_plane(point=(0, 0, 0), normal=(1, 0, 0)),
            (0.02, 0.1, 0),
        ),
        (
            boundary_models.rigid_sphere(center=(0, 0, 0), radius=1),
            (0.8, 0.1, 0.2),
        ),
    ],
)
def test_fixed_multiplier_total_tangent_and_functional_direction(surface, center):
    trace = boundary_models.ContactTrace(
        point_ids=(17,), node_ids=((0, 1, 2),),
        shape_values=((0.2, 0.3, 0.5),), weights=(0.75,),
    )
    nodes = np.asarray((
        np.asarray(center) + (0.02, 0, 0),
        np.asarray(center) + (0, 0.03, 0),
        np.asarray(center) - (0.008, 0.018, 0),
    ))
    state = _state(
        trace, surface, penalty=12.0, reference_nodes=nodes,
        initial_multiplier=(1.0,),
    )

    def contribution(values):
        evaluation = trace.evaluate(values)
        record = boundary_models.ContactProjectionRecord(
            trace.point_ids, surface.project(evaluation.query_points),
        )
        response = state.evaluate(evaluation, record)
        curvature = surface.gap_differential(record.projection).hessian
        return integrate_fixed_trace_augmented_normal(
            evaluation, record, response, curvature,
            surface_reference_point=(0, 0, 0),
        )

    base = contribution(nodes)
    assert base.active_points == 1
    direction = np.asarray(((0.2, -0.1, 0.3), (-0.1, 0.05, -0.2), (0.1, 0.2, 0.4)))
    step = 1e-7
    plus = contribution(nodes + step * direction)
    minus = contribution(nodes - step * direction)
    np.testing.assert_allclose(
        base.tangent_action(direction),
        (plus.nodal_structural_residual - minus.nodal_structural_residual) / (2 * step),
        rtol=1e-7, atol=1e-8,
    )
    np.testing.assert_allclose(
        np.sum(base.nodal_structural_residual * direction),
        (plus.algorithmic_functional - minus.algorithmic_functional) / (2 * step),
        rtol=1e-7, atol=1e-8,
    )
    np.testing.assert_allclose(base.contact_force_on_structure, -base.contact_force_on_surface)
    assert base.surface_generalized_moment.shape == (3,)


def test_changing_point_identity_is_not_silently_remapped():
    plane = boundary_models.rigid_plane(point=(0, 0, 0), normal=(1, 0, 0))
    trace = _single_trace()
    state = _state(trace, plane, penalty=10.0)
    evaluation, record = _record(plane, (-0.1, 0, 0), trace)
    with pytest.raises(ValueError, match="identities differ"):
        state.evaluate(
            evaluation,
            boundary_models.ContactProjectionRecord((18,), record.projection),
        )
    changed = _single_trace(weight=2.0)
    changed_evaluation, changed_record = _record(plane, (-0.1, 0, 0), changed)
    with pytest.raises(ValueError, match="trace basis changed"):
        state.evaluate(changed_evaluation, changed_record)
    with pytest.raises(ValueError, match="trace basis differs"):
        _state(trace, plane, penalty=10.0, reference_nodes=((0.1, 0, 0),)).restore_accepted(
            state.accepted_snapshot(),
        )
    other_plane = boundary_models.rigid_plane(point=(0.1, 0, 0), normal=(1, 0, 0))
    other_evaluation, other_record = _record(other_plane, (-0.1, 0, 0), trace)
    with pytest.raises(ValueError, match="rigid-surface identity changed"):
        state.evaluate(other_evaluation, other_record)
