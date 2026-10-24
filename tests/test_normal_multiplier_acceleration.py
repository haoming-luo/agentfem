# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Numerical controls do not license State updates or relax acceptance."""

import numpy as np
import pytest

from agentfem import boundary_models, steps
from agentfem.mechanics._normal_multiplier_acceleration import ProjectedMultiplierAitken
from test_normal_augmentation import _record, _single_trace, _state


@pytest.mark.parametrize("pressure_scale", [1e-250, 1., 1e250])
@pytest.mark.parametrize("area_scale", [1e-250, 1., 1e250])
def test_linear_map_secant_is_unit_invariant_and_reaches_exact_fixed_point(
    pressure_scale, area_scale,
):
    acceleration = ProjectedMultiplierAitken([area_scale])
    value = np.zeros(1)
    for _ in range(2):
        plain = .5 * value + pressure_scale
        omega = acceleration.relaxation(value, plain)
        value = np.maximum(0., value + omega * (plain - value))
    assert omega == pytest.approx(2.)
    np.testing.assert_allclose(value / pressure_scale, [2.], atol=1e-14, rtol=0.)
    assert acceleration.resets == 0


def test_vector_secant_uses_area_metric_not_point_count():
    accelerator = ProjectedMultiplierAitken([1., 3.])
    assert accelerator.relaxation([0., 0.], [1., 2.]) == 1.
    actual = accelerator.relaxation([1., 2.], [1.5, 2.4])
    previous, change = np.array([1., 2.]), np.array([-.5, -1.6])
    expected = -np.dot([1., 3.] * previous, change) / np.dot([1., 3.] * change, change)
    assert actual == pytest.approx(expected)


def test_activity_change_degenerate_and_wrong_direction_secants_reset():
    accelerator = ProjectedMultiplierAitken([1., 1.])
    accelerator.relaxation([0., 0.], [1., 1.])
    assert accelerator.relaxation([1., 1.], [0., 2.]) == 1.
    assert accelerator.resets == 1
    assert accelerator.relaxation([0., 2.], [0., 3.]) == 1.
    # Same active set but residual changed to a constant direction; next identical
    # residual cannot define a secant and must not divide by zero.
    assert accelerator.relaxation([0., 3.], [0., 4.]) == 1.
    assert accelerator.resets >= 2
    fresh = ProjectedMultiplierAitken([1.])
    fresh.relaxation([0.], [1.])
    assert fresh.relaxation([1.], [3.]) == 1.  # negative proposed omega
    assert fresh.resets == 1
    zero = ProjectedMultiplierAitken([1.])
    zero.relaxation([0.], [0.])
    assert zero.relaxation([0.], [0.]) == 1.


def test_relaxation_is_bounded_and_near_degenerate_updates_are_not_extrapolated():
    accelerator = ProjectedMultiplierAitken([1.])
    accelerator.relaxation([0.], [1.])
    assert accelerator.relaxation([1.], [1.999]) == 10.
    accelerator = ProjectedMultiplierAitken([1.])
    accelerator.relaxation([0.], [1.])
    assert accelerator.relaxation([100.], [90.]) == .1
    accelerator = ProjectedMultiplierAitken([1.])
    accelerator.relaxation([0.], [1.])
    assert accelerator.relaxation([1.], [2.-1e-12]) == 1.


@pytest.mark.parametrize("weights", [[], [0.], [-1.], [np.nan], [np.inf], [[1.]]])
def test_invalid_area_metric_is_rejected(weights):
    with pytest.raises(ValueError, match="weights"):
        ProjectedMultiplierAitken(weights)


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [-1.], [1., 2.]])
def test_bad_fixed_point_input_does_not_change_history(values):
    acceleration = ProjectedMultiplierAitken([1.])
    acceleration.relaxation([0.], [1.])
    with pytest.raises(ValueError, match="fixed-trace"):
        acceleration.relaxation([1.], values)
    assert acceleration.relaxation([1.], [1.5]) == pytest.approx(2.)


@pytest.mark.parametrize("option", ["unknown", "", None, True, 1, [], np.array([1, 2])])
def test_bad_public_acceleration_policy_is_rejected(option):
    with pytest.raises(ValueError, match="acceleration"):
        steps.normal_augmentation(
            penetration_tolerance=1e-9, complementarity_tolerance=1e-8,
            length_unit="m", pressure_unit="Pa", acceleration=option,
        )


@pytest.mark.parametrize("relaxation", [0., .01, 11., np.nan, np.inf, True, "2"])
def test_invalid_relaxation_is_atomic_and_needs_equilibrium_witness(relaxation):
    trace = _single_trace()
    plane = boundary_models.rigid_plane(point=(0., 0., 0.), normal=(1., 0., 0.))
    state = _state(trace, plane, penalty=10.)
    evaluation, record = _record(plane, (-.1, 0., 0.), trace)
    with pytest.raises(RuntimeError, match="converged projection"):
        state.augment_after_converged(evaluation, record, relaxation=relaxation)
    state.mark_equilibrium_converged(
        evaluation, record, total_residual_norm=0., total_residual_tolerance=1e-12,
    )
    with pytest.raises(ValueError, match="relaxation"):
        state.augment_after_converged(evaluation, record, relaxation=relaxation)
    assert state.augmentation_count == 0
    np.testing.assert_array_equal(state.iterate_multiplier, [0.])
    state.augment_after_converged(evaluation, record, relaxation=2.)
    np.testing.assert_array_equal(state.iterate_multiplier, [2.])
    np.testing.assert_array_equal(state.accepted_multiplier, [0.])
    with pytest.raises(ValueError, match="equilibrium"):
        state.commit(evaluation, record)  # update invalidated the witness
    state.rollback()
    np.testing.assert_array_equal(state.iterate_multiplier, [0.])


def test_relaxed_multiplier_remains_unilateral():
    trace = _single_trace()
    plane = boundary_models.rigid_plane(point=(0., 0., 0.), normal=(1., 0., 0.))
    state = _state(trace, plane, penalty=10., initial_multiplier=[1.])
    evaluation, record = _record(plane, (.1, 0., 0.), trace)
    state.mark_equilibrium_converged(
        evaluation, record, total_residual_norm=0., total_residual_tolerance=1e-12,
    )
    state.augment_after_converged(evaluation, record, relaxation=2.)
    np.testing.assert_array_equal(state.iterate_multiplier, [0.])
