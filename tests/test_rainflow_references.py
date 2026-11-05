# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Published numerical counting reference, not a material-life validation."""

import numpy as np
import pytest

from agentfem.constitutive.fatigue import BasquinCurve, assess_history, rainflow_cycles, turning_points


def test_nested_reversals_match_published_mathworks_count_range_mean_table():
    # Numerical reference: MathWorks rainflow documentation, "Algorithms".
    # https://www.mathworks.com/help/signal/ref/rainflow.html
    # Independent published count/range/mean values; no MATLAB execution claim.
    history = [-2., 1., -3., 5., -1., 3., -4., 4., -3., 1., -2., 3., 2., 6.]
    expected = [(0.5, 3., -0.5), (0.5, 4., -1.), (1., 4., 1.),
                (0.5, 8., 1.), (1., 3., -0.5), (1., 1., 2.5),
                (1., 7., 0.5), (0.5, 9., 0.5), (0.5, 10., 1.)]
    actual = [(c.count, c.stress_range, c.mean_stress) for c in rainflow_cycles(history)]
    assert actual == expected
    # Holding a reversal longer does not create additional fatigue cycles.
    assert rainflow_cycles(np.repeat(history, 3)) == rainflow_cycles(history)
    # Additional monotone samples preserve the same turning points.
    time = np.arange(len(history))
    dense = np.interp(np.linspace(0., len(history) - 1., 131), time, history)
    assert rainflow_cycles(dense) == rainflow_cycles(history)


@pytest.mark.parametrize("scale", [1.e-200, 1.e-3, 1., 1.e200])
def test_counting_is_scale_invariant_without_slope_product_underflow(scale):
    history = np.array([0., 1., 0., 2., 0., 1., 0.])
    reference = rainflow_cycles(history)
    with np.errstate(over="raise", under="raise", invalid="raise"):
        actual = rainflow_cycles(history * scale)
    assert [c.count for c in actual] == [c.count for c in reference]
    np.testing.assert_allclose([c.stress_range / scale for c in actual],
                               [c.stress_range for c in reference])
    np.testing.assert_allclose([c.mean_stress / scale for c in actual],
                               [c.mean_stress for c in reference])


def test_finite_large_mean_is_not_rejected_due_to_intermediate_sum_overflow():
    cycles = rainflow_cycles([8.e307, 1.e308, 8.e307])
    assert len(cycles) == 2
    assert all(c.mean_stress == pytest.approx(9.e307) for c in cycles)


def test_unrepresentable_range_is_rejected_not_silently_counted():
    with pytest.raises(ValueError, match="stress_range"):
        rainflow_cycles([-1.e308, 1.e308, -1.e308])


@pytest.mark.parametrize("history", [[0., 1.j, 0.], [0., np.nan, 0.], [0., np.inf, 0.]])
def test_invalid_history_cannot_be_reinterpreted_as_real_cycles(history):
    with pytest.raises(ValueError):
        turning_points(history)
    with pytest.raises(ValueError):
        assess_history(history, BasquinCurve(1000., -.1))
