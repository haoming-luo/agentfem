import numpy as np
import pytest

from tools.p92_local_feasibility import admissible, linear_step, same_card, signed_step, signed_errors


def test_linear_tradeoff_and_original_box():
    step, maximum = linear_step([2., 1.], [[1.], [-1.]], [0.], 1.)
    np.testing.assert_allclose(step, [-.5], atol=1e-10)
    assert maximum == pytest.approx(1.5)
    step, maximum = linear_step([2.], [[1.]], [-.9], 1.)
    assert step[0] == pytest.approx(-.1)
    assert maximum == pytest.approx(1.9)


def test_card_identity_permits_roundoff_not_refitting():
    assert same_card({'x': (1., 1e-8)}, {'x': (np.nextafter(1., 2.), 1e-8)})
    assert not same_card({'x': (1., 1e-8)}, {'x': (1., 1.000001e-8)})
    assert not same_card({'x': 0.}, {'x': 1e-100})
    assert not same_card({'x': 'own'}, {'x': 'shared'})


def test_local_improvement_does_not_erase_previous_pass_or_protocol():
    assert admissible([3., 2., .5], [2., 1.5, .9], ['RF', 'SLC', 'RF'])
    assert not admissible([3., 2., .5], [2., 1.5, 1.1], ['RF', 'SLC', 'RF'])
    assert not admissible([3., 2.], [2., 2.1], ['RF', 'SLC'])
    with pytest.raises(ValueError):
        linear_step([2.], [[np.nan]], [0.], 1.)


def test_signed_minimax_preserves_both_sides_at_zero_crossings():
    step, maximum = signed_step([.01, -.01], [[1.], [1.]], [0.], 1.)
    assert step[0] == pytest.approx(0., abs=1e-12)
    assert maximum == pytest.approx(.01)
    step, maximum = signed_step([2., -1.], [[1.], [1.]], [0.], 1.)
    assert step[0] == pytest.approx(-.5)
    assert maximum == pytest.approx(1.5)
    with pytest.raises(ValueError):
        signed_step([np.nan], [[1.]], [0.], 1.)


def test_signed_metric_retains_gate_magnitude_and_unknowns_fail():
    comparison = dict(records=[dict(cycle=1, peak_error_MPa=-2., valley_error_MPa=3.)])
    field = dict(last_predicted=1., last_observed=2., early_secant_predicted_per_s=-2.,
                 early_secant_observed_per_s=-3.)
    prediction = dict(hold_diagnostics=[dict(cycle=1, segment=2, fields={"stress_MPa": field})])
    keys = [("peak_stress", 1, None), ("valley_stress", 1, None),
            ("hold_stress_MPa_last", 1, 2), ("hold_stress_MPa_rms", 1, 2),
            ("early_stress_MPa_secant", 1, 2)]
    magnitudes = [1., 2., 3., 4., 5.]
    result = signed_errors(prediction, comparison, keys, magnitudes)
    np.testing.assert_array_equal(result, [-1., 2., -3., 4., 5.])
    np.testing.assert_array_equal(abs(result), magnitudes)
    with pytest.raises(ValueError, match="Unknown"):
        signed_errors(prediction, comparison, [("unrecognized", 1, None)], [1.])
