import numpy as np
import pytest

from tools.p92_history_kinetics_split import split_derivative, COORDINATES
from tools.p92_spectrum_recovery_joint import NAMES


def test_product_chain_rule_and_second_order_difference_error():
    errors = []
    for h in (.001, .0005):
        samples = []
        for p in (1-h, 1+h):
            incoming = p*p
            samples.append([[p*incoming], [incoming], [p]])
        d = split_derivative(*samples, h)
        assert d["full"][0] == pytest.approx(3+h*h, abs=1e-12)
        assert d["history"][0] == pytest.approx(2., abs=1e-12)
        assert d["kinetics"][0] == pytest.approx(1., abs=1e-12)
        errors.append(d["chain_rule_residual"][0])
    assert errors[0]/errors[1] == pytest.approx(4., rel=1e-5)


def test_signed_cancellation_is_not_a_positive_fraction():
    low = np.array([[0.], [-2.], [2.]])
    d = split_derivative(low, -low, .5)
    assert d["full"] == [0.]
    assert d["history"] == [4.]
    assert d["kinetics"] == [-4.]
    assert d["chain_rule_relative_error"] == 0.


def test_coordinates_match_declared_mechanisms():
    assert [NAMES[i] for i in COORDINATES] == ["A.reference_flow", "B.reference_flow",
        "B.static_recovery1", "B.static_recovery2", "recoverable.slow_modulus", "recoverable.slow_time"]


@pytest.mark.parametrize("low,high,step", [([], [], .1), ([[1]], [[2]], .1),
    ([[1], [2], [3]], [[2], [3], [np.nan]], .1), ([[1], [2], [3]], [[2], [3], [4]], 0)])
def test_bad_derivative_inputs_rejected(low, high, step):
    with pytest.raises(ValueError):
        split_derivative(low, high, step)
