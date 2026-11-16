import math

import numpy as np
import pytest
from scipy.optimize import brentq

from tools.p92_dwell_life_reference import hold_factor, predict_600, screen


def factor(minutes):
    return hold_factor(minutes, alpha=.6, beta=.68, g=9.2, h=1185.,
                       temperature_C=600., amplitude=.004)


def test_zero_hold_and_asymptotic_bound():
    assert factor(0.) == 1.
    assert factor(1e300) == pytest.approx(.4)
    values = [factor(t) for t in np.logspace(-8, 6, 60)]
    assert all(.4 <= v <= 1. for v in values)
    assert np.all(np.diff(values) <= 0)


def test_logarithmic_implementation_matches_direct_equation():
    expected = .6 * .68**(3*math.exp(9.2*(600-20)/1510)*math.exp(-1185*.004))+.4
    assert factor(3.) == pytest.approx(expected, rel=1e-14)


def test_separate_scalar_inversion_of_modified_equation():
    result = predict_600(amplitude=.004, tensile_minutes=3.)
    d = factor(3.)
    independent = brentq(lambda n: 397/134509*(2*n/d)**(-.068)
                         + .341*(2*n/d)**(-.61)-.004, 1., 1e6)
    assert result["cycles"] == pytest.approx(independent, rel=1e-10)
    assert predict_600(amplitude=.004)["cycles"] == result["base_cycles"]


@pytest.mark.parametrize("minutes", [-1, float("nan"), float("inf"), True, "1"])
def test_invalid_duration_rejected(minutes):
    with pytest.raises(ValueError):
        factor(minutes)


@pytest.mark.parametrize("options", [dict(tensile_minutes=61),
    dict(compressive_minutes=11), dict(tensile_minutes=1, compressive_minutes=1)])
def test_unverified_paths_rejected(options):
    with pytest.raises(ValueError):
        predict_600(amplitude=.004, **options)


def test_no_dwell_extrapolation_to_other_amplitudes():
    with pytest.raises(ValueError):
        predict_600(amplitude=.003, tensile_minutes=1.)


def test_reproduction_retains_all_observations_without_qualification():
    report = screen()
    assert len(report["records"]) == 7
    assert not report["engineering_qualified"]
    assert not report["independent_validation"]
    assert report["records"][0]["observed_cycles"] == 1303
    assert all(r["cycles"] > 0 for r in report["records"])
