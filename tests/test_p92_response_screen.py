from copy import deepcopy

import pytest

from tools.p92_response_screen import screen


def fixture():
    fields = dict(rms=0., observed_max_abs=100., observed_excursion=50.,
        last_observed=50., last_predicted=50.,
        early_secant_observed_per_s=1., early_secant_predicted_per_s=1.,
        late_secant_observed_per_s=.1, late_secant_predicted_per_s=.1)
    holds = [dict(cycle=1, segment=i, stress_control=False, duration_s=180.,
        fields={"stress_MPa": deepcopy(fields)}, sample_times_s=[1., 2., 150., 180.],
        late_secant_times_s=[150., 180.]) for i in (1, 3)]
    pred = dict(protocol="RF", cycles=[dict(cycle=1)], reversed_post_hold_ramps=[],
                hold_diagnostics=holds)
    comp = dict(records=[dict(cycle=1, observed_peak_MPa=100., observed_valley_MPa=-100.,
                             peak_error_MPa=0., valley_error_MPa=0.)])
    return pred, comp


def test_response_screen_pass_is_not_engineering_or_life_admission():
    result = screen(*fixture(), requested_cycles=1)
    assert result["status"] == "passed"
    assert not result["engineering_qualified"] and not result["life_qualified"]


def test_good_extrema_do_not_hide_bad_hold_or_secant():
    pred, comp = fixture()
    pred["hold_diagnostics"][0]["fields"]["stress_MPa"]["last_predicted"] += 6.
    result = screen(pred, comp, requested_cycles=1)
    assert result["status"] == "failed"
    pred, comp = fixture()
    pred["hold_diagnostics"][0]["fields"]["stress_MPa"]["early_secant_predicted_per_s"] = 1.3
    assert screen(pred, comp, requested_cycles=1)["status"] == "failed"


def test_unresolved_slope_and_missing_coverage_are_not_passed():
    pred, comp = fixture()
    pred["hold_diagnostics"][0]["fields"]["stress_MPa"]["late_secant_observed_per_s"] = .0001
    assert screen(pred, comp, requested_cycles=1)["status"] == "unresolved"
    pred, comp = fixture()
    pred["hold_diagnostics"].pop()
    assert screen(pred, comp, requested_cycles=1)["status"] == "unresolved"
    pred, comp = fixture()
    comp["records"] = []
    assert screen(pred, comp, requested_cycles=1)["status"] == "unresolved"


def test_incomplete_or_artificially_continued_path_fails():
    pred, comp = fixture()
    assert screen(pred, comp, requested_cycles=2)["status"] == "failed"
    pred["incompatible_continuation"] = True
    assert screen(pred, comp, requested_cycles=1)["status"] == "failed"
    with pytest.raises(ValueError):
        screen(pred, comp, requested_cycles=0)


def test_stress_hold_checks_strain_response_not_zero_prescribed_stress_slope():
    pred, comp = fixture()
    pred["protocol"] = "SLC"
    hold = deepcopy(pred["hold_diagnostics"][0])
    hold["segment"], hold["stress_control"] = 5, True
    hold["fields"]["strain"] = dict(rms=0., observed_max_abs=.001,
        observed_excursion=.001, last_observed=.001, last_predicted=.001,
        early_secant_observed_per_s=.0001, early_secant_predicted_per_s=.0001,
        late_secant_observed_per_s=.00001, late_secant_predicted_per_s=.00001)
    hold["fields"]["stress_MPa"]["early_secant_observed_per_s"] = 0.
    hold["fields"]["stress_MPa"]["late_secant_observed_per_s"] = 0.
    pred["hold_diagnostics"].append(hold)
    assert screen(pred, comp, requested_cycles=1)["status"] == "passed"
    hold["fields"]["strain"]["last_predicted"] = .00111
    assert screen(pred, comp, requested_cycles=1)["status"] == "failed"
