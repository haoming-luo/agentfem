from copy import deepcopy

import numpy as np
import pytest

from tests.test_p92_measured_controls import data
from tools.p92_measured_controls import measured_path, replay
from tools.p92_published_recovery_reference import published_parameters
from tools.p92_two_mechanism import Mechanism, integrate
from tools.p92_waveform_controls import attach_waveforms, integrate_waveform


def test_waveform_matches_piecewise_elastic_solution_without_mutating_state():
    p = published_parameters(600)
    m = (Mechanism(p, enabled=False),)*2
    old = np.zeros(10)
    old[0], old[9] = 12., .0003
    original = old.copy()
    w = dict(times=[0., .3, 1.], deviation=[0., .0001, 0.])
    sol = integrate_waveform(old, 1., .002, w, m)
    q = np.array([0., .2, .3, .7, 1.])
    expected = old[9]+(.002-old[9])*q+np.interp(q, w["times"], w["deviation"])
    np.testing.assert_allclose(sol.sol(q)[9], expected, atol=1e-13)
    np.testing.assert_allclose(sol.sol(q)[0], 12+p.E*(expected-old[9]), atol=1e-8)
    np.testing.assert_array_equal(old, original)
    with pytest.raises(ValueError, match="observations"):
        sol.sol(np.array([1.1]))


def test_zero_shape_matches_original_nonlinear_ramp_and_knot_refinement():
    m = (Mechanism(published_parameters(600)),)*2
    old = np.zeros(10)
    ref = integrate(old, 2., .001, m, rtol=1e-10)
    for knots in ([0., 1., 2.], [0., .5, 1., 1.5, 2.]):
        sol = integrate_waveform(old, 2., .002,
            dict(times=knots, deviation=np.zeros(len(knots))), m, rtol=1e-10)
        np.testing.assert_allclose(sol.y[:, -1], ref.y[:, -1], atol=1e-6, rtol=1e-8)


@pytest.mark.parametrize("protocol", ["RF", "SLR", "SLC"])
def test_attachment_changes_only_ramps_and_retains_observation_identity(protocol):
    source = data(protocol)
    path = measured_path(source, cycles=1)
    result = attach_waveforms(path, source)
    for before, after in zip(path["segments"], result["segments"], strict=True):
        assert "waveform" not in before
        np.testing.assert_array_equal(before["sample_times"], after["sample_times"])
        assert before["source_indices"] == after["source_indices"]
        if before["stress_control"]:
            assert "waveform" not in after
    assert "waveform" not in result["segments"][1]  # peak dwell
    if protocol != "RF":
        assert "waveform" not in result["segments"][3]


def test_hold_observations_do_not_become_prescribed_strain():
    source = data("SLC")
    path = attach_waveforms(measured_path(source, cycles=1), source)
    p = published_parameters(600)
    _, result = replay(path, (Mechanism(p, enabled=False),)*2)
    assert result["segment_endpoints"][3]["state"][9] == pytest.approx(0., abs=1e-12)
    ramp = next(r for r in result["waveform_diagnostics"] if r["segment"] == 4)
    assert ramp["initial_strain_offset"] == pytest.approx(-.001, abs=1e-8)
    changed = deepcopy(path)
    changed["segments"][3]["measured_strain"] += .5
    _, replayed = replay(changed, (Mechanism(p, enabled=False),)*2)
    assert result["segment_endpoints"] == replayed["segment_endpoints"]


def test_waveform_does_not_read_future_cycle_at_duplicate_boundary():
    source = data("RF")
    path = measured_path(source, cycles=1)
    before = attach_waveforms(path, source)
    source["time_s"] = np.r_[source["time_s"], source["time_s"][-1]]
    source["strain"] = np.r_[source["strain"], 100.]
    source["cycle"] = np.r_[source["cycle"], 2]
    after = attach_waveforms(path, source)
    for a, b in zip(before["segments"], after["segments"], strict=True):
        if "waveform" in a:
            np.testing.assert_array_equal(a["waveform"]["deviation"], b["waveform"]["deviation"])


def test_waveform_refuses_changed_identity_and_invalid_support():
    source = data()
    path = measured_path(source, cycles=1)
    source["sha256"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        attach_waveforms(path, source)
    source["sha256"] = path["source_sha256"]
    source["time_s"][10] = source["time_s"][9]
    with pytest.raises(ValueError, match="continuous"):
        attach_waveforms(path, source)


@pytest.mark.parametrize("times,dev", [([0., 0., 1.], [0., 0., 0.]),
    ([0., 1.], [0., 1.]), ([0., 1.], [0., float("nan")])])
def test_invalid_waveforms_are_rejected(times, dev):
    with pytest.raises(ValueError, match="waveform"):
        integrate_waveform(np.zeros(10), 1., .002, dict(times=times, deviation=dev), ())


def test_nonlinear_shaped_ramp_matches_manual_restart_and_independent_solver():
    m = (Mechanism(published_parameters(600)),)*2
    old = np.zeros(10)
    w = dict(times=[0., .4, 1.], deviation=[0., .0001, 0.])
    full = integrate_waveform(old, 1., .002, w, m, rtol=1e-10)
    intermediate = integrate(old, .4, .0009/.4, m, rtol=1e-10).y[:, -1]
    end = integrate(intermediate, .6, .0011/.6, m, rtol=1e-10).y[:, -1]
    np.testing.assert_allclose(full.y[:, -1], end, atol=1e-8, rtol=1e-10)
    bdf = integrate_waveform(old, 1., .002, w, m, method="BDF", rtol=1e-10)
    np.testing.assert_allclose(full.y[:, -1], bdf.y[:, -1], atol=1e-6, rtol=1e-8)


def test_waveform_refuses_complex_or_malformed_state():
    for old in (np.ones(9), np.ones(10)*complex(0, 1), np.full(10, np.nan)):
        with pytest.raises(ValueError):
            integrate_waveform(old, 1., .002, dict(times=[0., 1.], deviation=[0., 0.]), ())


def test_strain_waveform_is_rejected_on_stress_control():
    source = data()
    path = attach_waveforms(measured_path(source, cycles=1), source)
    path["segments"][3]["waveform"] = deepcopy(path["segments"][2]["waveform"])
    m = (Mechanism(published_parameters(600), enabled=False),)*2
    with pytest.raises(ValueError, match="stress-controlled"):
        replay(path, m)


def test_shaped_brackets_retain_same_observations():
    source = data()
    paths = [attach_waveforms(measured_path(source, cycles=1, switch_bound=side), source)
             for side in ("lower", "upper")]
    for lo, hi in zip(paths[0]["segments"], paths[1]["segments"], strict=True):
        assert lo["source_indices"] == hi["source_indices"]
        assert lo["tail"]["source_indices"] == hi["tail"]["source_indices"]


def test_recorded_prefix_uses_dwell_drift_but_never_stress_hold_strain():
    source = data()
    path = measured_path(source, cycles=1)
    # Perturb a measured strain-controlled dwell without changing switches.
    source["strain"][(source["time_s"] > 20) & (source["time_s"] < 40)] += .0001
    prefix = attach_waveforms(path, source, mode="recorded_prefix")
    assert all("waveform" in s for s in prefix["segments"][:3])
    assert all("waveform" not in s for s in prefix["segments"][3:])
    p = published_parameters(600)
    m = (Mechanism(p, enabled=False),)*2
    state = np.zeros(10)
    start = 0.
    for seg in prefix["segments"][:3]:
        sol = integrate_waveform(state, seg["duration"], seg["target"], seg["waveform"], m)
        q = seg["waveform"]["times"]
        np.testing.assert_allclose(sol.sol(q)[9],
            np.interp(start+q, source["time_s"], source["strain"]), atol=1e-12)
        state = sol.y[:, -1]
        start += seg["duration"]
    _, result = replay(prefix, m)
    assert result["segment_endpoints"][3]["state"][9] == pytest.approx(0., abs=1e-12)
