from copy import deepcopy

import numpy as np
import pytest

from tools.p92_measured_controls import measured_path, replay
from tools.p92_published_recovery_reference import Parameters, published_parameters
from tools.p92_two_mechanism import Mechanism


def data(protocol="SLC"):
    if protocol == "RF":
        knots = [0, 2, 182, 186, 366, 368]
        strain = [0, .002, .002, -.002, -.002, 0]
    elif protocol == "SLR":
        knots = [0, 2, 182, 184, 1984, 1986, 2166, 2168]
        strain = [0, -.002, -.002, 0, 0, .002, .002, 0]
    else:
        knots = [0, 2, 182, 184, 1984, 1985, 2165, 2167]
        strain = [0, -.002, -.002, 0, .001, .002, .002, 0]
    t = np.arange(0, knots[-1]+.1, .2)
    return dict(protocol=protocol, time_s=t, strain=np.interp(t, knots, strain),
                stress_Pa=np.zeros_like(t), cycle=np.ones_like(t), sha256="synthetic")


@pytest.mark.parametrize("protocol,segments", [("RF", 5), ("SLR", 7), ("SLC", 7)])
def test_measured_switches_are_bracketed_and_preserve_source_indices(protocol, segments):
    source = data(protocol)
    path = measured_path(source, cycles=1)
    assert len(path["segments"]) == segments
    for event in path["switches"]:
        assert event["bracket_s"][0] <= event["time_s"] <= event["bracket_s"][-1]
    for seg in path["segments"]:
        np.testing.assert_array_equal(seg["measured_strain"], source["strain"][seg["source_indices"]])
    if protocol == "SLC":
        assert path["segments"][3]["stress_control"]
        assert path["segments"][3]["target"] is None
        assert path["segments"][4]["duration"] == pytest.approx(1.)


def test_missing_cycles_and_gaps_are_not_invented():
    source = data()
    with pytest.raises(ValueError, match="Missing"):
        measured_path(source, cycles=2)
    source["time_s"][100:] += 10
    with pytest.raises(ValueError, match="discontinuous"):
        measured_path(source, cycles=1)


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_switch_bracket_sensitivity_uses_recorded_endpoints(side):
    path = measured_path(data(), cycles=1, switch_bound=side)
    for event in path["switches"]:
        assert event["time_s"] == event["bracket_s"][0 if side == "lower" else -1]
    assert all(seg["duration"] > 0 for seg in path["segments"])


def test_frozen_elastic_replay_does_not_prescribe_measured_creep_or_reset_history():
    source = data()
    path = measured_path(source, cycles=1)
    original = deepcopy(path)
    p = published_parameters(600)
    mechanisms = (Mechanism(p, enabled=False),)*2
    residual, record = replay(path, mechanisms)
    # The measured hold reaches .001; elastic model remains at zero strain.
    assert record["segment_endpoints"][3]["state"][9] == pytest.approx(0., abs=1e-14)
    assert np.linalg.norm(residual) > 0
    assert record["cycles"][0]["end_state"][9] == pytest.approx(0., abs=1e-14)
    assert not record["engineering_qualified"]
    for a, b in zip(path["segments"], original["segments"], strict=True):
        np.testing.assert_array_equal(a["measured_strain"], b["measured_strain"])
    _, fine = replay(path, mechanisms, method="BDF", rtol=1e-10)
    np.testing.assert_allclose(record["cycles"][0]["end_state"],
                               fine["cycles"][0]["end_state"], atol=1e-8)


def test_nominal_timing_mismatch_is_reported_not_silently_called_exact():
    from tools.p92_measured_controls import departure
    t = np.array([179.8, 180., 180.2, 180.4, 180.6, 180.8])
    e = np.array([0., 0., 0., 0., .00001, .00021])
    event = departure(t, e, np.arange(len(t)), {"time_s": 0.}, 180., 1)
    assert not event["nominal_within_bracket"]
    assert event["nominal_time_s"] == 180.
    assert event["time_s"] == pytest.approx(.5*sum(event["bracket_s"]))


def test_training_window_does_not_include_later_cycle_observations():
    single = data("RF")
    source = dict(protocol="RF", sha256="synthetic")
    for key in ("time_s", "strain", "stress_Pa", "cycle"):
        parts = []
        for i in range(4):
            values = single[key].copy()
            if key == "time_s":
                values += i*368
            if key == "cycle":
                values += i
            parts.append(values if i == 0 else values[1:])
        source[key] = np.concatenate(parts)
    before = measured_path(source, cycles=3)
    source["strain"][source["cycle"] == 4] = 1e10
    source["stress_Pa"][source["cycle"] == 4] = 1e10
    after = measured_path(source, cycles=3)
    assert before["switches"] == after["switches"]
    for seg in after["segments"]:
        assert np.all(source["cycle"][seg["source_indices"]] <= 3)


def test_seed_requires_frozen_first_cycle_and_same_base():
    from dataclasses import asdict
    from hashlib import sha256
    import json
    from tools.p92_measured_fit import seed_parameters
    from tools.p92_two_mechanism_fit import decode
    calibration = {"parameters": asdict(published_parameters(600))}
    cards = [asdict(m) for m in decode([0.]*6, Parameters(**calibration["parameters"]))]
    seed = dict(mechanisms=cards, training_cycles=[1], fit={"success": True},
                parameter_sha256=sha256(json.dumps(cards, sort_keys=True).encode()).hexdigest(),
                base_calibration_sha256=sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest(),
                scaled_parameters=[0.]*6)
    assert len(seed_parameters(seed, calibration)) == 6
    seed["training_cycles"] = [1, 4]
    with pytest.raises(ValueError, match="first-cycle"):
        seed_parameters(seed, calibration)


def test_frozen_replay_stops_before_inventing_reverse_ramp(monkeypatch):
    from types import SimpleNamespace
    from tools import p92_measured_controls as module
    calls = []

    def fake(state, *args, **kwargs):
        calls.append(state.copy())
        end = state.copy()
        end[9] = .003
        return SimpleNamespace(y=np.column_stack([state, end]),
                               sol=lambda t: np.repeat(end[:, None], len(t), axis=1))

    monkeypatch.setattr(module, "integrate", fake)
    common = dict(cycle=1, duration=1., sample_times=np.array([.5, 1.]),
                  measured_stress=np.zeros(2), measured_strain=np.zeros(2))
    path = dict(protocol="SLC", initial_stress=0., initial_strain=0., source_sha256="fixture",
                segments=[dict(common, stress_control=True, target=None),
                          dict(common, stress_control=False, target=.002)])
    _, result = replay(path, ())
    assert len(calls) == 1
    assert not result["cycles"]
    assert len(result["segment_endpoints"]) == 1
    assert result["reversed_post_hold_ramps"][0]["cycle"] == 1
    assert not result["incompatible_continuation"]


def test_frozen_prediction_rejects_changed_parameter_identity():
    from tools.p92_measured_fit import frozen_predictions
    with pytest.raises(ValueError, match="intact"):
        frozen_predictions(dict(parameters=[], parameter_sha256="changed", fit={"success": True}), [])


def test_archived_measured_fit_retains_failed_cycle_not_false_qualification():
    import json
    from pathlib import Path
    from hashlib import sha256
    report = json.loads((Path(__file__).resolve().parents[1]/
        "evidence/creep/2026-11-17-measured-controls-fit.json").read_text())
    assert report["training_cycles"] == [1, 2, 3]
    assert not report["engineering_qualified"] and not report["life_qualified"]
    assert report["parameter_sha256"] == sha256(json.dumps(report["parameters"], sort_keys=True).encode()).hexdigest()
    for prediction in report["frozen_predictions"]:
        assert prediction["parameter_sha256"] == report["parameter_sha256"]
        assert prediction["numerical_refinement_passed"]
        assert not prediction["incompatible_continuation"]
    slc = next(p for p in report["frozen_predictions"] if p["protocol"] == "SLC")
    assert [r["cycle"] for r in slc["cycles"]] == [1, 2, 3, 4]
    assert slc["reversed_post_hold_ramps"][0]["cycle"] == 5
    for side in ("lower", "upper"):
        assert slc["switch_bracket_sensitivity"][side]["reversed_post_hold_ramps"][0]["cycle"] == 5


def test_time_softening_coordinate_replacement_preserves_seed_and_fast_branch():
    import json
    from pathlib import Path
    from dataclasses import asdict
    from tools.p92_measured_fit import time_softening_coordinates
    root = Path(__file__).resolve().parents[1]
    seed = json.loads((root/"evidence/creep/2026-11-17-measured-controls-fit.json").read_text())
    base = json.loads((root/"evidence/creep/2026-11-16-bam-positive-strength.json").read_text())
    initial, decoder, fixed = time_softening_coordinates(seed, base)
    assert initial.shape == (6,)
    assert initial[2] == 0.
    assert fixed == seed["scaled_parameters"][2]
    assert [asdict(m) for m in decoder(initial)] == seed["parameters"]
    changed = initial.copy()
    changed[2] = -1.
    a, b = decoder(changed)
    original_a, original_b = decoder(initial)
    assert a == original_a
    assert b.parameters.gamma_r1 == pytest.approx(original_b.parameters.gamma_r1*.1)
    assert b.parameters.gamma_r2 == pytest.approx(original_b.parameters.gamma_r2*.1)
    assert b.parameters.k == original_b.parameters.k
    both_initial, both_decoder, _ = time_softening_coordinates(seed, base, both=True)
    assert [asdict(m) for m in both_decoder(both_initial)] == seed["parameters"]
    both_changed = both_initial.copy()
    both_changed[1:3] = [-1., -.5]
    fast, slow = both_decoder(both_changed)
    assert fast.parameters.k == original_a.parameters.k
    assert fast.parameters.gamma1 == original_a.parameters.gamma1
    assert fast.parameters.gamma_r1 == pytest.approx(original_a.parameters.gamma_r1*.1)
    assert slow.parameters.gamma_r1 == pytest.approx(original_b.parameters.gamma_r1*10**(-.5))
    seed["training_cycles"] = [1, 2, 3, 5]
    with pytest.raises(ValueError, match="early-cycle"):
        time_softening_coordinates(seed, base)


def test_hold_metrics_do_not_replace_stress_controlled_creep_with_measurements():
    p = published_parameters(600)
    _, result = replay(measured_path(data(), cycles=1), (Mechanism(p, enabled=False),)*2)
    hold = next(h for h in result["hold_diagnostics"] if h["stress_control"])
    assert hold["fields"]["strain"]["last_observed"] == pytest.approx(.001)
    assert hold["fields"]["strain"]["last_predicted"] == pytest.approx(0., abs=1e-14)
    assert hold["fields"]["strain"]["rms"] > 0
    assert hold["fields"]["strain"]["early_secant_predicted_per_s"] == pytest.approx(0.)
    assert hold["sample_times_s"][-1] <= hold["duration_s"]


def test_late_hold_observation_does_not_change_fitting_or_history():
    p = published_parameters(600)
    path = measured_path(data(), cycles=1)
    without_tail = dict(path, segments=[{k: v for k, v in s.items() if k != "tail"}
                                       for s in path["segments"]])
    mechanisms = (Mechanism(p, enabled=False),)*2
    residual, result = replay(path, mechanisms)
    old_residual, old = replay(without_tail, mechanisms)
    np.testing.assert_array_equal(residual, old_residual)
    assert result["segment_endpoints"] == old["segment_endpoints"]
    hold = next(h for h in result["hold_diagnostics"] if h["stress_control"])
    assert len(hold["late_secant_source_indices"]) == 2
    assert hold["late_secant_times_s"][0] >= .8*hold["duration_s"]
    assert hold["fields"]["strain"]["late_secant_predicted_per_s"] == 0.


@pytest.mark.parametrize("variant", ["slow", "both"])
def test_archived_time_fit_retains_accuracy_gap_despite_completed_cycles(variant):
    import json
    from pathlib import Path
    from hashlib import sha256
    from tools.p92_measured_fit import coordinate_names
    report = json.loads((Path(__file__).resolve().parents[1] /
        f"evidence/creep/2026-11-17-{variant}-time-identification.json").read_text())
    assert report["training_cycles"] == [1, 2, 3]
    assert not report["engineering_qualified"] and not report["life_qualified"]
    assert len(report["scaled_parameters"]) == 6
    assert report["coordinate_names"] == coordinate_names(report["parameter_direction"])
    assert report["parameter_sha256"] == sha256(json.dumps(report["parameters"], sort_keys=True).encode()).hexdigest()
    for prediction in report["frozen_predictions"]:
        assert prediction["parameter_sha256"] == report["parameter_sha256"]
        assert prediction["numerical_refinement_passed"]
        assert [c["cycle"] for c in prediction["cycles"]] == list(range(1, 11))
        assert not prediction["reversed_post_hold_ramps"]
        assert not prediction["incompatible_continuation"]
    slc = next(p for p in report["frozen_predictions"] if p["protocol"] == "SLC")
    first = next(h for h in slc["hold_diagnostics"] if h["stress_control"])
    strain = first["fields"]["strain"]
    assert strain["last_predicted"] < strain["last_observed"]
    assert strain["early_secant_predicted_per_s"] < strain["early_secant_observed_per_s"]
