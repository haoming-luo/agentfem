"""Relative recorded stress variation, never an observed-state reset.

Research diagnostic for the two smooth spectrum kernels only. Prescribed
stress is an input, so its agreement with observation is not a response pass.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_recoverable_spectrum as spectrum
from tools import p92_recoverable_mobility_reference as mobility
from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_waveform_controls import attach_waveforms
from tools.p92_recoverable_frozen import decode_material
from tools.p92_two_rate_controls import replay
from tools.p92_two_rate_frozen import numerical_difference
from tools.p92_bounded_softening import compare_candidate
from tools.p92_response_screen import screen
from tools.p92_measured_fit import write_new
from tools.p92_recorded_hold_audit import recorded_strain_segments
from tools.p92_waveform_experiment import refine_knots


def stress_path(old, times, delta, material, kernel, *, method="Radau", rtol=1e-9):
    if kernel not in (spectrum, mobility):
        raise ValueError("Stress-ripple diagnostic requires an audited smooth spectrum kernel")
    if any(np.iscomplexobj(v) for v in (old, times, delta)):
        raise ValueError("Require real state and controls")
    times, delta = np.asarray(times, float), np.asarray(delta, float)
    state = np.array(old, dtype=float, copy=True)
    if (state.shape != (12,) or times.ndim != 1 or len(times) < 2 or delta.shape != times.shape
            or times[0] != 0 or delta[0] != 0 or np.any(np.diff(times) <= 0)
            or not all(np.isfinite(x).all() for x in (state, times, delta))
            or np.any(state[[3, 6, 9]] < 0) or method not in {"Radau", "BDF"}
            or not np.isfinite(rtol) or not 0 < rtol < 1):
        raise ValueError("Invalid stress-ripple controls")
    spec = material.spectrum if kernel is mobility else material
    bounds = np.array([v for b in spec.backbone.branches for v in b.saturation])
    if np.any(abs(state[[1, 2, 4, 5]]) > bounds+1e-7):
        raise ValueError("Initial backstress outside invariant domain")
    modulus = spec.backbone.modulus
    start_stress = state[0]
    initial_energy = kernel.stored_energy(state, material)
    initial_work, initial_dissipation = state[8:10]
    parts = []
    for i, (a, b) in enumerate(zip(times[:-1], times[1:], strict=True)):
        slope = (delta[i+1]-delta[i])/(b-a)

        def rhs(t, y, slope=slope):
            rate = kernel.rates(y, 0., material, stress_control=True)
            rate[0] = slope
            rate[7] += slope/modulus
            rate[8] += y[0]*slope/modulus
            return rate

        sol = solve_ivp(rhs, (0., b-a), state, method=method, rtol=rtol, dense_output=True,
            atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13])
        if not sol.success or not np.isfinite(sol.y).all():
            raise RuntimeError("Stress-ripple integration failed")
        if abs(sol.y[0, -1]-start_stress-delta[i+1]) > 1e-7:
            raise RuntimeError("Stress control not reproduced")
        state = sol.y[:, -1].copy()
        parts.append(sol)
    balance = state[8]-initial_work-state[9]+initial_dissipation-kernel.stored_energy(state, material)+initial_energy
    if abs(balance) > 1e-7+1e-7*abs(state[8]-initial_work):
        raise RuntimeError("Stress-ripple mechanical balance failed")
    joined = np.concatenate([p.y for p in parts], axis=1)
    if np.any(abs(joined[[1, 2, 4, 5]]) > bounds[:, None]+1e-6):
        raise RuntimeError("Backstress invariant violated")
    history = joined[[3, 6, 9]]
    tolerance = 10*(np.array([1e-13, 1e-13, 1e-11])[:, None]+rtol*abs(history))
    if np.any(np.maximum.accumulate(history, axis=1)-history > tolerance):
        raise RuntimeError("Accumulated state decreased")

    def evaluate(query):
        q = np.asarray(query, float)
        if q.ndim != 1 or not np.isfinite(q).all() or np.any((q < 0) | (q > times[-1])):
            raise ValueError("Observation outside ripple interval")
        ids = np.minimum(np.searchsorted(times, q, side="right")-1, len(parts)-1)
        out = np.empty((12, len(q)))
        for i in np.unique(ids):
            selected = ids == i
            out[:, selected] = parts[i].sol(q[selected]-times[i])
        return out

    return SimpleNamespace(y=joined, sol=evaluate)


def controls(path, source):
    result, start = [], 0.
    for segment, event in zip(path["segments"], path["switches"], strict=True):
        end = event["time_s"]
        if segment["stress_control"]:
            t, stress = source["time_s"], source["stress_Pa"]/1e6
            ids = np.flatnonzero((t > start) & (t < end))
            times = np.r_[0., t[ids]-start, segment["duration"]]
            values = np.r_[np.interp(start, t, stress), stress[ids], np.interp(end, t, stress)]
            result.append((times, values-values[0]))
        start = end
    return result


def wrapper(kernel, waves):
    index = 0

    def integrate(old, duration, rate, material, *, stress_control=False, method="Radau", rtol=1e-9):
        nonlocal index
        if not stress_control:
            return kernel.integrate(old, duration, rate, material, method=method, rtol=rtol)
        if index >= len(waves):
            raise ValueError("Unexpected additional stress hold")
        times, delta = waves[index]
        if times[-1] != duration:
            raise ValueError("Stress segment identity changed")
        result = stress_path(old, times, delta, material, kernel, method=method, rtol=rtol)
        index += 1
        return result

    return SimpleNamespace(MODEL_ID=kernel.MODEL_ID, STATE_LAYOUT=kernel.STATE_LAYOUT,
        integrate=integrate, stored_energy=kernel.stored_energy, count=lambda: index)


def evaluate(card, source_dir):
    kernel, material = decode_material(card["parameters"], card["model_id"])
    source = read_source(source_dir/"Data_SLC.lis", protocol="SLC")
    if source["sha256"] != card["source_sha256"]["SLC"]:
        raise ValueError("Source identity changed")
    path = attach_waveforms(measured_path(source, cycles=3), source)
    waves, results = controls(path, source), {}
    for name, method, rtol in (("nominal", "Radau", 1e-9), ("relative_ripple", "Radau", 1e-9),
                               ("independent", "BDF", 1e-10)):
        proxy = kernel if name == "nominal" else wrapper(kernel, waves)
        _, prediction = replay(path, material, kernel=proxy, method=method, rtol=rtol)
        if name != "nominal" and proxy.count() != len(waves):
            raise ValueError("Incomplete stress-ripple replay")
        results[name] = prediction
        print(json.dumps(dict(arm=name, balance=prediction["mechanical_balance_MPa"])), flush=True)
    numerical = numerical_difference(results["relative_ripple"], results["independent"], extra_strain_indices=(10, 11))
    reports = {k: screen(v, compare_candidate(source, v), requested_cycles=3) for k, v in results.items()}
    return dict(schema="agentfem.research.relative-stress-ripple.v1", model_id=card["model_id"],
        parameters=card["parameters"], source_sha256=source["sha256"], numerical=numerical,
        results=results, response_screens=reports, fitted=False, engineering_qualified=False,
        acceptance_thresholds_changed=False, prescribed_stress_is_not_validation=True,
        scope="Relative recorded stress about predicted entry; not authenticated commands or a state reset")


def refine_stress_controls(waves):
    """Bisect unchanged piecewise-linear controls; never smooth observations."""
    refined = []
    for times, delta in waves:
        times, delta = np.asarray(times, float), np.asarray(delta, float)
        if (times.ndim != 1 or len(times) < 2 or times.shape != delta.shape
                or times[0] != 0 or delta[0] != 0 or np.any(np.diff(times) <= 0)
                or not np.isfinite([*times, *delta]).all()):
            raise ValueError("Invalid stress-control refinement")
        new = np.sort(np.r_[times, .5*(times[:-1]+times[1:])])
        refined.append((new, np.interp(new, times, delta)))
    return refined


def evaluate_mixed(card, source_dir):
    kernel, material = decode_material(card["parameters"], card["model_id"])
    if kernel not in (spectrum, mobility):
        raise ValueError("Mixed diagnostic requires an audited smooth spectrum kernel")
    source = read_source(source_dir/"Data_SLC.lis", protocol="SLC")
    if source["sha256"] != card["source_sha256"]["SLC"]:
        raise ValueError("Source identity changed")
    path = measured_path(source, cycles=3)
    nominal, recorded = attach_waveforms(path, source), recorded_strain_segments(path, source)
    waves, results = controls(path, source), {}
    arms = (("nominal", nominal, False, "Radau", 1e-9),
            ("strain_only", recorded, False, "Radau", 1e-9),
            ("stress_only", nominal, True, "Radau", 1e-9),
            ("combined", recorded, True, "Radau", 1e-9),
            ("independent", recorded, True, "BDF", 1e-10),
            ("refined", refine_knots(recorded), True, "Radau", 1e-10))
    for name, control, ripple, method, rtol in arms:
        current_waves = refine_stress_controls(waves) if name == "refined" else waves
        proxy = wrapper(kernel, current_waves) if ripple else kernel
        _, prediction = replay(control, material, kernel=proxy, method=method, rtol=rtol)
        if ripple and proxy.count() != len(waves):
            raise ValueError("Incomplete mixed-control replay")
        results[name] = prediction
        print(json.dumps(dict(arm=name, balance=prediction["mechanical_balance_MPa"])), flush=True)
    numerical = {name: numerical_difference(results["combined"], results[name], extra_strain_indices=(10, 11))
                 for name in ("independent", "refined")}
    numerical["passed"] = all(item["passed"] for item in numerical.values())
    reports = {k: screen(v, compare_candidate(source, v), requested_cycles=3) for k, v in results.items()}
    return dict(schema="agentfem.research.mixed-control-sensitivity.v1", model_id=card["model_id"],
        parameters=card["parameters"], source_sha256=source["sha256"], numerical=numerical,
        results=results, response_screens=reports, fitted=False, engineering_qualified=False,
        life_qualified=False, acceptance_thresholds_changed=False, prescribed_stress_is_not_validation=True,
        scope="Four fixed-card input arms; relative stress and continuous recorded strain; no measured creep state reset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--combined-strain", action="store_true", help="Four-arm fixed-card mixed-control sensitivity")
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    names = ("p92_stress_ripple_audit.py", "p92_recoverable_frozen.py", "p92_recoverable_spectrum.py",
             "p92_recoverable_mobility_reference.py", "p92_two_rate_reference.py", "p92_two_rate_controls.py",
             "p92_measured_controls.py", "p92_waveform_controls.py", "p92_bam_source.py",
             "p92_hold_observations.py", "p92_response_screen.py", "p92_bounded_softening.py",
             "p92_recorded_hold_audit.py", "p92_waveform_experiment.py", "p92_two_rate_frozen.py")

    def identities():
        return {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}

    before = identities()
    result = (evaluate_mixed if args.combined_strain else evaluate)(json.loads(raw), args.source_dir)
    if identities() != before or args.frozen.read_bytes() != raw:
        raise RuntimeError("Stress-ripple inputs changed during replay")
    result.update(driver_sha256=before, frozen_sha256=sha256(raw).hexdigest())
    write_new(args.output, result)
