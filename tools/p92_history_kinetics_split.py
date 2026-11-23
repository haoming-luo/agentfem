"""Causal incoming-history versus direct hold-kinetics sensitivities."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools import p92_recoverable_spectrum as kernel
from tools.p92_bam_source import read_source
from tools.p92_hold_observations import summarize
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_recoverable_frozen import decode_material
from tools.p92_response_screen import POLICY
from tools.p92_spectrum_recovery_joint import decode, NAMES
from tools.p92_two_rate_controls import replay
from tools.p92_waveform_controls import attach_waveforms

COORDINATES = (1, 10, 16, 17, 20, 21)
METRICS = ("first_predicted", "last_predicted", "early_secant_predicted_per_s",
           "late_secant_predicted_per_s")


def split_derivative(lower, upper, step):
    low, high = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (low.ndim != 2 or low.shape[0] != 3 or high.shape != low.shape
            or not low.shape[1] or not np.isfinite(low).all() or not np.isfinite(high).all()
            or not np.isfinite(step) or step <= 0):
        raise ValueError("Require full/history/kinetic observations and positive difference step")
    slopes = (high-low)/(2*step)
    closure = slopes[0]-slopes[1]-slopes[2]
    relative = float(np.linalg.norm(closure)/max(np.linalg.norm(slopes), 1e-30))
    return dict(full=slopes[0].tolist(), history=slopes[1].tolist(), kinetics=slopes[2].tolist(),
                chain_rule_residual=closure.tolist(), chain_rule_relative_error=relative)


def scales(segment, field):
    stress = max(POLICY["stress_absolute_MPa"], POLICY["stress_relative"]*field["observed_max_abs"])
    result = [stress, stress]
    for window in ("early", "late"):
        times = np.array(segment["sample_times"] if window == "early" else segment["tail"]["times"])
        if window == "early":
            times = times[times <= min(10., .1*segment["duration"])]
        value = field.get(f"{window}_secant_observed_per_s")
        if (value is None or len(times) < 2 or abs(value*(times[-1]-times[0])) <
                POLICY["minimum_resolution_units"]*POLICY["stress_resolution_MPa"]):
            raise ValueError("Require resolved stress secants for causal sensitivity")
        result.append(POLICY["secant_relative"]*abs(value))
    return np.array(result)


def evaluate(frozen, directory):
    module, base = decode_material(frozen["parameters"], frozen["model_id"])
    if module is not kernel:
        raise ValueError("Require unchanged original spectrum")
    records = {}
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(directory/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != frozen["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        path = attach_waveforms(measured_path(source, cycles=1), source)
        index = next(i for i, s in enumerate(path["segments"]) if s["strain_hold"] and s["target"] > 0)
        segment = path["segments"][index]
        if segment.get("waveform") is not None or segment["stress_control"] or index == 0:
            raise ValueError("Require a post-loading constant-strain hold")
        _, causal = replay(path, base, kernel=kernel)
        incoming = np.array(causal["segments"][index-1]["end_state"])
        original_hold = next(h for h in causal["hold_diagnostics"] if h["segment"] == index)
        scale = scales(segment, original_hold["fields"]["stress_MPa"])

        def values(initial, material, segment=segment, index=index, scale=scale):
            sol = kernel.integrate(initial, segment["duration"], 0., material)
            f = summarize(segment, sol.sol, index=index, strain_index=7)["fields"]["stress_MPa"]
            return np.array([f[name] for name in METRICS])/scale

        rows = []
        for coordinate in COORDINATES:
            derivatives = []
            for step in (.001, .0005):
                samples = []
                for sign in (-1., 1.):
                    z = np.eye(22)[coordinate]*sign*step
                    material = decode(frozen["parameters"], z, (0., 0.))
                    _, changed = replay(path, material, kernel=kernel)
                    altered = np.array(changed["segments"][index-1]["end_state"])
                    samples.append([values(altered, material), values(altered, base), values(incoming, material)])
                derivatives.append(dict(step=step, **split_derivative(*samples, step)))
            d0 = np.array([derivatives[0][k] for k in ("full", "history", "kinetics")])
            d1 = np.array([derivatives[1][k] for k in ("full", "history", "kinetics")])
            error = float(np.linalg.norm(d1-d0)/max(np.linalg.norm(d1), 1e-30))
            passed = error <= .002 and all(d["chain_rule_relative_error"] <= .002 for d in derivatives)
            rows.append(dict(coordinate=NAMES[coordinate], derivatives=derivatives,
                             two_scale_relative_error=error, numerical_passed=passed))
            print(json.dumps(dict(protocol=protocol, coordinate=NAMES[coordinate],
                                  numerical_passed=passed)), flush=True)
        records[protocol] = dict(segment=index, source_indices=segment["source_indices"],
            metrics=METRICS, normalization=scale.tolist(), baseline=original_hold, rows=rows)
    return dict(schema="agentfem.research.history-kinetics-split.v1", records=records,
        parameters=frozen["parameters"], source_sha256=frozen["source_sha256"],
        model_id=frozen["model_id"], training_cycles=[1], fitted=False,
        engineering_qualified=False, life_qualified=False,
        scope="Counterfactual derivative decomposition, not state reset for material prediction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    names = ("p92_history_kinetics_split.py", "p92_hold_observations.py", "p92_spectrum_recovery_joint.py",
             "p92_full_sensitivity.py", "p92_recoverable_spectrum.py", "p92_two_rate_reference.py",
             "p92_two_rate_controls.py", "p92_measured_controls.py", "p92_waveform_controls.py",
             "p92_bam_source.py", "p92_response_screen.py", "p92_recoverable_frozen.py")
    before = {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    result = evaluate(json.loads(raw), args.source_dir)
    if raw != args.frozen.read_bytes() or any(
            sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() != h for n, h in before.items()):
        raise RuntimeError("Causal sensitivity input changed")
    result.update(driver_sha256=before, frozen_sha256=sha256(raw).hexdigest())
    write_new(args.output, result)
