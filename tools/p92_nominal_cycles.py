"""Continuous nominal BAM RF loading; not reconstruction of missing measurements."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from numbers import Real
from pathlib import Path

import numpy as np

from tools.p92_published_recovery_reference import (
    Parameters, integrate_segment, isotropic_strength_lower_bound,
)


def nominal_cycles(parameters, initial_state, *, cycles, method="Radau", rtol=1e-8):
    """Carry all five states across the published 368 s RF waveform.

    Peaks below are extrema on the adaptive solver samples, not certified
    continuous extrema. A failed segment publishes only completed cycles.
    """
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
        raise ValueError("cycles must be a positive integer")
    if (method not in {"Radau", "BDF"} or isinstance(rtol, (bool, np.bool_))
            or not isinstance(rtol, Real) or not np.isfinite(rtol) or not 0 < rtol < 1):
        raise ValueError("Require Radau/BDF and positive finite rtol")
    if np.iscomplexobj(initial_state):
        raise ValueError("Require real initial state")
    state = np.asarray(initial_state, dtype=float).copy()
    if (state.shape != (5,) or not np.isfinite(state).all() or state[4] < 0
            or state[3]+parameters.k <= 0):
        raise ValueError("Require explicit finite five-component initial state")
    records = []
    failure = None
    for cycle in range(1, cycles+1):
        trial = state.copy()
        peak, valley = float(trial[0]), float(trial[0])
        for segment, (duration, rate) in enumerate(((2., .001), (180., 0.),
                (4., -.001), (180., 0.), (2., .001))):
            try:
                solution = integrate_segment(trial, duration, rate, parameters,
                                             method=method, rtol=rtol)
            except RuntimeError as exc:
                failure = {"cycle": cycle, "segment": segment, "reason": str(exc)}
                break
            trial = solution.y[:, -1].copy()
            peak = max(peak, float(solution.y[0].max()))
            valley = min(valley, float(solution.y[0].min()))
        if failure is not None:
            break
        state = trial
        records.append({"cycle": cycle, "time_s": cycle*368.,
                        "sampled_peak_MPa": peak, "sampled_valley_MPa": valley,
                        "end_state": state.tolist()})
    return {"schema": "agentfem.research.p92-nominal-rf.v1", "method": method,
            "rtol": rtol, "requested_cycles": cycles, "completed_cycles": len(records),
            "failure": failure, "parameters": asdict(parameters),
            "initial_state": np.asarray(initial_state).tolist(),
            "last_completed_state": state.tolist(), "cycles": records,
            "sufficient_strength_lower_bound_MPa": isotropic_strength_lower_bound(parameters, initial_state[3]),
            "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10319173/ Table 1",
            "initial_state_assumption": "Supplied stress/history at nominal zero strain; measured initial strain offset is not replayed",
            "scope": "Nominal imposed waveform, not missing measured history; no damage or life prediction",
            "engineering_qualified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--cycles", type=int, default=1000)
    parser.add_argument("--method", choices=("Radau", "BDF"), default="Radau")
    args = parser.parse_args()
    frozen = json.loads(args.calibration.read_text())
    if not frozen["fit"]["success"]:
        parser.error("Require a converged frozen calibration")
    report = nominal_cycles(Parameters(**frozen["parameters"]), frozen["initial_state"],
                            cycles=args.cycles, method=args.method)
    report["calibration_record_sha256"] = sha256(
        json.dumps(frozen, sort_keys=True, allow_nan=False).encode()).hexdigest()
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(1 if report["failure"] else 0)
