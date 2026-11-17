"""Research-only Jiang-type recovery discriminator; not a P92 material card.

Tests a single mechanism on the frozen five-state candidate. This is NOT the
published two-mechanism Chaboche-Jiang model. No parameter is fitted here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from tools.p92_bounded_softening import bounded_rates, _trial_multiplier, compare_candidate
from tools.p92_published_recovery_reference import Parameters, isotropic_strength_lower_bound
from tools.validate_p92_loading_path import nominal_path


def corrected_rates(state, strain_rate, p, *, exponent):
    if not np.isfinite(exponent) or exponent < 0:
        raise ValueError("Require finite nonnegative ratcheting exponent")
    values = bounded_rates(state, strain_rate, p, limit=2., rate=.1)
    phi, _ = _trial_multiplier(state[4], limit=2., rate=.1)
    for i, C, a in ((1, p.C1, p.a1), (2, p.C2, p.a2)):
        # Current saturation is a/phi. Do not clip W to conceal an excursion.
        weight = (abs(state[i])*phi/a)**exponent
        values[i] += C*phi*(1-weight)*state[i]*values[4]
    return values


def segment(old, strain, duration, rate, p, *, exponent, stress_control=False, rtol=1e-8):
    old = np.asarray(old, dtype=float)
    if (old.shape != (5,) or not np.isfinite(old).all() or old[4] < 0
            or not np.isfinite([strain, duration, rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1 or p.E <= 0
            or min(p.a1, p.a2) <= 0 or isotropic_strength_lower_bound(p, old[3]) <= 0):
        raise ValueError("Invalid state, material or integration controls")

    def rhs(t, y):
        v = corrected_rates(y[:5], 0. if stress_control else rate, p, exponent=exponent)
        edot = -v[0]/p.E if stress_control else rate
        if stress_control:
            v[0] = 0.
        return np.r_[v, edot]

    solution = solve_ivp(rhs, (0., duration), np.r_[old, strain], method="Radau",
                         rtol=rtol, atol=[1e-8]*4+[1e-12]*2)
    if (not solution.success or not np.isfinite(solution.y).all()
            or np.any(np.diff(solution.y[4]) < -1e-10)
            or solution.y[4, -1] < old[4] or np.any(p.k+solution.y[3] <= 0)):
        raise RuntimeError("Ratcheting screen failed numerical/domain checks")
    return solution


def run(calibration, *, protocol, cycles, exponent, rtol=1e-8):
    if type(cycles) is not int or cycles < 1 or not calibration["fit"]["success"]:
        raise ValueError("Require positive cycles and converged frozen calibration")
    p = Parameters(**calibration["parameters"])
    state = np.asarray(calibration["initial_state"], dtype=float).copy()
    strain, records = 0., []
    if protocol == "SLC":
        path = [("ramp", 0., -.002), ("hold", 180., None),
                ("ramp", 0., 0.), ("stress", 1800., None),
                ("ramp", 0., .002), ("hold", 180., None), ("ramp", 0., 0.)]
    else:
        times, strains = nominal_path(protocol)
        path = [("fixed", dt, target) for dt, target in zip(np.diff(times), strains[1:])]
    for cycle in range(1, cycles+1):
        peak = valley = float(state[0])
        endpoints = []
        for kind, duration, target in path:
            if kind == "ramp":
                duration = abs(target-strain)/.001
            rate = (target-strain)/duration if target is not None else 0.
            sol = segment(state, strain, duration, rate, p, exponent=exponent,
                          stress_control=kind == "stress", rtol=rtol)
            state, strain = sol.y[:5, -1].copy(), float(sol.y[5, -1])
            peak, valley = max(peak, float(sol.y[0].max())), min(valley, float(sol.y[0].min()))
            endpoints.append(dict(control=kind, strain=strain, state=state.tolist()))
        records.append(dict(cycle=cycle, sampled_peak_MPa=peak, sampled_valley_MPa=valley,
                            end_state=state.tolist(), segment_endpoints=endpoints))
        # Stop instead of turning an overshooting creep hold into a reverse ramp
        # in subsequent cycles. This record is a failed protocol screen.
        if protocol == "SLC" and endpoints[3]["strain"] > .002:
            break
    from hashlib import sha256
    return dict(protocol=protocol, exponent=exponent, rtol=rtol, cycles=records,
                calibration_sha256=sha256(json.dumps(calibration, sort_keys=True,
                    allow_nan=False).encode()).hexdigest(), parameters=calibration["parameters"],
                requested_cycles=cycles, engineering_qualified=False, life_qualified=False,
                independent_validation=False,
                overshoot_policy="Finish failed cycle for diagnostics, then stop; extrema of that cycle are not an admissible protocol prediction",
                envelope_exceeded=protocol == "SLC" and records[-1]["segment_endpoints"][3]["strain"] > .002,
                scope="Fixed-base mechanism sensitivity; not the published two-strain model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new evidence path")
    from tools.p92_bam_source import read_source
    calibration = json.loads(args.calibration.read_text())
    reports = []
    for exponent in (0., 1., 3.):
        for protocol in ("RF", "SLR", "SLC"):
            report = run(calibration, protocol=protocol, cycles=10, exponent=exponent)
            report["comparison"] = compare_candidate(read_source(
                args.source_dir/f"Data_{protocol}.lis", protocol=protocol), report)
            reports.append(report)
            print(json.dumps(dict(exponent=exponent, protocol=protocol,
                cycles=len(report["cycles"]), envelope_exceeded=report["envelope_exceeded"],
                groups=report["comparison"]["groups"])), flush=True)
    args.output.write_text(json.dumps(reports, indent=2, allow_nan=False)+"\n")
