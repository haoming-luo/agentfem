"""Frozen two-mechanism protocol screen; no inferred life or automatic admission."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bounded_softening import compare_candidate
from tools.p92_published_recovery_reference import Parameters
from tools.p92_two_mechanism import Mechanism, integrate
from tools.validate_p92_loading_path import nominal_path


def run(mechanisms, *, initial_stress, protocol, cycles=10, rtol=1e-8, method="Radau"):
    if protocol not in {"RF", "SLR", "SLC"} or type(cycles) is not int or cycles < 1:
        raise ValueError("Require supported protocol and positive integer cycles")
    state = np.zeros(10)
    state[0] = initial_stress
    if protocol == "SLC":
        path = [("ramp", 0., -.002), ("hold", 180., None),
                ("ramp", 0., 0.), ("stress", 1800., None),
                ("ramp", 0., .002), ("hold", 180., None), ("ramp", 0., 0.)]
    else:
        times, strains = nominal_path(protocol)
        path = [("fixed", dt, target) for dt, target in zip(np.diff(times), strains[1:])]
    records, envelope_failure = [], None
    for cycle in range(1, cycles+1):
        peak = valley = float(state[0])
        endpoints = []
        for kind, duration, target in path:
            if kind == "ramp":
                duration = abs(target-state[9])/.001
            rate = (target-state[9])/duration if target is not None else 0.
            sol = integrate(state, duration, rate, mechanisms,
                            stress_control=kind == "stress", rtol=rtol, method=method)
            state = sol.y[:, -1].copy()
            peak = max(peak, float(sol.y[0].max()))
            valley = min(valley, float(sol.y[0].min()))
            endpoints.append(dict(control=kind, duration_s=duration, state=state.tolist()))
            if kind == "stress" and state[9] > .002:
                envelope_failure = dict(cycle=cycle, strain=float(state[9]),
                                        target=.002, segment_endpoints=endpoints)
                break  # Do not invent a reverse ramp or count a partial cycle.
        if envelope_failure:
            break
        records.append(dict(cycle=cycle, sampled_peak_MPa=peak, sampled_valley_MPa=valley,
                            end_state=state.tolist(), segment_endpoints=endpoints))
    return dict(protocol=protocol, mechanisms=[asdict(m) for m in mechanisms],
                initial_stress_MPa=initial_stress, initial_hidden_state="virgin per specimen",
                requested_cycles=cycles, cycles=records, envelope_failure=envelope_failure,
                method=method, rtol=rtol, engineering_qualified=False, life_qualified=False,
                independent_validation=False, scope="Nominal path development screen; partial cycles excluded from extrema, failure retained")


def candidates(calibration):
    """Prespecified mechanism sensitivities, deliberately NOT fitted constants."""
    p = Parameters(**calibration["parameters"])
    fast = Mechanism(p)
    # First test simply adding a slower branch: an important negative control.
    slow = Mechanism(replace(p, Z=3*p.Z))
    # Second tests suppressing time recovery of fast backstress while retaining
    # the slow mechanism. This is a mechanism ablation, not a material calibration.
    retained = Mechanism(replace(p, gamma1=0., gamma2=0.))
    result = {"single_reference": (fast, Mechanism(p, enabled=False)),
              "added_slower_flow": (fast, slow),
              "retained_fast_backstress": (retained, slow)}
    # Separate follow-up sensitivities preserve the flow at 120 MPa overstress
    # while varying stress sensitivity. This reference is a numerical pivot,
    # not an experimental calibration point or an identified mechanism scale.
    for exponent in (6., 12.):
        Z = 120./((120./p.Z)**(p.n/exponent))
        result[f"retained_fast_n{int(exponent)}"] = (
            Mechanism(replace(retained.parameters, n=exponent, Z=Z)), slow)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--method", choices=("Radau", "BDF"), default="Radau")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new evidence path")
    from tools.p92_bam_source import read_source
    calibration = json.loads(args.calibration.read_text())
    identity = sha256(json.dumps(calibration, sort_keys=True, allow_nan=False).encode()).hexdigest()
    reports = []
    for name, mechanisms in candidates(calibration).items():
        for protocol in ("RF", "SLR", "SLC"):
            data = read_source(args.source_dir/f"Data_{protocol}.lis", protocol=protocol)
            report = run(mechanisms, initial_stress=float(data["stress_Pa"][0]/1e6),
                         protocol=protocol, rtol=args.rtol, method=args.method)
            report.update(candidate=name, calibration_sha256=identity, source_sha256=data["sha256"])
            if report["cycles"]:
                report["comparison"] = compare_candidate(data, report)
            reports.append(report)
            print(json.dumps(dict(candidate=name, protocol=protocol, complete_cycles=len(report["cycles"]),
                failure_cycle=(report["envelope_failure"] or {}).get("cycle"),
                groups=report.get("comparison", {}).get("groups"))), flush=True)
    args.output.write_text(json.dumps(reports, indent=2, allow_nan=False)+"\n")
