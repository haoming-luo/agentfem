"""BAM SLC nominal mixed-control material-point screen, not a life model.

The stress-controlled 1800 s hold starts at the stress reached at zero strain.
Its evolving strain is retained when prescribing the next ramp.
No strain-controlled dwell is substituted and no hidden state is reset.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from tools.p92_bounded_softening import (
    bounded_rates, bounded_jacobian, integrate_bounded, multiplier, compare_candidate,
)
from tools.p92_published_recovery_reference import Parameters, isotropic_strength_lower_bound


def stress_hold(old, strain, duration, parameters, *, limit, rate, rtol=1e-8):
    """Constant nominal axial stress; sixth state is total axial strain.

    From sigma_dot=E*(epsilon_dot-flow), prescribed sigma_dot=0 gives
    epsilon_dot=flow. All internal evolution equations remain unchanged.
    This is a small-strain uniaxial research closure, not a finite-strain
    stress-control adapter or a new global nonlinear solver.
    """
    old = np.asarray(old)
    if (np.iscomplexobj(old) or old.shape != (5,) or not np.isfinite(old).all()
            or not np.isfinite([strain, duration, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1 or parameters.E <= 0):
        raise ValueError("Require finite state, positive duration/modulus and valid tolerance")
    multiplier(old[4], limit=limit, rate=rate)
    if isotropic_strength_lower_bound(parameters, old[3]) <= 0:
        raise ValueError("Require positive sufficient isotropic strength")

    def rhs(t, y):
        values = bounded_rates(y[:5], 0., parameters, limit=limit, rate=rate)
        flow = -values[0]/parameters.E
        values[0] = 0.
        return np.r_[values, flow]

    def jac(t, y):
        matrix = np.zeros((6, 6))
        matrix[:5, :5] = bounded_jacobian(y[:5], parameters, limit=limit, rate=rate)
        matrix[5, :5] = -matrix[0, :5]/parameters.E
        matrix[0, :] = 0.
        return matrix

    solution = solve_ivp(rhs, (0., duration), np.r_[old, strain], method="Radau",
                         jac=jac, rtol=rtol, atol=[1e-8]*4+[1e-12]*2)
    if (not solution.success or not np.isfinite(solution.y).all()
            or np.any(np.diff(solution.y[4]) < -1e-10)
            or solution.y[4, -1] < old[4]
            or np.any(parameters.k+solution.y[3] <= 0)
            or not np.allclose(solution.y[0], old[0], rtol=0., atol=1e-8)):
        raise RuntimeError("Mixed-control stress hold failed numerical/domain checks")
    return solution


def run(calibration, *, cycles, limit=2., rate=.1, rtol=1e-8, ramp_mode="fixed_rate"):
    if type(cycles) is not int or cycles < 1 or not calibration["fit"]["success"]:
        raise ValueError("Require positive cycle count and a converged base calibration")
    if ramp_mode not in {"fixed_rate", "table_duration"}:
        raise ValueError("Select fixed_rate or table_duration for the post-creep ramp")
    p = Parameters(**calibration["parameters"])
    state = np.asarray(calibration["initial_state"], dtype=float).copy()
    strain = 0.
    records = []
    for cycle in range(1, cycles+1):
        peak, valley = float(state[0]), float(state[0])
        endpoints = []
        # Source: DOI 10.1016/j.dib.2023.109333. Methods specify 0.001/s;
        # Table 3 lists a nominal 2 s post-creep ramp. Both cannot hold
        # exactly when creep strain is nonzero. Expose this loading ambiguity
        # explicitly; fixed_rate follows Methods without resetting creep strain.
        for kind, duration, target in (("ramp", 2., -.002), ("strain", 180., None),
                                      ("ramp", 2., 0.), ("stress", 1800., None),
                                      ("ramp", 2., .002), ("strain", 180., None),
                                      ("ramp", 2., 0.)):
            if kind == "stress":
                solution = stress_hold(state, strain, duration, p,
                                       limit=limit, rate=rate, rtol=rtol)
                strain = float(solution.y[5, -1])
            else:
                if kind == "ramp" and ramp_mode == "fixed_rate":
                    duration = abs(target-strain)/.001
                    if duration <= 0:
                        raise RuntimeError("Zero-length strain ramp requires explicit protocol review")
                strain_rate = (target-strain)/duration if kind == "ramp" else 0.
                solution = integrate_bounded(state, duration, strain_rate, p,
                                             limit=limit, rate=rate, rtol=rtol)
                if kind == "ramp":
                    strain = target
            state = solution.y[:5, -1].copy()
            peak = max(peak, float(solution.y[0].max()))
            valley = min(valley, float(solution.y[0].min()))
            endpoints.append(dict(control=kind, duration_s=duration,
                                  strain=strain, state=state.tolist()))
        records.append(dict(cycle=cycle, sampled_peak_MPa=peak,
                            sampled_valley_MPa=valley, end_state=state.tolist(),
                            post_creep_target_overshot=endpoints[3]["strain"] > .002,
                            segment_endpoints=endpoints))
    from hashlib import sha256
    return dict(schema="agentfem.research.p92-mixed-control.v1", protocol="SLC",
                calibration_sha256=sha256(json.dumps(calibration, sort_keys=True,
                                          allow_nan=False).encode()).hexdigest(),
                parameters=calibration["parameters"], limit=limit, rate=rate, rtol=rtol,
                ramp_mode=ramp_mode,
                loading_ambiguity="Methods prescribe 0.001/s; Table 3 lists a nominal 2 s post-creep ramp",
                cycles=records, engineering_qualified=False, life_qualified=False,
                protocol_envelope_exceeded=any(r["post_creep_target_overshot"] for r in records),
                first_envelope_exit_cycle=next((r["cycle"] for r in records
                                               if r["post_creep_target_overshot"]), None),
                scope="Nominal mixed control and RF-calibrated response; no SLC fit or failure prediction",
                overshoot_policy="Continue toward prescribed target for diagnostics only; reversed post-creep ramps are outside the nominal SLC envelope",
                loading_source="https://doi.org/10.1016/j.dib.2023.109333 Table 3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--ramp-mode", choices=("fixed_rate", "table_duration"), default="fixed_rate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new evidence path")
    from tools.p92_bam_source import read_source
    data = read_source(args.source, protocol="SLC")
    report = run(json.loads(args.calibration.read_text()), cycles=args.cycles, ramp_mode=args.ramp_mode)
    report["comparison"] = compare_candidate(data, report)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report["comparison"]["groups"]))
