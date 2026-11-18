"""Research-only discriminator: saturation shift versus persistent softening.

The original 2020 reproduction is untouched. The alternative combines a
saturating recovery variable S with a separate linear contribution H*p.
Its recovery extension is a hypothesis, NOT the published 2013 or 2020 model
and NOT a damage-to-failure law. Existing parameters are not calibrated to it.
"""

import numpy as np
from scipy.integrate import solve_ivp

from tools.p92_published_recovery_reference import rates, rate_jacobian


def isotropic_reference(accumulated, *, Q, b, H, mode):
    """Virgin-state exact solutions with static recovery disabled."""
    p = np.asarray(accumulated)
    if (np.iscomplexobj(p) or not np.isfinite(p).all() or np.any(p < 0)
            or not np.isfinite([Q, b, H]).all() or b <= 0):
        raise ValueError("Require finite real parameters, b > 0 and p >= 0")
    saturation = -np.expm1(-b*p)
    if mode == "published_rate":
        return (Q+H/b)*saturation
    if mode == "separate_linear":
        return Q*saturation+H*p
    raise ValueError("Unknown isotropic reference mode")


def split_rates(state, strain_rate, parameters):
    """State sigma, x1, x2, S, p; total isotropic contribution R=S+H*p.

    Both dynamic and static recovery act on S. Flow sees S+H*p. This explicit
    choice must be recalibrated and independently validated before application.
    """
    p = parameters
    transformed = np.asarray(state, dtype=float).copy()
    transformed[3] += p.H*transformed[4]
    result = rates(transformed, strain_rate, p)
    speed = result[4]
    S, accumulated = state[3:]
    recovery = p.gamma_r1+p.gamma_r2*np.exp(-p.r*accumulated)
    result[3] = p.b*(p.Q-S)*speed+recovery*abs(p.Qr-S)**p.mr*np.sign(p.Qr-S)
    return result


def split_jacobian(state, parameters):
    p = parameters
    transformed = np.asarray(state, dtype=float).copy()
    transformed[3] += p.H*transformed[4]
    change = np.eye(5)
    change[3, 4] = p.H
    jac = rate_jacobian(transformed, p) @ change
    S, accumulated = state[3:]
    speed = split_rates(state, 0., p)[4]
    tail = p.gamma_r2*np.exp(-p.r*accumulated)
    recovery = p.gamma_r1+tail
    jac[3] = p.b*(p.Q-S)*jac[4]
    jac[3, 3] -= p.b*speed+recovery*p.mr*abs(p.Qr-S)**(p.mr-1.)
    jac[3, 4] -= p.r*tail*abs(p.Qr-S)**p.mr*np.sign(p.Qr-S)
    return jac


def integrate_split(state, duration, strain_rate, parameters, *, rtol=1e-8):
    """Detached trial with terminal positive-strength guard; no silent clipping."""
    old = np.asarray(state)
    if (np.iscomplexobj(old) or old.shape != (5,) or not np.isfinite(old).all()
            or old[4] < 0 or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1
            or parameters.k+old[3]+parameters.H*old[4] <= 0):
        raise ValueError("Invalid split-model initial state or integration controls")

    def strength(_time, candidate):
        return parameters.k+candidate[3]+parameters.H*candidate[4]
    strength.terminal = True
    strength.direction = -1
    solution = solve_ivp(lambda t, y: split_rates(y, strain_rate, parameters),
                         (0., duration), old.astype(float).copy(), method="Radau",
                         jac=lambda t, y: split_jacobian(y, parameters),
                         rtol=rtol, atol=[1e-8]*4+[1e-12], events=strength)
    if (not solution.success or solution.status == 1
            or not np.isfinite(solution.y).all()
            or np.any(np.diff(solution.y[4]) < -1e-10)):
        raise RuntimeError("Split-model trial failed or reached nonpositive strength")
    return solution


def run_frozen_candidate(calibration, *, cycles, rtol=1e-8):
    """No refitting: a structural sensitivity experiment, not a validation fit."""
    from tools.p92_published_recovery_reference import Parameters
    from hashlib import sha256
    import json
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
        raise ValueError("cycles must be a positive integer")
    if not calibration["fit"]["success"]:
        raise ValueError("Require converged frozen calibration")
    p = Parameters(**calibration["parameters"])
    state = np.asarray(calibration["initial_state"], dtype=float).copy()
    state[3] -= p.H*state[4]
    records, failure = [], None
    for cycle in range(1, cycles+1):
        trial = state.copy()
        peak, valley = float(trial[0]), float(trial[0])
        for segment, (duration, rate) in enumerate(((2., .001), (180., 0.),
                (4., -.001), (180., 0.), (2., .001))):
            try:
                solution = integrate_split(trial, duration, rate, p, rtol=rtol)
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
                        "end_state": state.tolist(),
                        "end_strength_MPa": float(p.k+state[3]+p.H*state[4])})
    return {"schema": "agentfem.research.p92-split-softening.v1",
            "parameters": calibration["parameters"], "requested_cycles": cycles,
            "completed_cycles": len(records), "failure": failure, "rtol": rtol,
            "calibration_record_sha256": sha256(json.dumps(calibration, sort_keys=True,
                                                         allow_nan=False).encode()).hexdigest(),
            "cycles": records, "last_completed_state": state.tolist(),
            "state_layout": ["sigma", "chi1", "chi2", "S", "p"],
            "total_isotropic": "S+H*p", "engineering_qualified": False,
            "life_qualified": False, "parameters_refitted": False,
            "scope": "Hypothesized split recovery law, nominal RF loading; not published-model reproduction or a failure-life prediction"}


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--cycles", type=int, default=1000)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_frozen_candidate(json.loads(args.calibration.read_text()),
                                  cycles=args.cycles, rtol=args.rtol)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("completed_cycles", "failure",
                                           "engineering_qualified", "life_qualified")}))
    raise SystemExit(1 if report["failure"] else 0)
