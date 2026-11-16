"""Research candidate: bounded dynamic-recovery evolution, no failure law.

The original published reproduction remains unchanged. A bounded Chaboche-type
recovery multiplier is a mechanism hypothesis, not a new P92 material card.
It leaves the proven isotropic-strength bound intact. Its parameters must be
calibrated separately; numerical admissibility is not experimental validation.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from tools.p92_published_recovery_reference import (
    Parameters, rates, rate_jacobian, isotropic_strength_lower_bound,
)


def multiplier(accumulated, *, limit, rate):
    """phi = 1+(limit-1)*(1-exp(-rate*p)), with 1 <= phi <= limit."""
    from numbers import Real
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
            or not np.isfinite(v) for v in (accumulated, limit, rate))
            or accumulated < 0 or limit < 1 or rate < 0):
        raise ValueError("Require p >= 0, limit >= 1, rate >= 0, all real finite")
    return _trial_multiplier(accumulated, limit=limit, rate=rate)


def _trial_multiplier(accumulated, *, limit, rate):
    # Radau's error estimator/Newton stages may probe slightly negative p.
    # Evaluate the smooth analytic extension there; do not clip a trial or
    # declare its internal stage an accepted material state.
    decay = np.exp(-rate*accumulated)
    return 1+(limit-1)*(-np.expm1(-rate*accumulated)), (limit-1)*rate*decay


def bounded_rates(state, strain_rate, parameters, *, limit, rate):
    result = rates(state, strain_rate, parameters)
    phi, _ = _trial_multiplier(state[4], limit=limit, rate=rate)
    result[1] -= parameters.C1*(phi-1)*state[1]*result[4]
    result[2] -= parameters.C2*(phi-1)*state[2]*result[4]
    return result


def bounded_jacobian(state, parameters, *, limit, rate):
    result = rate_jacobian(state, parameters)
    phi, derivative = _trial_multiplier(state[4], limit=limit, rate=rate)
    speed = rates(state, 0., parameters)[4]
    for i, C in ((1, parameters.C1), (2, parameters.C2)):
        result[i] -= C*(phi-1)*state[i]*result[4]
        result[i, i] -= C*(phi-1)*speed
        result[i, 4] -= C*derivative*state[i]*speed
    return result


def integrate_bounded(old, duration, strain_rate, parameters, *, limit, rate, rtol=1e-8):
    state = np.asarray(old)
    if (np.iscomplexobj(state) or state.shape != (5,) or not np.isfinite(state).all()
            or state[4] < 0 or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid state or integration controls")
    multiplier(state[4], limit=limit, rate=rate)
    bound = isotropic_strength_lower_bound(parameters, state[3])
    if bound <= 0:
        raise ValueError("Require a positive sufficient isotropic-strength bound")
    solution = solve_ivp(
        lambda t, y: bounded_rates(y, strain_rate, parameters, limit=limit, rate=rate),
        (0., duration), state.astype(float).copy(), method="Radau",
        jac=lambda t, y: bounded_jacobian(y, parameters, limit=limit, rate=rate),
        rtol=rtol, atol=[1e-8]*4+[1e-12],
    )
    if (not solution.success or not np.isfinite(solution.y).all()
            or np.any(np.diff(solution.y[4]) < -1e-10)
            or solution.y[4, -1] < state[4]
            or np.any(parameters.k+solution.y[3] <= 0)):
        raise RuntimeError("Bounded recovery integration failed its numerical/domain checks")
    return solution


def run_candidate(calibration, *, cycles, limit, rate, rtol=1e-8):
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
        raise ValueError("Require positive integer cycles")
    if not calibration["fit"]["success"]:
        raise ValueError("Require converged frozen base calibration")
    p = Parameters(**calibration["parameters"])
    state = np.asarray(calibration["initial_state"], dtype=float).copy()
    records = []
    for cycle in range(1, cycles+1):
        trial = state.copy()
        peak, valley = float(trial[0]), float(trial[0])
        for duration, strain_rate in ((2., .001), (180., 0.), (4., -.001), (180., 0.), (2., .001)):
            solution = integrate_bounded(trial, duration, strain_rate, p,
                                         limit=limit, rate=rate, rtol=rtol)
            trial = solution.y[:, -1].copy()
            peak = max(peak, float(solution.y[0].max()))
            valley = min(valley, float(solution.y[0].min()))
        state = trial
        records.append(dict(cycle=cycle, sampled_peak_MPa=peak,
                            sampled_valley_MPa=valley, end_state=state.tolist()))
    return dict(schema="agentfem.research.p92-bounded-recovery.v1", parameters=asdict(p),
                limit=limit, rate=rate, rtol=rtol, cycles=records,
                sufficient_strength_lower_bound_MPa=isotropic_strength_lower_bound(p),
                engineering_qualified=False, life_qualified=False,
                scope="Continuous nominal RF, bounded-recovery hypothesis; no refit, no life law")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--cycles", type=int, default=1000)
    parser.add_argument("--limit", type=float, required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_candidate(json.loads(args.calibration.read_text()), cycles=args.cycles,
                           limit=args.limit, rate=args.rate, rtol=args.rtol)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report["cycles"][-1]))
