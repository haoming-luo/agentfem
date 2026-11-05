# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Uniaxial equation reproduction of DOI 10.3390/met10030307, not FEM.

Published Table 4 parameters, MPa/seconds, isothermal 600 or 650 Celsius.
No fitting, temperature interpolation, life prediction or BAM transfer.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

SOURCE_SHA256 = "c00a71167969edaab6a1b52641140cb5a3f241e82a90b74b464f293f46fcd604"


@dataclass(frozen=True)
class Parameters:
    E: float
    k: float
    Q: float
    b: float
    H: float
    Qr: float
    gamma_r1: float
    gamma_r2: float
    r: float
    mr: float
    a1: float
    C1: float
    a2: float
    C2: float
    gamma1: float
    gamma2: float
    q: float
    m: float
    Z: float
    n: float


def published_parameters(temperature_C):
    if temperature_C == 600:
        return Parameters(161933., 180., -111.06, .81, -4., -105.74,
                          2.14e-8, 2.01e-8, 5.98, 2.80,
                          27.05, 25000., 72.39, 996.97, 3.e-9, 7.47e-7,
                          10., 3.87, 1028., 3.)
    if temperature_C == 650:
        return Parameters(134573., 132., -93.31, .66, -4., -86.67,
                          1.93e-8, 9.22e-8, 1.05, 2.46,
                          9.89, 24685.10, 49.87, 1055., 1.e-8, 2.e-6,
                          5.80, 3.90, 700., 3.80)
    raise ValueError("Published parameters exist at 600 and 650 Celsius only.")


def rates(state, strain_rate, p):
    """State: sigma, chi1, chi2, R, accumulated inelastic strain.

    Equations 2, 4, 7, 9--13; positive stress/backstress conventions.
    No separate Norton branch is superimposed on the unified flow law.
    """
    sigma, x1, x2, R, accumulated = state
    effective = sigma-x1-x2
    speed = (max(abs(effective)-R-p.k, 0.)/p.Z)**p.n
    flow = speed*np.sign(effective)
    gamma = p.gamma1+p.gamma2*np.exp(-p.q*accumulated)
    gamma_r = p.gamma_r1+p.gamma_r2*np.exp(-p.r*accumulated)
    return np.array([
        p.E*(strain_rate-flow),
        p.C1*(p.a1*flow-x1*speed)-gamma*abs(x1)**p.m*np.sign(x1),
        p.C2*(p.a2*flow-x2*speed)-gamma*abs(x2)**p.m*np.sign(x2),
        p.b*(p.Q-R)*speed+gamma_r*abs(p.Qr-R)**p.mr*np.sign(p.Qr-R)+p.H*speed,
        speed,
    ])


def rate_jacobian(state, p):
    """Analytical fixed-temperature Jacobian of the five coupled rate equations."""
    sigma, x1, x2, R, accumulated = state
    effective = sigma-x1-x2
    direction = np.sign(effective)
    overstress = abs(effective)-R-p.k
    speed = (max(overstress, 0.)/p.Z)**p.n
    gradient = np.zeros(5)
    if overstress > 0:
        gradient[:4] = p.n*(overstress/p.Z)**(p.n-1.)/p.Z * np.array([
            direction, -direction, -direction, -1.])
    flow_gradient = direction*gradient
    gamma_tail = p.gamma2*np.exp(-p.q*accumulated)
    gamma = p.gamma1+gamma_tail
    gamma_r_tail = p.gamma_r2*np.exp(-p.r*accumulated)
    gamma_r = p.gamma_r1+gamma_r_tail
    jac = np.zeros((5, 5))
    jac[0] = -p.E*flow_gradient
    for row, x, coefficient, saturation in ((1, x1, p.C1, p.a1), (2, x2, p.C2, p.a2)):
        jac[row] = coefficient*(saturation*flow_gradient-x*gradient)
        jac[row, row] -= coefficient*speed+gamma*p.m*abs(x)**(p.m-1.)
        jac[row, 4] += p.q*gamma_tail*abs(x)**p.m*np.sign(x)
    jac[3] = (p.b*(p.Q-R)+p.H)*gradient
    jac[3, 3] -= p.b*speed+gamma_r*p.mr*abs(p.Qr-R)**(p.mr-1.)
    jac[3, 4] -= p.r*gamma_r_tail*abs(p.Qr-R)**p.mr*np.sign(p.Qr-R)
    jac[4] = gradient
    return jac


def implicit_update(old, strain_increment, dt, parameters):
    """Local backward-Euler update and consistent axial tangent; not a 3D law.

    All internal variables are solved simultaneously. No state is committed or
    modified here, and a failed local solve returns no usable response.
    """
    if (np.iscomplexobj(old) or np.asarray(old).shape != (5,)
            or not np.isfinite(old).all() or old[4] < 0
            or any(isinstance(v, (bool, np.bool_)) or np.iscomplexobj(v)
                   for v in (strain_increment, dt))
            or not np.isfinite([strain_increment, dt]).all() or dt <= 0):
        raise ValueError("Require real finite local state, strain increment and positive dt.")
    old = np.asarray(old, dtype=float).copy()
    scale = np.array([parameters.k]*4+[1.])
    rate = strain_increment/dt

    def residual(candidate):
        return (candidate-old-dt*rates(candidate, rate, parameters))/scale

    def jacobian(candidate):
        return (np.eye(5)-dt*rate_jacobian(candidate, parameters))/scale[:, None]

    guess = old.copy()
    guess[0] += parameters.E*strain_increment
    solution = root(residual, guess, jac=jacobian, method="hybr", options={"xtol": 1.e-10})
    state = solution.x
    if (not solution.success or not np.isfinite(state).all()
            or np.max(np.abs(residual(state))) > 1.e-8
            or state[4] < old[4]-1.e-12 or state[3]+parameters.k <= 0):
        raise RuntimeError(f"Local implicit recovery update failed: {solution.message}")
    derivative = np.linalg.solve(np.eye(5)-dt*rate_jacobian(state, parameters),
                                 np.array([parameters.E, 0., 0., 0., 0.]))
    return state.copy(), float(derivative[0])


def substepped_update(old, strain_increment, dt, parameters, *, substeps):
    """Fixed-partition local reference with the complete axial chain tangent.

    Each local solve owns a detached candidate. The derivative includes every
    intermediate history update; it is not the last substep's fixed-old-state
    tangent. Choosing an adaptive partition and admitting a 3D provider remain
    separate tasks. No committed caller state is modified, even on failure.
    """
    if (isinstance(substeps, (bool, np.bool_))
            or not isinstance(substeps, (int, np.integer)) or substeps < 1):
        raise ValueError("substeps must be a positive integer.")
    if (any(isinstance(v, (bool, np.bool_)) or np.iscomplexobj(v)
            for v in (strain_increment, dt))
            or not np.isfinite([strain_increment, dt]).all() or dt <= 0):
        raise ValueError("Require real finite strain increment and positive dt.")
    current = np.asarray(old).copy()
    h = dt/substeps
    derivative = np.zeros(5)
    direct = np.array([parameters.E/substeps, 0., 0., 0., 0.])
    for _ in range(substeps):
        current, _ = implicit_update(current, strain_increment/substeps, h, parameters)
        derivative = np.linalg.solve(
            np.eye(5)-h*rate_jacobian(current, parameters), derivative+direct)
    return current.copy(), float(derivative[0])


def integrate_segment(state, duration, strain_rate, parameters, *, method="Radau", rtol=1.e-8):
    if method not in {"Radau", "BDF"}:
        raise ValueError("Use a declared stiff integrator: Radau or BDF.")
    if (np.iscomplexobj(state) or np.asarray(state).shape != (5,)
            or not np.isfinite(state).all() or state[4] < -1.e-12
            or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Require finite state and positive physical duration/tolerance.")
    old = np.asarray(state, dtype=float).copy()
    result = solve_ivp(lambda _t, y: rates(y, strain_rate, parameters), (0., duration), old,
                       method=method, rtol=rtol, atol=np.array([1.e-8]*4+[1.e-12]),
                       dense_output=True)
    if not result.success or not np.isfinite(result.y).all():
        raise RuntimeError(f"Material-point integration failed: {result.message}")
    if (np.any(np.diff(result.y[4]) < -1.e-10)
            or np.any(result.y[3]+parameters.k <= 0)):
        raise RuntimeError("Path left the positive-strength, monotone-history domain.")
    return result


def run_cycles(*, temperature_C=600, strain_range=.008, dwell=180., cycles=3,
               method="Radau", rtol=1.e-8):
    if (isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1
            or not np.isfinite([strain_range, dwell]).all() or strain_range <= 0 or dwell <= 0):
        raise ValueError("Require positive strain range, dwell and integer cycle count.")
    p = published_parameters(temperature_C)
    amplitude, strain_rate = strain_range/2., .001
    state = np.zeros(5)
    time, evaluations = 0., 0

    def segment(duration, rate):
        nonlocal state, time, evaluations
        result = integrate_segment(state, duration, rate, p, method=method, rtol=rtol)
        state = result.y[:, -1].copy()
        time += duration
        evaluations += result.nfev
        return result

    # Explicit startup assumption: virgin zero state, ramp to the tensile peak.
    segment(amplitude/strain_rate, strain_rate)
    records = []
    for cycle in range(1, cycles+1):
        peak = state.copy()
        hold = segment(dwell, 0.)
        records.append({"cycle": cycle, "hold_end_time_s": time,
                        "peak_stress_MPa": float(peak[0]),
                        "end_hold_stress_MPa": float(state[0]),
                        "relaxed_stress_MPa": float(peak[0]-state[0]),
                        "accumulated_inelastic_strain": float(state[4]),
                        "hold_stress_samples_MPa": hold.sol(np.linspace(0., dwell, 11))[0].tolist()})
        if cycle != cycles:
            segment(strain_range/strain_rate, -strain_rate)
            segment(strain_range/strain_rate, strain_rate)
    return {"source_doi": "10.3390/met10030307", "parameter_table": 4,
            "source_pdf_sha256": SOURCE_SHA256,
            "scope": "published uniaxial equations; no experimental validation or FEM claim",
            "temperature_C": temperature_C, "strain_range": strain_range,
            "strain_rate_per_s": strain_rate, "tensile_dwell_s": dwell,
            "startup": "zero state; first ramp from zero strain to positive half-range",
            "method": method, "rtol": rtol, "parameters": asdict(p),
            "cycles": records, "final_state": state.tolist(), "rhs_evaluations": evaluations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=int, choices=(600, 650), default=600)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--dwell", type=float, default=180.)
    parser.add_argument("--strain-range", type=float, default=.008)
    parser.add_argument("--method", choices=("Radau", "BDF"), default="Radau")
    parser.add_argument("--rtol", type=float, default=1.e-8)
    parser.add_argument("--published-campaign", action="store_true",
                        help="Run Table 2 conditions to their reported half-lives, not predicted lives.")
    parser.add_argument("--output", type=Path, help="Write a new JSON evidence file; never overwrite.")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Output exists; choose a new evidence path.")
    if args.published_campaign:
        rows = []
        conditions = ((600, .008, 60., 1303), (600, .008, 180., 1190),
                      (600, .008, 600., 675), (600, .012, 180., 541),
                      (650, .008, 60., 870), (650, .008, 180., 853),
                      (650, .008, 600., 625), (650, .012, 180., 482))
        for temperature, strain_range, dwell, reported_life in conditions:
            started = perf_counter()
            row = run_cycles(temperature_C=temperature, cycles=reported_life//2,
                             dwell=dwell, strain_range=strain_range,
                             method=args.method, rtol=args.rtol)
            row["reported_life_used_to_select_run_length"] = reported_life
            row["elapsed_seconds"] = perf_counter()-started
            rows.append(row)
            print(f"Completed {temperature} C, range={strain_range:g}, dwell={dwell:g} s, "
                  f"cycles={reported_life//2}; {row['elapsed_seconds']:.1f} s", flush=True)
        report = {"scope": "published-equation reproduction; not life or experimental validation",
                  "reported_lives_are_inputs_not_predictions": True, "runs": rows}
    else:
        report = run_cycles(temperature_C=args.temperature, cycles=args.cycles,
                            dwell=args.dwell, strain_range=args.strain_range,
                            method=args.method, rtol=args.rtol)
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized+"\n")
        print(f"Evidence: {args.output}")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
