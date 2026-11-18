"""Bounded joint RF/SLR response experiment, not independent validation.

Fit each protocol's first three continuous cycles with equal protocol weight.
The original equations remain unchanged. This diagnoses whether transferable
parameters exist within a predeclared six-parameter family before adding terms.
Previously inspected later cycles remain development checks, not blind data.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from tools.p92_bam_source import read_source, replay_window
from tools.p92_bam_cyclic_screen import replay
from tools.p92_published_recovery_reference import published_parameters, isotropic_strength_lower_bound


def decode(values):
    z = np.asarray(values, dtype=float)
    if z.shape != (6,) or not np.isfinite(z).all():
        raise ValueError("Require six finite scaled parameters")
    E, gap, Z, q, b, log_recovery = z
    if E <= 0 or gap <= 0 or Z <= 0 or q <= 0 or b <= 0:
        raise ValueError("Require positive modulus, strength gap, viscosity, Q magnitude and b")
    p = published_parameters(600)
    Q = -100*q
    k = 100*gap-min(Q+p.H/b, p.Qr)
    recovery = 10.**log_recovery
    return replace(p, E=E*1e5, k=k, Z=Z*1000, Q=Q, b=b,
                   gamma_r1=p.gamma_r1*recovery, gamma_r2=p.gamma_r2*recovery)


def training_path(data, *, balance_strain=False):
    window = replay_window(data, first_cycle=1, last_cycle=3,
                           max_interval_s=2. if data["protocol"] == "RF" else 6.)
    h = window["histories"]
    t, e, s = h["strain"].abscissa, h["strain"].values, h["stress_Pa"].values/1e6
    dt = np.diff(t)
    weights = np.r_[dt[0]/2, (dt[:-1]+dt[1:])/2, dt[-1]/2]
    weights /= weights.sum()
    if balance_strain:
        de = np.abs(np.diff(e))
        if de.sum() == 0:
            raise ValueError("Strain-balanced calibration requires nonconstant strain")
        strain_weights = np.r_[de[0]/2, (de[:-1]+de[1:])/2, de[-1]/2]
        weights = .5*weights+.5*strain_weights/strain_weights.sum()
    return t, e, s, np.sqrt(weights), np.array([s[0], 0., 0., 0., 0.])


def fit_joint(rf, slr, *, max_nfev=40, balance_strain=False):
    if rf["protocol"] != "RF" or slr["protocol"] != "SLR":
        raise ValueError("Require separate RF and SLR records")
    if type(max_nfev) is not int or max_nfev < 1:
        raise ValueError("Require positive integer evaluation budget")
    started = perf_counter()
    paths = [training_path(data, balance_strain=balance_strain) for data in (rf, slr)]
    initial = np.array([1.551764711, .01, 1.192485, 1.1106, .81, 0.])
    lower, upper = [1., 1e-8, .3, .2, .1, -2.], [2., 1., 2., 1.5, 3., 2.]

    def residual(z):
        p = decode(z)
        return np.concatenate([w*(replay(t, e, p, old)[:, 0]-s)/np.sqrt(2)
                               for t, e, s, w, old in paths])

    fit = least_squares(residual, initial, bounds=(lower, upper), max_nfev=max_nfev,
                        ftol=1e-6, xtol=1e-6, gtol=1e-6)
    p = decode(fit.x)
    report = dict(schema="agentfem.research.p92-joint-protocol-fit.v1",
                  parameters=asdict(p), scaled_parameters=fit.x.tolist(),
                  scaled_bounds=[lower, upper],
                  scaled_names=["E/1e5", "strength_gap/100", "Z/1000", "-Q/100", "b", "log10_R_recovery_scale"],
                  fit=dict(success=bool(fit.success), message=fit.message,
                           cost=float(fit.cost), nfev=int(fit.nfev),
                           active_bounds=fit.active_mask.tolist(),
                           singular_values=np.linalg.svd(fit.jac, compute_uv=False).tolist()),
                  sufficient_strength_lower_bound_MPa=isotropic_strength_lower_bound(p),
                  training_cycles=[1, 2, 3], source_sha256={d["protocol"]: d["sha256"] for d in (rf, slr)},
                  initial_state_assumption="Initial measured stress, virgin hidden history separately per specimen; no later resets",
                  objective=("Equal protocol weights; within each protocol 50% physical time and 50% total strain variation"
                             if balance_strain else
                             "Equal protocol weights; within each protocol trapezoidal physical-time weights"),
                  engineering_qualified=False, independent_validation=False, life_qualified=False,
                  scope="Model-family discrimination on inspected records; no new constitutive terms or life fit")
    if fit.success:
        records = []
        for data in (rf, slr):
            h = replay_window(data, first_cycle=1, last_cycle=10,
                              max_interval_s=2. if data["protocol"] == "RF" else 6.)["histories"]
            t, e, s = h["strain"].abscissa, h["strain"].values, h["stress_Pa"].values/1e6
            old = np.array([s[0], 0., 0., 0., 0.])
            y = replay(t, e, p, old)
            fine = replay(t, e, p, old, method="Radau")
            ids = data["cycle"][(data["cycle"] >= 1) & (data["cycle"] <= 10)]
            for cycle in range(1, 11):
                mask = ids == cycle
                duration = np.diff(t[mask])
                error = fine[mask, 0]-s[mask]
                mse = np.sum((error[1:]**2+error[:-1]**2)*.5*duration)/duration.sum()
                records.append(dict(protocol=data["protocol"], cycle=cycle,
                                    role="calibration" if cycle <= 3 else "inspected_development",
                                    time_weighted_RMSE_MPa=float(np.sqrt(mse)),
                                    max_abs_error_MPa=float(np.max(np.abs(error))),
                                    observed_peak_MPa=float(s[mask].max()),
                                    predicted_peak_MPa=float(fine[mask, 0].max()),
                                    BE_Radau_max_difference_MPa=float(np.max(np.abs(y[mask, 0]-fine[mask, 0])))))
        report["cycles"] = records
    report["elapsed_seconds"] = perf_counter()-started
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rf", type=Path)
    parser.add_argument("slr", type=Path)
    parser.add_argument("--max-nfev", type=int, default=40)
    parser.add_argument("--balance-strain", action="store_true",
                        help="Predeclared 50:50 time/strain-variation objective; never a measurement uncertainty model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new evidence path")
    report = fit_joint(read_source(args.rf, protocol="RF"),
                       read_source(args.slr, protocol="SLR"), max_nfev=args.max_nfev,
                       balance_strain=args.balance_strain)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report["fit"]))
    raise SystemExit(0 if report["fit"]["success"] else 1)
