# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Frozen measured-input P92 response screen; not a calibrated engineering law."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from agentfem.results import HistoryResult
from agentfem.verification import history_errors
from tools.p92_bam_source import read_source, replay_window
from tools.p92_published_recovery_reference import (
    implicit_update, integrate_segment, published_parameters,
)


def replay(times, strains, parameters, initial_state, *, subdivisions=1, method="BE"):
    """Advance one continuous material history; observations never reset state."""
    t, e = np.asarray(times, dtype=float), np.asarray(strains, dtype=float)
    if (t.ndim != 1 or len(t) < 2 or e.shape != t.shape
            or not np.isfinite(t).all() or not np.isfinite(e).all() or np.any(np.diff(t) <= 0)):
        raise ValueError("Require finite aligned strain with strictly increasing physical time.")
    if (isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or subdivisions < 1):
        raise ValueError("subdivisions must be a positive integer.")
    if method not in {"BE", "Radau"}:
        raise ValueError("method must be BE or Radau.")
    old = np.asarray(initial_state, dtype=float).copy()
    if old.shape != (5,) or not np.isfinite(old).all() or old[4] < 0:
        raise ValueError("Provide explicit five-component initial state.")
    states = [old.copy()]
    for dt, de in zip(np.diff(t), np.diff(e), strict=True):
        if method == "Radau":
            old = integrate_segment(old, dt, de/dt, parameters, rtol=1e-8).y[:, -1].copy()
        else:
            for _ in range(subdivisions):
                old, _ = implicit_update(old, de/subdivisions, dt/subdivisions, parameters)
        states.append(old.copy())
    return np.asarray(states)


def calibrate(times, strains, observed, initial_state, *, max_nfev=30):
    """Only supplied training observations influence the three fitted constants."""
    base = published_parameters(600)
    t = np.asarray(times, dtype=float)
    dt = np.diff(t)
    weights = np.r_[dt[0]/2, (dt[:-1]+dt[1:])/2, dt[-1]/2]
    weights = np.sqrt(weights/weights.sum())
    def parameters(z):
        return replace(base, E=float(z[0]*1e5), k=float(z[1]*100), Z=float(z[2]*1000))
    def residual(z):
        return weights*(replay(t, strains, parameters(z), initial_state)[:, 0]-observed)
    fit = least_squares(residual, [base.E/1e5, base.k/100, base.Z/1000],
                        bounds=([1., .5, .3], [2., 2.5, 1.5]), max_nfev=max_nfev,
                        ftol=1e-6, xtol=1e-6, gtol=1e-6)
    return parameters(fit.x), {
        "success": bool(fit.success), "message": fit.message, "nfev": fit.nfev,
        "active_bounds": fit.active_mask.tolist(), "cost": float(fit.cost),
        "scaled_parameter_jacobian_singular_values": np.linalg.svd(fit.jac, compute_uv=False).tolist(),
    }


def screen(data, *, max_nfev=30, include_radau=False):
    start = perf_counter()
    window = replay_window(data, first_cycle=1, last_cycle=10, max_interval_s=2.)
    h = window["histories"]
    t, strain, observed = h["strain"].abscissa, h["strain"].values, h["stress_Pa"].values/1e6
    selected = (data["cycle"] >= 1) & (data["cycle"] <= 10)
    cycles = data["cycle"][selected]
    training = cycles <= 3
    initial = np.array([observed[0], 0., 0., 0., 0.])
    p, fit = calibrate(t[training], strain[training], observed[training], initial, max_nfev=max_nfev)
    states = replay(t, strain, p, initial)
    refined = replay(t, strain, p, initial, subdivisions=2)
    reports = []
    for cycle in range(1, 11):
        m = cycles == cycle
        predicted, actual = states[m, 0], observed[m]
        comparison = history_errors(HistoryResult("predicted", t[m], predicted, unit="MPa"),
            HistoryResult("measured", t[m], actual, unit="MPa"),
            reference_source=f"BAM RF sha256:{data['sha256']}; cycle {cycle}", max_interval=2.)
        def work(stress, increments=np.diff(strain[m])):
            return float(np.sum((stress[1:]+stress[:-1])*.5*increments)*1e6)
        reports.append({"cycle": cycle, "role": "calibration" if cycle <= 3 else "within_specimen_holdout",
            "stress_errors": comparison, "predicted_peak_MPa": float(predicted.max()),
            "observed_peak_MPa": float(actual.max()), "predicted_valley_MPa": float(predicted.min()),
            "observed_valley_MPa": float(actual.min()), "predicted_sampled_work_J_per_m3": work(predicted),
            "observed_sampled_work_J_per_m3": work(actual),
            "refinement_max_stress_change_MPa": float(np.max(abs(refined[m, 0]-predicted)))})
    report = {"schema": "agentfem.research.p92-bam-cyclic.v1", "source_sha256": data["sha256"],
        "fit": fit, "parameters": asdict(p), "fitted_names": ["E", "k", "Z"],
        "fixed_parameter_source": "DOI 10.3390/met10030307 Table 4 at 600 C; unvalidated transfer hypothesis",
        "nominal_temperature_C": 620, "measured_temperature_range_K": float(np.ptp(h["temperature_K"].values)),
        "initial_state": initial.tolist(), "final_state": states[-1].tolist(),
        "initial_state_assumption": "Measured initial stress; zero hidden history; no later resets",
        "cycles": reports, "engineering_qualified": False,
        "scope": "Uniaxial continuous-state response transfer screen, not structural or life validation",
        "work_scope": "Sampled nominal axial path work; not dissipation and no endpoint closure",
        "uncertainty": "No independent-specimen scatter inferred from within-path residuals"}
    if include_radau:
        independent = replay(t, strain, p, initial, method="Radau")
        report["independent_Radau_max_stress_difference_MPa"] = float(np.max(abs(independent[:, 0]-states[:, 0])))
        report["refined_BE_Radau_max_stress_difference_MPa"] = float(np.max(abs(independent[:, 0]-refined[:, 0])))
    report["elapsed_seconds"] = perf_counter()-start
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--max-nfev", type=int, default=30)
    parser.add_argument("--radau", action="store_true")
    args = parser.parse_args()
    print(json.dumps(screen(read_source(args.source, protocol="RF"), max_nfev=args.max_nfev,
                            include_radau=args.radau), indent=2, allow_nan=False))
