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
    implicit_update, integrate_segment, published_parameters, isotropic_strength_lower_bound,
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
    for index, (dt, de) in enumerate(zip(np.diff(t), np.diff(e), strict=True)):
        if method == "Radau":
            old = integrate_segment(old, dt, de/dt, parameters, rtol=1e-8).y[:, -1].copy()
        else:
            for substep in range(subdivisions):
                try:
                    old, _ = implicit_update(old, de/subdivisions, dt/subdivisions, parameters)
                except RuntimeError as exc:
                    raise RuntimeError(f"Replay interval {index}, time={t[index]:.17g}, "
                                       f"substep={substep}, dt={dt/subdivisions:.17g}, "
                                       f"de={de/subdivisions:.17g}, old={old.tolist()}: {exc}") from exc
        states.append(old.copy())
    return np.asarray(states)


def calibrate(times, strains, observed, initial_state, *, max_nfev=30,
              positive_strength=False):
    """Only supplied training observations influence the three fitted constants."""
    # Reject accidental (n, 1) broadcasting before constructing an objective.
    if any(np.iscomplexobj(v) for v in (times, strains, observed, initial_state)):
        raise ValueError("Calibration requires real measured histories and state.")
    t, e, s = (np.asarray(v, dtype=float) for v in (times, strains, observed))
    if (t.ndim != 1 or len(t) < 2 or e.shape != t.shape or s.shape != t.shape
            or not all(np.isfinite(v).all() for v in (t, e, s)) or np.any(np.diff(t) <= 0)):
        raise ValueError("Require aligned finite scalar calibration histories and increasing time.")
    if (isinstance(max_nfev, (bool, np.bool_))
            or not isinstance(max_nfev, (int, np.integer)) or max_nfev < 1):
        raise ValueError("max_nfev must be a positive integer.")
    base = published_parameters(600)
    if not isinstance(positive_strength, bool):
        raise ValueError("positive_strength must be boolean.")
    # Keep the original research screen reproducible. The opt-in alternative
    # changes only the admissible k lower bound, not the training/holdout split.
    minimum_k = 50.
    if positive_strength:
        old = np.asarray(initial_state, dtype=float)
        if old.shape != (5,) or not np.isfinite(old).all() or old[4] < 0:
            raise ValueError("Provide explicit five-component initial state.")
        minimum_k = max(minimum_k, base.k-isotropic_strength_lower_bound(base, old[3])+1.e-6)
        if minimum_k >= 250.:
            raise ValueError("Initial history requires strength outside the frozen calibration bounds.")
    dt = np.diff(t)
    weights = np.r_[dt[0]/2, (dt[:-1]+dt[1:])/2, dt[-1]/2]
    weights = np.sqrt(weights/weights.sum())
    def parameters(z):
        return replace(base, E=float(z[0]*1e5), k=float(z[1]*100), Z=float(z[2]*1000))
    def residual(z):
        return weights*(replay(t, e, parameters(z), initial_state)[:, 0]-s)
    fit = least_squares(residual, [base.E/1e5, max(base.k, minimum_k)/100, base.Z/1000],
                        bounds=([1., minimum_k/100, .3], [2., 2.5, 1.5]), max_nfev=max_nfev,
                        ftol=1e-6, xtol=1e-6, gtol=1e-6)
    return parameters(fit.x), {
        "success": bool(fit.success), "message": fit.message, "nfev": fit.nfev,
        "active_bounds": fit.active_mask.tolist(), "cost": float(fit.cost),
        "scaled_parameter_jacobian_singular_values": np.linalg.svd(fit.jac, compute_uv=False).tolist(),
        "positive_strength_constraint": positive_strength,
        "k_lower_bound_MPa": minimum_k,
        "sufficient_strength_lower_bound_MPa": isotropic_strength_lower_bound(parameters(fit.x), initial_state[3]),
    }


def screen(data, *, max_nfev=30, include_radau=False, positive_strength=False):
    start = perf_counter()
    window = replay_window(data, first_cycle=1, last_cycle=10, max_interval_s=2.)
    h = window["histories"]
    t, strain, observed = h["strain"].abscissa, h["strain"].values, h["stress_Pa"].values/1e6
    selected = (data["cycle"] >= 1) & (data["cycle"] <= 10)
    cycles = data["cycle"][selected]
    training = cycles <= 3
    initial = np.array([observed[0], 0., 0., 0., 0.])
    p, fit = calibrate(t[training], strain[training], observed[training], initial,
                       max_nfev=max_nfev, positive_strength=positive_strength)
    if not fit["success"]:
        raise RuntimeError(f"RF calibration did not converge; no holdout prediction: {fit}")
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
    if positive_strength:
        report["validation_design"] = "Model revision on previously inspected RF/SLR records, not fresh blind validation"
        for record in reports:
            if record["cycle"] > 3:
                record["role"] = "previously_inspected_model_revision_check"
    if include_radau:
        independent = replay(t, strain, p, initial, method="Radau")
        report["independent_Radau_max_stress_difference_MPa"] = float(np.max(abs(independent[:, 0]-states[:, 0])))
        report["refined_BE_Radau_max_stress_difference_MPa"] = float(np.max(abs(independent[:, 0]-refined[:, 0])))
    report["elapsed_seconds"] = perf_counter()-start
    return report


def transfer_screen(data, calibrated):
    """Freeze the RF fit and predict a distinct SLR specimen without refitting."""
    from hashlib import sha256
    from tools.p92_published_recovery_reference import Parameters

    if data["protocol"] != "SLR" or calibrated.get("schema") != "agentfem.research.p92-bam-cyclic.v1":
        raise ValueError("Provide SLR observations and the frozen RF cyclic screen.")
    if not calibrated["fit"]["success"]:
        raise ValueError("RF calibration did not converge.")
    window = replay_window(data, first_cycle=1, last_cycle=10, max_interval_s=6.)
    h = window["histories"]
    t, e, s = h["strain"].abscissa, h["strain"].values, h["stress_Pa"].values/1e6
    p = Parameters(**calibrated["parameters"])
    initial = np.array([s[0], 0., 0., 0., 0.])
    baseline = replay(t, e, p, initial, subdivisions=2)
    refined = replay(t, e, p, initial, subdivisions=4)
    cycles = data["cycle"][(data["cycle"] >= 1) & (data["cycle"] <= 10)]
    records = []
    for cycle in range(1, 11):
        mask = cycles == cycle
        errors = history_errors(HistoryResult("predicted", t[mask], refined[mask, 0], unit="MPa"),
            HistoryResult("measured", t[mask], s[mask], unit="MPa"),
            reference_source=f"BAM SLR sha256:{data['sha256']}; cycle {cycle}", max_interval=6.)
        records.append({"cycle": cycle, "stress_errors": errors,
                        "refinement_max_stress_change_MPa": float(np.max(abs(baseline[mask, 0]-refined[mask, 0])))})
    return {"schema": "agentfem.research.p92-bam-slr-transfer.v1",
            "source_sha256": data["sha256"], "calibration_source_sha256": calibrated["source_sha256"],
            "calibration_record_sha256": sha256(json.dumps(calibrated, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "parameters": asdict(p), "initial_state": initial.tolist(), "cycles": records,
            "final_state": refined[-1].tolist(), "engineering_qualified": False,
            "scope": "Distinct SLR specimen/protocol prediction with frozen RF parameters; no refitting or life inference",
            "initial_state_assumption": "Measured initial stress and zero hidden history; no cycle reset",
            "temperature_assumption": "Nominal isothermal 620 C; measured excursions not modeled",
            "measured_temperature_range_K": float(np.ptp(h["temperature_K"].values))}


def sensitivity_screen(data, calibrated):
    """Local deterministic parameter sensitivity; no fitting or probability band."""
    from tools.p92_published_recovery_reference import Parameters
    if (data["protocol"] != "RF" or calibrated.get("schema") != "agentfem.research.p92-bam-cyclic.v1"
            or not calibrated["fit"]["success"]):
        raise ValueError("Require RF inputs and a converged frozen RF calibration.")
    window = replay_window(data, first_cycle=1, last_cycle=10, max_interval_s=2.)
    h = window["histories"]
    t, e = h["strain"].abscissa, h["strain"].values
    p = Parameters(**calibrated["parameters"])
    initial = np.asarray(calibrated["initial_state"], dtype=float)
    if data["sha256"] != calibrated["source_sha256"]:
        raise ValueError("Sensitivity inputs must match the frozen calibration source.")
    baseline = replay(t, e, p, initial, subdivisions=2)[:, 0]
    records, columns = [], []
    # One-at-a-time relative perturbations diagnose influence, not uncertainty.
    for name in ("E", "k", "Z", "Q", "b", "Qr", "gamma_r1", "gamma_r2", "a2", "C2"):
        value = getattr(p, name)
        pair = []
        for factor in (.99, 1.01):
            modified = replace(p, **{name: value*factor})
            stress = replay(t, e, modified, initial, subdivisions=2)[:, 0]
            pair.append(stress)
            comparison = history_errors(HistoryResult("perturbed", t, stress, unit="MPa"),
                HistoryResult("baseline", t, baseline, unit="MPa"),
                reference_source=f"Frozen RF prediction; {data['sha256']}", max_interval=2.)
            records.append({"parameter": name, "factor": factor, "response_change": comparison})
        columns.append((pair[1]-pair[0])/.02)
    cycles = data["cycle"][(data["cycle"] >= 1) & (data["cycle"] <= 10)]
    training = cycles <= 3
    dt = np.diff(t[training])
    weights = np.r_[dt[0]/2, (dt[:-1]+dt[1:])/2, dt[-1]/2]
    weighted = np.column_stack(columns)[training]*np.sqrt(weights/weights.sum())[:, None]
    singular = np.linalg.svd(weighted, compute_uv=False)
    return {"schema": "agentfem.research.p92-local-sensitivity.v1",
            "source_sha256": data["sha256"], "parameters": asdict(p), "records": records,
            "training_relative_parameter_sensitivity_singular_values_MPa": singular.tolist(),
            "sensitivity_coordinates": "Dimensionless fractional parameter changes, central +/-1%; cycles 1--3 time weights",
            "identifiability_scope": "Local conditioning only; no observation-noise model, global uniqueness or covariance inference",
            "scope": "Deterministic one-at-a-time sensitivity at the frozen fit; no refit, covariance, or predictive interval",
            "engineering_qualified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--max-nfev", type=int, default=30)
    parser.add_argument("--radau", action="store_true")
    parser.add_argument("--positive-strength", action="store_true",
                        help="Constrain the fit using the scalar isotropic-strength invariant")
    parser.add_argument("--frozen-rf", type=Path,
                        help="Predict SLR using this completed RF JSON record without fitting")
    args = parser.parse_args()
    if args.frozen_rf:
        if args.radau or args.positive_strength:
            parser.error("--radau/--positive-strength apply to RF fitting, not frozen SLR transfer")
        report = transfer_screen(read_source(args.source, protocol="SLR"),
                                 json.loads(args.frozen_rf.read_text()))
    else:
        report = screen(read_source(args.source, protocol="RF"), max_nfev=args.max_nfev,
                        include_radau=args.radau, positive_strength=args.positive_strength)
    print(json.dumps(report, indent=2, allow_nan=False))
