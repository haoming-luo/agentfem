"""Conditional inherited-state audit; never a continuous material prediction."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from tools import p92_recoverable_spectrum as kernel
from tools.p92_bam_source import read_source
from tools.p92_hold_observations import summarize
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_recoverable_frozen import decode_material
from tools.p92_response_screen import POLICY
from tools.p92_two_rate_controls import replay
from tools.p92_waveform_controls import attach_waveforms


def state_geometry(state, material):
    state = np.asarray(state, dtype=float)
    a = np.array([v for branch in material.backbone.branches for v in branch.saturation])
    indices = [1, 2, 4, 5]
    if state.shape != (12,) or not np.isfinite(state).all() or np.any(abs(state[indices]) > a):
        raise ValueError("Require an interior or boundary spectrum state")
    directions = np.zeros((12, 5))
    directions[indices, np.arange(4)] = a
    scale = min(120./np.array(material.moduli))
    directions[10, 4], directions[11, 4] = scale, -scale
    lower = np.r_[np.maximum(-.3, (-a-state[indices])/a), -.3]
    upper = np.r_[np.minimum(.3, (a-state[indices])/a), .3]
    return directions, lower, upper


def hold_metrics(segment, solution):
    hold = summarize(segment, solution.sol, index=segment["segment_index"], strain_index=7)
    f = hold["fields"]["stress_MPa"]
    limit = max(POLICY["stress_absolute_MPa"], POLICY["stress_relative"]*f["observed_max_abs"])
    observed = np.asarray(segment["measured_stress"])
    predicted = solution.sol(np.asarray(segment["sample_times"]))[0]
    residual = list((predicted-observed)/limit/np.sqrt(len(observed)))
    errors = dict(rms=f["rms"]/limit, last=(f["last_predicted"]-f["last_observed"])/limit)
    residual.append(errors["last"])
    for window in ("early", "late"):
        times = np.asarray(hold["sample_times_s"] if window == "early" else hold["late_secant_times_s"])
        if window == "early":
            times = times[times <= min(10., .1*segment["duration"])]
        measured = f.get(f"{window}_secant_observed_per_s")
        if (measured is None or len(times) < 2 or
                abs(measured*(times[-1]-times[0])) < POLICY["minimum_resolution_units"]*POLICY["stress_resolution_MPa"]):
            raise ValueError("Conditional secant diagnostic requires resolved observations")
        value = (f[f"{window}_secant_predicted_per_s"]-measured)/(POLICY["secant_relative"]*abs(measured))
        errors[window] = value
        residual.append(value)
    return np.array(residual), dict(hold=hold, normalized_errors=errors,
        all_hold_gates_passed=bool(max(abs(v) for v in errors.values()) <= 1.))


def bounded_jacobian(response, coordinates, lower, upper, *, step=.0005):
    """Absolute scaled-coordinate differences, including at zero and bounds."""
    u, lo, hi = (np.asarray(v, dtype=float) for v in (coordinates, lower, upper))
    if (u.ndim != 1 or not len(u) or lo.shape != u.shape or hi.shape != u.shape
            or not np.isfinite([*u, *lo, *hi, step]).all() or step <= 0
            or np.any(lo >= hi) or np.any(u < lo) or np.any(u > hi)):
        raise ValueError("Require bounded finite coordinates and positive absolute step")
    columns = []
    for i in range(len(u)):
        a, b = u.copy(), u.copy()
        a[i], b[i] = max(lo[i], u[i]-step), min(hi[i], u[i]+step)
        columns.append((response(b)-response(a))/(b[i]-a[i]))
    return np.column_stack(columns)


def reconstruct(state, material, segment):
    directions, lower, upper = state_geometry(state, material)
    calls = 0

    def response(u, *, method="Radau", rtol=1e-9):
        nonlocal calls
        calls += 1
        if np.any(u < lower) or np.any(u > upper):
            raise ValueError("Conditional state outside declared box")
        initial = state+directions@u
        solution = kernel.integrate(initial, segment["duration"], 0., material, method=method, rtol=rtol)
        residual, report = hold_metrics(segment, solution)
        return residual, report, initial, solution

    initial, report, _, _ = response(np.zeros(5))
    matrices = []
    for step in (.001, .0005):
        columns = []
        for i in range(5):
            low, high = np.zeros(5), np.zeros(5)
            low[i], high[i] = max(lower[i], -step), min(upper[i], step)
            columns.append((response(high)[0]-response(low)[0])/(high[i]-low[i]))
        matrices.append(np.column_stack(columns))
    discrepancy = float(np.linalg.norm(matrices[1]-matrices[0])/max(np.linalg.norm(matrices[1]), 1e-30))
    singular = np.linalg.svd(matrices[1], compute_uv=False)
    result = dict(original_state=state.tolist(), state_directions=directions.tolist(),
        lower=lower.tolist(), upper=upper.tolist(), original=report,
        derivative_relative_discrepancy=discrepancy, singular_values=singular.tolist(),
        state_reconstruction=True, material_parameters_changed=False,
        state_reset_for_prediction=False, continuous_prediction=False,
        engineering_qualified=False, life_qualified=False)
    if discrepancy > .002:
        result.update(fit=dict(success=False, message="Derivative gate failed"), response_evaluations=calls)
        return result
    objective = lambda u: response(u)[0]
    fit = least_squares(objective, np.zeros(5), bounds=(lower, upper),
                        jac=lambda u: bounded_jacobian(objective, u, lower, upper),
                        max_nfev=20, ftol=1e-5, xtol=1e-5, gtol=1e-5)
    residual, candidate, incoming, reference = response(fit.x)
    _, independent, _, check = response(fit.x, method="BDF", rtol=1e-10)
    differences = abs(reference.sol(np.asarray(segment["sample_times"]))-check.sol(np.asarray(segment["sample_times"])))
    numerical = dict(stress_difference_MPa=float(differences[[0, 1, 2, 4, 5]].max()),
        strain_difference=float(differences[[3, 6, 7, 10, 11]].max()),
        work_dissipation_difference_MPa=float(differences[[8, 9]].max()))
    numerical["passed"] = bool(numerical["stress_difference_MPa"] <= 1e-4
        and numerical["strain_difference"] <= 1e-8 and numerical["work_dissipation_difference_MPa"] <= 1e-6)
    result.update(fit=dict(success=bool(fit.success), message=fit.message, nfev=fit.nfev,
        coordinates=fit.x.tolist(), cost=float(residual@residual/2), initial_cost=float(initial@initial/2)),
        reconstructed_state=incoming.tolist(), candidate=candidate, independent=independent,
        numerical=numerical, response_evaluations=calls,
        optimizer_difference=dict(kind="bounded_absolute_coordinate", step=.0005))
    return result


def evaluate(frozen, source_dir):
    module, material = decode_material(frozen["parameters"], frozen["model_id"])
    if module is not kernel:
        raise ValueError("Conditional discriminator requires unchanged original spectrum")
    records = {}
    for protocol in ("SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != frozen["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        path = attach_waveforms(measured_path(source, cycles=1), source)
        _, causal = replay(path, material, kernel=kernel)
        index = next(i for i, s in enumerate(path["segments"]) if i > 3 and s["strain_hold"])
        segment = dict(path["segments"][index], segment_index=index)
        if segment.get("waveform") is not None or segment["stress_control"]:
            raise ValueError("Require the nominal constant-strain post-dwell hold")
        state = np.array(causal["segments"][index-1]["end_state"])
        result = reconstruct(state, material, segment)
        result["causal_prediction"] = causal
        records[protocol] = result
        print(json.dumps(dict(protocol=protocol, fit=result.get("fit"),
                              numerical=result.get("numerical"))), flush=True)
    return dict(schema="agentfem.research.conditional-hold-state.v1", records=records,
        model_id=frozen["model_id"], parameters=frozen["parameters"], source_sha256=frozen["source_sha256"],
        material_parameters_changed=False, state_reset_for_prediction=False,
        engineering_qualified=False, life_qualified=False,
        scope="Bounded counterfactual state reconstruction, not a causal prediction or material calibration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    names = ("p92_hold_state_discriminator.py", "p92_recoverable_spectrum.py", "p92_two_rate_reference.py",
             "p92_hold_observations.py", "p92_response_screen.py", "p92_recoverable_frozen.py",
             "p92_measured_controls.py", "p92_waveform_controls.py", "p92_bam_source.py", "p92_two_rate_controls.py")

    def identities():
        return {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}

    before = identities()
    result = evaluate(json.loads(raw), args.source_dir)
    if before != identities() or raw != args.frozen.read_bytes():
        raise RuntimeError("Conditional diagnostic input changed")
    result.update(driver_sha256=before, frozen_sha256=sha256(raw).hexdigest())
    write_new(args.output, result)
