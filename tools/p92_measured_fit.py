"""Early three-cycle identification, frozen later-cycle development checks."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from tools.p92_bam_source import read_source
from tools.p92_bounded_softening import compare_candidate
from tools.p92_measured_controls import measured_path, replay, timing_sensitivity
from tools.p92_published_recovery_reference import Parameters
from tools.p92_two_mechanism_fit import decode
from tools.p92_two_mechanism import Mechanism
from tools.p92_two_mechanism_screen import run


def seed_parameters(seed, calibration):
    """Only reuse the declared first-cycle fit, with unchanged identities."""
    identity = sha256(json.dumps(seed["mechanisms"], sort_keys=True).encode()).hexdigest()
    base = sha256(json.dumps(calibration, sort_keys=True, allow_nan=False).encode()).hexdigest()
    reconstructed = [asdict(m) for m in decode(seed["scaled_parameters"], Parameters(**calibration["parameters"]))]
    if (seed["training_cycles"] != [1] or not seed["fit"]["success"]
            or seed["parameter_sha256"] != identity or reconstructed != seed["mechanisms"]
            or seed.get("base_calibration_sha256", base) != base):
        raise ValueError("Require intact first-cycle calibration seed, not a later-cycle refit")
    return np.asarray(seed["scaled_parameters"], dtype=float)


def time_softening_coordinates(seed, calibration, *, both=False):
    """Replace weak fast-static direction, retaining six fitted coordinates."""
    base = Parameters(**calibration["parameters"])
    old = np.asarray(seed["scaled_parameters"], dtype=float)
    cards = [asdict(m) for m in decode(old, base)]
    digest = sha256(json.dumps(cards, sort_keys=True).encode()).hexdigest()
    if (seed["training_cycles"] != [1, 2, 3] or not seed["fit"]["success"]
            or cards != seed["parameters"] or digest != seed["parameter_sha256"]):
        raise ValueError("Require unchanged early-cycle seed with original parameter coordinates")
    fixed_fast = float(old[2])
    initial = old.copy()
    initial[2] = 0.  # unit slow time-isotropic scale reproduces the seed exactly
    if both:
        initial[1] = 0.  # replace weak fast strength margin by fast time scale

    def decoder(z):
        z = np.asarray(z, dtype=float)
        if z.shape != (6,) or not np.isfinite(z).all():
            raise ValueError("Require six finite parameters")
        coordinates = z.copy()
        coordinates[2] = fixed_fast
        if both:
            coordinates[1] = old[1]
        fast, slow = decode(coordinates, base)
        if both:
            p = fast.parameters
            fast = replace(fast, parameters=replace(p, gamma_r1=p.gamma_r1*10.**z[1],
                                                   gamma_r2=p.gamma_r2*10.**z[1]))
        p = slow.parameters
        slow = replace(slow, parameters=replace(p, gamma_r1=p.gamma_r1*10.**z[2],
                                               gamma_r2=p.gamma_r2*10.**z[2]))
        return fast, slow

    return initial, decoder, fixed_fast


def coordinate_names(direction):
    names = ["log10_fast_flow_at_120MPa_per_s", "log10_fast_strength_margin_MPa",
             "log10_fast_static_recovery_scale", "log10_slow_flow_at_120MPa_per_s",
             "log10_slow_strength_margin_MPa", "log10_slow_a2_scale"]
    if direction in {"slow_time_isotropic", "both_time_isotropic"}:
        names[2] = "log10_slow_time_isotropic_scale"
    if direction == "both_time_isotropic":
        names[1] = "log10_fast_time_isotropic_scale"
    if direction not in {"fast_static", "slow_time_isotropic", "both_time_isotropic"}:
        raise ValueError("Unknown parameter direction")
    return names


def identify(calibration, seed, sources, *, max_nfev=40, publish_freeze=None,
             parameter_direction="fast_static"):
    if type(max_nfev) is not int or max_nfev < 1:
        raise ValueError("Require positive evaluation budget")
    if len(sources) != 3 or {d["protocol"] for d in sources} != {"RF", "SLR", "SLC"}:
        raise ValueError("Require all three distinct protocols")
    base = Parameters(**calibration["parameters"])
    if parameter_direction in {"slow_time_isotropic", "both_time_isotropic"}:
        if any(seed["source_sha256"][d["protocol"]] != d["sha256"] for d in sources):
            raise ValueError("Time-softening experiment requires unchanged source identities")
        initial, decoder, fixed_fast = time_softening_coordinates(seed, calibration,
            both=parameter_direction == "both_time_isotropic")
    elif parameter_direction == "fast_static":
        initial, decoder, fixed_fast = seed_parameters(seed, calibration), lambda z: decode(z, base), None
    else:
        raise ValueError("Unknown parameter direction")
    paths = [measured_path(d, cycles=3) for d in sources]
    # Admit all observed control paths before spending the identification budget.
    for data in sources:
        measured_path(data, cycles=10)
    started, calls = perf_counter(), 0

    def objective(z, rtol=1e-7):
        nonlocal calls
        calls += 1
        mechanisms = decoder(z)
        value = np.concatenate([replay(p, mechanisms, rtol=rtol, allow_incompatible=True)[0] for p in paths])
        if calls % 20 == 1:
            print(json.dumps(dict(evaluation=calls, cost=float(.5*value@value))), flush=True)
        return value

    def jacobian(z):
        columns = []
        for i in range(6):
            delta = np.eye(6)[i]*.001
            columns.append((objective(z+delta)-objective(z-delta))/.002)
        return np.column_stack(columns)

    lower, upper = [-5., -5., -5., -7., -5., -.7], [-1., 2., 1., -2., 2., .7]
    if parameter_direction == "both_time_isotropic":
        upper[1] = 1.
    result = least_squares(objective, initial, bounds=(lower, upper), jac=jacobian,
                           x_scale="jac", max_nfev=max_nfev, ftol=1e-5, xtol=1e-5, gtol=1e-5)
    mechanisms = decoder(result.x)
    identity = [asdict(m) for m in mechanisms]
    digest = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    report = dict(schema="agentfem.research.measured-controls-fit.v1",
        training_cycles=[1, 2, 3], later_cycles_role="previously inspected development, not blind validation",
        parameters=identity, parameter_sha256=digest, scaled_parameters=result.x.tolist(),
        parameter_direction=parameter_direction, fixed_fast_static_log10=fixed_fast,
        coordinate_names=coordinate_names(parameter_direction),
        fixed_fast_strength_margin_log10=float(seed["scaled_parameters"][1]) if parameter_direction == "both_time_isotropic" else None,
        seed_sha256=sha256(json.dumps(seed, sort_keys=True).encode()).hexdigest(),
        base_sha256=sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest(),
        bounds=[lower, upper], fit=dict(success=bool(result.success), message=result.message,
            cost=float(result.cost), nfev=int(result.nfev), residual_evaluations=calls,
            active_bounds=result.active_mask.tolist(),
            singular_values=np.linalg.svd(result.jac, compute_uv=False).tolist()),
        control_scope="Measured crossing times and source dwell durations; constant predicted stress during SLC dwell; endpoint strain ramps retain all hidden state",
        initial_state_assumption="Measured initial stress/strain, virgin internal states separately for each specimen",
        objective_continuation="Incompatible trial ramps may be continued only to define fixed-size fit residuals; frozen prediction stops before them",
        switch_evidence={p["protocol"]: p["switches"] for p in paths},
        source_sha256={p["protocol"]: p["source_sha256"] for p in paths},
        training_indices={p["protocol"]: [s["source_indices"] for s in p["segments"]] for p in paths},
        engineering_qualified=False, life_qualified=False, independent_validation=False)
    if publish_freeze is not None:
        publish_freeze(report)
    if result.success:
        report["training_refinement_max_scaled_residual_change"] = float(
            np.max(abs(objective(result.x)-objective(result.x, rtol=1e-9))))
        report["early_hold_diagnostics"] = {
            p["protocol"]: replay(p, mechanisms)[1]["hold_diagnostics"] for p in paths}
        report["frozen_predictions"] = frozen_predictions(report, sources)
    report["elapsed_seconds"] = perf_counter()-started
    return report


def frozen_predictions(report, sources):
    identity = report["parameters"]
    digest = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if digest != report["parameter_sha256"] or not report["fit"]["success"]:
        raise ValueError("Require intact converged frozen parameters")
    if len(sources) != 3 or {d["protocol"] for d in sources} != {"RF", "SLR", "SLC"}:
        raise ValueError("Require all three distinct protocols")
    mechanisms = tuple(Mechanism(Parameters(**m["parameters"]), m["exponent"], m["enabled"]) for m in identity)
    predictions = []
    for data in sources:
        if data["sha256"] != report["source_sha256"][data["protocol"]]:
            raise ValueError("Frozen source identity changed")
        path = measured_path(data, cycles=10)
        _, prediction = replay(path, mechanisms)
        _, fine = replay(path, mechanisms, rtol=1e-10, method="BDF")
        prediction["parameter_sha256"] = digest
        prediction["switch_evidence"] = path["switches"]
        prediction["switch_bracket_sensitivity"] = timing_sensitivity(data, mechanisms)
        prediction["comparison"] = compare_candidate(data, prediction)
        for row in prediction["comparison"]["records"]:
            row["role"] = "calibration" if row["cycle"] <= 3 else "inspected_development"
        prediction["comparison"].pop("groups")
        if ([x["cycle"] for x in prediction["reversed_post_hold_ramps"]] !=
                [x["cycle"] for x in fine["reversed_post_hold_ramps"]]):
            raise ValueError("Independent integrators disagree on control failure cycle")
        delta = np.abs(np.asarray([x["state"] for x in prediction["segment_endpoints"]])-
                       np.asarray([x["state"] for x in fine["segment_endpoints"]]))
        prediction["independent_integrator_max_state_difference"] = delta.max(axis=0).tolist()
        extrema = np.asarray([[a[k]-b[k] for k in ("sampled_peak_MPa", "sampled_valley_MPa")]
                              for a, b in zip(prediction["cycles"], fine["cycles"], strict=True)])
        prediction["independent_integrator_max_extrema_difference_MPa"] = float(np.max(abs(extrema)))
        prediction["numerical_refinement_passed"] = bool(
            np.max(abs(extrema)) < 1e-4 and delta[:, [0, 1, 2, 3, 5, 6, 7]].max() < 1e-4
            and delta[:, [4, 8, 9]].max() < 1e-8)
        nominal = run(mechanisms, initial_stress=path["initial_stress"], protocol=path["protocol"], cycles=10)
        prediction["autonomous_nominal_check"] = dict(completed_cycles=len(nominal["cycles"]),
            envelope_failure=nominal["envelope_failure"], cycles=nominal["cycles"])
        predictions.append(prediction)
    return predictions


def write_new(path, report):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("seed", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, default=40)
    parser.add_argument("--parameter-direction", choices=("fast_static", "slow_time_isotropic", "both_time_isotropic"), default="fast_static")
    parser.add_argument("--verify-frozen", type=Path, help="Recheck existing parameters without identification")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".frozen.json").exists():
        parser.error("Choose a new evidence path")
    sources = [read_source(args.source_dir/f"Data_{p}.lis", protocol=p) for p in ("RF", "SLR", "SLC")]
    if args.verify_frozen:
        report = json.loads(args.verify_frozen.read_text())
        started = perf_counter()
        report["frozen_predictions"] = frozen_predictions(report, sources)
        report["verification_seconds"] = perf_counter()-started
        report["prior_report_sha256"] = sha256(args.verify_frozen.read_bytes()).hexdigest()
    else:
        report = identify(json.loads(args.calibration.read_text()), json.loads(args.seed.read_text()),
            sources, max_nfev=args.max_nfev, parameter_direction=args.parameter_direction,
            publish_freeze=lambda report: write_new(args.output.with_suffix(".frozen.json"), report))
    report["verification_driver_sha256"] = {
        name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("p92_measured_fit.py", "p92_measured_controls.py", "p92_two_mechanism.py",
                     "p92_two_mechanism_fit.py", "p92_ratcheting_screen.py",
                     "p92_published_recovery_reference.py", "p92_bounded_softening.py")}
    report["coordinate_names"] = coordinate_names(report.get("parameter_direction", "fast_static"))
    write_new(args.output, report)
    print(json.dumps(report["fit"]), flush=True)
    raise SystemExit(0 if report["fit"]["success"] else 1)
