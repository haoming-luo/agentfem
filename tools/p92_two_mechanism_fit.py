"""Joint first-cycle identification; later inspected cycles are development only.

Nominal protocol inputs are explicit approximations to the recorded controller.
Stress/strain scales balance objectives, not experimental standard deviations.
"""

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
from tools.p92_published_recovery_reference import Parameters, isotropic_strength_lower_bound
from tools.p92_two_mechanism import Mechanism, integrate
from tools.p92_two_mechanism_screen import run
from tools.validate_p92_loading_path import nominal_path


def decode(z, base):
    z = np.asarray(z, dtype=float)
    if z.shape != (6,) or not np.isfinite(z).all():
        raise ValueError("Require six finite scaled parameters")
    threshold = base.k-isotropic_strength_lower_bound(base)
    # n=6 fixed. Z is parameterized by a common 120 MPa overstress pivot.
    fast = replace(base, n=6., Z=120./(10.**z[0])**(1/6),
                   k=threshold+10.**z[1],
                   gamma1=base.gamma1*10.**z[2], gamma2=base.gamma2*10.**z[2])
    slow = replace(base, Z=120./(10.**z[3])**(1/base.n),
                   k=threshold+10.**z[4], a2=base.a2*10.**z[5])
    return Mechanism(fast), Mechanism(slow)


def observations(data):
    protocol = data["protocol"]
    if protocol == "SLC":
        times, strains = np.array([0., 2., 182., 184., 1984.]), np.array([0., -.002, -.002, 0., 0.])
    else:
        times, strains = nominal_path(protocol)
    t, s, e = (data[k] for k in ("time_s", "stress_Pa", "strain"))
    first = np.flatnonzero(data["cycle"] == 1)
    if (len(first) < 3 or t[first[0]] != 0. or np.any(np.diff(t[first]) <= 0)
            or np.any(np.diff(t[first]) > (2. if protocol == "RF" else 6.))):
        raise ValueError("Require contiguous first-cycle observations starting at zero")
    segments = []
    for i, (lo, hi) in enumerate(zip(times[:-1], times[1:])):
        ids = np.flatnonzero((data["cycle"] == 1) & (t >= lo) & (t <= hi))
        if len(ids) < 3:
            raise ValueError("Insufficient first-cycle segment observations")
        # Keep actual recorded values, not interpolated experimental targets.
        # Log-spaced times retain early relaxation and late hold response.
        targets = lo+(hi-lo)*np.r_[0., np.geomspace(.001, 1., 15)]
        chosen = np.unique([ids[np.argmin(abs(t[ids]-target))] for target in targets])
        segments.append(dict(duration=hi-lo, rate=(strains[i+1]-strains[i])/(hi-lo),
            stress_control=protocol == "SLC" and i == 3,
            sample_times=t[chosen]-lo, measured_stress=s[chosen]/1e6,
            measured_strain=e[chosen], source_indices=chosen.tolist()))
    return dict(protocol=protocol, initial_stress=float(s[0]/1e6), segments=segments,
                source_sha256=data["sha256"])


def residual_path(path, mechanisms, *, rtol=2e-6):
    state = np.zeros(10)
    state[0] = path["initial_stress"]
    residuals = []
    for seg in path["segments"]:
        sol = integrate(state, seg["duration"], seg["rate"], mechanisms,
                        stress_control=seg["stress_control"], rtol=rtol)
        sample = sol.sol(seg["sample_times"])
        state = sol.y[:, -1].copy()
        channels = [(sample[0]-seg["measured_stress"])/100.]
        if seg["stress_control"]:
            channels.append((sample[9]-seg["measured_strain"])/.002)
        scale = np.sqrt(3*len(path["segments"])*len(channels)*len(sample[0]))
        residuals.extend(channel/scale for channel in channels)
    return np.concatenate(residuals)


def verify_frozen(report):
    """Independent integration check of already frozen predictions, no fit."""
    identity = report["mechanisms"]
    digest = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if digest != report["parameter_sha256"] or not report["fit"]["success"]:
        raise ValueError("Require an intact converged frozen parameter record")
    mechanisms = tuple(Mechanism(Parameters(**m["parameters"]), m["exponent"], m["enabled"])
                       for m in identity)
    records = []
    for coarse in report["frozen_predictions"]:
        if coarse["mechanisms"] != identity:
            raise ValueError("Prediction parameters differ from frozen identity")
        fine = run(mechanisms, initial_stress=coarse["initial_stress_MPa"],
                   protocol=coarse["protocol"], cycles=coarse["requested_cycles"],
                   rtol=1e-10, method="BDF")
        if ([r["cycle"] for r in coarse["cycles"]] != [r["cycle"] for r in fine["cycles"]]
                or (coarse["envelope_failure"] or {}).get("cycle") != (fine["envelope_failure"] or {}).get("cycle")):
            raise RuntimeError("Integration refinement changed the protocol exit")
        differences = [np.abs(np.array(a["end_state"])-b["end_state"])
                       for a, b in zip(coarse["cycles"], fine["cycles"], strict=True)]
        if coarse["envelope_failure"]:
            differences.append(np.abs(np.array(coarse["envelope_failure"]["segment_endpoints"][-1]["state"])
                                      -fine["envelope_failure"]["segment_endpoints"][-1]["state"]))
        delta = np.max(differences, axis=0)
        records.append(dict(protocol=coarse["protocol"], max_endpoint_stress_difference_MPa=float(delta[0]),
                            max_endpoint_strain_difference=float(max(delta[4], delta[8], delta[9])),
                            exit_cycle=(fine["envelope_failure"] or {}).get("cycle")))
    return records


def fit(calibration, sources, *, max_nfev=30, scaled_jacobian=False):
    if type(max_nfev) is not int or max_nfev < 1:
        raise ValueError("Positive evaluation budget required")
    if {d["protocol"] for d in sources} != {"RF", "SLR", "SLC"} or len(sources) != 3:
        raise ValueError("Require exactly RF, SLR, SLC")
    base = Parameters(**calibration["parameters"])
    paths = [observations(d) for d in sources]
    # Freeze this finite search box before looking at later-cycle predictions.
    initial = [-3., -1., -2., -4.5, -1., 0.]
    lower = [-5., -5., -5., -7., -5., -.7]
    upper = [-1., 2., 1., -2., 2., .7]
    started = perf_counter()
    count = 0

    def objective(z):
        nonlocal count
        count += 1
        m = decode(z, base)
        value = np.concatenate([residual_path(path, m, rtol=1e-8 if scaled_jacobian else 2e-6)
                                for path in paths])
        if count % 20 == 1:
            print(json.dumps(dict(evaluation=count, cost=float(.5*value@value))), flush=True)
        return value

    def derivative(z):
        # Absolute steps in dimensionless log coordinates avoid vanishing
        # relative perturbations when a parameter happens to be zero.
        columns = []
        for index in range(6):
            delta = np.zeros(6)
            delta[index] = 1e-3
            columns.append((objective(z+delta)-objective(z-delta))/(2e-3))
        return np.column_stack(columns)

    optimized = least_squares(objective, initial, bounds=(lower, upper), max_nfev=max_nfev,
                              jac=derivative if scaled_jacobian else "2-point",
                              x_scale="jac" if scaled_jacobian else 1.,
                              diff_step=1e-3, ftol=1e-5, xtol=1e-5, gtol=1e-5)
    mechanisms = decode(optimized.x, base)
    identity = [asdict(m) for m in mechanisms]
    report = dict(schema="agentfem.research.p92-dual-joint-fit.v1",
        base_calibration_sha256=sha256(json.dumps(calibration, sort_keys=True,
                                                 allow_nan=False).encode()).hexdigest(),
        mechanisms=identity, parameter_sha256=sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        scaled_parameters=optimized.x.tolist(), bounds=[lower, upper],
        scaled_names=["log10_fast_flow_at_120MPa", "log10_fast_strength_margin_MPa",
                      "log10_fast_static_recovery_scale", "log10_slow_flow_at_120MPa",
                      "log10_slow_strength_margin_MPa", "log10_slow_a2_scale"],
        fit=dict(success=bool(optimized.success), message=optimized.message, nfev=int(optimized.nfev),
                 actual_residual_evaluations=count, cost=float(optimized.cost),
                 active_bounds=optimized.active_mask.tolist(),
                 singular_values=np.linalg.svd(optimized.jac, compute_uv=False).tolist()),
        training_cycles=[1], objective="Equal protocol/segment weights, 100 MPa and .002 strain scales; both stress and strain during SLC hold",
        scaled_jacobian=scaled_jacobian,
        initial_state_assumption="Measured initial stress and virgin hidden states, separately per specimen",
        source_sha256={p["protocol"]: p["source_sha256"] for p in paths},
        training_indices={p["protocol"]: [s["source_indices"] for s in p["segments"]] for p in paths},
        engineering_qualified=False, independent_validation=False, life_qualified=False)
    # Freeze before inspecting later cycles; never refit on these outcomes.
    if optimized.success:
        coarse = objective(optimized.x)
        fine = np.concatenate([residual_path(p, mechanisms, rtol=1e-9) for p in paths])
        report["training_refinement_max_scaled_residual_change"] = float(max(abs(coarse-fine)))
        predictions = []
        for data in sources:
            prediction = run(mechanisms, initial_stress=float(data["stress_Pa"][0]/1e6),
                             protocol=data["protocol"], cycles=10)
            if prediction["cycles"]:
                prediction["comparison"] = compare_candidate(data, prediction)
                # The reused comparator knows the old RF split; replace roles
                # to reflect this experiment's three-protocol cycle-1 training.
                for row in prediction["comparison"]["records"]:
                    row["role"] = "calibration" if row["cycle"] == 1 else "inspected_development"
                prediction["comparison"].pop("groups")
            predictions.append(prediction)
        report["frozen_predictions"] = predictions
        report["frozen_numerical_verification"] = verify_frozen(report)
    report["elapsed_seconds"] = perf_counter()-started
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--max-nfev", type=int, default=30)
    parser.add_argument("--scaled-jacobian", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new evidence path")
    report = fit(json.loads(args.calibration.read_text()),
                 [read_source(args.source_dir/f"Data_{p}.lis", protocol=p) for p in ("RF", "SLR", "SLC")],
                 max_nfev=args.max_nfev, scaled_jacobian=args.scaled_jacobian)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report["fit"]), flush=True)
    raise SystemExit(0 if report["fit"]["success"] else 1)
