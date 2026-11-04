# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""RF-only sinh mechanism screen; SLR is diagnostic, not a blind holdout.

This independent scalar comparator is not a new FEM backend or a P92 material
card. No downloads, experimental acceptance thresholds or state resets are
hidden in the experiment. Optional FEM runs use the ordinary creep Step.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

try:
    from tools.validate_p92_loading_path import (
        compare_path, estimate_modulus, nominal_path, read_cycle, refined_times,
    )
    from tools.validate_p92_relaxation import SOURCE, read_hold, time_weights
except ModuleNotFoundError:
    from validate_p92_loading_path import (
        compare_path, estimate_modulus, nominal_path, read_cycle, refined_times,
    )
    from validate_p92_relaxation import SOURCE, read_hold, time_weights


@dataclass(frozen=True)
class SinhParameters:
    """E times creep rate, normalized at 200 MPa; exponent >= 1."""

    rate_at_200_MPa_per_s: float
    stress_scale_MPa: float
    exponent: float = 1.

    def __post_init__(self):
        values = tuple(asdict(self).values())
        if (any(np.iscomplexobj(v) for v in values) or not np.isfinite(values).all()
                or self.rate_at_200_MPa_per_s <= 0 or self.stress_scale_MPa <= 0
                or self.exponent < 1):
            raise ValueError("Require real finite positive rate/scale and exponent >= 1.")

    def rate_slope(self, stress):
        """Signed stress-relaxation rate and derivative, independently coded."""
        stress = float(stress)
        if not np.isfinite(stress):
            raise ValueError("Stress must be finite.")
        scale, exponent = self.stress_scale_MPa, self.exponent

        def log_sinh(x):
            return x + np.log(-np.expm1(-2. * x)) - np.log(2.)

        normalization = log_sinh(200. / scale)
        if stress == 0.:
            slope = (np.exp(np.log(self.rate_at_200_MPa_per_s) - normalization) / scale
                     if exponent == 1. else 0.)
            return 0., float(slope)
        x = abs(stress) / scale
        log_rate = np.log(self.rate_at_200_MPa_per_s) + exponent * (log_sinh(x) - normalization)
        with np.errstate(over="raise", invalid="raise"):
            rate = float(np.exp(log_rate))
            slope = rate * exponent / (scale * np.tanh(x))
        if not np.isfinite(slope):
            raise FloatingPointError("Unrepresentable sinh derivative.")
        return np.copysign(rate, stress), slope

    def coefficient(self, young):
        if not np.isfinite(young) or young <= 0:
            raise ValueError("Young modulus must be finite and positive.")
        x = 200. / self.stress_scale_MPa
        log_sinh = x + np.log(-np.expm1(-2. * x)) - np.log(2.)
        value = np.exp(np.log(self.rate_at_200_MPa_per_s / young) - self.exponent * log_sinh)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Coefficient is not representable in the core material.")
        return float(value)


def integrate_segment(times, initial, loading_rate, parameters):
    if any(np.iscomplexobj(v) for v in (times, initial, loading_rate)):
        raise ValueError("Integration inputs must be real.")
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or not np.isfinite([initial, loading_rate]).all()):
        raise ValueError("Integration requires increasing finite times and finite stress/loading.")

    def rhs(_t, state):
        return [loading_rate - parameters.rate_slope(state[0])[0]]

    def jac(_t, state):
        return [[-parameters.rate_slope(state[0])[1]]]

    solution = solve_ivp(rhs, (times[0], times[-1]), [initial], method="Radau",
                         jac=jac, rtol=1.e-9, atol=1.e-9, dense_output=True)
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("Independent sinh integration failed.")
    return solution.sol


def continuous_response(times, strain, young, parameters):
    if any(np.iscomplexobj(v) for v in (times, strain, young)):
        raise ValueError("Path must be real.")
    times, strain = np.asarray(times, dtype=float), np.asarray(strain, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or strain.shape != times.shape
            or not np.isfinite(times).all() or not np.isfinite(strain).all()
            or times[0] != 0 or strain[0] != 0 or np.any(np.diff(times) <= 0)
            or not np.isfinite(young) or young <= 0):
        raise ValueError("Path must start at zero, advance strictly and have a positive modulus.")
    segments, stress = [], 0.
    for i, duration in enumerate(np.diff(times)):
        solution = integrate_segment(times[i:i + 2], stress,
                                     young * (strain[i + 1] - strain[i]) / duration, parameters)
        stress = float(solution(times[i + 1])[0])
        segments.append(solution)

    def evaluate(query):
        if np.iscomplexobj(query):
            raise ValueError("Query must be real.")
        query = np.asarray(query, dtype=float)
        if not np.isfinite(query).all() or np.any(query < 0) or np.any(query > times[-1]):
            raise ValueError("Cannot extrapolate beyond the loading path.")
        flat = query.ravel()
        indices = np.minimum(np.searchsorted(times, flat, side="right") - 1, len(segments) - 1)
        values = np.empty_like(flat)
        for i in np.unique(indices):
            selected = indices == i
            values[selected] = segments[i](flat[selected])[0]
        return values.reshape(query.shape)

    return evaluate


def fit_profile(hold, *, exponents=(1., 2., 4., 8.)):
    """Fix an exponent grid before observing SLR; fit only the RF first hold."""
    if hold.family != "RF":
        raise ValueError("Calibration accepts RF only; SLR has a diagnostic role.")
    if (np.asarray(hold.stress).shape != np.asarray(hold.time).shape
            or not np.isfinite(hold.stress).all() or np.any(hold.stress <= 0)):
        raise ValueError("RF calibration requires finite positive aligned stress samples.")
    times = hold.time - hold.time[0]
    weights = np.sqrt(time_weights(times))
    candidates = []
    for exponent in exponents:
        def residual(values, exponent=exponent):
            parameters = SinhParameters(float(np.exp(values[0])), float(np.exp(values[1])), exponent)
            solution = integrate_segment(times, float(hold.stress[0]), 0., parameters)
            return weights * (solution(times)[0] - hold.stress)

        starts = []
        for scale in (20., 80., 320.):
            fit = least_squares(residual, np.log([1.5, scale]),
                                bounds=(np.log([1.e-5, 5.]), np.log([1.e3, 2000.])),
                                ftol=1.e-9, xtol=1.e-9, gtol=1.e-8, max_nfev=200)
            starts.append(fit)
        successful = [f for f in starts if f.success and np.isfinite(f.x).all()]
        if not successful:
            raise RuntimeError(f"No converged RF calibration for exponent {exponent}.")
        best = min(successful, key=lambda f: f.cost)
        parameters = SinhParameters(float(np.exp(best.x[0])), float(np.exp(best.x[1])), exponent)
        candidates.append({"parameters": asdict(parameters),
                           "RF_fit_rmse_MPa": float(np.sqrt(2 * best.cost)),
                           "active_bounds": best.active_mask.tolist(),
                           "multistart": [{"success": bool(f.success), "cost": float(f.cost),
                                           "evaluations": f.nfev} for f in starts]})
    if not candidates:
        raise ValueError("At least one exponent is required.")
    selected = min(range(len(candidates)), key=lambda i: candidates[i]["RF_fit_rmse_MPa"])
    return {"source": hold.summary(), "candidates": candidates, "selected_index": selected,
            "selection": "lowest RF-only time-weighted first-hold error on predeclared exponent grid",
            "bounds": {"rate_MPa_per_s": [1.e-5, 1.e3], "scale_MPa": [5., 2000.]},
            "exponents": list(exponents), "SLR_used_for_calibration": False,
            "initial_stress_conditioned_on_measurement_for_hold_fit_only": True}


def conditional_low_stress(cycle, parameters):
    """Separate post-hoc relaxation test, never a reset in a full path."""
    if cycle.family != "SLR":
        raise ValueError("Long low-stress discriminator requires SLR.")
    mask = (cycle.time >= 194.) & (cycle.time <= 1980.)
    times, observed = cycle.time[mask], cycle.stress[mask]
    if len(times) < 10:
        raise ValueError("Insufficient low-stress hold samples.")
    solution = integrate_segment(times, float(observed[0]), 0., parameters)
    predicted = solution(times)[0]
    return {"role": "post-hoc conditional discriminator; not full-path prediction",
            "initial_stress_conditioned_on_measurement": True,
            "window_s": times[[0, -1]].tolist(), "initial_MPa": float(observed[0]),
            "predicted_endpoint_MPa": float(predicted[-1]), "observed_endpoint_MPa": float(observed[-1]),
            "rmse_MPa": float(np.sqrt(np.dot(time_weights(times), (predicted - observed)**2))),
            "strain_peak_to_peak": float(np.ptp(cycle.strain[mask]))}


def fem_path(times, strain, young, parameters, *, refinement=1):
    from mpi4py import MPI
    from agentfem import amplitudes, constitutive, fields, mesh, models, results, solvers, steps, studies, units

    # Validate through the same scalar input contract without integrating.
    times, strain = np.asarray(times, dtype=float), np.asarray(strain, dtype=float)
    from_fit = {"rate_at_200_MPa_per_s": parameters.rate_at_200_MPa_per_s, "exponent": parameters.exponent}
    try:
        from tools.validate_p92_loading_path import validate_path
    except ModuleNotFoundError:
        from validate_p92_loading_path import validate_path
    validate_path(times, strain, young, from_fit)
    stations = refined_times(times, refinement)
    domain = mesh.cuboid((0., 0., 0.), (1., 1., 1.), (1, 1, 1),
                         cell_type="hexahedron", comm=MPI.COMM_SELF)
    model = models.create(study=studies.creep_solid(), mesh=domain, units=units.n_mm_mpa())
    u = model.field(fields.displacement(domain))
    model.material(constitutive.isotropic_sinh(
        young=young, poisson=.3, density=1., coefficient=parameters.coefficient(young),
        stress_scale=parameters.stress_scale_MPa, exponent=parameters.exponent))
    for component, axis in enumerate(("x", "y", "z")):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0., tag=component + 1), component=component, value=0.)
    model.fix(u, on=mesh.face(domain, axis="x", value=1., tag=4), component=0, value=1.)
    return model.step(target=u, duration=float(times[-1]),
                      amplitude=amplitudes.tabular(times.tolist(), strain.tolist()),
                      incrementation=steps.at(*(stations / times[-1])), progress=False,
                      material_points=(results.material_point("gauge", cell=0),),
                      solver_options=solvers.newton(relative_tolerance=1.e-10, absolute_tolerance=1.e-9))


def screen_cycle(cycle, young, parameters, *, fem=False):
    strain = cycle.strain - cycle.strain[0]
    evaluate = continuous_response(cycle.time, strain, young, parameters)
    times, nominal_strain = nominal_path(cycle.family)
    nominal = continuous_response(times, nominal_strain, young, parameters)
    report = {"source": cycle.summary(), "measured_path": compare_path(cycle, evaluate, cycle.time[-1]),
              "nominal_path": compare_path(cycle, nominal, times[-1]),
              "initial_stress_or_internal_state_resets": 0,
              "strain_reference_offset_removed": float(cycle.strain[0])}
    if fem:
        runs = []
        for refinement in (1, 2, 4):
            started = perf_counter()
            step = fem_path(cycle.time, strain, young, parameters, refinement=refinement)
            try:
                result = step.solve_result()
            except RuntimeError as exc:
                runs.append({"refinement": refinement, "failure": str(exc),
                             "accepted_time_s": step.accepted_time,
                             "last_attempt": (step.attempted_increments[-1].as_dict()
                                              if step.attempted_increments else None),
                             "wall_seconds": perf_counter() - started})
                report["fem"] = {"runs": runs, "passed": False}
                return report
            stress, total = result.histories["gauge.S"], result.histories["gauge.E"]
            runs.append({"refinement": refinement, "increments": len(stress.abscissa),
                         "max_scalar_error_MPa": float(np.max(np.abs(stress.values[:, 0, 0] - evaluate(stress.abscissa)))),
                         "max_lateral_stress_MPa": float(np.max(np.abs(stress.values[:, 1:, :]))),
                         "max_strain_error": float(np.max(np.abs(total.values[:, 0, 0] - np.interp(total.abscissa, cycle.time, strain)))),
                         "wall_seconds": perf_counter() - started})
        errors = [r["max_scalar_error_MPa"] for r in runs]
        report["fem"] = {"runs": runs,
                         "criterion": "decreasing max scalar error, finest <1 MPa; lateral <1e-6 MPa; strain <1e-10",
                         "passed": bool(errors[0] > errors[1] > errors[2] and errors[-1] < 1.
                                        and all(r["max_lateral_stress_MPa"] < 1.e-6 and r["max_strain_error"] < 1.e-10 for r in runs))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf", type=Path, required=True)
    parser.add_argument("--slr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fem", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new evidence path.")
    rf, slr = read_cycle(args.rf, "RF"), read_cycle(args.slr, "SLR")
    modulus = estimate_modulus(rf)
    profile = fit_profile(read_hold(args.rf, "RF"))
    parameters = SinhParameters(**profile["candidates"][profile["selected_index"]]["parameters"])
    print(json.dumps({"stage": "RF calibration complete", "parameters": asdict(parameters)}), flush=True)
    import scipy
    report = {"schema": "agentfem.p92-sinh-screen.v1", "source": SOURCE,
              "attribution": "Sonntag et al. (2022), BAM/Zenodo 7198219 v1, CC BY 4.0",
              "scope": "RF-only candidate screen; SLR already used for diagnosis, not blind validation",
              "temperature_assumption_K": 893.15, "temperature_drift_modeled": False,
              "stress_measure": "nominal experimental stress approximated by small-strain axial Cauchy stress",
              "modulus": modulus, "fit_profile": profile,
              "experimental_validation_passed": False,
              "implementation": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                                 "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 "source_tool_hashes": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                                       for name in ("validate_p92_loading_path.py", "validate_p92_relaxation.py")}},
              "cycles": {}}
    # Retain sensitivity across every predeclared candidate. Do not reselect
    # the RF winner using these already-seen SLR diagnostics.
    for candidate in profile["candidates"]:
        trial = SinhParameters(**candidate["parameters"])
        candidate["SLR_conditional_low_stress"] = conditional_low_stress(slr, trial)
    for cycle in (rf, slr):
        report["cycles"][cycle.family] = screen_cycle(cycle, modulus["selected_MPa"], parameters, fem=args.fem)
        print(json.dumps({"stage": cycle.family, "rmse_MPa": report["cycles"][cycle.family]["measured_path"]["time_weighted_rmse_MPa"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.fem and not all(row["fem"]["passed"] for row in report["cycles"].values()):
        raise SystemExit("FEM numerical gate failed; report retained.")


if __name__ == "__main__":
    main()
