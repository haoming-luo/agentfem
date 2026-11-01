# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Full first-cycle diagnostic of the frozen conditional Norton baseline.

This is a research comparator, not a P92 material asset or a new FEM solver.
Measured strain drives the ordinary creep Step; measured stress never resets
the response. A separately integrated scalar ODE checks numerical realization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

try:
    from tools.validate_p92_relaxation import (
        CHECKSUMS, SOURCE, Hold, _parse_lis, comparison, fit_relaxation, read_hold, time_weights,
    )
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from validate_p92_relaxation import (
        CHECKSUMS, SOURCE, Hold, _parse_lis, comparison, fit_relaxation, read_hold, time_weights,
    )


def read_cycle(path, family):
    raw = Path(path).read_bytes()
    if family not in CHECKSUMS or hashlib.md5(raw).hexdigest() != CHECKSUMS[family]:
        raise ValueError("Source checksum differs from reviewed Zenodo v1.")
    data = _parse_lis(raw)
    chosen = data[data[:, 1] == 1]
    if len(chosen) < 10 or chosen[0, 0] != 0. or np.any(np.diff(chosen[:, 0]) <= 0):
        raise ValueError("First cycle must start at zero with strictly increasing samples.")
    return Hold(family, chosen[:, 0], chosen[:, 2], chosen[:, 3], chosen[:, 4],
                hashlib.sha256(raw).hexdigest())


def estimate_modulus(cycle):
    """Predeclared RF early-ramp windows; never optimize against the holdout."""
    if cycle.family != "RF":
        raise ValueError("Only RF supplies the modulus diagnostic.")
    estimates = []
    for lo, hi in ((.15, .60), (.20, .70), (.30, .80)):
        mask = (cycle.time >= lo) & (cycle.time <= hi)
        strain, stress = cycle.strain[mask], cycle.stress[mask]
        if len(strain) < 4 or np.ptp(strain) <= 0:
            raise ValueError("Insufficient early-ramp samples for a slope.")
        slope, intercept = np.polyfit(strain, stress, 1)
        if not np.isfinite(slope) or slope <= 0:
            raise ValueError("Early-ramp stiffness is not positive and finite.")
        estimates.append({"window_s": [lo, hi], "samples": len(strain),
                          "effective_young_MPa": float(slope),
                          "intercept_MPa": float(intercept),
                          "fit_rmse_MPa": float(np.sqrt(np.mean((slope * strain + intercept - stress)**2)))})
    return {"selected_MPa": estimates[0]["effective_young_MPa"],
            "selection": "first predeclared RF window; other windows are sensitivity, not selection",
            "scope": "effective initial ramp slope, not independently measured elastic modulus",
            "intercept_not_applied_as_initial_stress": True, "windows": estimates}


def nominal_path(family):
    # Data in Brief 49 (2023) 109333, Tables 1 and 2. SLC is NOT strain-controlled.
    if family == "RF":
        return np.array([0., 2., 182., 186., 366., 368.]), np.array([0., .002, .002, -.002, -.002, 0.])
    if family == "SLR":
        return (np.array([0., 2., 182., 184., 1984., 1986., 2166., 2168.]),
                np.array([0., -.002, -.002, 0., 0., .002, .002, 0.]))
    raise ValueError("Only RF and SLR have reviewed strain-control paths.")


def validate_path(times, strain, young, fit):
    if any(np.iscomplexobj(value) for value in (times, strain, young, *fit.values())):
        raise ValueError("Path and effective parameters must be real.")
    times, strain = np.asarray(times, dtype=float), np.asarray(strain, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or strain.shape != times.shape
            or not np.isfinite(times).all() or not np.isfinite(strain).all()
            or times[0] != 0. or strain[0] != 0. or np.any(np.diff(times) <= 0)):
        raise ValueError("Path must have aligned finite samples, start at zero, and advance strictly.")
    values = [young, fit["rate_at_200_MPa_per_s"], fit["exponent"]]
    if not np.isfinite(values).all() or young <= 0 or values[1] <= 0 or values[2] < 1:
        raise ValueError("Invalid effective material parameters.")
    return times, strain


def continuous_response(times, strain, young, fit):
    """Independent scalar Radau integration, with a fixed zero initial stress.

    dσ/dt = E*dε/dt - r200*sign(σ)*(abs(σ)/200)**n. Restarting the
    integrator at input knots does not reset stress or accumulated history.
    """
    from scipy.integrate import solve_ivp

    times, strain = validate_path(times, strain, young, fit)
    rate, exponent = fit["rate_at_200_MPa_per_s"], fit["exponent"]
    segments, stress = [], [0.]
    for i, dt in enumerate(np.diff(times)):
        loading_rate = young * (strain[i + 1] - strain[i]) / dt
        def rhs(_t, s, loading_rate=loading_rate):
            return loading_rate - rate * np.sign(s) * (np.abs(s) / 200.) ** exponent
        def jac(_t, s):
            return np.array([[-rate * exponent / 200. * (abs(s[0]) / 200.) ** (exponent - 1)]])
        solution = solve_ivp(rhs, (times[i], times[i + 1]), [stress[-1]], method="Radau",
                             jac=jac, rtol=1.e-9, atol=1.e-9, dense_output=True)
        if not solution.success or not np.isfinite(solution.y).all():
            raise RuntimeError("Independent scalar integration failed.")
        segments.append(solution.sol)
        stress.append(float(solution.y[0, -1]))

    def evaluate(query):
        query = np.asarray(query, dtype=float)
        if not np.isfinite(query).all() or np.any(query < 0) or np.any(query > times[-1]):
            raise ValueError("Comparison cannot extrapolate beyond the input path.")
        flat = query.ravel()
        indices = np.minimum(np.searchsorted(times, flat, side="right") - 1, len(segments) - 1)
        result = np.empty_like(flat)
        for i in np.unique(indices):
            selected = indices == i
            result[selected] = segments[i](flat[selected])[0]
        return result.reshape(query.shape)

    return evaluate


def refined_times(times, refinement):
    if isinstance(refinement, bool) or not isinstance(refinement, int) or refinement < 1:
        raise ValueError("Refinement must be a positive integer.")
    return np.concatenate([np.linspace(a, b, refinement + 1)[1:]
                           for a, b in zip(times[:-1], times[1:], strict=True)])


def fem_path(times, strain, young, fit, *, refinement=1):
    from mpi4py import MPI
    from agentfem import amplitudes, constitutive, fields, mesh, models, results, solvers, steps, studies, units

    times, strain = validate_path(times, strain, young, fit)
    stations = refined_times(times, refinement)
    domain = mesh.cuboid((0., 0., 0.), (1., 1., 1.), (1, 1, 1),
                         cell_type="hexahedron", comm=MPI.COMM_SELF)
    model = models.create(study=studies.creep_solid(), mesh=domain, units=units.n_mm_mpa())
    u = model.field(fields.displacement(domain))
    model.material(constitutive.isotropic_power_law(
        young=young, poisson=.3, density=1., reference_stress=200.,
        coefficient=fit["rate_at_200_MPa_per_s"] / young, stress_exponent=fit["exponent"]))
    for component, axis in enumerate(("x", "y", "z")):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0., tag=component + 1), component=component, value=0.)
    model.fix(u, on=mesh.face(domain, axis="x", value=1., tag=4), component=0, value=1.)
    return model.step(target=u, duration=float(times[-1]),
                      amplitude=amplitudes.tabular(times.tolist(), strain.tolist()),
                      incrementation=steps.at(*(stations / times[-1])), progress=False,
                      material_points=(results.material_point("gauge", cell=0),),
                      solver_options=solvers.newton(relative_tolerance=1.e-10, absolute_tolerance=1.e-9))


def compare_path(cycle, evaluate, end):
    mask = cycle.time <= end
    times, observed = cycle.time[mask], cycle.stress[mask]
    predicted = evaluate(times)
    error = predicted - observed
    knots, _ = nominal_path(cycle.family)
    phases = []
    for a, b in zip(knots[:-1], knots[1:], strict=True):
        selected = (times >= a) & (times <= b)
        if np.count_nonzero(selected) >= 2:
            phases.append({"nominal_window_s": [a, b],
                           "rmse_MPa": float(np.sqrt(np.dot(time_weights(times[selected]), error[selected]**2))),
                           "max_abs_error_MPa": float(np.max(np.abs(error[selected])))})
    return {"samples": len(times), "time_weighted_rmse_MPa": float(np.sqrt(np.dot(time_weights(times), error**2))),
            "maximum_absolute_error_MPa": float(np.max(np.abs(error))),
            "predicted_stress_range_MPa": [float(predicted.min()), float(predicted.max())],
            "observed_stress_range_MPa": [float(observed.min()), float(observed.max())],
            "endpoint_error_MPa": float(error[-1]), "nominal_phase_errors": phases,
            "initial_stress_conditioned_on_measurement": False,
            "initial_stress_MPa": 0., "measured_initial_stress_MPa": float(observed[0]),
            "phase_alignment": "raw physical time; no peak alignment or fitted phase shift"}


def run_cycle(cycle, young, fit, *, fem=False):
    strain = cycle.strain - cycle.strain[0]
    evaluate = continuous_response(cycle.time, strain, young, fit)
    nominal_time, nominal_strain = nominal_path(cycle.family)
    nominal = continuous_response(nominal_time, nominal_strain, young, fit)
    report = {"source": cycle.summary(), "duration_s": float(cycle.time[-1]),
              "measured_path": compare_path(cycle, evaluate, cycle.time[-1]),
              "nominal_path": compare_path(cycle, nominal, nominal_time[-1]),
              "strain_reference_offset_removed": float(cycle.strain[0]),
              "initial_stress_or_internal_state_resets": 0,
              "temperature_assumption_K": 893.15, "temperature_drift_modeled": False}
    if fem:
        runs = []
        for refinement in (1, 2, 4):
            started = perf_counter()
            step = fem_path(cycle.time, strain, young, fit, refinement=refinement)
            result = step.solve_result()
            stress, total = result.histories["gauge.S"], result.histories["gauge.E"]
            errors = stress.values[:, 0, 0] - evaluate(stress.abscissa)
            runs.append({"refinement": refinement, "increments": len(stress.abscissa),
                         "max_scalar_reference_error_MPa": float(np.max(np.abs(errors))),
                         "max_lateral_stress_MPa": float(np.max(np.abs(stress.values[:, 1:, :]))),
                         "max_prescribed_strain_error": float(np.max(np.abs(
                             total.values[:, 0, 0] - np.interp(total.abscissa, cycle.time, strain)))),
                         "result_trust": result.trust_level, "wall_seconds": perf_counter() - started})
        errors = [run["max_scalar_reference_error_MPa"] for run in runs]
        report["fem"] = {"runs": runs, "error_ratios": [a / b for a, b in zip(errors[:-1], errors[1:])],
                         "criterion": "decreasing max scalar error, finest <1 MPa; lateral <1e-6 MPa; strain <1e-10",
                         "passed": bool(errors[0] > errors[1] > errors[2] and errors[2] < 1.
                                        and all(r["max_lateral_stress_MPa"] < 1.e-6
                                                and r["max_prescribed_strain_error"] < 1.e-10 for r in runs))}
    return report


def modulus_sensitivity(cycle, modulus, fit):
    """Hold the relaxation fit fixed; no best-error modulus is selected."""
    values = [w["effective_young_MPa"] for w in modulus["windows"]]
    # 151 GPa is used for mechanism-map normalization in the source article
    # (Sonntag et al., IJF 168, 107381, section 3.3); it is not a fit here.
    values.append(151_000.)
    rows = []
    for young in sorted(set(values)):
        response = continuous_response(cycle.time, cycle.strain - cycle.strain[0], young, fit)
        comparison = compare_path(cycle, response, cycle.time[-1])
        rows.append({"young_MPa": young, "time_weighted_rmse_MPa": comparison["time_weighted_rmse_MPa"],
                     "nominal_phase_errors": comparison["nominal_phase_errors"]})
    return {"scope": "diagnostic sensitivity only; SLR errors do not select parameters",
            "article_reference": "https://doi.org/10.1016/j.ijfatigue.2022.107381",
            "runs": rows}


def low_stress_diagnostic(cycle, fit):
    """Post-hoc discriminator; NOT a reset in the full-path FEM experiment."""
    if cycle.family != "SLR":
        raise ValueError("The reviewed long zero-strain hold belongs to SLR.")
    mask = (cycle.time >= 194.) & (cycle.time <= 1980.)
    if np.count_nonzero(mask) < 10:
        raise ValueError("Insufficient long-hold observations.")
    hold = Hold(cycle.family, cycle.time[mask], cycle.stress[mask], cycle.strain[mask],
                cycle.temperature[mask], cycle.source_sha256)
    return {"role": "post-hoc conditional diagnostic; not an independent validation or new fit",
            "question": "Does the same high-stress relaxation law transfer even with measured low-stress initialization?",
            "window_rule_s": [194., 1980.], "result": comparison(hold, fit)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf", type=Path, required=True)
    parser.add_argument("--slr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fem", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new report path.")
    rf, slr = read_cycle(args.rf, "RF"), read_cycle(args.slr, "SLR")
    fit = fit_relaxation(read_hold(args.rf, "RF"))
    modulus = estimate_modulus(rf)
    import agentfem
    import scipy
    report = {"schema": "agentfem.public-loading-path-diagnostic.v1", "source": SOURCE,
              "license": "CC-BY-4.0", "attribution": "Sonntag et al. (2022), Zenodo 7198219 v1",
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "baseline_tool_sha256": hashlib.sha256(Path(__file__).with_name("validate_p92_relaxation.py").read_bytes()).hexdigest(),
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "agentfem": agentfem.__version__},
              "scope": "first-cycle scalar baseline through ordinary homogeneous small-strain FEM, not specimen/life validation",
              "stress_convention": "experimental nominal stress approximated by infinitesimal axial Cauchy stress",
              "split": "RF early ramp + frozen RF first-hold law; no SLR stress enters parameters or initial conditions",
              "fit": fit, "effective_modulus": modulus,
              "cycles": [run_cycle(cycle, modulus["selected_MPa"], fit, fem=args.fem) for cycle in (rf, slr)],
              "SLR_modulus_sensitivity": modulus_sensitivity(slr, modulus, fit),
              "SLR_low_stress_diagnostic": low_stress_diagnostic(slr, fit),
              "experimental_validation_passed": False,
              "experimental_acceptance": "unassigned; no independent tolerance or validated full-cycle law"}
    report["numerical_verification_passed"] = (
        all(item["fem"]["passed"] for item in report["cycles"]) if args.fem else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.output), "numerical_verification_passed": report["numerical_verification_passed"],
                      "experimental_validation_passed": False}))
    return 1 if report["numerical_verification_passed"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
