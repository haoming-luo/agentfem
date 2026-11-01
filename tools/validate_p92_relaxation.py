# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Auditable conditional relaxation experiment, not a P92 material database.

Data: Sonntag et al., Zenodo 7198219 v1, CC BY 4.0. Original files are supplied
explicitly; this tool never downloads data or changes the numerical environment.
RF cycle 1 fits a two-parameter effective Norton relaxation law; SLR cycle 1 is
held out. Only measured initial stress is used from SLR during prediction.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

SOURCE = "https://doi.org/10.5281/zenodo.7198219"
CHECKSUMS = {
    "RF": "bcd4654b4fe014e2c08ec49fd1c3a97b",
    "SLR": "5214e2becfe6c797584e1f158c9a1f14",
}


@dataclass(frozen=True)
class Hold:
    family: str
    time: np.ndarray
    stress: np.ndarray
    strain: np.ndarray
    temperature: np.ndarray
    source_sha256: str

    def summary(self):
        return {
            "family": self.family, "cycle": 1, "source": SOURCE,
            "source_sha256": self.source_sha256, "license": "CC-BY-4.0",
            "samples": len(self.time), "absolute_window_s": self.time[[0, -1]].tolist(),
            "initial_nominal_stress_MPa": float(self.stress[0]),
            "strain_peak_to_peak": float(np.ptp(self.strain)),
            "temperature_range_K": [float(self.temperature.min()), float(self.temperature.max())],
        }


def _parse_lis(raw: bytes):
    text = raw.decode("cp1252")
    if text.count("[Data]") != 1:
        raise ValueError("Expected one complete [Data] section.")
    lines = text.split("[Data]", 1)[1].strip().splitlines()
    if (len(lines) < 3
            or lines[0].split() != ["Time", "Cycle", "Stress", "Strain", "Temperature"]
            or lines[1].split() != ["s", "No", "MPa", "%", "°C"]):
        raise ValueError("Unreviewed BAM columns/units; do not guess conversions.")
    data = np.loadtxt(io.StringIO("\n".join(lines[2:])), ndmin=2)
    if data.shape[1] != 5 or len(data) < 3 or not np.isfinite(data).all():
        raise ValueError("Incomplete or nonfinite BAM records.")
    if np.any(np.diff(data[:, 0]) < 0) or np.any(np.diff(data[:, 1]) < 0):
        raise ValueError("BAM time/cycle order is invalid.")
    if np.any(data[:, 1] < 1) or np.any(data[:, 1] != np.floor(data[:, 1])):
        raise ValueError("Cycle identifiers must be positive integers.")
    # Repeated rounded timestamps elsewhere in the raw file remain intact.
    # Sparse saved cycles are not filled or interpolated across missing cycles.
    data[:, 3] *= .01
    data[:, 4] += 273.15
    return data


def read_hold(path, family):
    raw = Path(path).read_bytes()
    if family not in CHECKSUMS or hashlib.md5(raw).hexdigest() != CHECKSUMS[family]:
        raise ValueError("Source checksum differs from the reviewed Zenodo v1 file.")
    data = _parse_lis(raw)
    # Predefined first hold, after ramp/servo settling and before reversal.
    # Actual sample endpoints (not invented 8/180 s stations) enter the report.
    chosen = data[(data[:, 1] == 1) & (data[:, 0] >= 8.) & (data[:, 0] <= 180.)]
    if len(chosen) < 10 or np.any(np.diff(chosen[:, 0]) <= 0):
        raise ValueError("Selected hold lacks a strictly increasing time history.")
    sign = 1 if family == "RF" else -1
    if np.any(sign * chosen[:, 2] <= 0):
        raise ValueError("Selected hold does not have its declared stress sign.")
    return Hold(family, chosen[:, 0], chosen[:, 2], chosen[:, 3], chosen[:, 4],
                hashlib.sha256(raw).hexdigest())


def time_weights(times):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("At least two finite times are required.")
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("Times must be strictly increasing.")
    weights = np.zeros_like(times)
    weights[:-1] += .5 * dt
    weights[1:] += .5 * dt
    return weights / (times[-1] - times[0])


def relaxation(times, initial, rate_at_200, exponent):
    """Exact signed solution of ds/dt=-r200*sign(s)*(abs(s)/200)**n."""
    times = np.asarray(times, dtype=float)
    parameters = np.asarray([initial, rate_at_200, exponent], dtype=float)
    if (not np.isfinite(times).all() or np.any(times < 0)
            or not np.isfinite(parameters).all() or rate_at_200 <= 0 or exponent < 1):
        raise ValueError("Invalid relaxation domain.")
    if exponent == 1:
        return initial * np.exp(-rate_at_200 / 200. * times)
    factor = rate_at_200 / 200. * (abs(initial) / 200.) ** (exponent - 1)
    return initial * np.exp(-np.log1p((exponent - 1) * factor * times) / (exponent - 1))


def fit_relaxation(hold):
    from scipy.optimize import least_squares

    t = hold.time - hold.time[0]
    weights = time_weights(hold.time)
    # Time weighting avoids over-counting the densely sampled start of a hold.
    def residual(x):
        return np.sqrt(weights) * (relaxation(t, hold.stress[0], np.exp(x[0]), x[1]) - hold.stress)
    fits = [least_squares(residual, [np.log(.5), n],
                          bounds=([np.log(1.e-5), 1.], [np.log(1.e3), 30.]),
                          ftol=1.e-11, xtol=1.e-11, gtol=1.e-9)
            for n in (2., 8., 16.)]
    fit = min(fits, key=lambda item: item.cost)
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError("Calibration did not converge.")
    return {"rate_at_200_MPa_per_s": float(np.exp(fit.x[0])),
            "exponent": float(fit.x[1]), "bound_active": bool(np.any(fit.active_mask)),
            "bounds": {"rate_at_200_MPa_per_s": [1.e-5, 1.e3], "exponent": [1., 30.]},
            "objective": "trapezoidal time-weighted squared stress error",
            "identified_quantity": "E*A; E and A are not separately identified",
            "fit_source": hold.summary(), "multi_start_costs": [float(f.cost) for f in fits]}


def comparison(hold, fit):
    predicted = relaxation(hold.time - hold.time[0], hold.stress[0],
                           fit["rate_at_200_MPa_per_s"], fit["exponent"])
    error = predicted - hold.stress
    return {**hold.summary(),
            "time_weighted_rmse_MPa": float(np.sqrt(np.dot(time_weights(hold.time), error**2))),
            "maximum_absolute_error_MPa": float(np.max(np.abs(error))),
            "endpoint_error_MPa": float(error[-1]),
            "predicted_endpoint_MPa": float(predicted[-1]),
            "measured_endpoint_MPa": float(hold.stress[-1]),
            "initial_stress_conditioned_on_measurement": True,
            "acceptance": "not_assigned: no independent experimental tolerance supplied"}


def fem_step(fit, *, initial, duration, count, young=100_000.):
    """Ordinary FEM realization of the fitted scalar ODE, not a P92 E fit."""
    from mpi4py import MPI
    from agentfem import constitutive, fields, mesh, models, results, solvers, steps, studies, units

    domain = mesh.cuboid((0., 0., 0.), (1., 1., 1.), (1, 1, 1),
                         cell_type="hexahedron", comm=MPI.COMM_SELF)
    model = models.create(study=studies.creep_solid(), mesh=domain, units=units.n_mm_mpa())
    u = model.field(fields.displacement(domain))
    model.material(constitutive.isotropic_power_law(
        young=young, poisson=.3, density=1., reference_stress=200.,
        coefficient=fit["rate_at_200_MPa_per_s"] / young, stress_exponent=fit["exponent"]))
    for component, axis in enumerate(("x", "y", "z")):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0., tag=component + 1),
                  component=component, value=0.)
    model.fix(u, on=mesh.face(domain, axis="x", value=1., tag=4),
              component=0, value=initial / young)
    return model.step(target=u, duration=duration, incrementation=steps.fixed(count),
                      material_points=(results.material_point("gauge", cell=0),),
                      progress=False, solver_options=solvers.newton(
                          relative_tolerance=1.e-10, absolute_tolerance=1.e-9))


def fem_evidence(hold, fit, counts=(32, 64, 128)):
    initial = float(hold.stress[0])
    duration = float(hold.time[-1] - hold.time[0])
    evidence = []
    for count in counts:
        started = perf_counter()
        step = fem_step(fit, initial=initial, duration=duration, count=count)
        result = step.solve_result()
        history = result.histories["gauge.S"]
        expected = relaxation(history.abscissa, initial, fit["rate_at_200_MPa_per_s"], fit["exponent"])
        error = history.values[:, 0, 0] - expected
        evidence.append({"increments": count, "max_abs_time_error_MPa": float(np.max(np.abs(error))),
                         "endpoint_time_error_MPa": float(error[-1]),
                         "lateral_stress_max_MPa": float(np.max(np.abs(history.values[:, 1:, :]))),
                         "result_trust": result.trust_level,
                         "wall_seconds": perf_counter() - started})
    errors = [run["max_abs_time_error_MPa"] for run in evidence]
    ratios = [a / b for a, b in zip(errors[:-1], errors[1:])]
    passed = len(errors) >= 3 and errors[-1] < 1. and all(1.5 < r < 2.5 for r in ratios)
    return {"scope": "homogeneous conditional relaxation; numerical E=100000 MPa, not measured P92 E",
            "no_spatial_convergence_claim": True, "runs": evidence,
            "time_error_ratios": ratios,
            "criterion": "final max time error <1 MPa and consecutive error ratios between 1.5 and 2.5",
            "numerical_verification_passed": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf", type=Path, required=True)
    parser.add_argument("--slr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fem", action="store_true")
    args = parser.parse_args()
    # Never silently overwrite a prior frozen experiment.
    if args.output.exists():
        parser.error("Output already exists; choose a new report path.")
    rf, slr = read_hold(args.rf, "RF"), read_hold(args.slr, "SLR")
    fit = fit_relaxation(rf)
    import scipy
    import agentfem
    report = {
        "schema": "agentfem.public-relaxation-experiment.v1", "source": SOURCE,
        "attribution": "Sonntag, Juergens, Uhlemann, Skrotzki, Olbricht (2022), Zenodo v1, CC BY 4.0",
        "scope": "conditional first-hold effective Norton baseline, not full-cycle material or life validation",
        "split": {"calibration": "RF cycle 1 tensile hold", "held_out": "SLR cycle 1 compressive hold"},
        "window_policy": "first-cycle raw time 8..180 s; no fabricated endpoints or cycle interpolation",
        "stress_measure": "nominal experimental stress approximated by small-strain axial stress",
        "temperature_policy": "nominally isothermal; measured drift reported, not fitted",
        "parameters": fit, "calibration": comparison(rf, fit), "held_out": comparison(slr, fit),
        "experimental_validation_passed": False,
        "implementation": {"tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "agentfem": agentfem.__version__},
    }
    if args.fem:
        report["fem_verification"] = fem_evidence(slr, fit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.output), "calibration": report["calibration"]["time_weighted_rmse_MPa"],
                      "held_out": report["held_out"]["time_weighted_rmse_MPa"]}))
    if args.fem and not report["fem_verification"]["numerical_verification_passed"]:
        raise SystemExit("FEM numerical verification failed; retained report is not experimental validation.")


if __name__ == "__main__":
    main()
