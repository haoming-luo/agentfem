# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded periodic plastic-contact mesh/increment/penalty sensitivity.

Use the public cyclic example and retain scalar observables at common physical
stations. This is a sensitivity study, not an independent material/continuum
reference, a performance comparison, or an automatic parameter tuner.
No full field archives are produced. Failed candidates remain in the report.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
from time import perf_counter

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI

import agentfem
from agentfem import provenance, solvers


def observe(step, factor):
    provider = step.contact_provider
    samples = provider.accepted_point_samples()
    contribution = provider.evaluate()
    u = step.solution
    state = step.response.state.committed_state_vectors()
    weights = samples["reference_weights"]
    pressure = samples["normal_pressures"]
    return {
        "factor": factor,
        "displacement_l2": float(np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(u, u) * ufl.dx)))),
        "force_on_structure": contribution.contact_force_on_structure.tolist(),
        "pressure_area_mean": float(weights @ pressure / weights.sum()),
        "maximum_pressure": float(pressure.max(initial=0.)),
        "active_reference_area": float(weights[pressure > 0.].sum()),
        "maximum_penetration": float(samples["penetrations"].max(initial=0.)),
        # This example has equal-volume affine tetrahedra and the same
        # quadrature rule on every cell. The arithmetic quadrature mean is
        # therefore a reference-volume mean, not a nodal smoothed peak.
        "peeq_reference_volume_mean": float(state[:, 9].mean()),
        "maximum_peeq": float(state[:, 9].max(initial=0.)),
        "normal_relative_work": step.accepted_history_recorders[
            "normal_contact_work:interior_plane"
        ].records[-1]["work"],
        "mpc_equation_mismatch": float(step.constraint.mismatch(factor)),
    }


def differences(coarse, fine):
    rows = []
    for a, b in zip(coarse["stations"], fine["stations"], strict=True):
        assert a["factor"] == b["factor"]
        row = {"factor": a["factor"]}
        for key in ("displacement_l2", "pressure_area_mean", "peeq_reference_volume_mean",
                    "maximum_peeq", "normal_relative_work"):
            delta = abs(a[key] - b[key])
            # Near zero do not conceal the absolute difference behind a
            # manufactured relative percentage.
            row[key] = {
                "absolute_difference": delta,
                "relative_to_fine": delta / abs(b[key]) if abs(b[key]) > 1e-12 else None,
            }
        row["force_absolute_difference"] = float(np.linalg.norm(
            np.asarray(a["force_on_structure"]) - b["force_on_structure"]
        ))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--increments", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--penalties", nargs="+", type=float, default=[10000.])
    parser.add_argument("--yield-stress", type=float, default=2.,
                        help="Physical Pa, not a solver tuning parameter. High values enable a separate elastic-limit diagnostic.")
    parser.add_argument("--affine-end-face-support", action="store_true")
    parser.add_argument("--factor-solver", default=None,
                        help="Explicit installed PETSc factor package (for example mumps); default unchanged.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1:
        parser.error("The bounded affine augmentation route is serial only.")
    if args.output.exists():
        parser.error("Use a new evidence path; prior reports are not overwritten.")
    if not np.isfinite(args.yield_stress) or args.yield_stress <= 0.:
        parser.error("yield-stress must be finite and positive.")
    for name in ("cells", "increments"):
        values = getattr(args, name)
        if values != sorted(set(values)) or any(value < 4 or value % 4 for value in values):
            parser.error(f"{name} must be increasing distinct positive multiples of four.")
    if len(set(args.penalties)) != len(args.penalties) or any(
        not np.isfinite(value) or value <= 0. for value in args.penalties
    ):
        parser.error("Penalties must be distinct finite positive values.")
    root = Path(__file__).resolve().parents[1]
    example = root / "examples/finite_j2_periodic_augmented_plane.py"
    build = runpy.run_path(str(example))["make_step"]
    report = {
        "schema": "agentfem.affine-plane-augmentation-sensitivity.v1",
        "scope": "serial_disjoint_normal_rows_cyclic_finite_J2_not_general_mortar_or_external_reference",
        "agentfem_version": agentfem.__version__,
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "example_sha256": hashlib.sha256(example.read_bytes()).hexdigest(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_identity": provenance.runtime_manifest()["identity"],
        "mpi_transport_environment": {"FI_PROVIDER": os.environ.get("FI_PROVIDER")},
        "material": {"young": 1000., "poisson": .3, "yield_stress": args.yield_stress, "hardening_modulus": 100.},
        "units": {"length": "m", "stress": "Pa", "force": "N", "work": "J"},
        "physical_path": "fixed plane x=1.005; periodic stretch 1/1.02/0.99/1.02 at 0/0.25/0.75/1",
        "fixture": ("affine_x_motion_on_z_end_faces" if args.affine_end_face_support
                    else "point_pinned_periodic_corner_class"),
        "area_interpretation": "active_reference_area counts strictly positive multipliers, including numerical remnants; not a thresholded physical contact-area certificate",
        "numerical_tolerances": {"penetration": 1e-8, "complementarity": 1e-7},
        "linear_solver": solvers.direct_solver(package=args.factor_solver).summary(),
        "records": [], "comparisons": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    for cells in args.cells:
        for increments in args.increments:
            for penalty in args.penalties:
                row = {"cells": cells, "increments": increments, "penalty": penalty,
                       "status": "running", "stations": []}
                started = perf_counter()
                report["records"].append(row)
                save()
                try:
                    step = build(cells=cells, increments=increments, penalty=penalty,
                                 affine_end_face_support=args.affine_end_face_support,
                                 linear_solver=solvers.direct_solver(package=args.factor_solver),
                                 yield_stress=args.yield_stress)
                    step.output_every = None
                    for factor in (.25, .5, .75, 1.):
                        step.solve(until=factor)
                        row["stations"].append(observe(step, factor))
                        row["setup_solve_observe_seconds"] = perf_counter() - started
                        save()
                        print(json.dumps({"cells": cells, "increments": increments,
                                          "penalty": penalty, "accepted_station": factor}), flush=True)
                    row.update(
                        status="completed",
                        accepted_increments=len(step.accepted_increments),
                        outer_updates=sum(i.checks["normal_augmentations"] for i in step.accepted_increments),
                        newton_iterations=sum(i.iterations for i in step.accepted_increments),
                        maximum_penetration_all_stations=max(
                            i.checks["normal_acceptance"]["maximum_penetration"] for i in step.accepted_increments
                        ),
                        maximum_complementarity_all_stations=max(
                            i.checks["normal_acceptance"]["maximum_complementarity"] for i in step.accepted_increments
                        ),
                    )
                except (RuntimeError, ValueError) as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                row["setup_solve_observe_seconds"] = perf_counter() - started
                save()
                print(json.dumps({key: value for key, value in row.items() if key != "stations"}), flush=True)
    completed = [row for row in report["records"] if row["status"] == "completed"]
    for a in completed:
        for b in completed:
            changed = [key for key in ("cells", "increments", "penalty") if a[key] != b[key]]
            if len(changed) == 1 and a[changed[0]] < b[changed[0]]:
                report["comparisons"].append({
                    "varied": changed[0],
                    "coarse": {key: a[key] for key in ("cells", "increments", "penalty")},
                    "fine": {key: b[key] for key in ("cells", "increments", "penalty")},
                    "station_differences": differences(a, b),
                })
    report["all_numerical_acceptance_passed"] = all(
        row["status"] == "completed" and row["maximum_penetration_all_stations"] <= 1e-8
        and row["maximum_complementarity_all_stations"] <= 1e-7
        for row in report["records"]
    )
    report["continuum_or_external_verification"] = False
    save()
    if not report["all_numerical_acceptance_passed"]:
        raise RuntimeError("At least one declared candidate failed; inspect the retained report.")


if __name__ == "__main__":
    main()
