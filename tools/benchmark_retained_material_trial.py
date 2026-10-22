# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded ABBA screen of scoped reuse in the same AL/MPC trajectory.

This measures an implicit contact path, not explicit-composite performance.
Use PYTHONPATH for source development or run with an installed candidate.
The reference disables only the selected private reuse scope, never physics.
The preconditioner mode instead compares explicit direct and managed GMRES
policies at declared tolerances; it does not require bitwise-identical arithmetic.
Combined mode compares fresh material/linear work plus direct solves against
retained material, the shared linear workspace and managed preconditioning.
It measures their interaction; individual percentages must not be added.
"""

import argparse
from contextlib import nullcontext
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import runpy
from time import perf_counter

import numpy as np

import agentfem
from agentfem import solvers
from agentfem.checkpointing import atomic_write_text
from agentfem.mechanics._affine_normal_augmentation import AffinePlaneNormalAugmentation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--increments", type=int, default=16)
    parser.add_argument("--mechanism", choices=("material", "linear", "preconditioner", "combined"), default="material")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path; existing evidence is not overwritten.")
    make = runpy.run_path(str(
        Path(__file__).resolve().parents[1] / "examples/finite_j2_periodic_augmented_plane.py"
    ))["make_step"]
    baseline = None
    rows = []
    # Preserve unsuccessful experiments too. Writes are outside the measured
    # solve interval; no failed run can disappear behind the final average.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_report = {"scope": "serial built-in J2, affine periodic fixed-plane augmentation",
                      "cells": args.cells, "increments": args.increments,
                      "mechanism": args.mechanism, "order": "ABBA", "runs": rows,
                      "tool_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                      "agentfem_version": agentfem.__version__,
                      "status": "running"}
    def save_partial():
        atomic_write_text(args.output, json.dumps(partial_report, indent=2, allow_nan=False) + "\n")
    save_partial()
    for enabled in (False, True, True, False):
        linear = solvers.direct_solver(package="mumps")
        if enabled and args.mechanism in ("preconditioner", "combined"):
            linear = solvers.LinearSolverOptions(
                ksp_type="gmres", pc_type="lu", factor_solver_type="mumps",
                rtol=1e-12, atol=1e-14, max_it=50, preconditioner_reuse="adaptive")
        try:
            step = make(cells=args.cells, increments=args.increments, penalty=40000.,
                        affine_end_face_support=True, linear_solver=linear)
        except Exception as exc:
            rows.append({"reuse_enabled": enabled, "status": "failed_setup",
                         "error_type": type(exc).__name__, "error": str(exc)})
            partial_report["status"] = "failed"
            save_partial()
            raise
        step.output_every = None
        if not enabled and args.mechanism in ("material", "combined"):
            step.state_transaction._trial_response_reuse_scope = nullcontext
        original_scope = AffinePlaneNormalAugmentation._linear_workspace_scope
        if not enabled and args.mechanism in ("linear", "combined"):
            AffinePlaneNormalAugmentation._linear_workspace_scope = lambda *a: nullcontext(None)
        snapshots = []
        started = perf_counter()
        try:
            for station in (.25, .5, .75, 1.):
                step.solve(until=station)
                response = step.response
                snapshots.append([
                    step.solution.x.array.copy(), response.state.committed_state_vectors().copy(),
                    response.first_piola_stress.values.copy(), response.tangent.values.copy(),
                    step.contact_provider.accepted_point_samples()["normal_pressures"].copy(),
                    np.asarray([record["work"] for record in step.accepted_history_recorders[
                        "normal_contact_work:interior_plane"
                    ].records]),
                ])
        except Exception as exc:
            rows.append({"reuse_enabled": enabled, "status": "failed_solve",
                         "completed_stations": len(snapshots),
                         "solve_and_observe_seconds": perf_counter() - started,
                         "error_type": type(exc).__name__, "error": str(exc)})
            partial_report["status"] = "failed"
            save_partial()
            raise
        finally:
            AffinePlaneNormalAugmentation._linear_workspace_scope = original_scope
        elapsed = perf_counter() - started
        if baseline is None:
            baseline = snapshots
        equal = all(np.array_equal(a, b) for group, reference in zip(snapshots, baseline, strict=True)
                    for a, b in zip(group, reference, strict=True))
        differences = [[float(np.max(np.abs(a-b), initial=0.))
                        for a, b in zip(group, reference, strict=True)]
                       for group, reference in zip(snapshots, baseline, strict=True)]
        parity = (all(np.allclose(a, b, rtol=1e-7, atol=1e-9)
                      for group, reference in zip(snapshots, baseline, strict=True)
                      for a, b in zip(group, reference, strict=True))
                  if args.mechanism in ("preconditioner", "combined") else equal)
        row = {
            "status": "completed" if parity else "failed_comparison",
            "reuse_enabled": enabled, "solve_and_observe_seconds": elapsed,
            "material_response_reuses": sum(
                item.checks.get("material_response_reuses", 0) for item in step.accepted_increments
            ),
            "all_station_arrays_equal": equal,
            "accepted_by_declared_comparison": parity,
            "station_max_absolute_differences": differences,
            "accepted_increments": len(step.accepted_increments),
            "newton_iterations": sum(item.iterations for item in step.accepted_increments),
        }
        if args.mechanism in ("preconditioner", "combined"):
            row["linear_policy"] = linear.summary()
            row["preconditioner_reuses"] = sum(
                item.checks.get("linear_preconditioner_reuses", 0) for item in step.accepted_increments)
            row["preconditioner_builds"] = sum(
                item.checks.get("linear_preconditioner_builds", 0) for item in step.accepted_increments)
            row["rebuild_retries"] = sum(
                item.checks.get("linear_preconditioner_rebuild_retries", 0) for item in step.accepted_increments)
            row["maximum_true_linear_residual_ratio"] = max(
                item.checks.get("maximum_true_linear_residual_ratio", 0.)
                for item in step.accepted_increments)
        rows.append(row)
        save_partial()
        print(json.dumps(row), flush=True)
        if not parity:
            break
    report = {
        "scope": "serial built-in J2, affine periodic fixed-plane augmentation",
        "tool_sha256": partial_report["tool_sha256"],
        "agentfem_version": agentfem.__version__,
        "package_name": agentfem.__name__, "platform": platform.platform(),
        "python": platform.python_version(),
        "threads": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        )},
        "source_sha256": {name: sha256((
                              Path(agentfem.__file__).parent / name.removeprefix("src/agentfem/")
                              if name.startswith("src/agentfem/") else
                              Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                          for name in (
                              "src/agentfem/mechanics/_retained_trial_response.py",
                              "src/agentfem/mechanics/finite_strain_plasticity.py",
                              "src/agentfem/constitutive/quadrature.py",
                              "src/agentfem/mechanics/_affine_normal_augmentation.py",
                              "src/agentfem/_affine_equilibrium.py",
                              "src/agentfem/solvers.py",
                              "src/agentfem/_adaptive_linear_correction.py",
                              "examples/finite_j2_periodic_augmented_plane.py",
                          )},
        "cells": args.cells, "increments": args.increments,
        "mechanism": args.mechanism,
        "order": "ABBA", "runs": rows,
        "comparison": "U, committed state, first Piola, tangent, pressures and normal work at four stations",
        "comparison_tolerance": ({"rtol": 1e-7, "atol": 1e-9}
                                 if args.mechanism in ("preconditioner", "combined") else "exact"),
        "exclusions": "Not explicit performance, MPI, mixed pressure, external material, or continuum validation",
        "status": "completed" if len(rows) == 4 and all(row["accepted_by_declared_comparison"] for row in rows) else "failed",
    }
    if report["status"] == "completed":
        before = np.mean([row["solve_and_observe_seconds"] for row in rows if not row["reuse_enabled"]])
        after = np.mean([row["solve_and_observe_seconds"] for row in rows if row["reuse_enabled"]])
        report["observed_elapsed_reduction"] = 1. - after / before
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not all(row["accepted_by_declared_comparison"] for row in rows):
        raise RuntimeError("The reuse candidate exceeded the declared trajectory comparison.")


if __name__ == "__main__":
    main()
