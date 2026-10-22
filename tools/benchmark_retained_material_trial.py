# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded ABBA screen of retained J2 trials in the same AL/MPC trajectory.

This measures an implicit contact path, not explicit-composite performance.
Use PYTHONPATH for source development or run with an installed candidate.
The reference disables only the private retention scope, never physics.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--increments", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path; existing evidence is not overwritten.")
    make = runpy.run_path(str(
        Path(__file__).resolve().parents[1] / "examples/finite_j2_periodic_augmented_plane.py"
    ))["make_step"]
    baseline = None
    rows = []
    for enabled in (False, True, True, False):
        step = make(cells=args.cells, increments=args.increments, penalty=40000.,
                    affine_end_face_support=True,
                    linear_solver=solvers.direct_solver(package="mumps"))
        step.output_every = None
        if not enabled:
            step.state_transaction._trial_response_reuse_scope = nullcontext
        snapshots = []
        started = perf_counter()
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
        elapsed = perf_counter() - started
        if baseline is None:
            baseline = snapshots
        equal = all(np.array_equal(a, b) for group, reference in zip(snapshots, baseline, strict=True)
                    for a, b in zip(group, reference, strict=True))
        row = {
            "reuse_enabled": enabled, "solve_and_observe_seconds": elapsed,
            "material_response_reuses": sum(
                item.checks.get("material_response_reuses", 0) for item in step.accepted_increments
            ),
            "all_station_arrays_equal": equal,
            "accepted_increments": len(step.accepted_increments),
            "newton_iterations": sum(item.iterations for item in step.accepted_increments),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if not equal:
            break
    report = {
        "scope": "serial built-in J2, affine periodic fixed-plane augmentation",
        "package_name": agentfem.__name__, "platform": platform.platform(),
        "python": platform.python_version(),
        "threads": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        )},
        "source_sha256": {name: sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                          for name in (
                              "src/agentfem/mechanics/_retained_trial_response.py",
                              "src/agentfem/mechanics/finite_strain_plasticity.py",
                              "src/agentfem/mechanics/_affine_normal_augmentation.py",
                              "src/agentfem/_affine_equilibrium.py",
                              "examples/finite_j2_periodic_augmented_plane.py",
                          )},
        "cells": args.cells, "increments": args.increments,
        "order": "ABBA", "runs": rows,
        "comparison": "Exact U, committed state, first Piola, tangent, pressures and normal work at four stations",
        "exclusions": "Not explicit performance, MPI, mixed pressure, external material, or continuum validation",
    }
    if len(rows) == 4:
        before = np.mean([row["solve_and_observe_seconds"] for row in rows if not row["reuse_enabled"]])
        after = np.mean([row["solve_and_observe_seconds"] for row in rows if row["reuse_enabled"]])
        report["observed_elapsed_reduction"] = 1. - after / before
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not all(row["all_station_arrays_equal"] for row in rows):
        raise RuntimeError("The retained-response candidate changed the trajectory.")


if __name__ == "__main__":
    main()
