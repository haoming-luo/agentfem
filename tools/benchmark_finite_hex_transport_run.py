# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Alternating private full trajectories with identical constitutive equations.

Includes integration, geometry admission, spectral checks, transactions and
scheduled monitoring. Excludes preparation and disk I/O. Run without concurrent
numerical workloads before interpreting timings. Not an industrial benchmark.
"""

import argparse
import json
import os
from pathlib import Path
from statistics import median
import subprocess
from time import perf_counter

import numpy as np

from benchmark_material_transport import OrderedProvider, runtime_metadata
from verify_finite_hex_patch import prepare


def run(size, steps, repeats):
    timings = {"ordered": [], "columnar": []}
    errors = []
    for repetition in range(repeats + 1):
        order = tuple(timings) if repetition % 2 == 0 else tuple(reversed(timings))
        answers = {}
        for mode in order:
            step, history, internal, coordinates, rate = prepare(
                size,
                steps,
                material_adapter=OrderedProvider if mode == "ordered" else None,
            )
            start = perf_counter()
            step.run()
            elapsed = perf_counter() - start
            if repetition:
                timings[mode].append(elapsed)
            answers[mode] = {
                **{
                    name: getattr(history, name).value.x.array.copy()
                    for name in ("u", "v", "a")
                },
                **{
                    name: value.x.array.copy()
                    for name, value in step.residual._fields().items()
                },
            }
            t = steps * step.dt
            stretch = 1 + rate * t
            bulk = 100 / (3 * (1 - 2 * 0.3))
            expected_stress = np.eye(3) * 3 * bulk * np.log(stretch) / stretch**3
            error = {
                "displacement": float(
                    np.max(
                        np.abs(
                            history.u.value.x.array.reshape(-1, 3)
                            - rate * t * coordinates
                        )
                    )
                ),
                "stress": float(
                    np.max(
                        np.abs(internal.response.cauchy_stress.values - expected_stress)
                    )
                ),
                "energy": float(
                    abs(
                        internal.cells.volume
                        @ internal.response.strain_energy_density.values.reshape(-1)
                        - 4.5 * bulk * np.log(stretch) ** 2
                    )
                ),
            }
            if (
                error["displacement"] > 1e-9
                or error["stress"] > 1e-6
                or error["energy"] > 1e-7
            ):
                raise AssertionError(f"Affine reference mismatch: {error}")
            errors.append(error)
            print(
                json.dumps(
                    {
                        "repetition": repetition,
                        "warmup": repetition == 0,
                        "mode": mode,
                        "seconds": elapsed,
                        "errors": error,
                    }
                ),
                flush=True,
            )
            del step, history, internal
        for name, values in answers["ordered"].items():
            np.testing.assert_allclose(
                answers["columnar"][name], values, rtol=2e-12, atol=2e-12
            )
    return {
        "schema": "agentfem.private-finite-hex-transport-run.v1",
        "cells": size**3,
        "steps": steps,
        "repeats": repeats,
        "scope": "private_serial_affine_trajectory_including_transactions_monitoring_no_setup_io",
        "identical_material_equations": True,
        "nodal_and_material_fields_agree": True,
        "seconds": timings,
        "median_seconds": {mode: median(values) for mode, values in timings.items()},
        "maximum_absolute_reference_error": {
            name: max(error[name] for error in errors) for name in errors[0]
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "numpy_version": np.__version__,
        "runtime": runtime_metadata(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        not 1 <= args.size <= 24
        or not 200 <= args.steps <= 1000
        or not 2 <= args.repeats <= 7
    ):
        parser.error("Use size 1..24, steps 200..1000 and repeats 2..7")
    if args.output is not None and args.output.exists():
        parser.error("Refusing to overwrite evidence")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    )
    result = run(args.size, args.steps, args.repeats)
    result.update(revision=revision, tracked_source_dirty=dirty)
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
