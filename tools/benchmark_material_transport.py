# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Alternating full private trials: ordered-point versus columnar transport.

Equations, tangent, geometry admission, fields and rejection rules are unchanged.
The comparison excludes setup, commit, restore and I/O. Do not run concurrently
with other numerical jobs when using its timings as performance evidence.
"""

import argparse
import json
import os
from pathlib import Path
from statistics import median
import subprocess
from time import perf_counter

import numpy as np

from benchmark_finite_hex_trial import make_case


class OrderedProvider:
    """Expose the retained object-batch interface of the identical material."""

    def __init__(self, material):
        self.name = material.name
        self.state_schema = material.state_schema
        self.tangent_convention = material.tangent_convention
        self.stored_energy_component_names = material.stored_energy_component_names
        self.update = material.update
        self.update_batch = material.update_batch
        self.summary = material.summary


def run(size, repeats):
    cases = {}
    for mode in ("ordered", "columnar"):
        u, residual = make_case(size)
        if mode == "ordered":
            residual.material = OrderedProvider(residual.material)
        snapshot = residual.transaction_snapshot()
        u.x.array[:] = (0.02 * u.function_space.tabulate_dof_coordinates()).ravel()
        cases[mode] = (residual, snapshot)
    timings = {mode: [] for mode in cases}
    for repetition in range(repeats + 1):
        order = tuple(cases) if repetition % 2 == 0 else tuple(reversed(cases))
        responses = {}
        for mode in order:
            residual, snapshot = cases[mode]
            residual.update_time(1e-6)
            start = perf_counter()
            vector = residual.assemble_vector()
            elapsed = perf_counter() - start
            if repetition:
                timings[mode].append(elapsed)
            responses[mode] = {
                "force": vector.array.copy(),
                **{
                    name: value.x.array.copy()
                    for name, value in residual._fields().items()
                },
            }
            vector.destroy()
            residual.restore(snapshot)
        for name, values in responses["ordered"].items():
            np.testing.assert_allclose(
                responses["columnar"][name], values, rtol=2e-12, atol=2e-12
            )
    return {
        "cells": size**3,
        "repeats": repeats,
        "scope": "private_serial_finite_hex_trial_no_commit_restore_or_io",
        "identical_material_equations": True,
        "force_and_response_fields_agree": True,
        "seconds": timings,
        "median_seconds": {mode: median(values) for mode, values in timings.items()},
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "numpy_version": np.__version__,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.size <= 24 or not 3 <= args.repeats <= 30:
        parser.error("Use size 1..24 and repeats 3..30")
    if args.output is not None and args.output.exists():
        parser.error("Refusing to overwrite evidence")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    )
    result = run(args.size, args.repeats)
    result.update(revision=revision, tracked_source_dirty=dirty)
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
