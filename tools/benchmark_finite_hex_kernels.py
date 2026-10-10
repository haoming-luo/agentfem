# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Controlled old-algorithm comparisons; never a universal solver speed claim.

Run when other numerical jobs are idle, with BLAS threads limited to one.
Geometry and boundary timings isolate kernels. The paired trajectory comparison
isolates the isotropic mass-gradient spectral shortcut: both routes retain the
same constitutive model, signed-curvature checks, time step and energy ledger.
"""

import argparse
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import basix
import numpy as np

from agentfem.elements._hex_validity import _derivatives
from verify_finite_hex_plastic_path import prepare


def paired(reference, optimized, *, repetitions=11, equivalent=None):
    old, new = reference(), optimized()
    if equivalent is None:
        np.testing.assert_allclose(old, new, rtol=2e-12, atol=1e-13)
    else:
        equivalent(old, new)
    samples = {"reference": [], "optimized": []}
    for repeat in range(repetitions):
        order = (("reference", reference), ("optimized", optimized))
        for name, operation in order if repeat % 2 == 0 else order[::-1]:
            started = perf_counter()
            operation()
            samples[name].append(perf_counter() - started)
    medians = {name: float(np.median(values)) for name, values in samples.items()}
    return {
        "seconds": samples,
        "median_seconds": medians,
        "median_speedup": medians["reference"] / medians["optimized"],
    }


def kernels():
    rng = np.random.default_rng(983)
    coordinates = np.tile(basix.cell.geometry(basix.CellType.hexahedron), (4096, 1, 1))
    coordinates += rng.uniform(-0.04, 0.04, coordinates.shape)
    jacobian = np.einsum(
        "cai,qaj->cqij",
        coordinates - coordinates.mean(1, keepdims=True),
        _derivatives(),
    )
    geometry = paired(
        lambda: np.linalg.det(jacobian),
        lambda: np.einsum(
            "...i,...i->...",
            jacobian[..., 0],
            np.cross(jacobian[..., 1], jacobian[..., 2]),
        ),
    )
    index = np.arange(0, 274625 * 3, 3, dtype=np.int64)
    values = rng.normal(size=len(index))
    target = np.zeros(274625 * 3)

    def scalar_assignment():
        for dof, value in zip(index, values):
            target[dof] = value
        return target.copy()

    def array_assignment():
        target[index] = values
        return target.copy()

    return {
        "geometry_4096_cells_27_determinants": geometry,
        "prescribed_component_274625_dofs": paired(scalar_assignment, array_assignment),
    }


def trajectory(*, size, optimized):
    step, xyz = prepare(size, 1000)
    if not optimized:
        # Zero flags select the retained general signed-congruence algorithm.
        # No physics, tangent or timestep is replaced.
        cells = step.residual.internal.cells
        cells._isotropic_mass_gradient_bound = np.zeros(len(cells.coordinates))
    started = perf_counter()
    step.run()
    elapsed = perf_counter() - started
    material = step.residual.internal.response
    return elapsed, {
        "u": step.state.u.value.x.array.copy(),
        "v": step.state.v.value.x.array.copy(),
        "stress": material.cauchy_stress.values.copy(),
        "state": material.state.committed_state_vectors().copy(),
        "history": step.history_records,
    }


def full_trajectory(size, pairs):
    # Discard one short warmup from each branch, including all JIT imports.
    trajectory(size=1, optimized=False)
    trajectory(size=1, optimized=True)
    samples = {"reference": [], "optimized": []}
    max_difference = {}
    for pair in range(pairs):
        outputs = {}
        for name in (
            ("reference", "optimized") if pair % 2 == 0 else ("optimized", "reference")
        ):
            elapsed, outputs[name] = trajectory(
                size=size, optimized=name == "optimized"
            )
            samples[name].append(elapsed)
            print(
                json.dumps({"pair": pair + 1, "mode": name, "seconds": elapsed}),
                flush=True,
            )
        for name in ("u", "v", "stress", "state"):
            np.testing.assert_allclose(
                outputs["reference"][name],
                outputs["optimized"][name],
                rtol=1e-12,
                atol=1e-12,
            )
            max_difference[name] = max(
                max_difference.get(name, 0),
                float(
                    np.max(
                        np.abs(outputs["reference"][name] - outputs["optimized"][name])
                    )
                ),
            )
        for old, new in zip(
            outputs["reference"]["history"], outputs["optimized"]["history"]
        ):
            assert old.keys() == new.keys()
            for key in old:
                if isinstance(old[key], (int, float)):
                    np.testing.assert_allclose(
                        old[key], new[key], rtol=1e-12, atol=1e-12
                    )
                else:
                    assert old[key] == new[key]
    medians = {name: float(np.median(values)) for name, values in samples.items()}
    return {
        "cells": size**3,
        "steps": 1000,
        "pairs": pairs,
        "seconds": samples,
        "median_seconds": medians,
        "median_speedup": medians["reference"] / medians["optimized"],
        "maximum_absolute_response_difference": max_difference,
        "scope": "regular_cell_spectral_shortcut_only_same_plastic_trajectory_not_arbitrary_geometry",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.size <= 24 or not 3 <= args.pairs <= 9 or args.output.exists():
        parser.error("Use size 1..24, pairs 3..9, and a new output file.")
    report = {
        "schema": "agentfem.finite-hex-optimization-evidence.v1",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "tracked_worktree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], text=True
            ).strip()
        ),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "kernel_measurements": kernels(),
        "trajectory": full_trajectory(args.size, args.pairs),
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["trajectory"], indent=2))


if __name__ == "__main__":
    main()
