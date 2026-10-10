# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Controlled old-algorithm comparisons; never a universal solver speed claim.

Run when other numerical jobs are idle, with BLAS threads limited to one.
Geometry and boundary timings isolate kernels. The paired trajectory comparison
compares retained general spectral algebra, the old geometry contraction and
the old nine-column tangent algebra against their optimized equivalents. Both
routes retain the constitutive equations, checks, time step and energy ledger.
"""

import argparse
import ast
from contextlib import ExitStack
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from unittest.mock import patch

import basix
import numpy as np

from agentfem.elements._hex_validity import _derivatives, _jacobians
from agentfem.constitutive.finite_strain_plasticity import FiniteStrainJ2Logarithmic
from verify_finite_hex_plastic_path import prepare


REFERENCE_REVISION = "0c0df953"


@lru_cache(maxsize=1)
def reference_tangent():
    """Load only our committed pre-optimization method from this trusted repo.

    This benchmark intentionally has no remote-code download or caller-selected
    module. Pinning the source avoids maintaining a second constitutive law.
    """
    revision = subprocess.check_output(
        ["git", "rev-parse", REFERENCE_REVISION], text=True
    ).strip()
    source = subprocess.check_output(
        [
            "git",
            "show",
            revision + ":src/agentfem/constitutive/finite_strain_plasticity.py",
        ],
        text=True,
    )
    owner = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "FiniteStrainJ2Logarithmic"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_analytic_algorithmic_tangent_batch"
    )
    namespace = {"np": np}
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            "committed-reference-tangent",
            "exec",
        ),
        namespace,
    )
    return namespace[method.name], {
        "revision": revision,
        "source_sha256": sha256(source.encode()).hexdigest(),
    }


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
    from agentfem import constitutive

    rng = np.random.default_rng(983)
    coordinates = np.tile(basix.cell.geometry(basix.CellType.hexahedron), (4096, 1, 1))
    coordinates += rng.uniform(-0.04, 0.04, coordinates.shape)
    jacobian = np.einsum(
        "cai,qaj->cqij",
        coordinates - coordinates.mean(1, keepdims=True),
        _derivatives(),
    )
    centered = coordinates - coordinates.mean(1, keepdims=True)
    contraction = paired(
        lambda: np.einsum("cai,qaj->cqij", centered, _derivatives()),
        lambda: _jacobians(centered),
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

    material = constitutive.finite_strain_j2_logarithmic(
        young=100, poisson=0.3, yield_stress=1, hardening_modulus=5
    )
    gradients = np.eye(3) + rng.uniform(-0.04, 0.04, (4096, 3, 3))
    states = np.tile(material.state_schema.initial_state(), (4096, 1))
    baseline = material._integrate_batch(gradients, states)

    def tangent_equivalence(old, new):
        if np.linalg.norm(new - old) > 2e-12 * np.linalg.norm(old):
            raise AssertionError("Directional tangent batching changed the derivative.")

    tangent = paired(
        lambda: reference_tangent()[0](material, gradients, states, baseline=baseline),
        lambda: material._analytic_algorithmic_tangent_batch(
            gradients, states, baseline=baseline
        ),
        equivalent=tangent_equivalence,
    )

    return {
        "analytic_material_tangent_4096_points": tangent,
        "geometry_4096_cells_jacobian_contraction": contraction,
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
    with ExitStack() as comparison:
        if not optimized:
            comparison.enter_context(
                patch.object(
                    FiniteStrainJ2Logarithmic,
                    "_analytic_algorithmic_tangent_batch",
                    reference_tangent()[0],
                )
            )
            comparison.enter_context(
                patch(
                    "agentfem.elements._hex_validity._jacobians",
                    lambda centered: np.einsum(
                        "cai,qaj->cqij", centered, _derivatives()
                    ),
                )
            )
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
        "scope": "regular_cell_geometry_spectral_and_direction_batch_algebra_same_plastic_trajectory_not_arbitrary_geometry",
        "reference_tangent": reference_tangent()[1],
        "preparation_and_disk_io_included": False,
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
