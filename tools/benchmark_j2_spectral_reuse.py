# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired discrete-update and whole-Step timings for call-local J2 reuse.

The reference is our pinned pre-reuse analytic tangent, not another material
law. No physical parameter, tolerance, increment or thread count is tuned.
Results apply to these workloads and machine, not the explicit composite case.
Run with PYTHONPATH=src and one BLAS thread while other numerical jobs are idle.
"""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
from time import perf_counter
from unittest.mock import patch

import numpy as np

from agentfem.constitutive.finite_strain_plasticity import FiniteStrainJ2Logarithmic
from agentfem.constitutive.material_array_batch import MaterialPointArrayBatchInput


REFERENCE_REVISION = "bd9b3118"


def reference_tangent(root):
    source = subprocess.check_output([
        "git", "show", f"{REFERENCE_REVISION}:src/agentfem/constitutive/finite_strain_plasticity.py",
    ], cwd=root, text=True)
    owner = next(n for n in ast.parse(source).body
                 if isinstance(n, ast.ClassDef) and n.name == "FiniteStrainJ2Logarithmic")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_analytic_algorithmic_tangent_batch")
    scope = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]),
                 "pinned-agentfem-tangent", "exec"), scope)
    return scope[method.name], sha256(source.encode()).hexdigest()


def paired(operation, *, reference, repeats, rng):
    current = FiniteStrainJ2Logarithmic._analytic_algorithmic_tangent_batch
    methods = {"reference": reference, "reused": current}
    expected = None
    samples, differences = {key: [] for key in methods}, {}
    execution_order = []
    for repeat in range(-1, repeats):
        order = list(methods) if repeat < 0 else rng.permutation(list(methods))
        execution_order.append(dict(repeat=repeat, warmup=repeat < 0, order=list(order)))
        for name in order:
            with patch.object(FiniteStrainJ2Logarithmic,
                              "_analytic_algorithmic_tangent_batch", methods[name]):
                seconds, values = operation()
            if expected is None:
                expected = values
            if values.keys() != expected.keys():
                raise AssertionError("Compared result channels differ.")
            for key, value in values.items():
                np.testing.assert_allclose(value, expected[key], rtol=1e-11, atol=1e-11)
                differences[key] = max(differences.get(key, 0.),
                                       float(np.max(np.abs(value - expected[key]), initial=0.)))
            if repeat >= 0:
                samples[name].append(seconds)
    medians = {k: float(np.median(v)) for k, v in samples.items()}
    return dict(seconds=samples, median_seconds=medians,
                execution_order=execution_order,
                elapsed_reduction_fraction=1-medians["reused"]/medians["reference"],
                maximum_absolute_difference=differences)


def material_update(point_count):
    material = FiniteStrainJ2Logarithmic(young=210_000, poisson=.3,
                                      yield_stress=250, hardening_modulus=1000)
    rng = np.random.default_rng(771)
    gradients = np.eye(3) + rng.uniform(-.04, .04, (point_count, 3, 3))
    states = np.tile(material.state_schema.initial_state(), (point_count, 1))
    states = material._integrate_batch(gradients, states)["state"]
    new = gradients.copy()
    new[:, 0, 1] += .025
    request = MaterialPointArrayBatchInput(
        deformation_gradient_old=gradients, deformation_gradient_new=new,
        state_old=states, state_schema=material.state_schema,
        time=0., time_increment=.1,
    )

    def run():
        start = perf_counter()
        response = material.update_array_batch(request)
        elapsed = perf_counter() - start
        return elapsed, dict(stress=response.cauchy_stress, tangent=response.consistent_tangent,
                             state=response.state_new, energy=response.strain_energy_density,
                             dissipation=response.dissipation_density_increment)
    return run


def whole_step(build, cells):
    def run():
        _, step = build(cells=cells, penalty=5000.)
        start = perf_counter()
        result = step.solve_result()
        elapsed = perf_counter() - start
        if result.status != "completed":
            raise RuntimeError(f"Unexpected result status {result.status}")
        values = dict(u=step.solution.x.array.copy(),
                      state=step.response.state.committed_state_vectors().copy(),
                      force=step.contact_provider.evaluate().contact_force_on_structure.copy(),
                      iterations=np.array([[i.iterations, i.normal_augmentations,
                                            i.line_search_evaluations]
                                           for i in step.accepted_increments]))
        for index, snapshot in enumerate(step.snapshots):
            values[f"factor_{index}"] = np.array(snapshot.load_factor)
            values[f"u_{index}"] = snapshot.solution.x.array.copy()
            for name, value in snapshot.fields.items():
                values[f"field_{index}_{name}"] = getattr(value, "function", value).x.array.copy()
        return elapsed, values
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cells", nargs="+", type=int, default=[2, 4, 6])
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 3 or args.points < 1 or any(c < 1 for c in args.cells):
        parser.error("Require >=3 repeats and positive batch/mesh sizes.")
    if args.output.exists():
        parser.error("Use a fresh output path.")
    root = Path(__file__).resolve().parents[1]
    reference, digest = reference_tangent(root)
    build = runpy.run_path(str(root / "examples/finite_j2_augmented_plane.py"))["make_step"]
    rng = np.random.default_rng(20261019)
    comparisons = {f"material_{args.points}_points": paired(
        material_update(args.points), reference=reference, repeats=args.repeats, rng=rng,
    )}
    for cells in args.cells:
        comparisons[f"whole_step_{cells}_cells_per_edge"] = paired(
            whole_step(build, cells), reference=reference, repeats=args.repeats, rng=rng,
        )
    report = dict(
        schema="agentfem.j2-call-local-spectrum-benchmark.v1",
        scope="specific_material_batch_and_serial_implicit_plane_contact_not_general_speedup",
        reference_revision=REFERENCE_REVISION, reference_source_sha256=digest,
        current_source_sha256=sha256((root / "src/agentfem/constitutive/finite_strain_plasticity.py").read_bytes()).hexdigest(),
        benchmark_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        platform=platform.platform(), numpy_version=np.__version__,
        thread_environment={k: os.environ.get(k) for k in
                            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        excluded="imports_setup_JIT_warmup_checkpoint_export",
        comparisons=comparisons,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: {"medians": v["median_seconds"],
                          "reduction": v["elapsed_reduction_fraction"]}
                      for k, v in comparisons.items()}, indent=2))


if __name__ == "__main__":
    main()
