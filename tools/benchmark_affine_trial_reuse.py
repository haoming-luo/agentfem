# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare the serial affine path with a locally available frozen Git revision.

Research only: imports test fixtures, never modifies installed source, and
temporarily substitutes the historical Python load-path function in memory.
All physical inputs, material and Operator implementations stay identical.
This isolates the Procedure change rather than comparing different models.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter

import numpy as np

from agentfem import _nonlinear_problems, solvers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="a5b5a8fc")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2 or args.output.exists():
        parser.error("Use at least two repetitions and a fresh output path.")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tests"))
    from test_affine_equilibrium_reuse import _step
    from test_affine_j2_automatic_restart import _automatic_periodic_j2_step

    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.baseline + "^{commit}"],
        cwd=root, text=True,
    ).strip()
    source = subprocess.check_output(
        ["git", "show", revision + ":src/agentfem/solvers.py"], cwd=root, text=True,
    )
    definition = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "solve_affine_nonlinear_path")
    namespace = dict(vars(solvers))
    exec(compile(ast.Module(body=[definition], type_ignores=[]),
                 f"{revision}:solvers.py", "exec"), namespace)
    baseline = namespace["solve_affine_nonlinear_path"]
    current = _nonlinear_problems.solve_affine_nonlinear_path
    records, comparisons = [], []
    try:
        for kind in ("sphere", "moving", "friction", "automatic_no_contact"):
            expected = None
            for repeat in range(-1, args.repeats):
                order = ("baseline", "reuse") if repeat % 2 else ("reuse", "baseline")
                for policy in order:
                    _nonlinear_problems.solve_affine_nonlinear_path = (
                        baseline if policy == "baseline" else current
                    )
                    step = (_automatic_periodic_j2_step() if kind == "automatic_no_contact"
                            else _step(kind))
                    started = perf_counter()
                    step.solve()
                    elapsed = perf_counter() - started
                    arrays = [step.solution.x.array.copy(),
                              step.response.state.committed_state_vectors().copy()]
                    for snapshot in step.snapshots:
                        arrays.append(snapshot.solution.x.array.copy())
                        for name in sorted(snapshot.fields):
                            value = snapshot.fields[name]
                            arrays.append(getattr(value, "function", value).x.array.copy())
                    if kind != "automatic_no_contact":
                        arrays.append(step.contact_provider.evaluate()
                                      .assembly.nodal_structural_residual.copy())
                    path = [(item.load_factor, item.iterations, item.accepted_step_lengths,
                             item.linear_converged_reasons)
                            for item in step.accepted_increments]
                    if expected is None:
                        expected = arrays, path
                    if path != expected[1]:
                        raise AssertionError("Accepted Newton paths differ.")
                    differences = []
                    for actual, reference in zip(arrays, expected[0], strict=True):
                        np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-10)
                        differences.append(float(np.max(np.abs(actual-reference), initial=0.)))
                    records.append(dict(
                        case=kind, policy=policy, repeat=repeat, warmup=repeat < 0,
                        solve_seconds=elapsed, newton_iterations=sum(i.iterations for i in step.accepted_increments),
                        accepted_increments=len(step.accepted_increments),
                        attempted_increments=len(step.attempted_increments),
                        maximum_array_difference=max(differences),
                    ))
            medians = {policy: statistics.median(
                row["solve_seconds"] for row in records
                if row["case"] == kind and row["policy"] == policy and not row["warmup"]
            ) for policy in ("baseline", "reuse")}
            comparisons.append(dict(case=kind, median_seconds=medians,
                                    elapsed_reduction=1-medians["reuse"]/medians["baseline"]))
    finally:
        _nonlinear_problems.solve_affine_nonlinear_path = current
    report = dict(
        schema="agentfem.affine-trial-reuse-screen.v1", baseline_revision=revision,
        scope="serial_small_implicit_paths_not_composite_or_universal_speedup",
        baseline_function_sha256=hashlib.sha256(ast.unparse(definition).encode()).hexdigest(),
        candidate_sha256=hashlib.sha256((root/"src/agentfem/_affine_equilibrium.py").read_bytes()).hexdigest(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        platform=platform.platform(), records=records, comparisons=comparisons,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
