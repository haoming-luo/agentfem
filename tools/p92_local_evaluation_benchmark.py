"""Bounded A/B research trajectory timing against a fixed repository baseline.

Run with python -m tools.p92_local_evaluation_benchmark in a full source
checkout. This is not an installed application template or an industrial
performance guarantee. The baseline is trusted source from this repository.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import types

import numpy as np
import scipy
import agentfem

from tools import p92_tensor_material_reference as provider
from tools.p92_nonuniform_structure import build, observation

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "8d77c6887cc6d7c0ceab8b3e3fa817df2e95c9e7"


def snapshot(step):
    values = {"U": step.solution.x.array.copy(), "S": step.response.cauchy_stress.values.copy()}
    for name, state in step.state.committed.items():
        values["state." + name] = state.values.copy()
    for history in step.material_point_histories:
        values["history." + history.name] = history.values.copy()
        values["time." + history.name] = history.abscissa.copy()
    return values


def run(*, resolution):
    if type(resolution) is not int or resolution < 1:
        raise ValueError("resolution must be a positive integer")
    if not Path(agentfem.__file__).resolve().is_relative_to(ROOT / "src"):
        raise RuntimeError(
            "This source benchmark requires this checkout's AgentFEM. "
            "Use an editable installation or PYTHONPATH=src."
        )
    checkout = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    source = subprocess.run(
        ["git", "show", BASELINE + ":tools/p92_tensor_normalization.py"], cwd=ROOT,
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout
    baseline = types.ModuleType("p92_verified_repository_baseline")
    exec(compile(source, "<verified-repository-baseline>", "exec"), baseline.__dict__)
    candidate = provider.implicit_tensor_update
    rows, reference = [], None
    warm, _, _ = build(resolution=1)
    warm.solve()
    try:
        for mode in ("baseline", "candidate", "candidate", "baseline", "baseline", "candidate"):
            provider.implicit_tensor_update = (baseline.implicit_tensor_update if mode == "baseline" else candidate)
            start = perf_counter()
            step, left, right = build(resolution=resolution)
            built = perf_counter()
            step.solve(until=4 / 64)
            loaded = observation(step, left, right)
            step.solve()
            held = observation(step, left, right)
            end = perf_counter()
            arrays = snapshot(step)
            if reference is None:
                reference = arrays
            if arrays.keys() != reference.keys():
                raise AssertionError("Snapshot keys changed")
            exact = all(np.array_equal(arrays[k], reference[k]) for k in arrays)
            difference = max(float(np.max(np.abs(arrays[k] - reference[k]), initial=0.)) for k in arrays)
            row = dict(mode=mode, total_seconds=end - start, solve_seconds=end - built,
                       build_seconds=built - start, exact=exact, maximum_absolute_difference=difference,
                       arrays=len(arrays), increments=len(step.accepted_increments),
                       loaded_reaction_N=loaded["tip_reaction_N"], held_reaction_N=held["tip_reaction_N"])
            print(json.dumps(row), flush=True)
            rows.append(row)
            if not exact:
                raise AssertionError("Equivalent execution did not preserve accepted arrays")
    finally:
        provider.implicit_tensor_update = candidate
    return dict(
        framework_checkout_commit=checkout,
        framework_source_mode="verified_checkout",
        baseline_commit=BASELINE, resolution=resolution, cells=4 * resolution**3,
        baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        candidate_source_sha256=hashlib.sha256((ROOT / "tools/p92_tensor_normalization.py").read_bytes()).hexdigest(),
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        machine=platform.machine(),
        timing_scope="build, ramp and hold with observations; after small-mesh warm-up; no visualization files; serial",
        comparison_scope="all displacement, stress, committed states and selected accepted histories; not experimental validation",
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new evidence path")
    report = run(resolution=args.resolution)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
