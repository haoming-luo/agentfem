"""Compare opt-in numerical rules, never report them as equivalent algorithms."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from agentfem.constitutive._corotational import polar_strain_increment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=65536)
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.points, args.chunk, args.repeats) < 1 or args.output.exists():
        parser.error("Use positive sizes and a new output path.")
    rng = np.random.default_rng(1182)
    n = args.points

    def rotation():
        q, _ = np.linalg.qr(rng.normal(size=(n, 3, 3)))
        q[:, :, 0] *= np.linalg.det(q)[:, None]
        return q

    axes = rotation()
    r0 = rotation()
    r1 = rotation()
    u0 = (axes * rng.uniform(0.8, 1.2, size=(n, 1, 3))) @ axes.transpose(0, 2, 1)
    delta = rng.normal(size=(n, 3, 3)) * 0.001
    delta = (delta + delta.transpose(0, 2, 1)) / 2
    old = r0 @ u0
    new = r1 @ (u0 + delta)
    results = {}
    timings = {"exact": [], "midpoint": []}

    def run(rule):
        answer = np.empty_like(old)
        start = perf_counter()
        for k in range(0, n, args.chunk):
            answer[k : k + args.chunk] = polar_strain_increment(
                old[k : k + args.chunk], new[k : k + args.chunk], integration_rule=rule
            ).strain_increment
        return perf_counter() - start, answer

    for rule in timings:
        run(rule)
    for repeat in range(args.repeats):
        for rule in ("exact", "midpoint") if repeat % 2 == 0 else ("midpoint", "exact"):
            elapsed, result = run(rule)
            timings[rule].append(elapsed)
            results[rule] = result
    source = (
        Path(__file__).resolve().parents[1]
        / "src/agentfem/constitutive/_corotational.py"
    )
    report = {
        "scope": "moving-point kinematic kernel only; different integration rules",
        "points": n,
        "chunk": args.chunk,
        "repeats": args.repeats,
        "seconds": timings,
        "median_speed_ratio_exact_over_midpoint": float(
            np.median(timings["exact"]) / np.median(timings["midpoint"])
        ),
        "maximum_absolute_strain_difference": float(
            abs(results["exact"] - results["midpoint"]).max()
        ),
        "maximum_exact_strain": float(abs(results["exact"]).max()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "default_changed": False,
        "full_solver_speedup_measured": False,
        "limitations": [
            "no material history or contact",
            "not equal finite increments",
            "requires trajectory/time refinement before adoption",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
