"""Paired same-model screen of certified smooth-face reuse (research only).

Run from the checkout with the FEniCSx environment. Both variants keep the
same mesh, material, penalty, whole-face proof, Newton options and Gauss-rule
cache. The reference merely disables the *smooth-plane shortcut*, forcing
the existing certified two-feature integration. This is not an Abaqus or
production large-model benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import test_contact_face_science_research as fixture  # noqa: E402
from agentfem.boundary_models import (  # noqa: E402
    contact_feature_pair_geometry_research as discovery,
)


def run(*, repetitions: int, warmup: int) -> dict[str, object]:
    if repetitions < 1 or warmup < 0:
        raise ValueError("Need positive measured repetitions and nonnegative warmup.")
    original = discovery.integrate_certified_smooth_closed_face
    measurements: dict[str, list[float]] = {"two_feature": [], "smooth_face": []}
    first_solution = None
    first_energy = None

    def unavailable(*args, **kwargs):
        raise NotImplementedError("Benchmark reference deliberately skips the fast path.")

    try:
        for pair_index in range(warmup + repetitions):
            order = (
                ("two_feature", "smooth_face")
                if pair_index % 2 == 0 else ("smooth_face", "two_feature")
            )
            for mode in order:
                discovery.integrate_certified_smooth_closed_face = (
                    original if mode == "smooth_face" else unavailable
                )
                _, step = fixture._elastic_column(
                    cells=2, increments=3, penalty=500.0,
                    poisson=0.3, surface_mode="closed",
                )
                start = time.perf_counter()
                result = step.solve_result()
                elapsed = time.perf_counter() - start
                if result.status != "completed":
                    raise AssertionError("One paired contact solve did not complete.")
                solution = step.solution.x.array.copy()
                energy = result.quantity("wide_closed_tool_contact_energy")
                if first_solution is None:
                    first_solution, first_energy = solution, energy
                else:
                    np.testing.assert_allclose(
                        solution, first_solution, rtol=1e-8, atol=1e-10,
                    )
                    np.testing.assert_allclose(
                        energy, first_energy, rtol=1e-8, atol=1e-10,
                    )
                if pair_index >= warmup:
                    measurements[mode].append(elapsed)
    finally:
        discovery.integrate_certified_smooth_closed_face = original
    medians = {name: statistics.median(values) for name, values in measurements.items()}
    return {
        "scope": "single_process_two_cell_three_increment_research_step",
        "paired_repetitions": repetitions,
        "warmup_pairs": warmup,
        "median_seconds": medians,
        "ratio_two_feature_over_smooth": (
            medians["two_feature"] / medians["smooth_face"]
        ),
        "measurements_seconds": measurements,
        "solution_and_contact_energy_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    options = parser.parse_args()
    print(json.dumps(run(repetitions=options.repetitions, warmup=options.warmup), indent=2))


if __name__ == "__main__":
    main()
