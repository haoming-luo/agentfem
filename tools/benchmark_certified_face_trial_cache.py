"""Paired exact-trial reuse screen for the bounded research face Step.

This measures a small serial finite-J2 solve, not production contact speed.
The reference disables only the in-memory certificate reuse; both variants
still prove every distinct displacement/pose and solve the same equations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from agentfem.boundary_models import _certified_face_trial_provider as face  # noqa: E402


def run(
    *, repetitions: int, warmup: int, cells: int, increments: int,
) -> dict[str, object]:
    if repetitions < 1 or warmup < 0 or cells < 1 or increments < 1:
        raise ValueError("Need positive repetitions, cells and increments.")
    original_evaluate = face.DolfinxCertifiedFaceContactTrialProvider.evaluate
    original_geometry = face.evaluate_certified_closed_p1_contact_operator
    measurements: dict[str, list[float]] = {"recompute": [], "reuse": []}
    geometry_calls: dict[str, list[int]] = {"recompute": [], "reuse": []}
    first_solution = None
    first_energy = None
    first_material = None
    first_face_path = None
    first_energy_path = None
    counter = 0

    def without_reuse(provider):
        provider._last_trial = None
        return original_evaluate(provider)

    def count_geometry(*args, **kwargs):
        nonlocal counter
        counter += 1
        return original_geometry(*args, **kwargs)

    try:
        face.evaluate_certified_closed_p1_contact_operator = count_geometry
        for pair_index in range(warmup + repetitions):
            order = (
                ("recompute", "reuse") if pair_index % 2 == 0
                else ("reuse", "recompute")
            )
            for mode in order:
                face.DolfinxCertifiedFaceContactTrialProvider.evaluate = (
                    without_reuse if mode == "recompute" else original_evaluate
                )
                _, step = fixture._elastic_column(
                    cells=cells, increments=increments, penalty=500.0,
                    poisson=0.3, surface_mode="closed",
                )
                counter = 0
                start = time.perf_counter()
                result = step.solve_result()
                elapsed = time.perf_counter() - start
                if result.status != "completed":
                    raise AssertionError("One paired contact solve did not complete.")
                solution = step.solution.x.array.copy()
                energy = result.quantity("wide_closed_tool_contact_energy")
                material = (
                    step.state_transaction.response.state.committed_state_vectors().copy()
                )
                face_path = json.dumps(step.contact_face_path, sort_keys=True)
                energy_path = json.dumps(
                    [asdict(item) for item in step.energy_history], sort_keys=True,
                )
                if first_solution is None:
                    first_solution, first_energy = solution, energy
                    first_material = material
                    first_face_path = face_path
                    first_energy_path = energy_path
                else:
                    np.testing.assert_allclose(
                        solution, first_solution, rtol=1e-8, atol=1e-10,
                    )
                    np.testing.assert_allclose(
                        energy, first_energy, rtol=1e-8, atol=1e-10,
                    )
                    np.testing.assert_allclose(
                        material, first_material, rtol=1e-12, atol=1e-14,
                    )
                    if face_path != first_face_path or energy_path != first_energy_path:
                        raise AssertionError("Contact or work history changed under reuse.")
                if pair_index >= warmup:
                    measurements[mode].append(elapsed)
                    geometry_calls[mode].append(counter)
    finally:
        face.DolfinxCertifiedFaceContactTrialProvider.evaluate = original_evaluate
        face.evaluate_certified_closed_p1_contact_operator = original_geometry
    medians = {key: statistics.median(values) for key, values in measurements.items()}
    return {
        "scope": "single_process_certified_face_research_step",
        "cells_per_edge": cells,
        "increments": increments,
        "paired_repetitions": repetitions,
        "warmup_pairs": warmup,
        "median_seconds": medians,
        "ratio_recompute_over_reuse": medians["recompute"] / medians["reuse"],
        "geometry_proof_calls": geometry_calls,
        "solution_material_contact_and_work_histories_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cells", type=int, default=2)
    parser.add_argument("--increments", type=int, default=3)
    options = parser.parse_args()
    print(json.dumps(run(
        repetitions=options.repetitions, warmup=options.warmup,
        cells=options.cells, increments=options.increments,
    ), indent=2))


if __name__ == "__main__":
    main()
