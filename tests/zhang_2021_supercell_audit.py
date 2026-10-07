"""Content-bound geometric supercell audit for the Zhang periodic cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from zhang_2021_plane_strain_promotion import (
    _benchmark_identity,
    _scientific_identity,
    load_candidate,
)


SCHEMA = "agentfem.zhang-2021-geometric-supercell-audit.v1"
STRESS_TOLERANCE = 1.0e-2
ENERGY_TOLERANCE = 5.0e-2
MICRO_MACRO_TOLERANCE = 1.0e-8
PERIODIC_TOLERANCE = 1.0e-10
EXPECTED_REPETITIONS = ((1, 1), (1, 2), (2, 1), (2, 2))


def _relative(reference, candidate) -> float:
    first = np.asarray(reference, dtype=float)
    second = np.asarray(candidate, dtype=float)
    return float(
        np.linalg.norm(first - second) / max(float(np.linalg.norm(first)), 1.0e-30)
    )


def assess(runs) -> dict[str, object]:
    by_repetitions = {
        tuple(int(value) for value in run["candidate"]["cell_repetitions"]): run
        for run in runs
    }
    complete_family = len(runs) == len(EXPECTED_REPETITIONS) and set(
        by_repetitions
    ) == set(EXPECTED_REPETITIONS)
    baseline = by_repetitions.get((1, 1))
    common_implementation = bool(
        runs
        and len({_scientific_identity(run["payload"]) for run in runs}) == 1
        and len({_benchmark_identity(run["payload"]) for run in runs}) == 1
        and len(
            {
                json.dumps(
                    {
                        key: run["candidate"].get(key)
                        for key in (
                            "formulation",
                            "mesh_size",
                            "quadrature_degree",
                            "macroscopic_deformation_gradient",
                        )
                    },
                    sort_keys=True,
                )
                for run in runs
            }
        )
        == 1
        and len(
            {
                json.dumps(
                    [
                        step["load_factor"]
                        for step in run["payload"]["accepted_increment_path"]
                    ]
                )
                for run in runs
            }
        )
        == 1
    )
    comparisons = {}
    if baseline is not None:
        base_payload = baseline["payload"]
        base_stress = base_payload["first_piola"]
        base_energy = base_payload["mixed_elastic_energy_diagnostics"][
            "primal_elastic_energy_density"
        ]
        for repetitions in EXPECTED_REPETITIONS:
            run = by_repetitions.get(repetitions)
            if run is None:
                continue
            payload = run["payload"]
            stress_error = _relative(base_stress, payload["first_piola"])
            energy_error = _relative(
                base_energy,
                payload["mixed_elastic_energy_diagnostics"][
                    "primal_elastic_energy_density"
                ],
            )
            hill_mandel = float(payload["maximum_hill_mandel_relative_error"])
            periodic = float(payload["periodic_equation_mismatch"])
            passed = bool(
                stress_error <= STRESS_TOLERANCE
                and energy_error <= ENERGY_TOLERANCE
                and hill_mandel <= MICRO_MACRO_TOLERANCE
                and periodic <= PERIODIC_TOLERANCE
            )
            comparisons[f"{repetitions[0]}x{repetitions[1]}"] = {
                "passed": passed,
                "global_cells": int(payload["global_cells"]),
                "reference_cell_area": float(payload["reference_cell_area"]),
                "first_piola_relative_l2_change": stress_error,
                "primal_elastic_energy_relative_change": energy_error,
                "maximum_hill_mandel_relative_error": hill_mandel,
                "periodic_equation_mismatch": periodic,
                "artifact": {"path": run["path"], "sha256": run["sha256"]},
            }
    passed = bool(
        complete_family
        and common_implementation
        and len(comparisons) == len(EXPECTED_REPETITIONS)
        and all(item["passed"] for item in comparisons.values())
    )
    return {
        "schema": SCHEMA,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "periodic_cell_size_invariant": passed,
        "content_bound": True,
        "benchmark_promotion_authorized": False,
        "complete_family": complete_family,
        "common_implementation_and_controls": common_implementation,
        "tolerances": {
            "first_piola_relative_l2": STRESS_TOLERANCE,
            "primal_elastic_energy_relative": ENERGY_TOLERANCE,
            "hill_mandel_relative": MICRO_MACRO_TOLERANCE,
            "periodic_equation_mismatch": PERIODIC_TOLERANCE,
            "authority": "AgentFEM controlled diagnostic; not published",
        },
        "comparisons": comparisons,
        "decision_scope": (
            "1x1, 1x2, 2x1, and 2x2 exact-geometry supercells independently "
            "remeshed at one common target size; this is a geometric cell-size "
            "invariance check, not an exact discrete-topology tiling proof"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs=4, type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    evidence = assess(tuple(load_candidate(path) for path in arguments.candidates))
    payload = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not evidence["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
