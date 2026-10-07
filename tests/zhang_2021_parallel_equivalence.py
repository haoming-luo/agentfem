"""Content-bound serial/MPI equivalence audit for the Zhang candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from zhang_2021_plane_strain_promotion import (
    _benchmark_identity,
    _discretization_identity,
    _scientific_identity,
    load_candidate,
)


SCHEMA = "agentfem.zhang-2021-serial-mpi-equivalence.v1"


def _relative(reference, candidate) -> tuple[float, float]:
    first = np.asarray(reference, dtype=float)
    second = np.asarray(candidate, dtype=float)
    maximum_absolute = float(np.max(np.abs(first - second), initial=0.0))
    scale = max(float(np.linalg.norm(first)), float(np.linalg.norm(second)), 1.0)
    return maximum_absolute, float(np.linalg.norm(first - second) / scale)


def assess(serial, parallel, *, tolerance: float = 1.0e-10) -> dict[str, object]:
    first = serial["payload"]
    second = parallel["payload"]
    serial_ranks = int(serial["candidate"]["mpi_ranks"])
    parallel_ranks = int(parallel["candidate"]["mpi_ranks"])
    invariant_parameters = (
        "formulation",
        "mesh_size",
        "global_cells",
        "discretization_fingerprint",
        "quadrature_degree",
        "macroscopic_deformation_gradient",
        "cell_repetitions",
        "reference_cell_area",
    )
    identity_checks = {
        "rank_scope": serial_ranks == 1 and parallel_ranks > 1,
        "scientific_runtime": _scientific_identity(first)
        == _scientific_identity(second),
        "benchmark_implementation": _benchmark_identity(first)
        == _benchmark_identity(second),
        "discretization": _discretization_identity(first)
        == _discretization_identity(second),
        "candidate_parameters": all(
            serial["candidate"].get(name) == parallel["candidate"].get(name)
            for name in invariant_parameters
        ),
        "accepted_increment_path": [
            item["load_factor"] for item in first["accepted_increment_path"]
        ]
        == [item["load_factor"] for item in second["accepted_increment_path"]],
    }
    observables = {
        "first_piola": (first["first_piola"], second["first_piola"]),
        "primal_elastic_energy_density": (
            first["mixed_elastic_energy_diagnostics"]["primal_elastic_energy_density"],
            second["mixed_elastic_energy_diagnostics"]["primal_elastic_energy_density"],
        ),
        "periodic_equation_mismatch": (
            first["periodic_equation_mismatch"],
            second["periodic_equation_mismatch"],
        ),
        "maximum_hill_mandel_relative_error": (
            first["maximum_hill_mandel_relative_error"],
            second["maximum_hill_mandel_relative_error"],
        ),
        "pressure_block_residual_norm": (
            first["pressure_block_residual_norm"],
            second["pressure_block_residual_norm"],
        ),
        "maximum_quadrature_pressure_constraint_defect": (
            first["maximum_quadrature_pressure_constraint_defect"],
            second["maximum_quadrature_pressure_constraint_defect"],
        ),
    }
    numerical = {}
    for name, (reference, candidate) in observables.items():
        maximum_absolute, relative = _relative(reference, candidate)
        numerical[name] = {
            "maximum_absolute": maximum_absolute,
            "relative_l2": relative,
            "passed": maximum_absolute <= tolerance and relative <= tolerance,
        }
    passed = all(identity_checks.values()) and all(
        item["passed"] for item in numerical.values()
    )
    return {
        "schema": SCHEMA,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "serial_mpi_equivalent": passed,
        "content_bound": True,
        "benchmark_promotion_authorized": False,
        "tolerance": float(tolerance),
        "serial_ranks": serial_ranks,
        "parallel_ranks": parallel_ranks,
        "identity_checks": identity_checks,
        "numerical_checks": numerical,
        "artifacts": {
            "serial": {"path": serial["path"], "sha256": serial["sha256"]},
            "parallel": {
                "path": parallel["path"],
                "sha256": parallel["sha256"],
            },
        },
        "decision_scope": (
            "same executable mesh and load path across one and multiple MPI "
            "ranks; no Table 5 promotion authority"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", type=Path)
    parser.add_argument("parallel", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    arguments = parser.parse_args()
    evidence = assess(
        load_candidate(arguments.serial),
        load_candidate(arguments.parallel),
        tolerance=arguments.tolerance,
    )
    payload = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not evidence["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
