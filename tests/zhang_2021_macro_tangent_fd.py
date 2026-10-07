"""Independently check the Zhang-cell condensed macro tangent.

Every perturbed solve follows the identical history through the penultimate
accepted state.  Only one final macroscopic deformation-gradient component is
changed, so the centered difference checks the same fixed-old-state derivative
claimed by ``homogenized_algorithmic_tangent`` rather than differentiating a
different loading history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from agentfem import results
from agentfem.provenance import content_fingerprint


COMPONENT_ORDER = ("11", "21", "12", "22")
BASE_GRADIENT = np.asarray(((1.0, 0.1), (0.0, 1.0)))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _column_major(matrix) -> tuple[str, ...]:
    # argparse treats a negative scientific-notation token such as ``-1e-6``
    # as a possible option when it follows an ``nargs`` argument.  Fixed-point
    # text preserves the same float while remaining an unambiguous value.
    return tuple(f"{value:.17f}" for value in np.asarray(matrix).reshape(-1, order="F"))


def _run_candidate(
    *,
    output: Path,
    mesh_size: float,
    quadrature_degree: int,
    increments: int,
    penultimate: np.ndarray,
    final: np.ndarray,
    tangent: bool,
) -> dict[str, object]:
    driver = Path(__file__).with_name("zhang_2021_plane_strain_driver.py")
    coordinate = (increments - 1.0) / increments
    command = [
        sys.executable,
        str(driver),
        "--mesh-size",
        str(mesh_size),
        "--quadrature-degree",
        str(quadrature_degree),
        "--increments",
        str(increments),
        "--macro-gradient",
        *_column_major(final),
        "--penultimate-gradient",
        *_column_major(penultimate),
        "--penultimate-coordinate",
        f"{coordinate:.17g}",
        "--output",
        str(output),
    ]
    if not tangent:
        command.append("--skip-tangent")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if completed.returncode:
        raise RuntimeError(
            "Zhang tangent perturbation failed:\n" + completed.stdout + completed.stderr
        )
    evidence = output / "zhang_2021_table5_plane_strain_assessment.json"
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    if payload.get("result_status") != "completed":
        raise RuntimeError(f"Perturbed candidate did not complete: {evidence}")
    if payload.get("identity_stable_during_run") is not True:
        raise RuntimeError(f"Perturbed candidate identity drifted: {evidence}")
    return {
        "path": str(evidence),
        "sha256": _sha256(evidence),
        "payload": payload,
    }


def _same_preincrement_problem(reference, candidate) -> bool:
    left = reference["discretization_identity"]
    right = candidate["discretization_identity"]
    same_domain = all(
        left[name] == right[name] for name in ("mesh", "cell_tags", "facet_tags")
    )
    left_path = reference["candidate"]["deformation_gradient_path"]
    right_path = candidate["candidate"]["deformation_gradient_path"]
    same_path_prefix = (
        left_path["coordinates"] == right_path["coordinates"]
        and left_path["gradients"][:-1] == right_path["gradients"][:-1]
    )
    invariants = (
        "formulation",
        "mesh_size",
        "global_cells",
        "quadrature_degree",
        "accepted_increments",
        "requested_fixed_increments",
        "mpi_ranks",
    )
    same_parameters = all(
        reference["candidate"][name] == candidate["candidate"][name]
        for name in invariants
    )
    left_source = reference["runtime"]["manifest"]["identity"]["execution"]["source"]
    right_source = candidate["runtime"]["manifest"]["identity"]["execution"]["source"]
    same_source = left_source.get("scientific_runtime_sha256") == right_source.get(
        "scientific_runtime_sha256"
    )
    return bool(
        same_domain
        and same_path_prefix
        and same_parameters
        and same_source
        and reference["benchmark_implementation"]
        == candidate["benchmark_implementation"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-size", type=float, default=0.30)
    parser.add_argument("--quadrature-degree", type=int, default=4)
    parser.add_argument("--increments", type=int, default=10)
    parser.add_argument("--relative-step", type=float, default=1.0e-6)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.increments < 2:
        parser.error("--increments must be at least 2")
    if not np.isfinite(arguments.relative_step) or arguments.relative_step <= 0.0:
        parser.error("--relative-step must be finite and positive")

    arguments.output.mkdir(parents=True, exist_ok=True)
    coordinate = (arguments.increments - 1.0) / arguments.increments
    penultimate = np.eye(2) + coordinate * (BASE_GRADIENT - np.eye(2))
    base = _run_candidate(
        output=arguments.output / "base",
        mesh_size=arguments.mesh_size,
        quadrature_degree=arguments.quadrature_degree,
        increments=arguments.increments,
        penultimate=penultimate,
        final=BASE_GRADIENT,
        tangent=True,
    )
    base_payload = base["payload"]
    tangent = base_payload.get("homogenized_algorithmic_tangent") or {}
    if tangent.get("values") is None:
        raise RuntimeError("Base candidate did not produce a condensed tangent.")

    plus_columns = []
    minus_columns = []
    perturbation_records = []
    all_payloads = [base_payload]
    steps = np.full(4, float(arguments.relative_step))
    for column, component in enumerate(COMPONENT_ORDER):
        row, col = np.unravel_index(column, (2, 2), order="F")
        direction = np.zeros((2, 2), dtype=float)
        direction[row, col] = 1.0
        records = {}
        for sign, label in ((1.0, "plus"), (-1.0, "minus")):
            final = BASE_GRADIENT + sign * steps[column] * direction
            record = _run_candidate(
                output=arguments.output / f"{label}_{component}",
                mesh_size=arguments.mesh_size,
                quadrature_degree=arguments.quadrature_degree,
                increments=arguments.increments,
                penultimate=penultimate,
                final=final,
                tangent=False,
            )
            if not _same_preincrement_problem(base_payload, record["payload"]):
                raise RuntimeError(
                    f"Perturbation {label}_{component} changed the source, "
                    "executed mesh/tags, solver coordinates, or pre-final path."
                )
            records[label] = record
            all_payloads.append(record["payload"])
        plus_columns.append(np.asarray(records["plus"]["payload"]["first_piola"]))
        minus_columns.append(np.asarray(records["minus"]["payload"]["first_piola"]))
        perturbation_records.append(
            {
                "component": component,
                "step": steps[column],
                "plus": {
                    "path": records["plus"]["path"],
                    "sha256": records["plus"]["sha256"],
                },
                "minus": {
                    "path": records["minus"]["path"],
                    "sha256": records["minus"]["sha256"],
                },
            }
        )

    check = results.check_homogenized_algorithmic_tangent(
        tangent["values"],
        plus_first_piola=np.column_stack(plus_columns),
        minus_first_piola=np.column_stack(minus_columns),
        perturbation_steps=steps,
        component_order=COMPONENT_ORDER,
        relative_tolerance=arguments.relative_tolerance,
    )
    implementation = {
        "driver_sha256": _sha256(Path(__file__)),
        "candidate_driver_sha256": _sha256(
            Path(__file__).with_name("zhang_2021_plane_strain_driver.py")
        ),
    }
    clean_source = all(
        not payload["runtime"]["manifest"]["identity"]["execution"]["source"].get(
            "tracked_dirty",
            True,
        )
        for payload in all_payloads
    )
    record = {
        "schema": "agentfem.zhang-2021-macro-tangent-fd.v1",
        "status": "passed" if check.passed else "failed",
        "accepted": check.passed,
        "benchmark_promotion_authorized": False,
        "content_bound": clean_source,
        "candidate": {
            "mesh_size": arguments.mesh_size,
            "quadrature_degree": arguments.quadrature_degree,
            "increments": arguments.increments,
            "relative_step": arguments.relative_step,
        },
        "implementation": implementation,
        "base": {"path": base["path"], "sha256": base["sha256"]},
        "perturbations": perturbation_records,
        "check": check.as_dict(),
        "decision_scope": (
            "fixed-pre-increment-state consistency of AgentFEM's condensed "
            "homogenized tangent; no Table 5 promotion authority"
        ),
    }
    record["fingerprint"] = content_fingerprint(record)
    output = arguments.output / "macro_tangent_finite_difference.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        "Zhang macro tangent fixed-old-state check "
        f"| {record['status'].upper()} "
        f"| relative_error={check.relative_frobenius_error:.6g} "
        f"| max_column_error={max(check.column_relative_errors):.6g}"
    )
    print(f"Evidence: {output}")
    return 0 if check.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
