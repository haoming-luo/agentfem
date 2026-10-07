"""Content-bound convergence audit for the Zhang et al. Table 5 candidate.

The audit consumes candidate JSON files written by
``zhang_2021_plane_strain_driver.py``.  It derives refinement decisions from
the archived observables and refuses slices in which more than the declared
axis changes.  It deliberately cannot promote the external benchmark yet:
formulation, cell-size, MPI, restart, and finite-difference macro-tangent
evidence remain independent gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np


SCHEMA = "agentfem.zhang-2021-table5-convergence-audit.v1"
OBSERVABLE_TOLERANCES = {
    "first_piola": 5.0e-3,
    "elastic_energy_density": 5.0e-3,
    "effective_tangent": 1.0e-2,
}
INDEPENDENT_PROMOTION_GATES = (
    "plane_strain_formulation_converged",
    "periodic_cell_size_invariant",
    "serial_mpi_equivalent",
    "restart_equivalent",
    "macro_tangent_finite_difference_consistent",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(candidate: dict[str, object]) -> dict[str, object]:
    try:
        source = candidate["runtime"]["manifest"]["identity"]["execution"]["source"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Candidate has no AgentFEM source identity.") from exc
    if not isinstance(source, dict):
        raise ValueError("Candidate source identity must be a mapping.")
    return source


def _scientific_identity(candidate: dict[str, object]) -> str:
    source = _source(candidate)
    identity = source.get("scientific_runtime_sha256") or source.get(
        "package_tree_sha256"
    )
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity.lower())
    ):
        raise ValueError("Candidate has no valid scientific runtime SHA-256.")
    return identity.lower()


def _benchmark_identity(candidate: dict[str, object]) -> tuple[tuple[str, str], ...]:
    implementation = candidate.get("benchmark_implementation")
    if not isinstance(implementation, dict) or implementation.get("schema") != (
        "agentfem.benchmark-implementation-identity.v1"
    ):
        raise ValueError("Candidate has no benchmark implementation identity.")
    files = implementation.get("files")
    if not isinstance(files, (list, tuple)) or len(files) < 2:
        raise ValueError("Candidate benchmark implementation files are incomplete.")
    identities = []
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("Benchmark implementation records must be mappings.")
        name = record.get("name")
        digest = record.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise ValueError("Candidate has an invalid benchmark file identity.")
        identities.append((name, digest.lower()))
    if len({name for name, _digest in identities}) != len(identities):
        raise ValueError("Benchmark implementation file names must be unique.")
    return tuple(sorted(identities))


def _discretization_identity(candidate: dict[str, object]) -> str:
    identity = candidate.get("discretization_identity")
    if not isinstance(identity, dict) or identity.get("schema") != (
        "agentfem.external-benchmark-discretization.v1"
    ):
        raise ValueError("Candidate has no executable discretization identity.")
    fingerprint = identity.get("fingerprint")
    selected = candidate.get("candidate")
    if (
        not isinstance(selected, dict)
        or selected.get("discretization_fingerprint") != fingerprint
    ):
        raise ValueError("Candidate discretization fingerprints disagree.")
    digest = (
        fingerprint.removeprefix("sha256:") if isinstance(fingerprint, str) else None
    )
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest.lower())
    ):
        raise ValueError("Candidate has no valid discretization SHA-256.")
    return digest.lower()


def load_candidate(path: Path) -> dict[str, object]:
    """Load one candidate and retain the exact artifact identity."""

    selected = Path(path)
    payload = json.loads(selected.read_text(encoding="utf-8"))
    if payload.get("candidate_schema") != "agentfem.external-benchmark-candidate.v2":
        raise ValueError(f"Unsupported Zhang candidate schema: {selected}")
    candidate = payload.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"Candidate parameters are missing: {selected}")
    if payload.get("result_status") != "completed":
        raise ValueError(f"Candidate solve did not complete: {selected}")
    if payload.get("identity_stable_during_run") is not True:
        raise ValueError(
            f"Candidate identity was not stable during execution: {selected}"
        )
    _scientific_identity(payload)
    _benchmark_identity(payload)
    _discretization_identity(payload)
    return {
        "path": str(selected),
        "sha256": _sha256(selected),
        "payload": payload,
        "candidate": candidate,
    }


def _observable(run, name: str) -> np.ndarray:
    payload = run["payload"]
    if name == "first_piola":
        value = payload.get("first_piola")
    elif name == "elastic_energy_density":
        diagnostics = payload.get("mixed_elastic_energy_diagnostics") or {}
        value = diagnostics.get("primal_elastic_energy_density")
    elif name == "effective_tangent":
        tangent = payload.get("homogenized_algorithmic_tangent") or {}
        value = tangent.get("values")
    else:  # pragma: no cover - internal programming guard
        raise KeyError(name)
    if value is None:
        raise KeyError(name)
    array = np.asarray(value, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"Observable {name!r} is empty or non-finite.")
    return array


def _relative_change(coarser: np.ndarray, finer: np.ndarray) -> float:
    difference = float(np.linalg.norm(finer - coarser))
    return difference / max(float(np.linalg.norm(finer)), np.finfo(float).tiny)


def _stable(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _axis_audit(
    runs,
    *,
    name: str,
    coordinate: Callable[[dict[str, object]], float],
    fine_order: Callable[[dict[str, object]], float],
    invariant_parameters: tuple[str, ...],
    require_decreasing: bool = True,
) -> dict[str, object]:
    selected = tuple(sorted(runs, key=lambda run: fine_order(run["candidate"])))
    coordinates = tuple(float(coordinate(run["candidate"])) for run in selected)
    unique = len(set(coordinates)) == len(coordinates)
    invariant = {
        parameter: tuple(
            sorted({_stable(run["candidate"].get(parameter)) for run in selected})
        )
        for parameter in invariant_parameters
    }
    setup_consistent = bool(
        len(selected) >= 3
        and unique
        and all(len(values) == 1 for values in invariant.values())
    )
    checks: dict[str, object] = {}
    for observable, tolerance in OBSERVABLE_TOLERANCES.items():
        try:
            values = tuple(_observable(run, observable) for run in selected)
        except (KeyError, TypeError, ValueError) as exc:
            checks[observable] = {
                "status": "incomplete",
                "passed": False,
                "message": str(exc),
                "tolerance": tolerance,
            }
            continue
        changes = tuple(
            _relative_change(values[index - 1], values[index])
            for index in range(1, len(values))
        )
        # Coarse levels may lie outside the asymptotic regime.  The promotion
        # decision therefore checks that the final refinement improves on the
        # immediately preceding pair, while preserving every earlier change
        # in the evidence record.
        decreasing = bool(len(changes) >= 2 and changes[-1] <= changes[-2] + 1.0e-14)
        passed = bool(
            setup_consistent
            and changes
            and (decreasing or not require_decreasing)
            and changes[-1] <= tolerance
        )
        checks[observable] = {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "successive_relative_changes": changes,
            "decreasing": decreasing,
            "decreasing_required": require_decreasing,
            "finest_change": None if not changes else changes[-1],
            "tolerance": tolerance,
            "contract_origin": "AgentFEM_project_contract_not_author_tolerance",
        }
    incomplete = bool(
        not setup_consistent
        or any(check["status"] == "incomplete" for check in checks.values())
    )
    passed = bool(not incomplete and all(check["passed"] for check in checks.values()))
    return {
        "axis": name,
        "status": "passed" if passed else "incomplete" if incomplete else "failed",
        "passed": passed,
        "coordinates_coarse_to_fine": coordinates,
        "setup_consistent": setup_consistent,
        "invariant_parameters": invariant,
        "checks": checks,
        "artifacts": tuple(
            {"path": run["path"], "sha256": run["sha256"]} for run in selected
        ),
    }


def assess_convergence(
    *,
    increment_runs=(),
    mesh_runs=(),
    quadrature_runs=(),
) -> dict[str, object]:
    """Derive available convergence axes from candidate file contents."""

    groups = {
        "load_increment_path_converged": tuple(increment_runs),
        "mesh_converged": tuple(mesh_runs),
        "quadrature_converged": tuple(quadrature_runs),
    }
    all_runs = tuple(run for runs in groups.values() for run in runs)
    source_identities = {_scientific_identity(run["payload"]) for run in all_runs}
    benchmark_identities = {_benchmark_identity(run["payload"]) for run in all_runs}
    clean = bool(all_runs) and all(
        not _source(run["payload"]).get("tracked_dirty", True) for run in all_runs
    )
    common_source = bool(all_runs) and len(source_identities) == 1
    common_benchmark = bool(all_runs) and len(benchmark_identities) == 1
    audits: dict[str, object] = {}
    if increment_runs:
        audits["load_increment_path_converged"] = _axis_audit(
            increment_runs,
            name="fixed_load_increments",
            coordinate=lambda item: float(item["requested_fixed_increments"]),
            fine_order=lambda item: float(item["requested_fixed_increments"]),
            invariant_parameters=(
                "formulation",
                "mesh_size",
                "global_cells",
                "quadrature_degree",
                "mpi_ranks",
                "discretization_fingerprint",
            ),
        )
    if mesh_runs:
        audits["mesh_converged"] = _axis_audit(
            mesh_runs,
            name="mesh_size",
            coordinate=lambda item: float(item["mesh_size"]),
            fine_order=lambda item: -float(item["mesh_size"]),
            invariant_parameters=(
                "formulation",
                "quadrature_degree",
                "requested_fixed_increments",
                "mpi_ranks",
            ),
        )
    if quadrature_runs:
        audits["quadrature_converged"] = _axis_audit(
            quadrature_runs,
            name="quadrature_degree",
            coordinate=lambda item: float(item["quadrature_degree"]),
            fine_order=lambda item: float(item["quadrature_degree"]),
            invariant_parameters=(
                "formulation",
                "mesh_size",
                "global_cells",
                "requested_fixed_increments",
                "mpi_ranks",
                "discretization_fingerprint",
            ),
            require_decreasing=False,
        )
    derived = {name: bool(audits.get(name, {}).get("passed")) for name in groups}
    missing = tuple(name for name, passed in derived.items() if not passed) + tuple(
        INDEPENDENT_PROMOTION_GATES
    )
    return {
        "schema": SCHEMA,
        "status": "incomplete",
        "accepted": False,
        "benchmark_promotion_authorized": False,
        "content_bound": bool(clean and common_source and common_benchmark),
        "source": {
            "clean": clean,
            "common_scientific_runtime": common_source,
            "common_benchmark_implementation": common_benchmark,
            "scientific_runtime_sha256": tuple(
                sorted(str(item) for item in source_identities)
            ),
            "benchmark_implementation": tuple(
                tuple({"name": name, "sha256": digest} for name, digest in identity)
                for identity in sorted(benchmark_identities)
            ),
        },
        "derived_convergence": derived,
        "axis_audits": audits,
        "missing_promotion_evidence": missing,
        "decision_scope": (
            "content-bound mesh, increment, and quadrature diagnostics; "
            "independent formulation/MPI/restart/tangent gates remain required"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--increment-run", action="append", type=Path, default=[])
    parser.add_argument("--mesh-run", action="append", type=Path, default=[])
    parser.add_argument("--quadrature-run", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = assess_convergence(
        increment_runs=tuple(load_candidate(path) for path in arguments.increment_run),
        mesh_runs=tuple(load_candidate(path) for path in arguments.mesh_run),
        quadrature_runs=tuple(
            load_candidate(path) for path in arguments.quadrature_run
        ),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(arguments.output)
    print(f"Zhang Table 5 convergence audit | {report['status'].upper()}")
    print(f"Evidence: {arguments.output}")


if __name__ == "__main__":
    main()
