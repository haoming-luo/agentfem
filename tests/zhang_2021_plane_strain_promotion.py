"""Content-bound convergence audit for the Zhang et al. Table 5 candidate.

The audit consumes candidate JSON files written by
``zhang_2021_plane_strain_driver.py``.  It derives refinement decisions from
the archived observables and refuses slices in which more than the declared
axis changes.  It deliberately cannot promote the external benchmark yet:
formulation, cell-size, MPI, and restart evidence remain independent gates.
Fixed-old-state finite-difference macro-tangent evidence can be supplied as a
separate, content-bound perturbation study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def _canonical_fingerprint(record: object) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _verified_artifact(reference, *, root: Path) -> Path:
    if not isinstance(reference, dict):
        raise ValueError("Tangent evidence artifact reference must be a mapping.")
    path = Path(str(reference.get("path", "")))
    selected = path if path.is_absolute() else root / path
    digest = reference.get("sha256")
    if (
        not selected.is_file()
        or not isinstance(digest, str)
        or _sha256(selected) != digest
    ):
        raise ValueError(f"Tangent evidence artifact hash mismatch: {selected}")
    return selected


def load_lifecycle_evidence(*, supercell=None, parallel=None, restarts=()):
    """Recompute audits from hashed candidates and verify restart archives.

    This closes only the stated lifecycle gates.  Element interpolation and
    coarse supercell evidence do not establish spatial or formulation convergence.
    """
    from zhang_2021_parallel_equivalence import assess as assess_parallel
    from zhang_2021_supercell_audit import assess as assess_supercell

    gates = {}
    artifacts = []

    def read(path):
        selected = Path(path)
        payload = json.loads(selected.read_text(encoding="utf-8"))
        artifacts.append({"path": str(selected), "sha256": _sha256(selected)})
        return selected, payload

    if supercell is not None:
        selected, payload = read(supercell)
        if payload.get("schema") != "agentfem.zhang-2021-geometric-supercell-audit.v1":
            raise ValueError("Unsupported supercell evidence schema.")
        runs = tuple(
            load_candidate(_verified_artifact(item["artifact"], root=selected.parent))
            for item in payload["comparisons"].values()
        )
        gates["periodic_cell_size_invariant"] = assess_supercell(runs)["passed"]
    if parallel is not None:
        selected, payload = read(parallel)
        if payload.get("schema") != "agentfem.zhang-2021-serial-mpi-equivalence.v1":
            raise ValueError("Unsupported parallel evidence schema.")
        runs = [
            load_candidate(
                _verified_artifact(payload["artifacts"][name], root=selected.parent)
            )
            for name in ("serial", "parallel")
        ]
        gates["serial_mpi_equivalent"] = assess_parallel(*runs)["passed"]
    directions = set()
    restart_passed = bool(restarts)
    for path in restarts:
        selected, payload = read(path)
        unsealed = dict(payload)
        fingerprint = unsealed.pop("fingerprint", None)
        if (
            payload.get("schema") != "agentfem.zhang-2021-plane-strain-restart.v1"
            or fingerprint != _canonical_fingerprint(unsealed)
        ):
            raise ValueError("Restart evidence schema or fingerprint mismatch.")
        references = payload["checkpoint"]["artifacts"]
        manifest_path = _verified_artifact(references["manifest"], root=selected.parent)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name in ("nodal_state", "quadrature_state"):
            actual = _verified_artifact(references[name], root=selected.parent)
            original = _verified_artifact(manifest[name], root=manifest_path.parent)
            if actual.resolve() != original.resolve():
                raise ValueError(
                    "Restart evidence does not reference the manifest archive."
                )
        writer, reader = (
            int(payload["writer_rank_count"]),
            int(payload["reader_rank_count"]),
        )
        if (
            writer != int(manifest["writer_rank_count"])
            or manifest.get("portable") is not True
        ):
            raise ValueError("Restart rank or portability identity mismatch.")
        directions.add((writer, reader))
        checks = payload["comparison"]["checks"]
        restart_passed &= bool(
            payload.get("passed") is True
            and checks
            and all(item.get("passed") is True for item in checks.values())
        )
    if restarts:
        gates["restart_equivalent"] = bool(
            restart_passed and {(1, 2), (2, 1)} <= directions
        )
    return {"derived_gates": gates, "artifacts": artifacts}


def load_macro_tangent_check(path: Path) -> dict[str, object]:
    """Load one fixed-old-state tangent check and verify its artifact graph."""

    selected = Path(path)
    payload = json.loads(selected.read_text(encoding="utf-8"))
    if payload.get("schema") != "agentfem.zhang-2021-macro-tangent-fd.v1":
        raise ValueError(f"Unsupported macro-tangent evidence schema: {selected}")
    fingerprint = payload.get("fingerprint")
    unsealed = dict(payload)
    unsealed.pop("fingerprint", None)
    if fingerprint != _canonical_fingerprint(unsealed):
        raise ValueError(f"Macro-tangent evidence fingerprint mismatch: {selected}")
    if (
        payload.get("status") != "passed"
        or payload.get("accepted") is not True
        or payload.get("content_bound") is not True
    ):
        raise ValueError(
            f"Macro-tangent evidence is not accepted and content-bound: {selected}"
        )
    root = selected.parent
    base_path = _verified_artifact(payload.get("base"), root=root)
    base = load_candidate(base_path)
    references = []
    for perturbation in payload.get("perturbations", ()):
        if not isinstance(perturbation, dict):
            raise ValueError("Tangent perturbation record must be a mapping.")
        references.extend((perturbation.get("plus"), perturbation.get("minus")))
    if len(references) != 8:
        raise ValueError(
            "A 2D macro-tangent check requires eight perturbation artifacts."
        )
    for reference in references:
        _verified_artifact(reference, root=root)
    check = payload.get("check")
    candidate = payload.get("candidate")
    implementation = payload.get("implementation")
    if not isinstance(check, dict) or check.get("passed") is not True:
        raise ValueError("Macro-tangent numerical check did not pass.")
    if not isinstance(candidate, dict) or not isinstance(implementation, dict):
        raise ValueError("Macro-tangent candidate identity is incomplete.")
    return {
        "path": str(selected),
        "sha256": _sha256(selected),
        "payload": payload,
        "candidate": candidate,
        "check": check,
        "implementation": implementation,
        "scientific_runtime": _scientific_identity(base["payload"]),
        "benchmark_implementation": _benchmark_identity(base["payload"]),
        "discretization": _discretization_identity(base["payload"]),
    }


def assess_macro_tangent_sensitivity(runs) -> dict[str, object]:
    """Audit a three-level centered-difference perturbation sequence."""

    selected = tuple(sorted(runs, key=lambda run: -run["candidate"]["relative_step"]))
    steps = tuple(float(run["candidate"]["relative_step"]) for run in selected)
    invariant_parameters = ("mesh_size", "quadrature_degree", "increments")
    common_problem = bool(
        len(selected) >= 3
        and len(set(steps)) == len(steps)
        and all(steps[index] > steps[index + 1] for index in range(len(steps) - 1))
        and all(
            len({json.dumps(run["candidate"].get(name)) for run in selected}) == 1
            for name in invariant_parameters
        )
        and len({run["scientific_runtime"] for run in selected}) == 1
        and len({run["benchmark_implementation"] for run in selected}) == 1
        and len({run["discretization"] for run in selected}) == 1
        and len({_stable(run["implementation"]) for run in selected}) == 1
    )
    analytical = tuple(
        np.asarray(run["check"]["analytical"], dtype=float) for run in selected
    )
    finite_difference = tuple(
        np.asarray(run["check"]["finite_difference"], dtype=float) for run in selected
    )
    analytical_consistent = bool(
        analytical
        and all(np.array_equal(analytical[0], value) for value in analytical[1:])
    )
    relative_errors = tuple(
        float(run["check"]["relative_frobenius_error"]) for run in selected
    )
    successive_changes = tuple(
        _relative_change(finite_difference[index - 1], finite_difference[index])
        for index in range(1, len(finite_difference))
    )
    observed_orders = tuple(
        math.log(relative_errors[index - 1] / relative_errors[index])
        / math.log(steps[index - 1] / steps[index])
        for index in range(1, len(selected))
        if relative_errors[index] > 0.0 and relative_errors[index - 1] > 0.0
    )
    all_checks_passed = bool(
        selected and all(run["check"].get("passed") is True for run in selected)
    )
    second_order_entry = bool(observed_orders and observed_orders[0] >= 1.5)
    stable_finest = bool(successive_changes and successive_changes[-1] <= 1.0e-6)
    passed = bool(
        common_problem
        and analytical_consistent
        and all_checks_passed
        and second_order_entry
        and stable_finest
    )
    return {
        "schema": "agentfem.zhang-2021-macro-tangent-step-audit.v1",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "macro_tangent_finite_difference_consistent": passed,
        "benchmark_promotion_authorized": False,
        "common_problem": common_problem,
        "analytical_tangent_identical": analytical_consistent,
        "all_point_checks_passed": all_checks_passed,
        "relative_steps_coarse_to_fine": steps,
        "relative_frobenius_errors": relative_errors,
        "successive_finite_difference_changes": successive_changes,
        "observed_orders": observed_orders,
        "second_order_entry": second_order_entry,
        "stable_finest": stable_finest,
        "finest_change_tolerance": 1.0e-6,
        "artifacts": tuple(
            {"path": run["path"], "sha256": run["sha256"]} for run in selected
        ),
        "decision_scope": (
            "fixed-pre-increment-state macro-tangent step sensitivity; "
            "no Table 5 promotion authority"
        ),
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
    interface_refinement: bool = False,
) -> dict[str, object]:
    selected = tuple(sorted(runs, key=lambda run: fine_order(run["candidate"])))
    coordinates = tuple(float(coordinate(run["candidate"])) for run in selected)
    unique = len(set(coordinates)) == len(coordinates)
    invariant = {
        parameter: tuple(
            sorted({_stable(run["candidate"].get(parameter)) for run in selected})
        )
        for parameter in (
            *invariant_parameters,
            "geometry_source",
            "mesh_policy",
            "macroscopic_deformation_gradient",
            "cell_repetitions",
            "reference_cell_area",
            "deformation_gradient_path",
        )
    }
    if interface_refinement:
        invariant["mesh_policy"] = tuple(sorted({
            _stable({key: value for key, value in run["candidate"]["mesh_policy"].items()
                     if key != "interface_size"}) for run in selected
        }))
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
    interface_runs=(),
    quadrature_runs=(),
    tangent_runs=(),
    lifecycle_evidence=None,
) -> dict[str, object]:
    """Derive available convergence axes from candidate file contents."""

    groups = {
        "load_increment_path_converged": tuple(increment_runs),
        "mesh_converged": tuple(mesh_runs),
        "quadrature_converged": tuple(quadrature_runs),
    }
    interface_runs = tuple(interface_runs)
    all_runs = tuple(run for runs in groups.values() for run in runs) + interface_runs
    source_identities = {_scientific_identity(run["payload"]) for run in all_runs}
    benchmark_identities = {_benchmark_identity(run["payload"]) for run in all_runs}
    clean = bool(all_runs) and all(
        not _source(run["payload"]).get("tracked_dirty", True) for run in all_runs
    )
    common_source = bool(all_runs) and len(source_identities) == 1
    common_benchmark = bool(all_runs) and len(benchmark_identities) == 1
    audits: dict[str, object] = {}
    interface_audit = None
    if interface_runs:
        for run in interface_runs:
            item = run["candidate"]
            policy = item.get("mesh_policy") or {}
            if policy.get("kind") != "interface_distance_threshold" or set(policy) != {
                "kind", "interface_size", "transition_distance", "distance_sampling"
            }:
                raise ValueError("Interface study requires the explicit supported mesh policy.")
            size = float(policy["interface_size"])
            if not np.isfinite(size) or not 0 < size < float(item["mesh_size"]):
                raise ValueError("Invalid interface refinement coordinate.")
        interface_audit = _axis_audit(
            interface_runs, name="interface_size",
            coordinate=lambda item: float(item["mesh_policy"]["interface_size"]),
            fine_order=lambda item: -float(item["mesh_policy"]["interface_size"]),
            invariant_parameters=("formulation", "mesh_size", "quadrature_degree",
                                  "requested_fixed_increments", "mpi_ranks"),
            interface_refinement=True,
        )
        interface_audit["scope"] = "Interface-size sensitivity at fixed background mesh; not global spatial convergence."
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
    tangent_audit = (
        assess_macro_tangent_sensitivity(tangent_runs) if tangent_runs else None
    )
    content_bound = bool(
        (all_runs or tangent_runs)
        and (not all_runs or (clean and common_source and common_benchmark))
        and (
            not tangent_runs
            or all(run["payload"].get("content_bound") is True for run in tangent_runs)
        )
    )
    independent_gates = tuple(
        name
        for name in INDEPENDENT_PROMOTION_GATES
        if not (lifecycle_evidence or {}).get("derived_gates", {}).get(name, False)
        and (
            name != "macro_tangent_finite_difference_consistent"
            or tangent_audit is None
            or not tangent_audit["passed"]
        )
    )
    missing = (
        tuple(name for name, passed in derived.items() if not passed)
        + independent_gates
    )
    return {
        "schema": SCHEMA,
        "status": "incomplete",
        "accepted": False,
        "benchmark_promotion_authorized": False,
        "content_bound": content_bound,
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
        "interface_refinement_audit": interface_audit,
        "macro_tangent_step_audit": tangent_audit,
        "lifecycle_evidence": lifecycle_evidence,
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
    parser.add_argument("--interface-run", action="append", type=Path, default=[])
    parser.add_argument("--quadrature-run", action="append", type=Path, default=[])
    parser.add_argument("--tangent-run", action="append", type=Path, default=[])
    parser.add_argument("--supercell-audit", type=Path)
    parser.add_argument("--parallel-audit", type=Path)
    parser.add_argument("--restart-audit", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = assess_convergence(
        increment_runs=tuple(load_candidate(path) for path in arguments.increment_run),
        mesh_runs=tuple(load_candidate(path) for path in arguments.mesh_run),
        interface_runs=tuple(load_candidate(path) for path in arguments.interface_run),
        quadrature_runs=tuple(
            load_candidate(path) for path in arguments.quadrature_run
        ),
        tangent_runs=tuple(
            load_macro_tangent_check(path) for path in arguments.tangent_run
        ),
        lifecycle_evidence=load_lifecycle_evidence(
            supercell=arguments.supercell_audit,
            parallel=arguments.parallel_audit,
            restarts=arguments.restart_audit,
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
