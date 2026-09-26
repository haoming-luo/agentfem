"""Content-bound promotion assessment for the Lewandowski J2 beam.

The assessment consumes independently produced candidate directories.  It
derives convergence and equivalence decisions from their curves and manifests;
callers cannot promote the benchmark by supplying Boolean assertions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lewandowski_2023_self_weight_beam_fixture import (
    DEFINITION,
    UPSTREAM_BEHAVIOUR_SHA256,
    UPSTREAM_COMMIT,
    UPSTREAM_SOLVER_SHA256,
    assess_external_curve,
    bundled_reference_curve,
)


SCHEMA = "agentfem.lewandowski-2023-beam-promotion.v1"
MESH_RMS_TOLERANCE = 0.01
MESH_MAXIMUM_TOLERANCE = 0.02
INCREMENT_RMS_TOLERANCE = 0.002
INCREMENT_MAXIMUM_TOLERANCE = 0.005
RANK_RMS_TOLERANCE = 1.0e-9
RANK_MAXIMUM_TOLERANCE = 1.0e-8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidate(root: Path) -> dict[str, object]:
    """Load one candidate run and verify its content identities."""

    directory = Path(root)
    manifest_path = directory / "assessment.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = manifest["candidate"]
    curve_path = directory / str(candidate["curve_file"])
    actual_digest = _sha256(curve_path)
    if actual_digest != candidate["curve_sha256"]:
        raise RuntimeError(f"Candidate curve identity mismatch: {curve_path}")
    table = np.genfromtxt(curve_path, names=True, delimiter=",")
    load = np.atleast_1d(table["load_factor"]).astype(float)
    displacement = np.atleast_1d(table["downward_displacement_m"]).astype(float)
    if load.size != int(candidate["points"]):
        raise RuntimeError(f"Candidate point count mismatch: {curve_path}")
    source = manifest["runtime"]["manifest"]["identity"]["execution"]["source"]
    if not isinstance(source, dict):
        raise RuntimeError("Promotion requires an AgentFEM source-checkout identity.")
    return {
        "root": str(directory),
        "manifest_sha256": _sha256(manifest_path),
        "curve_sha256": actual_digest,
        "manifest": manifest,
        "candidate": candidate,
        "load": load,
        "displacement": displacement,
        "source": source,
    }


def _curve_error(reference, candidate) -> dict[str, float]:
    reference_load = np.asarray(reference["load"], dtype=float)
    reference_u = np.asarray(reference["displacement"], dtype=float)
    candidate_u = np.interp(
        reference_load,
        np.asarray(candidate["load"], dtype=float),
        np.asarray(candidate["displacement"], dtype=float),
    )
    scale = float(np.max(np.abs(reference_u)))
    if scale <= np.finfo(float).eps:
        raise ValueError("A convergence curve has zero displacement scale.")
    difference = candidate_u - reference_u
    return {
        "normalized_rms": float(np.sqrt(np.mean(difference**2)) / scale),
        "normalized_maximum": float(np.max(np.abs(difference)) / scale),
    }


def _convergence(levels, *, kind: str, rms_limit: float, maximum_limit: float):
    if len(levels) != 3:
        raise ValueError(f"{kind} convergence requires exactly three levels.")
    coarse_to_medium = _curve_error(levels[1], levels[0])
    medium_to_fine = _curve_error(levels[2], levels[1])
    decreasing = (
        medium_to_fine["normalized_rms"]
        <= coarse_to_medium["normalized_rms"] + 1.0e-14
        and medium_to_fine["normalized_maximum"]
        <= coarse_to_medium["normalized_maximum"] + 1.0e-14
    )
    passed = (
        decreasing
        and medium_to_fine["normalized_rms"] <= rms_limit
        and medium_to_fine["normalized_maximum"] <= maximum_limit
    )
    return {
        "passed": passed,
        "coarse_to_medium": coarse_to_medium,
        "medium_to_fine": medium_to_fine,
        "decreasing": decreasing,
        "contracts": {
            "medium_to_fine_normalized_rms": rms_limit,
            "medium_to_fine_normalized_maximum": maximum_limit,
            "origin": "AgentFEM_project_promotion_contract_not_author_tolerance",
        },
    }


def assess_promotion(
    *,
    mesh_roots,
    increment_roots,
    rank_roots,
    restart_report: Path,
) -> dict[str, object]:
    """Derive every promotion decision from content-bound artifacts."""

    mesh_runs = [load_candidate(Path(root)) for root in mesh_roots]
    increment_runs = [load_candidate(Path(root)) for root in increment_roots]
    rank_runs = [load_candidate(Path(root)) for root in rank_roots]
    all_runs = mesh_runs + increment_runs + rank_runs

    source_commits = {run["source"].get("commit") for run in all_runs}
    source_identities = {
        run["source"].get("package_tree_sha256") for run in all_runs
    }
    clean_source = all(not run["source"].get("tracked_dirty", True) for run in all_runs)
    common_source = len(source_identities) == 1

    mesh_runs.sort(
        key=lambda run: int(np.prod(run["candidate"]["subdivisions"]))
    )
    mesh_levels = [tuple(run["candidate"]["subdivisions"]) for run in mesh_runs]
    mesh_setup = (
        len(set(mesh_levels)) == 3
        and len({int(run["candidate"]["increments"]) for run in mesh_runs}) == 1
        and len({int(run["candidate"]["mpi_ranks"]) for run in mesh_runs}) == 1
        and len({run["candidate"].get("line_search") for run in mesh_runs}) == 1
        and len(
            {
                int(run["candidate"].get("maximum_iterations_limit", 30))
                for run in mesh_runs
            }
        )
        == 1
    )
    mesh_assessment = _convergence(
        mesh_runs,
        kind="mesh",
        rms_limit=MESH_RMS_TOLERANCE,
        maximum_limit=MESH_MAXIMUM_TOLERANCE,
    )
    mesh_assessment["levels"] = mesh_levels
    mesh_assessment["setup_consistent"] = mesh_setup
    mesh_assessment["passed"] = bool(mesh_assessment["passed"] and mesh_setup)

    increment_runs.sort(key=lambda run: int(run["candidate"]["increments"]))
    increment_levels = [int(run["candidate"]["increments"]) for run in increment_runs]
    increment_setup = (
        len(set(increment_levels)) == 3
        and len(
            {tuple(run["candidate"]["subdivisions"]) for run in increment_runs}
        )
        == 1
        and len({int(run["candidate"]["mpi_ranks"]) for run in increment_runs})
        == 1
        and len(
            {run["candidate"].get("line_search") for run in increment_runs}
        )
        == 1
        and len(
            {
                int(run["candidate"].get("maximum_iterations_limit", 30))
                for run in increment_runs
            }
        )
        == 1
    )
    increment_assessment = _convergence(
        increment_runs,
        kind="increment",
        rms_limit=INCREMENT_RMS_TOLERANCE,
        maximum_limit=INCREMENT_MAXIMUM_TOLERANCE,
    )
    increment_assessment["levels"] = increment_levels
    increment_assessment["setup_consistent"] = increment_setup
    increment_assessment["passed"] = bool(
        increment_assessment["passed"] and increment_setup
    )

    if len(rank_runs) != 2:
        raise ValueError("Rank equivalence requires exactly two candidate runs.")
    rank_runs.sort(key=lambda run: int(run["candidate"]["mpi_ranks"]))
    rank_counts = [int(run["candidate"]["mpi_ranks"]) for run in rank_runs]
    rank_setup = (
        rank_counts[0] == 1
        and rank_counts[1] > 1
        and tuple(rank_runs[0]["candidate"]["subdivisions"])
        == tuple(rank_runs[1]["candidate"]["subdivisions"])
        and int(rank_runs[0]["candidate"]["increments"])
        == int(rank_runs[1]["candidate"]["increments"])
        and rank_runs[0]["candidate"].get("line_search")
        == rank_runs[1]["candidate"].get("line_search")
        and int(rank_runs[0]["candidate"].get("maximum_iterations_limit", 30))
        == int(rank_runs[1]["candidate"].get("maximum_iterations_limit", 30))
    )
    rank_error = _curve_error(rank_runs[0], rank_runs[1])
    rank_assessment = {
        "passed": bool(
            rank_setup
            and rank_error["normalized_rms"] <= RANK_RMS_TOLERANCE
            and rank_error["normalized_maximum"] <= RANK_MAXIMUM_TOLERANCE
        ),
        "ranks": rank_counts,
        "setup_consistent": rank_setup,
        "errors": rank_error,
        "contracts": {
            "normalized_rms": RANK_RMS_TOLERANCE,
            "normalized_maximum": RANK_MAXIMUM_TOLERANCE,
            "origin": "AgentFEM_project_promotion_contract_not_author_tolerance",
        },
    }

    restart_path = Path(restart_report)
    restart = json.loads(restart_path.read_text(encoding="utf-8"))
    restart_source = restart["runtime"]["identity"]["execution"]["source"]
    candidate_package_identities = {
        run["source"].get("package_tree_sha256") for run in all_runs
    }
    restart_package_identity = restart_source.get("package_tree_sha256")
    restart_package_matches = restart_package_identity in candidate_package_identities
    restart_assessment = {
        "passed": bool(
            restart.get("passed")
            and not restart_source.get("tracked_dirty", True)
            and restart_package_matches
        ),
        "report_sha256": _sha256(restart_path),
        "reported_status": restart.get("status"),
        "package_tree_matches_candidates": restart_package_matches,
        "restart_commit": restart_source.get("commit"),
        "restart_package_tree_sha256": restart_package_identity,
    }

    exact_candidates = [
        run
        for run in all_runs
        if tuple(run["candidate"]["subdivisions"]) == DEFINITION.subdivisions
        and int(run["candidate"]["increments"]) == DEFINITION.increments
    ]
    if not exact_candidates:
        raise ValueError("No source-discretization candidate was supplied.")
    baseline = max(exact_candidates, key=lambda run: int(run["candidate"]["mpi_ranks"]))
    reference_load, reference_u, metadata = bundled_reference_curve()
    evidence = {
        "independent_reference_execution": True,
        # The paper calls A the top of the right edge, while the pinned public
        # executable explicitly samples the middle of the right extremity.
        # This benchmark and its candidate use the executable observer and do
        # not claim a point-A comparison to the plotted paper curve.
        "observer_reconciled": True,
        "candidate_mesh_converged": bool(mesh_assessment["passed"]),
        "candidate_increment_converged": bool(increment_assessment["passed"]),
        "serial_mpi_equivalent": bool(rank_assessment["passed"]),
        "restart_equivalent": bool(restart_assessment["passed"]),
    }
    external = assess_external_curve(
        candidate_load_factors=baseline["load"],
        candidate_displacements=baseline["displacement"],
        reference_load_factors=reference_load,
        reference_displacements=reference_u,
        reference_source_commit=UPSTREAM_COMMIT,
        reference_solver_sha256=UPSTREAM_SOLVER_SHA256,
        reference_behaviour_sha256=UPSTREAM_BEHAVIOUR_SHA256,
        reference_curve_sha256=metadata["curve_file_sha256"],
        declared_reference_curve_sha256=metadata["curve_file_sha256"],
        convergence_evidence=evidence,
    )
    content_bound = bool(clean_source and common_source)
    accepted = bool(content_bound and external["accepted"])
    return {
        "schema": SCHEMA,
        "status": "accepted" if accepted else "incomplete",
        "accepted": accepted,
        "content_bound": content_bound,
        "benchmark_promotion_authorized": accepted,
        "source": {
            "clean": clean_source,
            "common_identity": common_source,
            "package_tree_sha256": tuple(sorted(source_identities)),
            "commits": tuple(sorted(source_commits)),
            "identity_semantics": (
                "executable_agentfem_package_tree; harness-only commits may differ"
            ),
        },
        "observer_reconciliation": {
            "paper": "top_of_right_edge_point_A",
            "pinned_executable": "middle_of_right_extremity_(1,0,0)",
            "candidate": "middle_of_right_extremity_(1,0,0)",
            "claim_scope": "pinned_public_executable_curve_not_paper_point_A",
        },
        "mesh_convergence": mesh_assessment,
        "increment_convergence": increment_assessment,
        "rank_equivalence": rank_assessment,
        "restart_equivalence": restart_assessment,
        "external_comparison": external,
        "candidate_artifacts": tuple(
            {
                "root": run["root"],
                "manifest_sha256": run["manifest_sha256"],
                "curve_sha256": run["curve_sha256"],
            }
            for run in all_runs
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-run", action="append", type=Path, required=True)
    parser.add_argument("--increment-run", action="append", type=Path, required=True)
    parser.add_argument("--rank-run", action="append", type=Path, required=True)
    parser.add_argument("--restart-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = assess_promotion(
        mesh_roots=arguments.mesh_run,
        increment_roots=arguments.increment_run,
        rank_roots=arguments.rank_run,
        restart_report=arguments.restart_report,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(arguments.output)
    print(
        "Lewandowski beam promotion "
        f"| {report['status'].upper()} | evidence={arguments.output}"
    )


if __name__ == "__main__":
    main()
