"""Evidence contracts for the Lewandowski et al. external beam route."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI

from lewandowski_2023_self_weight_beam_fixture import (
    DEFINITION,
    REQUIRED_PROMOTION_EVIDENCE,
    UPSTREAM_ARTIFACTS,
    UPSTREAM_BEHAVIOUR_SHA256,
    UPSTREAM_COMMIT,
    UPSTREAM_SOLVER_SHA256,
    assess_external_curve,
    bundled_reference_curve,
)
from lewandowski_2023_self_weight_beam_driver import _increment_performance
from lewandowski_2023_self_weight_beam_promotion import assess_promotion
from lewandowski_2023_self_weight_beam_restart_driver import (
    PRIMARY_NORMALIZED_TOLERANCE,
    RESPONSE_NORMALIZED_TOLERANCE,
    _global_array_error,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_increment_performance_preserves_stage_totals_and_ksp_evidence():
    records = (
        SimpleNamespace(
            increment=1,
            load_factor=0.5,
            total_seconds=2.0,
            material_update_seconds=0.4,
            residual_assembly_seconds=0.2,
            tangent_assembly_seconds=0.5,
            linear_solve_seconds=0.7,
            line_search_seconds=0.1,
            linear_solve_calls=3,
            linear_iterations=3,
            linear_converged_reasons=(4, 4, 4),
        ),
        SimpleNamespace(
            increment=2,
            load_factor=1.0,
            total_seconds=3.0,
            material_update_seconds=0.6,
            residual_assembly_seconds=0.3,
            tangent_assembly_seconds=0.8,
            linear_solve_seconds=1.0,
            line_search_seconds=0.2,
            linear_solve_calls=4,
            linear_iterations=4,
            linear_converged_reasons=(4, 4, 4, 4),
        ),
    )

    evidence = _increment_performance(records)

    assert evidence["accepted_increment_count"] == 2
    assert evidence["totals"]["total_seconds"] == pytest.approx(5.0)
    assert evidence["totals"]["linear_solve_seconds"] == pytest.approx(1.7)
    assert evidence["linear_solve_calls"] == 7
    assert evidence["linear_iterations"] == 7
    assert evidence["increments"][1]["linear_converged_reasons"] == [4, 4, 4, 4]


def test_restart_error_contract_respects_physical_channel_scale():
    stress = np.array([250.0e6, -125.0e6, 0.0])
    roundoff_shift = np.array([4.0e-3, -2.0e-3, 0.0])

    response = _global_array_error(
        stress,
        stress + roundoff_shift,
        comm=MPI.COMM_SELF,
        normalized_tolerance=RESPONSE_NORMALIZED_TOLERANCE,
    )
    primary = _global_array_error(
        np.array([1.0, 0.5]),
        np.array([1.0, 0.5 + 1.0e-8]),
        comm=MPI.COMM_SELF,
        normalized_tolerance=PRIMARY_NORMALIZED_TOLERANCE,
    )

    assert response["passed"]
    assert response["reference_scale"] == pytest.approx(250.0e6)
    assert response["maximum_normalized"] == pytest.approx(1.6e-11)
    assert not primary["passed"]


def test_lewandowski_2023_external_reference_is_reexecuted_and_pinned():
    load, displacement, metadata = bundled_reference_curve()

    assert load.size == displacement.size == 31
    assert load[0] == pytest.approx(0.0)
    assert load[-1] == pytest.approx(1.0)
    assert np.all(np.diff(load) > 0.0)
    assert displacement[-1] == pytest.approx(0.10979636395992733)
    assert metadata["source"]["upstream_commit"] == UPSTREAM_COMMIT
    assert metadata["source"]["solver_sha256"] == UPSTREAM_SOLVER_SHA256
    assert metadata["source"]["behaviour_sha256"] == UPSTREAM_BEHAVIOUR_SHA256
    assert metadata["claim_scope"].startswith("Independently reexecuted")


def test_full_size_mpi_candidate_evidence_is_content_bound_and_fail_closed():
    root = (
        PROJECT_ROOT
        / "evidence"
        / "finite_strain_j2"
        / "lewandowski_2023_mpi4_candidate"
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    curve_bytes = (root / "candidate_curve.csv").read_bytes()
    curve = np.genfromtxt(root / "candidate_curve.csv", names=True, delimiter=",")
    reference_load, reference_displacement, _metadata = bundled_reference_curve()
    reference = np.interp(curve["load_factor"], reference_load, reference_displacement)
    candidate = np.asarray(curve["downward_displacement_m"], dtype=float)
    scale = float(np.max(np.abs(reference)))
    rms = float(np.sqrt(np.mean((candidate - reference) ** 2)) / scale)
    maximum = float(np.max(np.abs(candidate - reference)) / scale)

    assert hashlib.sha256(curve_bytes).hexdigest() == manifest["candidate"][
        "curve_sha256"
    ]
    assert manifest["status"] == "incomplete"
    assert manifest["runtime"]["source_tracked_dirty"] is True
    assert manifest["comparison"]["curve_contract_passed"] is True
    assert rms == pytest.approx(manifest["comparison"]["normalized_rms_error"])
    assert maximum == pytest.approx(
        manifest["comparison"]["normalized_maximum_error"]
    )
    assert set(manifest["open_promotion_gates"]) >= {
        "clean committed candidate identity",
        "candidate mesh convergence",
        "candidate increment convergence",
        "serial/MPI equivalence",
        "checkpoint/restart equivalence",
    }


def test_lewandowski_source_and_scientific_inputs_are_frozen():
    assert UPSTREAM_COMMIT == "cb43561d5e36a9ef691ad2c308261448cef44e29"
    assert tuple(item.size_bytes for item in UPSTREAM_ARTIFACTS) == (7905, 577)
    assert tuple(item.sha256 for item in UPSTREAM_ARTIFACTS) == (
        UPSTREAM_SOLVER_SHA256,
        UPSTREAM_BEHAVIOUR_SHA256,
    )
    assert DEFINITION.subdivisions == (30, 5, 8)
    assert DEFINITION.increments == 30
    assert DEFINITION.observer == (1.0, 0.0, 0.0)
    assert DEFINITION.maximum_body_force == pytest.approx(50.0e6)
    assert DEFINITION.young == pytest.approx(210.0e9)
    assert DEFINITION.yield_stress == pytest.approx(250.0e6)
    assert DEFINITION.hardening_modulus == pytest.approx(1.0e6)


def test_external_curve_assessment_fails_closed_without_independent_oracle():
    load = np.linspace(0.0, 1.0, 6)
    displacement = 0.02 * load**2
    result = assess_external_curve(
        candidate_load_factors=load,
        candidate_displacements=displacement,
    )

    assert result["status"] == "incomplete"
    assert not result["accepted"]
    assert "independent_reference_curve" in result["missing_evidence"]
    assert "pinned_reference_source_identity" in result["missing_evidence"]
    assert "reference_curve_content_identity" in result["missing_evidence"]
    assert set(REQUIRED_PROMOTION_EVIDENCE).issubset(result["missing_evidence"])


def test_external_curve_comparator_requires_identity_and_all_evidence():
    load = np.linspace(0.0, 1.0, 7)
    reference = 0.03 * load**2
    evidence = {name: True for name in REQUIRED_PROMOTION_EVIDENCE}

    incomplete = assess_external_curve(
        candidate_load_factors=load,
        candidate_displacements=reference,
        reference_load_factors=load,
        reference_displacements=reference,
        reference_source_commit="mutable-master",
        reference_solver_sha256=UPSTREAM_SOLVER_SHA256,
        reference_behaviour_sha256=UPSTREAM_BEHAVIOUR_SHA256,
        reference_curve_sha256="a" * 64,
        declared_reference_curve_sha256="a" * 64,
        convergence_evidence=evidence,
    )
    assert incomplete["status"] == "incomplete"
    assert "pinned_reference_source_identity" in incomplete["missing_evidence"]

    accepted = assess_external_curve(
        candidate_load_factors=load,
        candidate_displacements=reference,
        reference_load_factors=load,
        reference_displacements=reference,
        reference_source_commit=UPSTREAM_COMMIT,
        reference_solver_sha256=UPSTREAM_SOLVER_SHA256,
        reference_behaviour_sha256=UPSTREAM_BEHAVIOUR_SHA256,
        reference_curve_sha256="a" * 64,
        declared_reference_curve_sha256="a" * 64,
        convergence_evidence=evidence,
    )
    assert accepted["status"] == "accepted"
    assert accepted["accepted"]
    assert accepted["contracts"]["origin"].endswith("not_author_tolerance")

    wrong_curve_identity = assess_external_curve(
        candidate_load_factors=load,
        candidate_displacements=reference,
        reference_load_factors=load,
        reference_displacements=reference,
        reference_source_commit=UPSTREAM_COMMIT,
        reference_solver_sha256=UPSTREAM_SOLVER_SHA256,
        reference_behaviour_sha256=UPSTREAM_BEHAVIOUR_SHA256,
        reference_curve_sha256="a" * 64,
        declared_reference_curve_sha256="b" * 64,
        convergence_evidence=evidence,
    )
    assert wrong_curve_identity["status"] == "incomplete"
    assert "reference_curve_content_identity" in wrong_curve_identity["missing_evidence"]

    with pytest.raises(ValueError, match="tightened, not relaxed"):
        assess_external_curve(
            candidate_load_factors=load,
            candidate_displacements=reference,
            maximum_normalized_rms_error=0.031,
        )


def test_content_bound_promotion_derives_evidence_from_artifacts(tmp_path):
    reference_load, reference_u, _metadata = bundled_reference_curve()
    source = {
        "commit": "1" * 40,
        "tracked_dirty": False,
        "package_tree_sha256": "2" * 64,
    }

    def candidate(name, *, subdivisions, increments, ranks, scale):
        root = tmp_path / name
        root.mkdir()
        curve = root / "candidate_curve.csv"
        np.savetxt(
            curve,
            np.column_stack((reference_load, scale * reference_u)),
            delimiter=",",
            header="load_factor,downward_displacement_m",
            comments="",
        )
        digest = hashlib.sha256(curve.read_bytes()).hexdigest()
        manifest = {
            "candidate": {
                "curve_file": curve.name,
                "curve_sha256": digest,
                "points": int(reference_load.size),
                "subdivisions": subdivisions,
                "increments": increments,
                "mpi_ranks": ranks,
            },
            "runtime": {
                "manifest": {
                    "identity": {
                        "execution": {"source": source},
                    }
                }
            },
        }
        (root / "assessment.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        return root

    mesh = (
        candidate(
            "mesh-coarse",
            subdivisions=(18, 3, 5),
            increments=30,
            ranks=2,
            scale=1.008,
        ),
        candidate(
            "mesh-medium",
            subdivisions=(24, 4, 6),
            increments=30,
            ranks=2,
            scale=1.003,
        ),
        candidate(
            "mesh-fine",
            subdivisions=(30, 5, 8),
            increments=30,
            ranks=2,
            scale=1.001,
        ),
    )
    increments = (
        candidate(
            "increment-coarse",
            subdivisions=(30, 5, 8),
            increments=10,
            ranks=2,
            scale=1.0015,
        ),
        candidate(
            "increment-medium",
            subdivisions=(30, 5, 8),
            increments=20,
            ranks=2,
            scale=1.0005,
        ),
        candidate(
            "increment-fine",
            subdivisions=(30, 5, 8),
            increments=30,
            ranks=2,
            scale=1.0001,
        ),
    )
    ranks = (
        candidate(
            "rank-serial",
            subdivisions=(30, 5, 8),
            increments=30,
            ranks=1,
            scale=1.0,
        ),
        candidate(
            "rank-mpi",
            subdivisions=(30, 5, 8),
            increments=30,
            ranks=2,
            scale=1.0,
        ),
    )
    restart = tmp_path / "restart.json"
    restart.write_text(
        json.dumps(
            {
                "passed": True,
                "status": "accepted",
                "runtime": {
                    "identity": {
                        "execution": {"source": source},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = assess_promotion(
        mesh_roots=mesh,
        increment_roots=increments,
        rank_roots=ranks,
        restart_report=restart,
    )

    assert report["status"] == "accepted"
    assert report["content_bound"]
    assert report["benchmark_promotion_authorized"]
    assert report["mesh_convergence"]["passed"]
    assert report["increment_convergence"]["passed"]
    assert report["rank_equivalence"]["passed"]
    assert report["restart_equivalence"]["passed"]
    assert report["observer_reconciliation"]["claim_scope"] == (
        "pinned_public_executable_curve_not_paper_point_A"
    )

    dirty = json.loads((mesh[0] / "assessment.json").read_text(encoding="utf-8"))
    dirty["runtime"]["manifest"]["identity"]["execution"]["source"][
        "tracked_dirty"
    ] = True
    (mesh[0] / "assessment.json").write_text(json.dumps(dirty), encoding="utf-8")
    rejected = assess_promotion(
        mesh_roots=mesh,
        increment_roots=increments,
        rank_roots=ranks,
        restart_report=restart,
    )
    assert rejected["status"] == "incomplete"
    assert not rejected["content_bound"]
    assert not rejected["benchmark_promotion_authorized"]
