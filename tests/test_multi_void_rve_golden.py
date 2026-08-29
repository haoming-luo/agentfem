"""Fast and opt-in contracts for the deterministic multi-void RVE driver."""

from __future__ import annotations

import copy
import os

import pytest
from mpi4py import MPI

from multi_void_rve_golden_driver import (
    SCHEMA,
    compare_refinement_evidence,
    deterministic_realization,
    evidence_policy,
    run_candidate,
)


def _synthetic_level(*, mesh_size, cells, macro, mean, p95, p99, maximum, geometry):
    return {
        "schema": SCHEMA,
        "accepted_invariant_gates": True,
        "execution": {"global_cells": cells},
        "identities": {
            "realization": {"fingerprint": "fixed-realization"},
            "scientific_input": {"mesh": {"nominal_size": mesh_size}},
        },
        "quantities": {
            "homogenized_first_piola_stress": [[macro, 0.0], [0.0, -0.5 * macro]],
            "peeq_mean": mean,
            "peeq_p95": p95,
            "peeq_p99": p99,
            "solid_reference_fraction": 0.988,
        },
        "diagnostics": {
            "peeq_maximum": maximum,
            "geometry_relative_error": geometry,
        },
    }


def test_multi_void_policy_separates_golden_quantities_from_local_diagnostics():
    policy = evidence_policy()

    assert policy["fixed_stack_golden_candidates"] == [
        "homogenized_first_piola_stress",
        "peeq_mean",
        "peeq_p95",
        "solid_reference_fraction",
    ]
    assert "peeq_p99" in policy["mesh_sensitive_diagnostics"]
    assert "peeq_maximum" in policy["excluded_from_golden"]
    assert "minimum_quadrature_j" in policy["excluded_from_golden"]
    assert "checkpoint/restart equivalence on the realized mesh" in policy[
        "promotion_requires"
    ]


def test_multi_void_realization_has_stable_periodic_scientific_identity():
    realization = deterministic_realization()
    identity = realization.scientific_identity()

    assert identity["fingerprint"] == (
        "d3f0879a06d1092926544b465109cdff4b979cf384b7db55bb0a67fd1291e49d"
    )
    assert len(identity["voids"]) == 4
    assert identity["observed_periodic_inter_void_clearance"] >= 0.04
    assert identity["actual_void_fraction"] == pytest.approx(
        4.0 * (4.0 * 3.141592653589793 * 0.09**3 / 3.0)
    )


def test_refinement_comparison_accepts_stable_global_and_weighted_quantities():
    coarse = _synthetic_level(
        mesh_size=0.20,
        cells=100,
        macro=100.3,
        mean=0.00301,
        p95=0.00320,
        p99=0.00350,
        maximum=0.0040,
        geometry=0.008,
    )
    medium = _synthetic_level(
        mesh_size=0.16,
        cells=200,
        macro=100.1,
        mean=0.003005,
        p95=0.00322,
        p99=0.00360,
        maximum=0.0042,
        geometry=0.006,
    )
    fine = _synthetic_level(
        mesh_size=0.12,
        cells=400,
        macro=100.0,
        mean=0.003003,
        p95=0.00324,
        p99=0.00370,
        maximum=0.0044,
        geometry=0.004,
    )

    certificate = compare_refinement_evidence(coarse, medium, fine)

    assert certificate["accepted"]
    assert certificate["gates"] == {
        "all_level_invariants": True,
        "macro_first_piola": True,
        "peeq_mean": True,
        "peeq_p95": True,
        "solid_reference_fraction": True,
        "finest_geometry_relative_error": True,
    }
    assert "not formal asymptotic convergence" in certificate["wording"]


def test_refinement_comparison_fails_closed_on_drift_or_identity_change():
    levels = [
        _synthetic_level(
            mesh_size=size,
            cells=cells,
            macro=100.0,
            mean=0.003,
            p95=p95,
            p99=0.0035,
            maximum=0.004,
            geometry=0.004,
        )
        for size, cells, p95 in (
            (0.20, 100, 0.0030),
            (0.16, 200, 0.0031),
            (0.12, 400, 0.0033),
        )
    ]
    certificate = compare_refinement_evidence(*levels)
    assert not certificate["accepted"]
    assert not certificate["gates"]["peeq_p95"]

    incompatible = copy.deepcopy(levels[-1])
    incompatible["identities"]["realization"]["fingerprint"] = "different"
    with pytest.raises(ValueError, match="different void realizations"):
        compare_refinement_evidence(levels[0], levels[1], incompatible)


@pytest.mark.skipif(
    os.environ.get("AGENTFEM_RUN_MULTI_VOID_RVE") != "1",
    reason="set AGENTFEM_RUN_MULTI_VOID_RVE=1 for the real multi-void solve",
)
def test_real_multi_void_candidate_passes_invariant_gates():
    evidence = run_candidate(comm=MPI.COMM_SELF)
    assert evidence["accepted_invariant_gates"]
    assert evidence["status"] == "candidate_requires_refinement_mpi_and_restart"


@pytest.mark.skipif(
    os.environ.get("AGENTFEM_RUN_MULTI_VOID_RVE_REFINEMENT") != "1",
    reason=(
        "set AGENTFEM_RUN_MULTI_VOID_RVE_REFINEMENT=1 for the three-level "
        "multi-void refinement certificate"
    ),
)
def test_real_multi_void_successive_refinement_certificate():
    evidence = [
        run_candidate(comm=MPI.COMM_SELF, mesh_size=size)
        for size in (0.20, 0.16, 0.12)
    ]
    certificate = compare_refinement_evidence(*evidence)
    assert certificate["accepted"]
