"""Fast and opt-in contracts for the deterministic multi-void RVE driver."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import benchmarks, provenance

from multi_void_rve_golden_driver import (
    SCHEMA,
    compare_rank_evidence,
    compare_refinement_evidence,
    deterministic_realization,
    evidence_policy,
    run_candidate,
)
from periodic_void_fixture import PERIODIC_VOID_GMSH_OPTIONS


BENCHMARK_ID = "agentfem.benchmark.finite_strain_j2_periodic_multi_void"


def _benchmark_card() -> dict[str, object]:
    path = (
        Path(benchmarks.__file__).resolve().parents[1]
        / "knowledge"
        / "benchmarks"
        / "finite_strain_j2_periodic_multi_void.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _fixed_stack_mismatches(card) -> tuple[str, ...]:
    expected = card["regression_identity"]["fixed_stack"]
    current = provenance.runtime_manifest()["identity"]
    mismatches = []
    platform = expected["platform"]
    if current["operating_system"]["system"] != platform["system"]:
        mismatches.append("operating system")
    if current["machine"] != platform["machine"]:
        mismatches.append("machine architecture")
    if current["python"] != expected["python"]:
        mismatches.append("Python")
    for name, version in expected["packages"].items():
        if name == "agentfem":
            continue
        if current["packages"].get(name) != version:
            mismatches.append(name)
    if current["mpi"] != expected["mpi"]:
        mismatches.append("MPI")
    if current["numerics"] != expected["numerics"]:
        mismatches.append("scalar/numeric contract")

    try:
        import gmsh
    except ImportError:
        mismatches.append("Gmsh is unavailable")
        return tuple(mismatches)
    current_gmsh = {
        "version": gmsh.__version__,
        "options": PERIODIC_VOID_GMSH_OPTIONS,
    }
    if current_gmsh != expected["gmsh"]:
        mismatches.append("Gmsh version/generation policy")
    return tuple(mismatches)


def _synthetic_level(*, mesh_size, cells, macro, mean, p95, p99, maximum, geometry):
    return {
        "schema": SCHEMA,
        "accepted_invariant_gates": True,
        "execution": {"global_cells": cells},
        "identities": {
            "realization": {"fingerprint": "fixed-realization"},
            "scientific_input": {
                "mesh": {
                    "nominal_size": mesh_size,
                    "generator": "synthetic-tetrahedra",
                },
                "material": {"young": 200000.0, "yield_stress": 200.0},
                "macroscopic_deformation_gradient": [
                    [1.004, 0.0, 0.0],
                    [0.0, 0.998, 0.0],
                    [0.0, 0.0, 0.998],
                ],
                "increments": 2,
                "quadrature_degree": 1,
                "solver": {"maximum_iterations": 25},
            },
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
    assert {
        "weighted_sample_count",
        "weighted_measure_matches_homogenization",
        "weighted_statistics_semantics",
    } <= set(policy["threshold_invariants"])
    assert "physical_weighted_statistics" not in policy["threshold_invariants"]


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


def test_multi_void_golden_contract_is_machine_readable_and_self_consistent():
    card = _benchmark_card()
    golden = benchmarks.golden_benchmark(BENCHMARK_ID)
    regression = card["regression_identity"]

    assert card["status"] == (
        "automated_fixed_stack_regression_with_refinement_mpi_restart"
    )
    assert not regression["reference_source"]["tracked_dirty"]
    assert regression["scientific_input_fingerprint"] == (
        provenance.content_fingerprint(regression["scientific_input"])
    )
    assert regression["scientific_input"]["mesh"]["generation_options"] == (
        PERIODIC_VOID_GMSH_OPTIONS
    )
    assert golden.reference_version == (
        "periodic-multi-void-j2-d3f0879a-bf6ae719-v1"
    )
    assert {item.name for item in golden.quantities} == {
        "homogenized_first_piola_stress",
        "peeq_mean",
        "peeq_p95",
        "solid_reference_fraction",
    }


def test_multi_void_fixed_stack_matches_versioned_golden():
    card = _benchmark_card()
    mismatches = _fixed_stack_mismatches(card)
    if mismatches:
        message = "Multi-void fixed-stack Golden is inapplicable: " + "; ".join(
            mismatches
        )
        if os.environ.get("AGENTFEM_REQUIRE_MULTI_VOID_RVE_GOLDEN") == "1":
            pytest.fail(message)
        pytest.skip(message)

    evidence = run_candidate(comm=MPI.COMM_SELF)
    assert evidence["accepted_invariant_gates"]
    regression = card["regression_identity"]
    assert evidence["identities"]["realization"] == regression[
        "scientific_input"
    ]["realization"]
    assert evidence["identities"]["portable_mesh"] == regression["mesh_identity"]
    assert evidence["identities"]["scientific_input"] == regression[
        "scientific_input"
    ]
    assert evidence["identities"]["scientific_input_fingerprint"] == regression[
        "scientific_input_fingerprint"
    ]

    actual = {
        "homogenized_first_piola_stress": np.asarray(
            evidence["quantities"]["homogenized_first_piola_stress"], dtype=float
        ).reshape(-1),
        "peeq_mean": evidence["quantities"]["peeq_mean"],
        "peeq_p95": evidence["quantities"]["peeq_p95"],
        "solid_reference_fraction": evidence["quantities"][
            "solid_reference_fraction"
        ],
    }
    for quantity in benchmarks.golden_benchmark(BENCHMARK_ID).quantities:
        quantity.assert_accepts(actual[quantity.name])


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


def test_refinement_comparison_rejects_changed_physics_or_numerics():
    levels = [
        _synthetic_level(
            mesh_size=size,
            cells=cells,
            macro=100.0,
            mean=0.003,
            p95=0.0032,
            p99=0.0035,
            maximum=0.004,
            geometry=0.004,
        )
        for size, cells in ((0.20, 100), (0.16, 200), (0.12, 400))
    ]
    mutations = (
        ("material", lambda record: record["material"].update(young=210000.0)),
        (
            "loading",
            lambda record: record["macroscopic_deformation_gradient"][0].__setitem__(
                0, 1.01
            ),
        ),
        ("increments", lambda record: record.update(increments=4)),
        ("quadrature", lambda record: record.update(quadrature_degree=2)),
        (
            "solver",
            lambda record: record["solver"].update(maximum_iterations=40),
        ),
    )
    for _name, mutate in mutations:
        incompatible = copy.deepcopy(levels[-1])
        mutate(incompatible["identities"]["scientific_input"])
        with pytest.raises(ValueError, match="mesh-independent"):
            compare_refinement_evidence(levels[0], levels[1], incompatible)


def test_rank_comparison_is_machine_readable_and_fails_closed():
    serial = _synthetic_level(
        mesh_size=0.16,
        cells=200,
        macro=100.0,
        mean=0.003,
        p95=0.0032,
        p99=0.0035,
        maximum=0.004,
        geometry=0.004,
    )
    serial["execution"]["rank_count"] = 1
    serial["identities"].update(
        {
            "portable_mesh": {"hash": "mesh"},
            "constraint": {"fingerprint": "constraint"},
            "scientific_input_fingerprint": "sha256:input",
        }
    )
    parallel = copy.deepcopy(serial)
    parallel["execution"]["rank_count"] = 2
    parallel["quantities"]["homogenized_first_piola_stress"][0][0] += 1.0e-11
    parallel["quantities"]["peeq_mean"] += 1.0e-13

    certificate = compare_rank_evidence(serial, parallel)
    assert certificate["accepted"]
    assert certificate["gates"] == {
        "serial_is_one_rank": True,
        "parallel_has_multiple_ranks": True,
        "all_invariant_gates": True,
        "identities": True,
        "macroscopic_first_piola": True,
        "weighted_and_geometric_scalars": True,
    }

    wrong_mesh = copy.deepcopy(parallel)
    wrong_mesh["identities"]["portable_mesh"]["hash"] = "different"
    assert not compare_rank_evidence(serial, wrong_mesh)["accepted"]

    drifting = copy.deepcopy(parallel)
    drifting["quantities"]["peeq_p95"] += 1.0e-3
    assert not compare_rank_evidence(serial, drifting)["accepted"]


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
