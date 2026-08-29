"""Evidence contracts for the Lewandowski et al. external beam route."""

from __future__ import annotations

import numpy as np
import pytest

from lewandowski_2023_self_weight_beam_fixture import (
    DEFINITION,
    REQUIRED_PROMOTION_EVIDENCE,
    UPSTREAM_ARTIFACTS,
    UPSTREAM_BEHAVIOUR_SHA256,
    UPSTREAM_COMMIT,
    UPSTREAM_SOLVER_SHA256,
    assess_external_curve,
)


@pytest.mark.skip(
    reason=(
        "External promotion gate: the pinned legacy FEniCS/MGIS/MFront source "
        "has not been independently re-executed in the current environment, "
        "and the publication provides no tabulated load-deflection oracle."
    )
)
def test_lewandowski_2023_external_numerical_promotion_gate():
    """Remove this skip only with archived independent execution evidence."""

    pytest.fail("Lewandowski 2023 external curve has not been promoted.")


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
