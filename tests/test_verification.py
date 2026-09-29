from __future__ import annotations

import numpy as np
import pytest

from agentfem import constraints, results, verification


def test_reference_claim_does_not_confuse_inapplicable_theory_with_failure():
    applicable = verification.VerificationClaim.compare(
        name="slender_beam_displacement",
        observable="tip_displacement",
        actual=1.01,
        expected=1.0,
        reference="Euler-Bernoulli beam theory",
        relative_tolerance=0.02,
        validity_domain="slender beam with negligible shear deformation",
    )
    outside_domain = verification.VerificationClaim.compare(
        name="deep_beam_displacement",
        observable="tip_displacement",
        actual=1.8,
        expected=1.0,
        reference="Euler-Bernoulli beam theory",
        relative_tolerance=0.02,
        validity_domain="slender beam with negligible shear deformation",
        applicable=False,
    )

    assert applicable.status == "passed"
    assert outside_domain.status == "inconclusive"
    assert verification.report(applicable).trust_level == "verified"
    assert verification.report(outside_domain).trust_level == "converged"


def test_convergence_study_requires_a_coarse_to_fine_sequence():
    samples = (
        verification.ConvergenceSample(0.25, 0.75, label="coarse"),
        verification.ConvergenceSample(0.125, 0.9375, label="medium"),
        verification.ConvergenceSample(0.0625, 0.984375, label="fine"),
    )
    study = verification.convergence_study(
        "hole_peak_stress_convergence",
        "peak_circumferential_stress",
        samples,
    )
    claim = study.verify(
        maximum_relative_change=0.05,
        minimum_observed_order=1.9,
    )

    assert claim.status == "passed"
    assert study.observed_order == pytest.approx(2.0)
    assert study.finest_relative_change < 0.05
    assert claim.evidence["characteristic_sizes"] == [0.25, 0.125, 0.0625]

    with pytest.raises(ValueError, match="coarse-to-fine"):
        verification.convergence_study(
            "wrong_order",
            "stress",
            tuple(reversed(samples)),
        )


def test_nonuniform_refinement_is_inconclusive_when_order_is_required():
    study = verification.ConvergenceStudy(
        name="nonuniform",
        observable="response",
        samples=(
            verification.ConvergenceSample(0.3, np.array([1.0, 2.0])),
            verification.ConvergenceSample(0.1, np.array([1.1, 2.1])),
            verification.ConvergenceSample(0.04, np.array([1.11, 2.11])),
        ),
    )

    claim = study.verify(
        maximum_relative_change=0.02,
        minimum_observed_order=1.0,
    )

    assert study.observed_order is None
    assert claim.status == "inconclusive"


def test_verification_report_requires_every_claim_before_trust_advances():
    passed = verification.VerificationClaim.compare(
        name="reference",
        observable="qoi",
        actual=1.0,
        expected=1.0,
        reference="closed form",
    )
    failed = verification.VerificationClaim.compare(
        name="mesh_independence",
        observable="qoi",
        actual=1.2,
        expected=1.0,
        reference="refined solution",
        relative_tolerance=0.05,
    )

    accepted = verification.report(passed)
    rejected = verification.report(passed, failed)

    accepted.require("verified")
    assert accepted.acceptable
    assert rejected.trust_level == "converged"
    with pytest.raises(RuntimeError, match="mesh_independence"):
        rejected.require("verified")


def test_engineering_quality_is_easy_but_does_not_claim_scientific_verification():
    result = results.SimulationResult(
        "linear_case",
        metadata={
            "solve": {
                "kind": "linear_solve_info",
                "converged": True,
                "converged_reason": 2,
            }
        },
    )
    result.add_quantity("maximum_displacement", 1.2e-3, unit="m")

    report = result.verify(
        "engineering",
        required_quantities=("maximum_displacement",),
    )

    assert report.acceptable
    assert report.trust_level == "converged"
    assert report.quality_policy == "engineering"
    assert all(item.kind == "runtime" for item in report.claims)
    report.require()


def test_release_quality_requires_real_scientific_evidence():
    result = results.SimulationResult(
        "release_case",
        metadata={"solve": {"converged": True}},
    )
    result.add_quantity("response", 2.0)

    incomplete = result.verify("release")
    assert incomplete.trust_level == "converged"
    assert not incomplete.acceptable
    with pytest.raises(RuntimeError, match="below required"):
        incomplete.require()

    golden = verification.VerificationClaim.compare(
        name="versioned_response",
        observable="response",
        actual=2.0,
        expected=2.0,
        reference="release Golden v1",
    )
    accepted = result.verify("release", claims=(golden,))

    assert accepted.trust_level == "verified"
    assert accepted.acceptable
    accepted.require()


def test_quality_contract_reports_missing_outputs_without_hiding_them():
    result = results.SimulationResult(
        "incomplete_output",
        metadata={"solve": {"converged": True}},
    )
    result.add_quantity("available", 1.0)

    report = result.verify(
        "engineering",
        required_quantities=("required",),
    )

    failed = {item.name: item for item in report.claims if item.status == "failed"}
    assert "required_outputs_present" in failed
    assert not report.acceptable
    with pytest.raises(RuntimeError, match="required_outputs_present"):
        report.require()


def test_runtime_verification_consumes_typed_constraint_work_evidence():
    class AffineProvider:
        name = "affine_periodic"

        @staticmethod
        def capabilities():
            return constraints.ConstraintCapabilities(
                kind="periodic_constraint",
                enforcement="exact_affine_reduction",
                reaction_evidence="provider_dual_required",
                work_evidence="provider_dual_path_required",
            )

    provider = AffineProvider()
    dual = constraints.constraint_dual(
        provider,
        force=(10.0,),
        resultant=(0.0, 0.0),
        source="full_residual_virtual_work",
    )
    work = constraints.constraint_work(
        provider,
        value=0.5,
        integration="accepted_path_trapezoidal",
        sample_count=3,
        source="accepted_affine_path",
    )
    contract = constraints.constraint_balance_contract(
        (provider,),
        provider_duals=(dual,),
        provider_work=(work,),
    )
    result = results.SimulationResult(
        "constraint_evidence",
        metadata={
            "solve": {"converged": True},
            "constraint_balance_contract": contract,
            "constraint_duals": contract["provider_duals"],
            "constraint_work_evidence": contract["provider_work"],
        },
    )
    result.add_quantity("response", 1.0)

    report = result.verify("engineering")
    claim = next(
        item for item in report.claims
        if item.name == "constraint_evidence_consistent"
    )

    assert claim.status == "passed"
    assert claim.actual["provider_work_count"] == 1


def test_runtime_verification_rejects_serialized_constraint_work_drift():
    class AffineProvider:
        name = "affine_periodic"

        @staticmethod
        def capabilities():
            return constraints.ConstraintCapabilities(
                kind="periodic_constraint",
                enforcement="exact_affine_reduction",
                reaction_evidence="provider_dual_required",
                work_evidence="provider_dual_path_required",
            )

    provider = AffineProvider()
    dual = constraints.constraint_dual(
        provider,
        force=(10.0,),
        resultant=(0.0, 0.0),
    )
    work = constraints.constraint_work(
        provider,
        value=0.5,
        integration="accepted_path_trapezoidal",
        sample_count=3,
    )
    contract = constraints.constraint_balance_contract(
        (provider,),
        provider_duals=(dual,),
        provider_work=(work,),
    )
    changed = dict(contract["provider_work"][0])
    changed["value"] = 0.75
    result = results.SimulationResult(
        "constraint_evidence_drift",
        metadata={
            "solve": {"converged": True},
            "constraint_balance_contract": contract,
            "constraint_duals": contract["provider_duals"],
            "constraint_work_evidence": (changed,),
        },
    )
    result.add_quantity("response", 1.0)

    report = result.verify("engineering")
    claim = next(
        item for item in report.claims
        if item.name == "constraint_evidence_consistent"
    )

    assert claim.status == "failed"
    assert "serialized provider work differs" in claim.actual["issues"][0]
