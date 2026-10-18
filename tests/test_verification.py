from __future__ import annotations

import numpy as np
import pytest

from agentfem import constraints, results, verification


@pytest.mark.parametrize("actual,expected", [([], []), ([], 0.0),
    (np.empty((0, 3)), np.empty((0, 3)))])
def test_reference_comparison_cannot_verify_an_empty_observation(actual, expected):
    claim = verification.VerificationClaim.compare(
        name="empty", observable="response", actual=actual, expected=expected,
        reference="independent reference",
    )
    assert claim.status == "inconclusive"
    assert "empty" in claim.message.lower()
    assert verification.report(claim).trust_level != "verified"


@pytest.mark.parametrize("actual,expected", [
    (np.ones((2, 1)), np.ones(2)), (1.0, np.ones(2)),
    (np.ones(2), np.ones(1)), (np.ones(2), np.ones(3)),
])
def test_reference_comparison_does_not_broadcast_distinct_sample_layouts(actual, expected):
    claim = verification.VerificationClaim.compare(
        name="layout", observable="response", actual=actual, expected=expected,
        reference="independent reference",
    )
    assert claim.status == "failed"
    assert "shape" in claim.message.lower()


def test_reference_comparison_retains_explicit_scalar_constant_reference():
    claim = verification.VerificationClaim.compare(
        name="stress_free", observable="stress", actual=np.zeros((4, 3, 3)),
        expected=0.0, reference="homogeneous free thermal expansion",
    )
    assert claim.status == "passed"


@pytest.mark.parametrize("side", ["actual", "expected"])
def test_reference_comparison_cannot_silently_drop_a_sample_mask(side):
    values = dict(actual=np.ones(2), expected=np.ones(2))
    values[side] = np.ma.array([1.0, 1.0], mask=[False, True])
    with pytest.raises(ValueError, match="common sample"):
        verification.VerificationClaim.compare(
            name="masked", observable="response", reference="independent reference",
            **values,
        )


def test_reference_comparison_retains_evidence_after_caller_reuses_storage():
    actual = np.ones((2, 3))
    expected = actual.copy()
    claim = verification.VerificationClaim.compare(
        name="station", observable="response", actual=actual, expected=expected,
        reference="independent reference",
    )
    actual[:] = 7
    expected[:] = 11
    assert claim.status == "passed"
    np.testing.assert_array_equal(claim.actual, np.ones((2, 3)))
    np.testing.assert_array_equal(claim.expected, np.ones((2, 3)))
    for value in (claim.actual, claim.expected):
        with pytest.raises(ValueError):
            value.flags.writeable = True


@pytest.mark.parametrize("actual,expected", [(1+2j, 1), (1, 1+2j),
    (np.array([1+2j]), np.array([1+3j]))])
def test_reference_comparison_rejects_lossy_complex_conversion(actual, expected):
    with pytest.raises(TypeError, match="real"):
        verification.VerificationClaim.compare(
            name="complex", observable="response", actual=actual, expected=expected,
            reference="independent reference",
        )


@pytest.mark.parametrize("actual,expected", [(np.nan, 0.0), (np.inf, np.inf),
    ([1.0, -np.inf], [1.0, -np.inf])])
def test_reference_comparison_does_not_verify_nonfinite_data(actual, expected):
    claim = verification.VerificationClaim.compare(
        name="finite", observable="response", actual=actual, expected=expected,
        reference="independent reference",
    )
    assert claim.status == "failed"


@pytest.mark.parametrize("name", ["absolute_tolerance", "relative_tolerance"])
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_reference_comparison_checks_tolerances_before_empty_data(name, value):
    with pytest.raises(ValueError, match="tolerances"):
        verification.VerificationClaim.compare(
            name="tolerance", observable="response", actual=[], expected=[],
            reference="independent reference", **{name: value},
        )


def test_outside_domain_remains_inconclusive_for_incompatible_reference_layout():
    claim = verification.VerificationClaim.compare(
        name="domain", observable="response", actual=np.ones(2), expected=np.ones(3),
        reference="independent reference", applicable=False,
    )
    assert claim.status == "inconclusive"


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


@pytest.mark.parametrize(
    "mixed,certified,expected_status",
    [(0, False, "passed"), (2, False, "failed"), (2, True, "passed")],
)
def test_quality_report_exposes_uncertified_sampled_feature_switch(
    mixed, certified, expected_status,
):
    result = results.SimulationResult(
        "bounded_contact_quality",
        metadata={
            "solve": {"converged": True},
            "rigid_contact_point_diagnostics": {
                "constraint": "tool_A",
                "sampled_mixed_feature_slave_faces": mixed,
                "feature_transition_integral_certified": certified,
            },
        },
    )
    result.add_quantity("force", 1.0)
    report = result.verify("engineering")
    claim = next(
        item for item in report.claims
        if item.name == "no_observed_uncertified_contact_feature_transition"
    )
    assert claim.status == expected_status
    assert claim.actual["scope"] == (
        "sampled_warning_not_complete_feature_coverage_proof"
    )
    if expected_status == "failed":
        assert not report.acceptable
        assert report.trust_level == "converged"
        exploratory = result.verify("exploratory")
        assert exploratory.acceptable
        assert all(
            item.name != "no_observed_uncertified_contact_feature_transition"
            for item in exploratory.claims
        )


def test_quality_report_refuses_malformed_multi_pair_feature_evidence():
    result = results.SimulationResult(
        "contact_quality_invalid",
        metadata={
            "solve": {"converged": True},
            "rigid_contact_point_diagnostics": (
                {
                    "constraint": "tool_A",
                    "sampled_mixed_feature_slave_faces": 0,
                    "feature_transition_integral_certified": False,
                },
                {"constraint": "tool_B", "sampled_mixed_feature_slave_faces": -1},
            ),
        },
    )
    result.add_quantity("force", 1.0)
    report = result.verify("engineering")
    claim = next(
        item for item in report.claims
        if item.name == "no_observed_uncertified_contact_feature_transition"
    )
    assert claim.status == "failed"
    assert "invalid feature evidence" in claim.actual["issues"][0]
