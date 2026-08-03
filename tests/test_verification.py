from __future__ import annotations

import numpy as np
import pytest

from agentfem import verification


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
