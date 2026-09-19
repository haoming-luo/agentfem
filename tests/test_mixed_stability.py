# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from agentfem import diagnostics


def test_discrete_inf_sup_reports_normalized_rank_and_beta():
    evidence = diagnostics.discrete_inf_sup(
        np.diag([2.0, 0.5]),
        primal_norm=np.eye(2),
        multiplier_norm=np.eye(2),
    )

    assert evidence.singular_values == pytest.approx((2.0, 0.5))
    assert evidence.beta == pytest.approx(0.5)
    assert evidence.rank == 2
    assert evidence.full_row_rank is True
    assert evidence.condition_number == pytest.approx(4.0)
    assert "refinement sequence" in evidence.as_dict()["interpretation"]


def test_discrete_inf_sup_is_invariant_to_consistent_basis_changes():
    coupling = np.array([[2.0, -1.0, 0.5], [0.3, 1.2, -0.7]])
    primal = np.array([[3.0, 0.2, 0.1], [0.2, 2.0, -0.1], [0.1, -0.1, 1.5]])
    multiplier = np.array([[1.5, 0.2], [0.2, 0.9]])
    baseline = diagnostics.discrete_inf_sup(
        coupling,
        primal_norm=primal,
        multiplier_norm=multiplier,
    )
    primal_basis = np.array([[1.0, 0.2, 0.0], [0.1, 1.3, -0.2], [0.0, 0.3, 0.8]])
    multiplier_basis = np.array([[1.2, -0.1], [0.25, 0.9]])
    transformed = diagnostics.discrete_inf_sup(
        multiplier_basis.T @ coupling @ primal_basis,
        primal_norm=primal_basis.T @ primal @ primal_basis,
        multiplier_norm=multiplier_basis.T @ multiplier @ multiplier_basis,
    )

    np.testing.assert_allclose(
        transformed.singular_values,
        baseline.singular_values,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert transformed.beta == pytest.approx(baseline.beta, rel=1.0e-13)


def test_discrete_inf_sup_fails_closed_on_rank_and_norm_defects():
    deficient = diagnostics.discrete_inf_sup(
        [[1.0, 0.0], [2.0, 0.0]],
        primal_norm=np.eye(2),
        multiplier_norm=np.eye(2),
    )
    assert deficient.full_row_rank is False
    assert deficient.beta == 0.0
    assert np.isinf(deficient.condition_number)

    with pytest.raises(ValueError, match="positive definite"):
        diagnostics.discrete_inf_sup(
            [[1.0]],
            primal_norm=[[0.0]],
            multiplier_norm=[[1.0]],
        )
    with pytest.raises(ValueError, match="nonnegative"):
        diagnostics.discrete_inf_sup(
            [[1.0]],
            primal_norm=[[1.0]],
            multiplier_norm=[[1.0]],
            rank_tolerance=-1.0,
        )


def _sample(size, beta, *, label=""):
    evidence = diagnostics.discrete_inf_sup(
        np.diag([2.0, beta]),
        primal_norm=np.eye(2),
        multiplier_norm=np.eye(2),
    )
    return diagnostics.DiscreteInfSupSample(size, evidence, label=label)


def test_discrete_inf_sup_study_records_a_nondecaying_refinement_family():
    study = diagnostics.DiscreteInfSupStudy(
        "mixed shell family stability",
        (
            _sample(0.4, 0.50, label="coarse"),
            _sample(0.2, 0.48, label="medium"),
            _sample(0.1, 0.47, label="fine"),
        ),
    )

    assert study.minimum_beta == pytest.approx(0.47)
    assert study.finest_beta == pytest.approx(0.47)
    assert study.beta_range_ratio == pytest.approx(0.94)
    assert study.endpoint_decay_order == pytest.approx(
        np.log(0.47 / 0.50) / np.log(0.1 / 0.4)
    )
    claim = study.verify(minimum_beta=0.4, maximum_decay_order=0.1)
    assert claim.status == "passed"
    assert claim.evidence["sample_count"] == 3


def test_discrete_inf_sup_study_rejects_decay_and_rank_loss():
    decaying = diagnostics.DiscreteInfSupStudy(
        "decaying family",
        (
            _sample(0.4, 0.4),
            _sample(0.2, 0.2),
            _sample(0.1, 0.1),
        ),
    )
    assert decaying.endpoint_decay_order == pytest.approx(1.0)
    assert decaying.verify(
        minimum_beta=0.05,
        maximum_decay_order=0.2,
    ).status == "failed"

    deficient = diagnostics.discrete_inf_sup(
        [[1.0, 0.0], [2.0, 0.0]],
        primal_norm=np.eye(2),
        multiplier_norm=np.eye(2),
    )
    rank_loss = diagnostics.DiscreteInfSupStudy(
        "rank loss",
        (
            _sample(0.4, 0.5),
            _sample(0.2, 0.5),
            diagnostics.DiscreteInfSupSample(0.1, deficient),
        ),
    )
    assert rank_loss.minimum_beta == 0.0
    assert rank_loss.endpoint_decay_order is None
    assert rank_loss.verify(minimum_beta=0.0).status == "failed"
    assert rank_loss.verify(
        minimum_beta=0.0,
        maximum_decay_order=0.1,
    ).status == "inconclusive"


def test_discrete_inf_sup_study_validates_sequence_and_thresholds():
    with pytest.raises(ValueError, match="at least three"):
        diagnostics.DiscreteInfSupStudy(
            "too short",
            (_sample(0.2, 0.5), _sample(0.1, 0.5)),
        )
    with pytest.raises(ValueError, match="coarse-to-fine"):
        diagnostics.DiscreteInfSupStudy(
            "unordered",
            (_sample(0.2, 0.5), _sample(0.1, 0.5), _sample(0.15, 0.5)),
        )
    study = diagnostics.DiscreteInfSupStudy(
        "valid",
        (_sample(0.4, 0.5), _sample(0.2, 0.5), _sample(0.1, 0.5)),
    )
    with pytest.raises(ValueError, match="minimum_beta"):
        study.verify(minimum_beta=-1.0)
    with pytest.raises(ValueError, match="maximum_decay_order"):
        study.verify(minimum_beta=0.1, maximum_decay_order=-1.0)
