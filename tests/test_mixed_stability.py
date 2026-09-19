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
