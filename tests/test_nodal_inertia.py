# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem.time._nodal_inertia import plan_nodal_inertia


def test_noncommuting_stiffness_damping_amplification_is_stable():
    rng = np.random.default_rng(942)
    for _ in range(30):
        a, b, c = rng.normal(size=(3, 6, 6))
        k1, k2, damping = a.T @ a, b.T @ b, c.T @ c
        physical = np.array([0.02, 0.1])
        dt = 0.2
        rows = lambda matrix: np.abs(matrix).sum(axis=1).reshape(2, 3)
        p = plan_nodal_inertia(
            physical,
            stiffness_rows={"body": rows(k1), "contact": rows(k2)},
            damping_rows={"viscosity": rows(damping)},
            target_increment=dt,
        )
        mass = np.repeat(p.numerical_nodal_mass, 3)
        k, d = (k1 + k2) / mass[:, None], damping / mass[:, None]
        amplification = np.block(
            [
                [2 * np.eye(6) - dt * dt * k - dt * d, -np.eye(6) + dt * d],
                [np.eye(6), np.zeros((6, 6))],
            ]
        )
        assert max(abs(np.linalg.eigvals(amplification))) <= 1 + 1e-12
        # Stronger independent Loewner check of the actual combined condition.
        h = dt / 0.8
        assert (
            np.linalg.eigvalsh(
                4 * np.diag(mass) - h * h * (k1 + k2) - 2 * h * damping
            ).min()
            >= -1e-12
        )
        np.testing.assert_array_equal(physical, [0.02, 0.1])


def test_additive_sources_damping_and_existing_mass_are_not_discarded():
    physical = np.ones(2)
    rows = np.full((2, 3), 100.0)
    p = plan_nodal_inertia(
        physical,
        stiffness_rows={"body": rows, "contact": rows},
        damping_rows={"damping": rows},
        target_increment=1.0,
        safety_factor=1.0,
        minimum_nodal_factors=[200.0, 1.0],
    )
    np.testing.assert_array_equal(p.nodal_factors, [200.0, 100.0])
    assert not p.nodal_factors.flags.writeable
    assert p.summary()["nonlinear_path_certified"] is False
    assert p.summary()["added_mass_ratio"] == 149.0


def test_zero_rows_no_artificial_mass_and_uniform_time_mass_relation():
    zero = np.zeros((2, 2))
    p = plan_nodal_inertia([1, 2], stiffness_rows={"zero": zero}, target_increment=1)
    np.testing.assert_array_equal(p.nodal_factors, [1, 1])
    a = plan_nodal_inertia(
        [1, 2], stiffness_rows={"K": zero + 1000}, target_increment=1
    )
    b = plan_nodal_inertia(
        [1, 2], stiffness_rows={"K": zero + 1000}, target_increment=2
    )
    np.testing.assert_allclose(b.numerical_nodal_mass, 4 * a.numerical_nodal_mass)


@pytest.mark.parametrize(
    "options",
    [
        {"stiffness_rows": {}},
        {"stiffness_rows": {"k": np.ones((2, 4))}},
        {"stiffness_rows": {"k": np.ones((2, 3)) * np.nan}},
        {"stiffness_rows": {"k": -np.ones((2, 3))}},
        {"damping_rows": {"k": np.ones((2, 3))}},
        {"damping_rows": {"d": np.ones((2, 2))}},
        {"target_increment": True},
        {"target_increment": 0},
        {"target_increment": 1e300},
        {"safety_factor": 2},
        {"minimum_nodal_factors": [0.5, 1]},
    ],
)
def test_invalid_or_unattributed_inputs_reject(options):
    kwargs = {"stiffness_rows": {"k": np.ones((2, 3))}, "target_increment": 1.0}
    kwargs.update(options)
    with pytest.raises(ValueError):
        plan_nodal_inertia([1, 1], **kwargs)
