# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from agentfem.time._element_inertia import plan_element_inertia


def test_selective_scaling_assembled_dense_spectrum_and_mass_conservation():
    rng = np.random.default_rng(73)
    nodes = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 0]])
    mass = rng.uniform(0.1, 1, nodes.shape)
    operators = []
    bounds = []
    for m in mass:
        a = rng.normal(size=(3, 3))
        k = a.T @ a
        operators.append(k)
        bounds.append(np.linalg.eigvalsh(k / np.sqrt(m[:, None] * m[None, :]))[-1])
    target = 0.6
    plan = plan_element_inertia(
        nodes, mass, np.array(bounds), target_body_increment=target
    )
    global_k = np.zeros((6, 6))
    for ids, k in zip(nodes, operators):
        global_k[np.ix_(ids, ids)] += k
    exact = np.linalg.eigvalsh(
        global_k
        / np.sqrt(
            plan.numerical_nodal_mass[:, None] * plan.numerical_nodal_mass[None, :]
        )
    )[-1]
    assert exact <= (2 * 0.8 / target) ** 2 * (1 + 1e-14)
    assert plan.physical_nodal_mass.sum() == pytest.approx(mass.sum())
    assert plan.numerical_nodal_mass.sum() == pytest.approx(
        (mass * plan.element_factors[:, None]).sum()
    )
    assert plan.summary()["whole_system_stability_certificate"] is False


def test_no_mass_reduction_and_only_controlling_elements_scale():
    nodes = np.array([[0, 1], [1, 2]])
    plan = plan_element_inertia(
        nodes,
        np.ones((2, 2)),
        np.array([1.0, 100.0]),
        target_body_increment=0.2,
        safety_factor=1.0,
    )
    np.testing.assert_allclose(plan.element_factors, [1.0, 1.0])
    raised = plan_element_inertia(
        nodes,
        np.ones((2, 2)),
        np.array([1.0, 400.0]),
        target_body_increment=0.2,
        safety_factor=1.0,
        minimum_nodal_factors=np.array([5.0, 1.0, 1.0]),
    )
    np.testing.assert_allclose(raised.element_factors, [1.0, 4.0])
    np.testing.assert_allclose(raised.nodal_factors, [5.0, 2.5, 4.0])


@pytest.mark.parametrize(
    "kind", ["negative_bound", "negative_mass", "gap", "complex", "zero_dt", "overflow"]
)
def test_invalid_element_plan_fails(kind):
    nodes = np.array([[0, 1]])
    mass = np.ones((1, 2))
    bounds = np.array([2.0])
    dt = 0.1
    if kind == "negative_bound":
        bounds[0] = -1
    if kind == "negative_mass":
        mass[0, 0] = -1
    if kind == "gap":
        nodes[0, 1] = 1000000000
    if kind == "complex":
        bounds = bounds.astype(complex) + 1j
    if kind == "zero_dt":
        dt = 0
    if kind == "overflow":
        dt = 1e-300
    with pytest.raises(ValueError):
        plan_element_inertia(nodes, mass, bounds, target_body_increment=dt)
