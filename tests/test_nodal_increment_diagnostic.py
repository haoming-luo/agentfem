import json
import numpy as np
import pytest
from agentfem.time._nodal_inertia import diagnose_nodal_increment


def test_additive_stiffness_not_minimum_of_separate_limits():
    rows = np.full((2, 3), 4.0)
    result = diagnose_nodal_increment(
        np.ones(2),
        stiffness_rows={"body": rows, "bond": rows},
        requested_increment=1.0,
        safety_factor=1.0,
    )
    assert result["selected_increment"] == pytest.approx(1 / np.sqrt(2))
    assert result["nodes_exceeding_supplied_bound"] == 2
    assert result["controllers"][0]["contribution_demand_ratios"] == {
        "body": 1.0,
        "bond": 1.0,
    }
    assert not result["nonlinear_path_certified"]
    json.dumps(result, allow_nan=False)


def test_damping_and_zero_modes():
    k = np.zeros((2, 2))
    c = np.array([[2.0, 0.0], [0.0, 0.0]])
    result = diagnose_nodal_increment(
        np.ones(2),
        stiffness_rows={"body": k},
        damping_rows={"viscosity": c},
        requested_increment=0.5,
        safety_factor=0.8,
    )
    assert result["selected_increment"] == pytest.approx(0.8)
    assert result["unbounded_nodes"] == 1
    assert result["requested_within_supplied_bound"]
    json.dumps(result, allow_nan=False)


def test_all_zero_is_explicitly_unbounded_not_certified():
    result = diagnose_nodal_increment(
        [1.0], stiffness_rows={"body": np.zeros((1, 3))}, requested_increment=1.0
    )
    assert result["selected_increment"] is None
    assert result["controllers"] == []
    assert not result["operator_coverage_certified"]
    json.dumps(result, allow_nan=False)


def test_damped_root_and_no_input_mutation():
    mass = np.array([2.0, 1.0])
    k = np.array([[3.0, 4.0], [9.0, 2.0]])
    c = np.ones_like(k)
    before = [a.copy() for a in (mass, k, c)]
    result = diagnose_nodal_increment(
        mass,
        stiffness_rows={"body": k},
        damping_rows={"damping": c},
        requested_increment=0.1,
        safety_factor=0.8,
    )
    for item in result["controllers"]:
        n, j = item["node_index"], item["component"]
        h = item["selected_increment"] / 0.8
        assert h * h * k[n, j] + 2 * h * c[n, j] == pytest.approx(4 * mass[n])
    for actual, expected in zip((mass, k, c), before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_bad_top_count(bad):
    with pytest.raises(ValueError):
        diagnose_nodal_increment(
            [1.0],
            stiffness_rows={"k": np.ones((1, 3))},
            requested_increment=0.1,
            top_count=bad,
        )


def test_nonfinite_input_rejected():
    with pytest.raises(ValueError):
        diagnose_nodal_increment(
            [1.0],
            stiffness_rows={"k": np.full((1, 3), np.nan)},
            requested_increment=0.1,
        )


def test_roundoff_at_planned_boundary_is_not_scientific_failure_or_pass():
    result = diagnose_nodal_increment(
        [1.0],
        stiffness_rows={"k": np.full((1, 3), 4.0)},
        requested_increment=1.0,
        safety_factor=1.0,
    )
    assert result["classification"] == "roundoff_boundary"
    assert result["roundoff_boundary_nodes"] == 1
    assert not result["nonlinear_path_certified"]
