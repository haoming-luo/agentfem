# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from agentfem import time


def test_time_input_plan_preserves_order_and_combines_effects():
    calls = []
    rhs = time.input_update(
        lambda value: calls.append(("rhs", value)),
        effects="rhs",
        name="force",
        identity={"amplitude": "ramp", "revision": 1},
    )
    output = time.input_update(
        lambda value: calls.append(("output", value)),
        effects="output",
        name="probe",
    )

    plan = time.compose_inputs(rhs, output)
    result = plan(2.5)

    assert result == (None, None)
    assert calls == [("rhs", 2.5), ("output", 2.5)]
    assert plan.changes_operator is False
    assert time.input_summary(plan)["effects"] == (
        "output",
        "right_hand_side",
    )
    assert plan.summary()["restart_identity_bound"] is False
    assert plan.summary()["changes_right_hand_side"] is True
    assert plan.summary()["changes_operator"] is False
    assert plan.summary()["changes_state"] is False
    assert plan.summary()["changes_output"] is True


def test_bare_callback_is_conservative_and_requires_operator_refresh():
    plan = time.compose_inputs(lambda _time: None)

    assert plan.changes_operator is True
    assert time.input_effects(plan) == frozenset(time.TimeInputEffect)
    assert plan.summary()["updates"][0]["declaration"] == "conservative"
    assert plan.summary()["restart_identity_bound"] is False


def test_invalid_or_empty_effect_declarations_fail_early():
    with pytest.raises(ValueError, match="Unknown time-input effect"):
        time.input_update(lambda _time: None, effects="sometimes")
    with pytest.raises(ValueError, match="at least one effect"):
        time.input_update(lambda _time: None, effects=())


@pytest.mark.parametrize("identity", [{"bad": object()}, {"bad": math.nan}])
def test_time_input_identity_must_be_archivable(identity):
    with pytest.raises(ValueError, match="finite JSON data"):
        time.input_update(
            lambda _time: None,
            effects="rhs",
            identity=identity,
        )


def test_explicit_step_reports_matrix_free_time_input_lifecycle():
    from agentfem._transient_problems import ExplicitDynamicsStep

    update = time.input_update(
        lambda _time: None,
        effects="operator",
        identity={"coefficient": "temperature_dependent"},
    )
    step = ExplicitDynamicsStep(
        name="explicit",
        state=SimpleNamespace(),
        integrator=SimpleNamespace(),
        residual=SimpleNamespace(),
        dt=0.1,
        steps=2,
        update_load=update,
        stability=SimpleNamespace(summary=lambda: {"kind": "preflight"}),
    )

    lifecycle = step.operator_lifecycle_summary()

    assert lifecycle["operator_policy"] == "evaluate_residual_each_increment"
    assert lifecycle["prepared_operator_reused"] is False
    assert lifecycle["time_inputs"]["changes_operator"] is True
    assert lifecycle["stability_scope"] == "caller_must_bound_complete_path"


def test_incremental_nonlinear_summary_retains_time_input_lifecycle():
    from agentfem._nonlinear_problems import IncrementalNonlinearVariationalProblem

    update = time.input_update(
        lambda _factor: None,
        effects=("right_hand_side", "output"),
        identity={"load_path": "ramp"},
    )
    step = IncrementalNonlinearVariationalProblem(
        residual_form=object(),
        solution=object(),
        factor=SimpleNamespace(value=0.0),
        value_path=SimpleNamespace(update=lambda _factor: None),
        update_load=update,
    )

    summary = step.summary()

    assert summary["time_inputs"]["effects"] == (
        "output",
        "right_hand_side",
    )
    assert summary["operator_lifecycle"] == {
        "kind": "incremental_nonlinear_operator_lifecycle",
        "time_inputs": summary["time_inputs"],
        "operator_policy": "assemble_residual_and_tangent_each_attempt",
        "prepared_operator_reused": False,
        "rollback_policy": "restore_accepted_load_coordinate",
    }


def test_affine_nonlinear_summary_declares_fixed_input_lifecycle():
    from agentfem._nonlinear_problems import AffineNonlinearVariationalProblem

    step = AffineNonlinearVariationalProblem(
        residual_form=object(),
        jacobian_form=object(),
        solution=SimpleNamespace(name="U"),
        constraint=SimpleNamespace(summary=lambda: {"kind": "affine"}),
    )

    summary = step.summary()

    assert summary["time_inputs"]["effects"] == ()
    assert summary["operator_lifecycle"]["operator_policy"] == (
        "assemble_reduced_residual_and_tangent_each_attempt"
    )
    assert summary["operator_lifecycle"]["prepared_operator_reused"] is False
