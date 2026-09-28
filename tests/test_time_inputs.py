# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

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
