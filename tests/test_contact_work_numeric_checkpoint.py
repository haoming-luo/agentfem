"""Columnar encoding must preserve every accepted work station and failure atomicity."""

import copy
import json

import numpy as np
import pytest

from agentfem._checkpoint_arrays import encode_tree, decode_tree
from agentfem.boundary_models.contact_work import PrescribedContactWorkState
from test_contact_work_state import _station


def state(count=4):
    result = PrescribedContactWorkState(identity="tool")
    if count:
        result.initialize(_station(0.0, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0)))
    for i in range(1, count):
        result.begin(
            _station(
                i * 0.01,
                (i * 0.1, 2.0, 3.0),
                (i * 0.003, -i * 0.002, i * 0.001),
                factor=i / count,
            )
        )
        result.commit()
    return result


@pytest.mark.parametrize("count", [0, 1, 4, 1000])
def test_numeric_checkpoint_roundtrip_legacy_equivalence(count):
    original = state(count)
    snapshot = original.checkpoint_snapshot()
    tree, arrays = encode_tree(snapshot)
    assert len(json.dumps(tree)) < 4000  # metadata grows with columns, not time steps
    assert len(arrays) == 8
    restored = PrescribedContactWorkState(identity="tool")
    restored.restore(decode_tree(tree, arrays))
    assert restored.snapshot() == original.snapshot()
    assert restored.summary() == original.summary()
    if count:
        snapshot["accepted"]["generalized_force"][:] = 999
        assert original.snapshot() == restored.snapshot()


@pytest.mark.parametrize(
    "corruption",
    [
        "time_order",
        "nan",
        "wrong_dtype",
        "wrong_length",
        "wrong_vector_shape",
        "fractional_count",
        "negative_count",
        "wrong_identity",
        "missing_column",
    ],
)
def test_corrupt_columnar_history_rejected_before_mutation(corruption):
    original = state()
    before = original.snapshot()
    before_summary = original.summary()
    damaged = copy.deepcopy(original.checkpoint_snapshot())
    columns = damaged["accepted"]
    if corruption == "time_order":
        columns["time"][2] = columns["time"][1]
    elif corruption == "nan":
        columns["generalized_force"][1, 0] = np.nan
    elif corruption == "wrong_dtype":
        columns["time"] = columns["time"].astype(np.float32)
    elif corruption == "wrong_length":
        columns["factor"] = columns["factor"][:-1]
    elif corruption == "wrong_vector_shape":
        columns["generalized_coordinate"] = np.zeros((4, 4))
    elif corruption == "fractional_count":
        columns["active_point_count"] = np.ones(4) * 0.5
    elif corruption == "negative_count":
        columns["invalid_point_count"][2] = -1
    elif corruption == "wrong_identity":
        damaged["identity"] = "another tool"
    elif corruption == "missing_column":
        columns.pop("contact_resultant")
    with pytest.raises(ValueError):
        original.restore(damaged)
    assert original.snapshot() == before
    assert original.summary() == before_summary


def test_trial_cannot_be_checkpointed_and_old_json_still_restores():
    original = state()
    legacy = original.snapshot()
    original.begin(_station(0.05, (1.0, 2.0, 3.0), (0.1, 0.2, 0.3)))
    with pytest.raises(RuntimeError):
        original.checkpoint_snapshot()
    original.restore(legacy)
    assert original.snapshot() == legacy
    original.begin(_station(0.05, (1.0, 2.0, 3.0), (0.1, 0.2, 0.3)))
    original.commit()
    saved = original.checkpoint_snapshot()
    restored = PrescribedContactWorkState(identity="tool")
    restored.restore(saved)
    assert restored.summary() == original.summary()
