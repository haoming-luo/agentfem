# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import numpy as np
import pytest

from agentfem._checkpoint_arrays import contains_arrays, decode_tree, encode_tree


@pytest.mark.parametrize(
    "array",
    [
        np.arange(12, dtype=np.float64).reshape(3, 4).T,
        np.array(1.25),
        np.empty((0, 3)),
        np.array([True, False]),
        np.array([1 + 2j, -3j]),
        np.array([1, 2], dtype=">i4"),
    ],
)
def test_numeric_tree_roundtrip_preserves_shape_dtype_and_values(array):
    tree, arrays = encode_tree(
        {"type": "array", "value": (array, [None, True, "x", 5])}
    )
    restored = decode_tree(tree, arrays)
    result = restored["value"][0]
    np.testing.assert_array_equal(result, array)
    assert result.dtype == array.dtype
    assert result.shape == array.shape
    assert isinstance(restored["value"], tuple)
    assert not np.shares_memory(result, array)
    assert contains_arrays(restored)
    assert not contains_arrays({"no": [1, 2, None]})


@pytest.mark.parametrize(
    "value",
    [
        np.array([np.nan]),
        np.array([object()]),
        np.array(["text"]),
        float("inf"),
        {1: "key"},
        {1, 2},
    ],
)
def test_non_numeric_or_nonfinite_payload_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        encode_tree(value)


@pytest.mark.parametrize(
    "failure", ["shape", "dtype", "missing", "extra", "nan", "tag"]
)
def test_corrupt_array_descriptors_fail_closed(failure):
    tree, arrays = encode_tree(np.ones((2, 3)))
    tree, arrays = deepcopy(tree), deepcopy(arrays)
    if failure == "shape":
        tree["shape"] = [6]
    elif failure == "dtype":
        tree["dtype"] = "<f4"
    elif failure == "missing":
        arrays.clear()
    elif failure == "extra":
        arrays["surprise"] = np.zeros(1)
    elif failure == "nan":
        arrays[tree["name"]][0, 0] = np.nan
    else:
        tree["type"] = "executable"
    with pytest.raises(ValueError):
        decode_tree(tree, arrays)


def test_duplicate_mapping_and_array_references_rejected():
    tree, arrays = encode_tree({"a": np.ones(1)})
    tree["items"].append(deepcopy(tree["items"][0]))
    with pytest.raises(ValueError, match="repeated"):
        decode_tree(tree, arrays)
    tree["items"][-1][0] = "b"
    with pytest.raises(ValueError, match="repeated"):
        decode_tree(tree, arrays)


def test_deep_nesting_rejected():
    value = 1
    for _ in range(66):
        value = [value]
    with pytest.raises(ValueError, match="nesting"):
        encode_tree(value)
