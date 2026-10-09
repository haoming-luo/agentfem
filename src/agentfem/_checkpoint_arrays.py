# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Non-executable, typed tree encoding for optional checkpoint arrays."""

import math

import numpy as np


def contains_arrays(value):
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, dict):
        return any(contains_arrays(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_arrays(item) for item in value)
    return False


def encode_tree(value):
    arrays = {}

    def encode(item, depth):
        if depth > 64:
            raise ValueError("Checkpoint auxiliary nesting exceeds 64 levels.")
        if isinstance(item, np.ndarray):
            if item.dtype.kind not in "biufc" or not np.isfinite(item).all():
                raise ValueError("Checkpoint arrays must have finite numeric values.")
            name = f"array_{len(arrays):06d}"
            arrays[name] = np.array(item, copy=True, order="C", subok=False)
            return {
                "type": "array",
                "name": name,
                "shape": list(item.shape),
                "dtype": item.dtype.str,
            }
        if isinstance(item, np.generic):
            item = item.item()
        if item is None or isinstance(item, (str, bool, int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("Checkpoint scalar must be finite.")
            return {"type": "scalar", "value": item}
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("Checkpoint mapping keys must be strings.")
            return {
                "type": "mapping",
                "items": [[key, encode(val, depth + 1)] for key, val in item.items()],
            }
        if isinstance(item, (tuple, list)):
            return {
                "type": "tuple" if isinstance(item, tuple) else "list",
                "items": [encode(val, depth + 1) for val in item],
            }
        raise TypeError(f"Unsupported checkpoint auxiliary type: {type(item).__name__}")

    return encode(value, 0), arrays


def decode_tree(tree, arrays):
    used = set()

    def decode(node, depth):
        if depth > 64 or not isinstance(node, dict):
            raise ValueError("Invalid checkpoint auxiliary tree.")
        kind = node.get("type")
        if kind == "array":
            if set(node) != {"type", "name", "shape", "dtype"}:
                raise ValueError("Invalid checkpoint array descriptor.")
            name = node["name"]
            if name in used or name not in arrays:
                raise ValueError("Missing or repeated checkpoint array.")
            value = np.asarray(arrays[name])
            if (
                value.dtype.kind not in "biufc"
                or value.dtype.str != node["dtype"]
                or list(value.shape) != node["shape"]
                or not np.isfinite(value).all()
            ):
                raise ValueError(
                    "Checkpoint array shape, dtype or finite-value mismatch."
                )
            used.add(name)
            return value.copy()
        if kind == "scalar" and set(node) == {"type", "value"}:
            value = node["value"]
            if value is None or isinstance(value, (str, bool, int)):
                return value
            if isinstance(value, float) and math.isfinite(value):
                return value
            raise ValueError("Invalid checkpoint scalar.")
        if kind in {"mapping", "tuple", "list"} and set(node) == {"type", "items"}:
            if not isinstance(node["items"], list):
                raise ValueError("Invalid checkpoint sequence.")
            if kind == "mapping":
                result = {}
                for pair in node["items"]:
                    if (
                        not isinstance(pair, list)
                        or len(pair) != 2
                        or not isinstance(pair[0], str)
                        or pair[0] in result
                    ):
                        raise ValueError("Invalid or repeated checkpoint mapping key.")
                    result[pair[0]] = decode(pair[1], depth + 1)
                return result
            values = [decode(item, depth + 1) for item in node["items"]]
            return tuple(values) if kind == "tuple" else values
        raise ValueError("Unknown checkpoint auxiliary node.")

    result = decode(tree, 0)
    if set(arrays) != used:
        raise ValueError("Unreferenced checkpoint arrays are not accepted.")
    return result
