"""Tolerance-normalized development objective; does not change admission gates.

This is a deterministic engineering tradeoff, not a noise likelihood. Unresolved
observations are reported separately and never silently counted as passes.
"""

import numpy as np


def residuals(report):
    values, unresolved, keys = [], [], []
    for item in report["checks"]:
        key = (item["name"], item.get("cycle"), item.get("segment"))
        if item["status"] == "unresolved":
            unresolved.append(key)
            continue
        if "error" not in item:
            if item["status"] != "passed":
                raise ValueError("Incomplete or invalid response cannot enter identification")
            continue
        error, limit = item["error"], item["limit"]
        if not np.isfinite([error, limit]).all() or error < 0 or limit <= 0:
            raise ValueError("Require finite nonnegative error and positive limit")
        values.append(error/limit)
        keys.append(key)
    if not values:
        raise ValueError("No resolved response checks")
    normalized = np.asarray(values)
    # L4 emphasizes violations without an arbitrary protocol-specific weight.
    # Convergence of this scalar objective is NOT simultaneous feasibility.
    return normalized**2, dict(keys=keys, unresolved=unresolved,
        maximum_normalized_error=float(normalized.max()),
        failed_count=int(np.count_nonzero(normalized > 1.)))
