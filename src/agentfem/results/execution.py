# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Bridge structured solver events into scientific result evidence."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def execution_records(events: Iterable[object]) -> tuple[dict[str, object], ...]:
    """Normalize solver events without depending on a particular procedure."""

    records = []
    for event in events:
        if hasattr(event, "as_dict"):
            records.append(event.as_dict())
        elif isinstance(event, dict):
            records.append(dict(event))
        else:
            raise TypeError(
                "Execution events must provide as_dict() or be mappings."
            )
    return tuple(records)


def add_execution_trace(result, events: Iterable[object], *, time_unit="s") -> tuple[dict[str, object], ...]:
    """Attach one complete execution trace and its standard histories.

    ``time_unit`` labels physical time without converting event values. The
    default preserves the historical seconds convention; transient Results
    supply the unit declared by their model or history requests.

    The event list includes failed attempts and hidden progress frames.  Only
    accepted increments are projected into monotone histories suitable for
    plotting and datasets; the complete trace remains in metadata for audit.
    """

    records = execution_records(events)
    result.metadata["execution"] = {
        "schema": "agentfem.solve-events",
        "schema_version": "0.1.0",
        "event_count": len(records),
        "events": list(records),
    }

    transient = [item for item in records if item["kind"] == "time_increment"]
    if transient:
        times = np.asarray([item["time"] for item in transient], dtype=float)
        result.add_history(
            "accepted_increment",
            times,
            np.asarray([item["increment"] for item in transient], dtype=float),
            abscissa_name="time",
            abscissa_unit=time_unit,
            description="All accepted time-integration increments.",
        )
        result.add_history(
            "time_increment",
            times,
            np.diff(np.concatenate(([0.0], times))),
            unit=time_unit,
            abscissa_name="time",
            abscissa_unit=time_unit,
            description="Accepted physical time increment at each step.",
        )

    accepted = [item for item in records if item["kind"] == "increment_converged"]
    if accepted:
        factors = np.asarray([item["target_factor"] for item in accepted], dtype=float)
        result.add_histories(
            factors,
            {
                "newton_residual": [item["residual_norm"] for item in accepted],
                "newton_iterations": [item["iteration"] for item in accepted],
                "increment_size": [
                    item["target_factor"] - item["start_factor"]
                    for item in accepted
                ],
            },
            abscissa_name="load_factor",
            abscissa_unit=None,
            descriptions={
                "newton_residual": "Equilibrium residual at accepted convergence.",
                "newton_iterations": "Iterations in each accepted increment.",
                "increment_size": "Accepted normalized load increment.",
            },
        )
    return records
