# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Private bookkeeping shared by transient Procedures, not a solver backend."""

from dataclasses import dataclass, field

import numpy as np


def boundary_dof_identity(bcs) -> tuple:
    """Snapshot live constrained indices without allocating one Python int per DOF.

    This is an exact, process-local identity, not a portable checkpoint hash.
    Values of prescribed fields are deliberately excluded: they affect the RHS.
    Re-read the arrays each time so changed membership is never hidden by caching.
    """

    records = []
    for bc in bcs:
        dofs, owned = bc.dof_indices()
        indices = np.asarray(dofs)
        records.append(
            (indices.dtype.str, indices.shape, indices.tobytes(), int(owned))
        )
    return tuple(records)


@dataclass
class OperatorLifecycleLedger:
    """Accumulate successful-solve evidence across prepared-system generations.

    Backend snapshots are cumulative *within* one prepared system. They must
    not replace Procedure totals when a released system is prepared again.
    This ledger is execution evidence, not accepted physical history, and is
    intentionally not restored as part of a physical checkpoint.
    """

    _counts: dict[str, int] = field(
        default_factory=lambda: {
            "matrix_assembly_count": 0,
            "rhs_assembly_count": 0,
            "solve_count": 0,
        }
    )
    _previous: dict[str, int] = field(default_factory=dict)
    _iterations_total: int = 0
    _iterations_maximum: int = 0

    def begin_prepared(self) -> None:
        """Start a new backend counter generation, retaining Procedure totals."""

        self._previous.clear()

    def record(self, summary, solve_info, *, accumulate: bool) -> None:
        if summary is None:
            return
        values = {name: int(summary.get(name, 0)) for name in self._counts}
        deltas = {
            name: value if accumulate else value - self._previous.get(name, 0)
            for name, value in values.items()
        }
        if any(value < 0 for value in values.values()) or any(
            delta < 0 for delta in deltas.values()
        ):
            raise ValueError(
                "Operator counters regressed without a new prepared generation."
            )
        iterations = 0 if solve_info is None else int(solve_info.iterations)
        if iterations < 0:
            raise ValueError("Operator iteration count must be nonnegative.")
        for name, delta in deltas.items():
            self._counts[name] += delta
        self._previous = {} if accumulate else values
        if solve_info is not None:
            self._iterations_total += iterations
            self._iterations_maximum = max(self._iterations_maximum, iterations)

    def summary(self) -> dict[str, int]:
        return {
            **self._counts,
            "matrix_refresh_count": max(0, self._counts["matrix_assembly_count"] - 1),
            "ksp_iterations_total": self._iterations_total,
            "ksp_iterations_maximum": self._iterations_maximum,
        }
