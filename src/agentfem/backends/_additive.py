# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""PETSc algebra for assembled-local plus matrix-free tangent operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from petsc4py import PETSc


TangentAction = Callable[[PETSc.Vec, PETSc.Vec], None]


class _AdditiveTangentContext:
    """Python ``Mat`` context implementing one additive tangent action."""

    def __init__(
        self,
        local_matrix: PETSc.Mat,
        actions: tuple[TangentAction, ...],
        constrained_local_dofs: np.ndarray,
    ) -> None:
        self.local_matrix = local_matrix
        self.actions = actions
        self.constrained_local_dofs = constrained_local_dofs
        self._projected = local_matrix.createVecRight()
        self._work = local_matrix.createVecLeft()

    def mult(self, matrix: PETSc.Mat, source: PETSc.Vec, target: PETSc.Vec) -> None:
        """Apply the local matrix and every declared matrix-free action."""

        del matrix
        self.local_matrix.mult(source, target)
        if not self.actions:
            return
        source.copy(self._projected)
        if self.constrained_local_dofs.size:
            self._projected.array[self.constrained_local_dofs] = 0.0
        for action in self.actions:
            self._work.set(0.0)
            returned = action(self._projected, self._work)
            if returned is not None:
                raise TypeError(
                    "A matrix-free tangent action must write into its target "
                    "PETSc Vec and return None."
                )
            if self.constrained_local_dofs.size:
                self._work.array[self.constrained_local_dofs] = 0.0
            target.axpy(1.0, self._work)

    def close(self) -> None:
        """Release work vectors owned by the Python matrix context."""

        projected = self._projected
        work = self._work
        self._projected = None
        self._work = None
        for vector in (projected, work):
            if vector is not None:
                vector.destroy()


@dataclass
class AdditiveTangentMatrix:
    """Owned shell operator paired with its assembled preconditioner matrix.

    The supplied local and preconditioner matrices remain caller-owned.
    ``close()`` destroys only the shell matrix and temporary vectors created
    by this object.
    """

    operator: PETSc.Mat
    preconditioner: PETSc.Mat
    _context: _AdditiveTangentContext
    action_count: int
    constrained_local_dofs: tuple[int, ...]
    _closed: bool = False

    @property
    def closed(self) -> bool:
        return self._closed

    def summary(self) -> dict[str, object]:
        """Return stable numerical ownership and preconditioning semantics."""

        return {
            "kind": "additive_tangent_matrix",
            "operator": "assembled_local_plus_matrix_free_actions",
            "preconditioner": (
                "assembled_local_matrix"
                if self.preconditioner is self._context.local_matrix
                else "independent_assembled_approximation"
            ),
            "matrix_free_action_count": int(self.action_count),
            "constrained_local_dofs": list(self.constrained_local_dofs),
            "closed": bool(self._closed),
        }

    def close(self) -> None:
        """Release owned PETSc resources without destroying the local matrix."""

        if self._closed:
            return
        operator = self.operator
        context = self._context
        self.operator = None
        self._context = None
        self._closed = True
        first_error = None
        for resource in (operator, context):
            try:
                resource.destroy() if resource is operator else resource.close()
            except Exception as exc:  # pragma: no cover - PETSc failure path
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self):
        if self._closed:
            raise RuntimeError("AdditiveTangentMatrix is closed.")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def create_additive_tangent_matrix(
    local_matrix: PETSc.Mat,
    actions: Iterable[TangentAction] = (),
    *,
    preconditioner_matrix: PETSc.Mat | None = None,
    constrained_local_dofs: Iterable[int] = (),
) -> AdditiveTangentMatrix:
    """Create ``A = A_local + sum(A_nonlocal)`` with an assembled ``P``.

    Each action receives a projected input vector and a zeroed output vector.
    It must overwrite the output and return ``None``.  Constrained local input
    entries are zeroed for matrix-free actions, and their output rows are
    suppressed.  The assembled local matrix therefore remains the sole owner
    of essential-boundary identity rows.  By default ``P`` is ``A_local``.
    A distinct assembled approximation may be supplied when the local physics
    alone is singular in directions stabilized by a nonlocal contribution.
    It changes preconditioning only, never the true operator action.
    """

    if not isinstance(local_matrix, PETSc.Mat):
        raise TypeError("local_matrix must be a petsc4py.PETSc.Mat.")
    row_sizes, column_sizes = local_matrix.getSizes()
    if row_sizes != column_sizes:
        raise ValueError("Additive tangent matrices must be square.")
    selected_preconditioner = (
        local_matrix
        if preconditioner_matrix is None
        else preconditioner_matrix
    )
    if not isinstance(selected_preconditioner, PETSc.Mat):
        raise TypeError("preconditioner_matrix must be a petsc4py.PETSc.Mat.")
    if selected_preconditioner.getSizes() != local_matrix.getSizes():
        raise ValueError(
            "preconditioner_matrix must match the additive operator sizes."
        )
    selected_actions = tuple(actions)
    if any(not callable(action) for action in selected_actions):
        raise TypeError("Every matrix-free tangent action must be callable.")
    local_rows = int(row_sizes[0])
    constrained = np.unique(
        np.asarray(tuple(constrained_local_dofs), dtype=np.int64)
    )
    if constrained.ndim != 1:
        raise ValueError("constrained_local_dofs must be one-dimensional.")
    if np.any(constrained < 0) or np.any(constrained >= local_rows):
        raise ValueError(
            "constrained_local_dofs must address locally owned matrix rows."
        )
    context = _AdditiveTangentContext(
        local_matrix,
        selected_actions,
        constrained.astype(np.int32, copy=False),
    )
    operator = PETSc.Mat().createPython(
        local_matrix.getSizes(),
        context=context,
        comm=local_matrix.comm,
    )
    operator.setUp()
    return AdditiveTangentMatrix(
        operator=operator,
        preconditioner=selected_preconditioner,
        _context=context,
        action_count=len(selected_actions),
        constrained_local_dofs=tuple(int(value) for value in constrained),
    )


__all__ = [
    "AdditiveTangentMatrix",
    "TangentAction",
    "create_additive_tangent_matrix",
]
