# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Serial PETSc row insertion shared by independently integrated contact laws.

This layer owns no geometry, quadrature, contact State or evidence.  The
caller supplies the already certified nodal residual and COO tangent, and
the owning Procedure still performs material/contact assembly before BC
lifting or MPC reduction.
"""

from __future__ import annotations

import numpy as np


def add_contact_nodal_residual(vector, nodal_residual) -> None:
    """Add one finite 3D local contact residual without changing its identity."""

    raw = np.asarray(nodal_residual)
    if raw.dtype.kind not in "fiu":
        raise ValueError("Contact residual needs real finite scalar values.")
    values = np.asarray(raw, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.all(np.isfinite(values))
        or vector.array.shape != (values.size,)
    ):
        raise ValueError("Contact residual does not match finite PETSc local rows.")
    vector.array[:] += values.reshape(-1)


def add_contact_coo_tangent(
    matrix,
    rows,
    columns,
    values,
    *,
    constrained_dofs=(),
) -> None:
    """Add free-row/column COO values to an already assembled serial matrix.

    Full entries must remain available to the owning Procedure for strong-BC
    lifting before this filtered insertion.  Neither this routine nor the
    contact law may replace prescribed rows or silently change MPC reduction.
    """

    from petsc4py import PETSc

    raw_rows = np.asarray(rows)
    raw_columns = np.asarray(columns)
    raw_constrained = np.asarray(constrained_dofs)
    if (
        raw_rows.dtype.kind not in "iu"
        or raw_columns.dtype.kind not in "iu"
        or (raw_constrained.size and raw_constrained.dtype.kind not in "iu")
    ):
        raise ValueError(
            "Contact matrix row and constraint identities must be integers."
        )
    row_ids = np.asarray(raw_rows, dtype=np.int64)
    column_ids = np.asarray(raw_columns, dtype=np.int64)
    entries = np.asarray(values, dtype=float)
    size = matrix.getSize()
    local_size = matrix.getLocalSize()
    if (
        row_ids.ndim != 1
        or row_ids.shape != column_ids.shape
        or row_ids.shape != entries.shape
        or size[0] != size[1]
        or local_size != size
        or np.any(row_ids < 0)
        or np.any(column_ids < 0)
        or np.any(row_ids >= size[0])
        or np.any(column_ids >= size[1])
        or not np.all(np.isfinite(entries))
    ):
        raise ValueError("Contact tangent needs finite serial COO matrix entries.")
    selected = np.asarray(raw_constrained, dtype=np.int64).reshape(-1)
    if np.any(selected < 0) or np.any(selected >= size[0]):
        raise ValueError("Constrained contact DOFs must address the tangent matrix.")
    if not len(row_ids):
        return
    if len(selected):
        free = ~np.isin(row_ids, selected) & ~np.isin(column_ids, selected)
        row_ids, column_ids, entries = (
            row_ids[free],
            column_ids[free],
            entries[free],
        )
    if not len(row_ids):
        return
    # DOLFINx inserted UFL entries first; PETSc requires a flush before
    # switching this same matrix to additive contact values.
    matrix.assemble(PETSc.Mat.AssemblyType.FLUSH)
    order = np.lexsort((column_ids, row_ids))
    row_ids, column_ids, entries = (
        row_ids[order],
        column_ids[order],
        entries[order],
    )
    # Merge equal (row, column) entries in the same stable lexicographic order
    # as the former per-row bincount. A single compressed-row insertion removes
    # Python/PETSc crossings per contact row without caching a projection,
    # changing the contact law, or requiring SciPy as a new core dependency.
    first = np.r_[True, (np.diff(row_ids) != 0) | (np.diff(column_ids) != 0)]
    groups = np.cumsum(first) - 1
    summed = np.bincount(groups, weights=entries)
    distinct_rows, distinct_columns = row_ids[first], column_ids[first]
    row_offsets = np.r_[0, np.flatnonzero(np.diff(distinct_rows)) + 1, len(distinct_rows)]
    matrix.setValuesIJV(
        row_offsets.astype(PETSc.IntType),
        distinct_columns.astype(PETSc.IntType),
        summed,
        rowmap=distinct_rows[row_offsets[:-1]].astype(PETSc.IntType),
        addv=PETSc.InsertMode.ADD_VALUES,
    )
