# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Mass-scaled row bounds for fixed interpolation penalty operators.

This bounds B.T W D B when each symmetric positive-semidefinite point D is
bounded above by stiffness * I. It does not include moving-projection geometric
terms, material softening, damping, or an entire nonlinear trajectory.
"""

import numpy as np


def penalty_stencil_row_sums(node_ids, shape_values, weights, stiffness, nodal_mass):
    """Bound block rows without assembling point or global stiffness matrices.

    Signed shape values allow a two-sided jump as well as a one-sided trace.
    Coincident/duplicate node indices are accumulated, not overwritten. Mass
    components may differ: their minimum bounds each inverse-square-root block.
    Temporary storage is bounded to 4096 integration points at a time.
    """
    nodes = np.asarray(node_ids)
    arrays = [
        np.asarray(value) for value in (shape_values, weights, stiffness, nodal_mass)
    ]
    if nodes.dtype.kind not in "iu" or nodes.ndim != 2 or nodes.shape[1] == 0:
        raise ValueError("Penalty node indices must be an integer point/node table.")
    if any(np.iscomplexobj(value) for value in arrays):
        raise ValueError("Penalty stability data must be real.")
    shape, measure, modulus, mass = [np.asarray(value, dtype=float) for value in arrays]
    if (
        shape.shape != nodes.shape
        or measure.shape != (len(nodes),)
        or modulus.shape not in ((), (len(nodes),))
        or mass.ndim != 2
        or mass.shape[1] not in (2, 3)
        or any(
            not np.isfinite(value).all() for value in (shape, measure, modulus, mass)
        )
        or np.any(measure < 0)
        or np.any(modulus < 0)
        or np.any(mass < 0)
    ):
        raise ValueError("Invalid finite nonnegative penalty measure, modulus or mass.")
    if np.any(nodes < 0) or (nodes.size and nodes.max() >= len(mass)):
        raise ValueError("Penalty node index lies outside the nodal mass table.")
    if nodes.size and np.any(mass[np.unique(nodes)] <= 0):
        raise ValueError("Referenced penalty nodes require positive component mass.")
    modulus = np.broadcast_to(modulus, (len(nodes),))
    rows = np.zeros(len(mass))
    minimum_mass = mass.min(axis=1)
    for start in range(0, len(nodes), 4096):
        region = slice(start, start + 4096)
        ids = nodes[region]
        scaled = np.abs(shape[region]) / np.sqrt(minimum_mass[ids])
        increments = (
            (measure[region] * modulus[region])[:, None]
            * scaled
            * scaled.sum(axis=1)[:, None]
        )
        np.add.at(rows, ids.ravel(), increments.ravel())
    if not np.isfinite(rows).all():
        raise ValueError("Penalty spectral row bound overflowed.")
    return np.repeat(rows[:, None], mass.shape[1], axis=1)
