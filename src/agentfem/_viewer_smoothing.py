# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Opt-in display averaging. Never modify source arrays or merge interfaces."""
import numpy as np


def region_choices(grid):
    return [name for name, values in grid.cell_data.items()
            if np.asarray(values).shape == (grid.n_cells,)
            and np.asarray(values).dtype.kind in 'iu'
            and np.unique(values).size <= 256]


def average_cell_colors(grid, values, region=None, method='average'):
    """Average the selected scalar at shared nodes, separately per region.

    No node welding, geometric smoothing, tensor recovery or time interpolation.
    Without a region array the user explicitly treats the mesh as one domain.
    The range remains the caller's original cell range, not the averaged range.
    """
    import pyvista as pv
    if method not in {'average', 'preserve_jumps'}:
        raise ValueError('Choose Average or Preserve jumps color smoothing.')
    values = np.asarray(values, dtype=float)
    if values.shape != (grid.n_cells,) or not np.isfinite(values).all():
        raise ValueError('Cell color averaging needs one finite scalar per cell.')
    if region and region not in region_choices(grid):
        raise ValueError('Choose an integer cell-region field with at most 256 regions.')
    detached = grid.copy(deep=True)
    # Only the selected color is averaged, not all physical arrays.
    detached.point_data.clear()
    detached.cell_data.clear()
    key = '__averaged_cell_color__'
    detached.cell_data[key] = values
    parts = []
    labels = np.asarray(grid.cell_data[region]) if region else np.zeros(grid.n_cells, dtype=int)
    for label in np.unique(labels):
        part = detached.extract_cells(np.flatnonzero(labels == label)) if region else detached
        averaged = part.cell_data_to_point_data(pass_cell_data=True)
        if method == 'preserve_jumps':
            # Suppress nodal averaging where incident cell values differ by
            # more than 20% of this region's range. Duplicate display nodes
            # only: no welding, source mutation or invented extrema.
            ids = part.cell_connectivity
            count = np.diff(part.offset)
            contributions = np.repeat(np.asarray(part.cell_data[key]), count)
            lo, hi = np.full(part.n_points, np.inf), np.full(part.n_points, -np.inf)
            np.minimum.at(lo, ids, contributions)
            np.maximum.at(hi, ids, contributions)
            jump = hi-lo > .2*np.ptp(part.cell_data[key])
            if np.any(jump):
                original_ids = '__viewer_point_id__'
                part.point_data[original_ids] = np.arange(part.n_points)
                split = part.separate_cells()
                mapping = np.asarray(split.point_data[original_ids])
                colors = np.asarray(averaged.point_data[key])[mapping].copy()
                local = np.repeat(np.asarray(split.cell_data[key]), np.diff(split.offset))
                split_ids = split.cell_connectivity
                mask = jump[mapping[split_ids]]
                colors[split_ids[mask]] = local[mask]
                split.point_data.clear()
                split.point_data[key] = colors
                averaged = split
        parts.append(averaged)
    result = pv.MultiBlock(parts).combine(merge_points=False) if len(parts) > 1 else parts[0]
    return result, np.asarray(result.point_data[key]).copy()
