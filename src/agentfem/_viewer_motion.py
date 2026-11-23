"""Presentation-only deformation sizing and bounded temporal interpolation."""

import numpy as np


def _displacement(grid, name):
    u = np.asarray(grid.point_data[name], dtype=float)
    if u.shape != (grid.n_points, 3) or not np.isfinite(u).all():
        raise ValueError("Display needs a finite three-component displacement.")
    return u


def displacement_metrics(grid, name):
    u = _displacement(grid, name)
    length = float(np.ptp(np.asarray(grid.points), axis=0).max())
    maximum = float(np.linalg.norm(u, axis=1).max())
    # Translation must not be mistaken for visible distortion. This subtraction
    # is used only for sizing; the displayed displacement is never altered.
    typical = float(np.quantile(np.linalg.norm(u - np.median(u, axis=0), axis=1), .95))
    return length, maximum, typical


def automatic_scale(metrics):
    length, maximum, typical = np.max(np.asarray(metrics, dtype=float), axis=0)
    if not np.isfinite([length, maximum, typical]).all() or length <= 0:
        raise ValueError("Cannot size deformation from degenerate geometry.")
    if maximum >= .05*length or typical <= 1.e-12*length:
        return 1.
    # Target 10% of extent, but no node may be magnified beyond 20% of it.
    scale = max(1., min(.10*length/typical, .20*length/maximum, 1.e6))
    quantum = 10.**(np.floor(np.log10(scale))-2)
    return float(np.floor(scale/quantum)*quantum)  # readable, never round above cap


def timeline_sample(times, fraction):
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Smooth playback needs strictly increasing saved coordinates.")
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Invalid playback position.")
    target = float(t[0] + fraction*(t[-1]-t[0]))
    left = min(int(np.searchsorted(t, target, side="right"))-1, len(t)-2)
    return left, float((target-t[left])/(t[left+1]-t[left])), target


def interpolate_displacement(left, right, name, alpha, choice=None):
    """Interpolate only a declared displacement on an identical reference mesh.

    Unselected fields are retained from the left state, never interpolated.
    The caller may display only geometry or this displacement, not those fields.
    """
    if choice is not None and tuple(choice[:2]) != ("point", name):
        raise ValueError("Smooth display supports displacement or solid color, not damage or other fields.")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Invalid interpolation weight.")
    if (left.n_points != right.n_points or left.n_cells != right.n_cells or
            not np.array_equal(left.cells, right.cells) or
            not np.array_equal(left.celltypes, right.celltypes) or
            not np.array_equal(left.points, right.points)):
        raise ValueError("Smooth display requires identical reference geometry and connectivity.")
    a, b = _displacement(left, name), _displacement(right, name)
    result = left.copy(deep=True)
    result.point_data[name] = (1-alpha)*a + alpha*b
    return result
