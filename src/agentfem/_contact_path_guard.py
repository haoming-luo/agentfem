# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Conservative outside-to-outside gap guard, not continuous collision detection.

For a linearly interpolated point and valid Q1 solid boundary, relative path
speed is bounded by point travel plus maximum facet-node travel. If a path
touches the solid, its two positive endpoint distances sum to at most that
bound. Reject unresolved paths instead of accepting a possible tunnel.
Self-intersecting/inverting surfaces and curved temporal trajectories are not
certified here. Negative/zero endpoint gaps require the ordinary contact law.
"""

import numpy as np


class ContactPathResolutionError(ValueError):
    """A discrete increment cannot resolve its swept contact path."""

    def __init__(self, indices):
        self.diagnostic = {
            "code": "contact_path_unresolved",
            "query_indices": tuple(int(i) for i in indices),
            "retry": "reduce_increment",
        }
        super().__init__(
            "Potential missed contact crossing: outside endpoints do not "
            "resolve the swept path; reduce the increment or use a "
            "validated continuous-contact procedure."
        )


def outside_gap_lower_bounds(gaps_old, secondary_increment, primary_increment):
    """Distance lower bounds for a closed boundary under linear nodal motion.

    An initially positive signed distance can decrease by at most relative
    point travel plus maximum boundary-node travel (Q1 has convex weights).
    Positive returned values certify separation THROUGH the increment, not
    exact endpoint distances. Nonpositive bounds request fresh narrow phase.
    """
    raw = tuple(
        np.asarray(x) for x in (gaps_old, secondary_increment, primary_increment)
    )
    if any(np.iscomplexobj(x) or not np.isfinite(x).all() for x in raw):
        raise ValueError("Contact separation bound requires finite real geometry.")
    gaps, secondary, primary = raw
    if (
        gaps.ndim != 1
        or secondary.shape != (len(gaps), 3)
        or primary.ndim != 2
        or primary.shape[1:] != (3,)
        or not len(primary)
    ):
        raise ValueError("Contact separation bound geometry layout differs.")
    translation = primary.mean(axis=0)
    travel = np.linalg.norm(secondary - translation, axis=1) + np.max(
        np.linalg.norm(primary - translation, axis=1)
    )
    scale = np.maximum(np.abs(gaps), travel)
    scale = np.maximum(scale, np.max(np.abs(primary)))
    if len(secondary):
        scale = np.maximum(scale, np.max(np.abs(secondary), axis=1))
    lower = gaps - travel - 128 * np.finfo(float).eps * scale
    if not np.isfinite(lower).all():
        raise ValueError("Contact separation bound overflowed.")
    return lower


def unresolved_outside_paths(
    gaps_old, gaps_new, secondary_increment, primary_increment
):
    raw = tuple(
        np.asarray(x)
        for x in (gaps_old, gaps_new, secondary_increment, primary_increment)
    )
    if any(np.iscomplexobj(x) or not np.isfinite(x).all() for x in raw):
        raise ValueError("Contact path guard requires finite real geometry.")
    old, new, secondary, primary = raw
    if (
        old.ndim != 1
        or new.shape != old.shape
        or secondary.shape != (len(old), 3)
        or primary.ndim != 2
        or primary.shape[1:] != (3,)
        or not len(primary)
    ):
        raise ValueError("Contact path guard geometry layout differs.")
    # Remove a common linear translation without assuming either body fixed.
    translation = primary.mean(axis=0)
    bound = np.linalg.norm(secondary - translation, axis=1) + np.max(
        np.linalg.norm(primary - translation, axis=1)
    )
    pad = 128 * np.finfo(float).eps * np.maximum(bound, old + new)
    return (old > 0) & (new > 0) & (bound + pad >= old + new)
