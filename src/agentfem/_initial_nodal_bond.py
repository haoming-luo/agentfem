# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private fixed original-node bond map, separate from contact initialization."""

import numpy as np

from ._interface_pairing import _make_pairing
from ._interface_quadrilateral import BilinearReferenceTrace
from ._q1_surface_search import Q1SurfaceSearch


def initial_nodal_q1_pairing(negative, positive, *, eligible_nodes, tolerance):
    """Bind explicitly eligible, already coincident nodes to original Q1 facets.

    No new bonds appear later; no coordinates are adjusted here. Nodal weights
    are the negative surface's integrated Q1 basis areas. This is a one-sided
    nodal integration rule, NOT common-refinement. It requires refinement and
    side-swap assessment. Finite-rotation use is limited to equal normal and
    tangential elastic stiffness, for which separation traction is central.
    """
    if not all(isinstance(s, BilinearReferenceTrace) for s in (negative, positive)):
        raise TypeError("Nodal bond requires bilinear Q1 reference traces.")
    mask = np.asarray(eligible_nodes)
    if mask.dtype.kind != "b" or mask.shape != (len(negative.vertices),):
        raise ValueError(
            "Original bond eligibility must be an explicit boolean node mask."
        )
    selected = np.flatnonzero(mask)
    if not len(selected):
        raise ValueError("Original bond set is empty.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Coincidence tolerance must be positive finite.")
    projection = Q1SurfaceSearch(
        positive.vertices[positive.quadrilaterals], node_ids=positive.quadrilaterals
    ).project(
        negative.vertices[selected],
        maximum_distance=tolerance,
        require_unique_normal=False,
    )
    # Choose one incident facet only to encode the same node as a Q1 vertex.
    # The measure includes all incident facets, not the arbitrarily chosen one.
    owners = {}
    for facet, nodes in enumerate(negative.quadrilaterals):
        for local, node in enumerate(nodes):
            owners.setdefault(int(node), (facet, local))
    if any(int(node) not in owners for node in selected):
        raise ValueError("Eligible node has no incident surface facet.")
    incidence = np.asarray([owners[int(node)] for node in selected])
    weights = np.eye(4)[incidence[:, 1]]
    return _make_pairing(
        negative,
        positive,
        float(tolerance),
        {
            "negative_nodes": negative.quadrilaterals[incidence[:, 0]],
            "positive_nodes": positive.quadrilaterals[projection["facet_indices"]],
            "negative_weights": weights,
            "positive_weights": projection["weights"],
            "weights": negative.nodal_measures[selected],
            "normals": projection["normals"],
            "reference_mismatch": projection["points"] - negative.vertices[selected],
        },
        "original-node-q1-isotropic-elastic",
    )
