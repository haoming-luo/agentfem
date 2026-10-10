# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private, explicit strain-free geometry proposal; never alters a live model."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from ._interface_quadrilateral import BilinearReferenceTrace
from ._q1_surface_search import Q1SurfaceSearch


@dataclass(frozen=True)
class ReferenceSurfaceAdjustment:
    trace: BilinearReferenceTrace
    original_fingerprint: str
    target_fingerprint: str
    selected_nodes: np.ndarray
    adjustment: np.ndarray
    fingerprint: str
    maximum_distance: float

    def summary(self):
        return {
            "kind": "explicit_reference_surface_adjustment",
            "fingerprint": self.fingerprint,
            "original": self.original_fingerprint,
            "target": self.target_fingerprint,
            "adjusted": self.trace.geometry_fingerprint,
            "selected_node_count": len(self.selected_nodes),
            "maximum_adjustment": float(
                np.linalg.norm(self.adjustment, axis=1).max(initial=0.0)
            ),
            "distance_limit": self.maximum_distance,
            "automatic_bond_eligibility": False,
            "bulk_mesh_validated": False,
            "live_model_modified": False,
        }


def propose_reference_surface_adjustment(
    negative, positive, *, selected_nodes, maximum_distance, tolerance
):
    """Project an explicitly chosen node set before mesh/State construction.

    The caller must transfer these coordinates through original node identities
    to a *new* bulk mesh and validate every affected element. No original-contact
    eligibility or vendor initialization rule is inferred from distance/sign.
    Rebuild the fixed bond on the validated new reference; do not subtract a
    finite initial gap from a live large-rotation jump law.
    """
    if not all(isinstance(s, BilinearReferenceTrace) for s in (negative, positive)):
        raise TypeError("Reference adjustment requires Q1 reference traces.")
    mask = np.asarray(selected_nodes)
    if mask.dtype.kind != "b" or mask.shape != (len(negative.vertices),):
        raise ValueError("Select reference nodes using an explicit boolean mask.")
    selected = np.flatnonzero(mask)
    if not len(selected):
        raise ValueError("Reference adjustment selection is empty.")
    if (
        isinstance(maximum_distance, (bool, complex))
        or not np.isscalar(maximum_distance)
        or not np.isfinite(maximum_distance)
        or maximum_distance <= 0
    ):
        raise ValueError("Reference adjustment distance must be positive finite.")
    result = Q1SurfaceSearch(
        positive.vertices[positive.quadrilaterals], node_ids=positive.quadrilaterals
    ).project(
        negative.vertices[selected],
        maximum_distance=maximum_distance,
        require_unique_normal=False,
    )
    coordinates = negative.vertices.copy()
    coordinates[selected] = result["points"]
    adjusted = BilinearReferenceTrace(
        coordinates, negative.quadrilaterals, tolerance=tolerance
    )
    delta = coordinates - negative.vertices
    selected.setflags(write=False)
    delta.setflags(write=False)
    digest = sha256(b"agentfem.reference-surface-adjustment.v1")
    for trace in (negative, positive, adjusted):
        digest.update(trace.geometry_fingerprint.encode())
    digest.update(mask.tobytes())
    digest.update(np.float64(maximum_distance).tobytes())
    return ReferenceSurfaceAdjustment(
        adjusted,
        negative.geometry_fingerprint,
        positive.geometry_fingerprint,
        selected,
        delta,
        digest.hexdigest(),
        float(maximum_distance),
    )
