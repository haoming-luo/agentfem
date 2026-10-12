# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit one-sided Q1 contact declaration, independent of time/state owners."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DeformableContactPair:
    secondary: object
    primary: object
    secondary_dofs: np.ndarray
    primary_dofs: np.ndarray
    candidate_nodes: np.ndarray
    penalty: float
    search_radius: float
    geometry_policy: str = "oriented_open_patch"
    name: str = "deformable_contact"

    def __post_init__(self):
        from .._interface_quadrilateral import BilinearReferenceTrace
        for trace, name in ((self.secondary, "secondary_dofs"), (self.primary, "primary_dofs")):
            if not isinstance(trace, BilinearReferenceTrace):
                raise TypeError("Deformable contact requires explicitly bilinear Q1 traces")
            indices = np.asarray(getattr(self, name))
            if indices.shape != (len(trace.vertices),) or indices.dtype.kind not in "iu" or np.any(indices < 0):
                raise ValueError("Contact maps require one nonnegative integer per trace node")
            object.__setattr__(self, name, np.frombuffer(indices.astype(np.int64).tobytes(), dtype=np.int64))
        mask = np.asarray(self.candidate_nodes)
        if mask.dtype.kind != "b" or mask.shape != (len(self.secondary.vertices),) or not mask.any():
            raise ValueError("Contact needs a nonempty explicit candidate-node mask")
        object.__setattr__(self, "candidate_nodes", np.frombuffer(mask.tobytes(), dtype=bool))
        for value in (self.penalty, self.search_radius):
            if isinstance(value, (bool, complex)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError("Positive finite penalty and search radius required")
        if self.geometry_policy not in ("oriented_open_patch", "reviewed_closed_solid"):
            raise ValueError("Unknown contact geometry policy")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Contact pair needs a nonempty name")

    def summary(self):
        from ..provenance import content_fingerprint
        return {"kind": "deformable_q1_contact_pair", "name": self.name,
                "secondary": self.secondary.geometry_fingerprint,
                "primary": self.primary.geometry_fingerprint,
                "mapping_identity": content_fingerprint({"secondary": self.secondary_dofs.tolist(),
                    "primary": self.primary_dofs.tolist(), "candidates": self.candidate_nodes.tolist()}),
                "penalty": float(self.penalty), "search_radius": float(self.search_radius),
                "geometry_policy": self.geometry_policy,
                "scope": "one_sided_reference_nodal_penalty_not_self_contact"}


def deformable_contact_pair(secondary, primary, **options):
    return DeformableContactPair(secondary, primary, **options)
