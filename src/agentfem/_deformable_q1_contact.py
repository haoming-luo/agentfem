# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private conservative current-configuration node/Q1 penalty contribution.

Not general hard contact: one-sided reference nodal integration, bounded
search, no friction, no self contact, no implicit tangent and no MPI. The
master connectivity must be oriented outwards by the geometry owner.
"""

from dataclasses import dataclass

import numpy as np

from ._interface_quadrilateral import BilinearReferenceTrace
from ._q1_surface_search import Q1SurfaceSearch
from .boundary_models.contact_response import FrictionlessPenaltyContactLaw
from .boundary_models.rigid import SurfaceProjection, _readonly_array
from .provenance import content_fingerprint


@dataclass(frozen=True)
class DeformableContactResponse:
    secondary_residual: np.ndarray
    primary_residual: np.ndarray
    potential_energy: float
    projection: SurfaceProjection
    primary_nodes: np.ndarray
    primary_weights: np.ndarray

    def __post_init__(self):
        for name in ("secondary_residual", "primary_residual", "primary_weights"):
            object.__setattr__(self, name, _readonly_array(getattr(self, name)))
        object.__setattr__(
            self, "primary_nodes", _readonly_array(self.primary_nodes, dtype=np.int64)
        )


class DeformableQ1Penalty:
    """Separate current contact search from a fixed original cohesive bond.

    ``candidate_nodes`` is explicit and never changes bond eligibility.
    Search escape or an ambiguous normal fails, rather than silently deleting
    a constraint. Off-normal edge/corner contact is not admitted: normal-only
    forces there would not be the gradient of this distance potential and
    would introduce a spurious moment.
    """

    def __init__(
        self,
        secondary,
        primary,
        *,
        candidate_nodes,
        penalty,
        search_radius,
        geometry_policy="oriented_open_patch",
    ):
        if not all(isinstance(s, BilinearReferenceTrace) for s in (secondary, primary)):
            raise TypeError("Deformable contact requires original Q1 traces.")
        mask = np.asarray(candidate_nodes)
        if (
            mask.dtype.kind != "b"
            or mask.shape != (len(secondary.vertices),)
            or not mask.any()
        ):
            raise ValueError(
                "Contact candidates require a nonempty explicit boolean node mask."
            )
        if not np.isfinite(search_radius) or search_radius <= 0:
            raise ValueError("Contact search radius must be positive finite.")
        if np.ndim(penalty) != 0:
            raise ValueError("This contact contribution requires a scalar penalty.")
        self.secondary, self.primary = secondary, primary
        if geometry_policy not in ("oriented_open_patch", "reviewed_closed_solid"):
            raise ValueError("Unknown deformable contact geometry policy.")
        self.geometry_policy = geometry_policy
        if geometry_policy == "reviewed_closed_solid":
            from ._closed_q1_surface import ClosedQ1Surface

            ClosedQ1Surface(primary.vertices, primary.quadrilaterals)
        self.nodes = _readonly_array(np.flatnonzero(mask), dtype=np.int64)
        self.weights = _readonly_array(secondary.nodal_measures[self.nodes])
        self.search_radius = float(search_radius)
        self.law = FrictionlessPenaltyContactLaw(penalty)
        self.identity = content_fingerprint(
            {
                "secondary": secondary.geometry_fingerprint,
                "primary": primary.geometry_fingerprint,
                "candidate_nodes": self.nodes.tolist(),
                "penalty": float(penalty),
                "search_radius": self.search_radius,
                "measure": "reference_nodal",
                "normal": "primary_connectivity_outward",
                "geometry_policy": self.geometry_policy,
            }
        )

    def evaluate(self, secondary_displacement, primary_displacement):
        current = []
        for trace, displacement in (
            (self.secondary, secondary_displacement),
            (self.primary, primary_displacement),
        ):
            raw = np.asarray(displacement)
            if (
                np.iscomplexobj(raw)
                or raw.shape != trace.vertices.shape
                or not np.isfinite(raw).all()
            ):
                raise ValueError(
                    "Contact displacement must match finite real trace geometry."
                )
            current.append(trace.vertices + raw)
        xs, xp = current
        query = xs[self.nodes]
        if self.geometry_policy == "reviewed_closed_solid":
            from ._closed_q1_surface import ClosedQ1Surface

            search = ClosedQ1Surface(xp, self.primary.quadrilaterals)
        else:
            search = Q1SurfaceSearch(
                xp[self.primary.quadrilaterals], node_ids=self.primary.quadrilaterals
            )
        found = search.project(query, maximum_distance=self.search_radius)
        separation = query - found["points"]
        gaps = found.get(
            "signed_gaps", np.einsum("ij,ij->i", separation, found["normals"])
        )
        tangential = separation - gaps[:, None] * found["normals"]
        scale = max(
            float(np.max(np.abs(xs))), float(np.max(np.abs(xp))), self.search_radius
        )
        if np.any(
            (gaps < 0)
            & (np.linalg.norm(tangential, axis=1) > 1024 * np.finfo(float).eps * scale)
        ):
            raise NotImplementedError(
                "Active off-normal edge/corner contact requires a normal-cone formulation."
            )
        projection = SurfaceProjection(
            surface_name="deformable_primary",
            surface_kind="deformable_q1",
            query_points=query,
            closest_points=found["points"],
            normals=found["normals"],
            signed_gaps=gaps,
            valid=np.ones(len(query), dtype=bool),
            status_codes=np.full(len(query), "ok"),
            method="current_q1_closest_point_reference_nodal_measure",
        )
        response = self.law.evaluate(projection)
        point_force = self.weights[:, None] * response.structural_residual_tractions
        secondary_force = np.zeros_like(xs)
        secondary_force[self.nodes] = point_force
        primary_force = np.zeros_like(xp)
        nodes = self.primary.quadrilaterals[found["facet_indices"]]
        np.add.at(
            primary_force,
            nodes.ravel(),
            (-found["weights"][:, :, None] * point_force[:, None, :]).reshape(-1, 3),
        )
        return DeformableContactResponse(
            secondary_force,
            primary_force,
            float(self.weights @ response.potential_densities),
            projection,
            nodes,
            found["weights"],
        )

    def summary(self):
        return {
            "kind": "private_deformable_q1_penalty",
            "identity": self.identity,
            "candidate_count": len(self.nodes),
            "search_radius": self.search_radius,
            "integration": "one_sided_reference_nodal",
            "friction": "none",
            "hard_contact_exact": False,
            "implicit_tangent": False,
            "mpi": False,
            "geometry_policy": self.geometry_policy,
            "self_intersection_certificate": False,
        }
