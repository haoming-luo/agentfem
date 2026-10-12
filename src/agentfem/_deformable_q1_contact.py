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
    projection: SurfaceProjection | None
    primary_nodes: np.ndarray | None
    primary_weights: np.ndarray | None
    separation_lower_bounds: np.ndarray | None = None
    projection_indices: np.ndarray | None = None
    zero_gap_indices: np.ndarray | None = None
    zero_gap_tolerance: float = 0.0

    def __post_init__(self):
        zero = np.asarray(
            [] if self.zero_gap_indices is None else self.zero_gap_indices,
            dtype=np.int64 if self.zero_gap_indices is None else None,
        )
        if (
            zero.ndim != 1 or zero.dtype.kind not in "iu"
            or len(np.unique(zero)) != len(zero)
            or not np.isfinite(self.zero_gap_tolerance)
            or self.zero_gap_tolerance < 0
            or (len(zero) and (self.zero_gap_tolerance <= 0
                              or self.separation_lower_bounds is None))
        ):
            raise ValueError("Invalid unresolved zero-gap evidence.")
        if self.separation_lower_bounds is not None:
            bounds = np.asarray(self.separation_lower_bounds)
            if (bounds.ndim != 1 or np.any(zero < 0)
                    or np.any(zero >= len(bounds))
                    or np.any(bounds[zero] != -self.zero_gap_tolerance)):
                raise ValueError("Zero-gap bounds must retain the resolution band.")
        object.__setattr__(self, "zero_gap_indices", _readonly_array(zero, dtype=np.int64))
        if self.projection_indices is not None and (
            self.projection is None or self.separation_lower_bounds is None
        ):
            raise ValueError(
                "Partial contact projections require bounds and a projection."
            )
        for name in ("secondary_residual", "primary_residual", "primary_weights"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _readonly_array(getattr(self, name)))
        if self.primary_nodes is not None:
            object.__setattr__(
                self,
                "primary_nodes",
                _readonly_array(self.primary_nodes, dtype=np.int64),
            )
        if self.projection is None:
            bound = np.asarray(self.separation_lower_bounds)
            if (
                bound.ndim != 1
                or np.iscomplexobj(bound)
                or not np.isfinite(bound).all()
                or np.any(np.delete(bound, zero) <= 0)
                or self.potential_energy != 0
                or self.primary_nodes is not None
                or self.primary_weights is not None
                or np.any(self.secondary_residual)
                or np.any(self.primary_residual)
            ):
                raise ValueError(
                    "Bound-only contact needs separation or zero-gap evidence and must be force-free."
                )
            object.__setattr__(self, "separation_lower_bounds", _readonly_array(bound))
        elif self.separation_lower_bounds is not None:
            bound = np.asarray(self.separation_lower_bounds)
            indices = (
                np.arange(len(self.projection.signed_gaps))
                if self.projection_indices is None
                else np.asarray(self.projection_indices)
            )
            if (
                np.iscomplexobj(bound)
                or bound.ndim != 1
                or (
                    self.projection_indices is None
                    and bound.shape != self.projection.signed_gaps.shape
                )
                or indices.dtype.kind not in "iu"
                or indices.shape != self.projection.signed_gaps.shape
                or np.any(indices < 0)
                or np.any(indices >= len(bound))
                or len(np.unique(indices)) != len(indices)
                or not np.isfinite(bound).all()
                or np.any(bound[indices] > self.projection.signed_gaps)
            ):
                raise ValueError("Contact separation lower bounds are inconsistent.")
            unprojected = np.ones(len(bound), dtype=bool)
            unprojected[indices] = False
            if np.any(~unprojected[zero]):
                raise ValueError("Zero-gap evidence must not invent a projection.")
            unprojected[zero] = False
            if np.any(bound[unprojected] <= 0):
                raise ValueError(
                    "Unprojected contact points need positive separation bounds."
                )
            if self.projection_indices is not None:
                object.__setattr__(
                    self, "projection_indices", _readonly_array(indices, dtype=np.int64)
                )
            object.__setattr__(self, "separation_lower_bounds", _readonly_array(bound))

    @property
    def gap_lower_bounds(self):
        return (
            self.separation_lower_bounds
            if self.separation_lower_bounds is not None
            else self.projection.signed_gaps
        )

    @property
    def path_gaps(self):
        # Preserve the existing endpoint guard for exact projected states.
        # A bound-only state is strictly outside, with a conservative distance.
        if self.projection_indices is not None:
            gaps = self.separation_lower_bounds.copy()
            gaps[self.projection_indices] = self.projection.signed_gaps
            return gaps
        return (
            self.projection.signed_gaps
            if self.projection is not None
            else self.separation_lower_bounds
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
        execution=None,
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
        self.execution = execution
        if geometry_policy == "reviewed_closed_solid":
            from ._closed_q1_surface import ClosedQ1Surface

            self._closed_search = ClosedQ1Surface(primary.vertices, primary.quadrilaterals,
                                                  execution=execution)
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
                "execution": None if execution is None else execution.summary(),
            }
        )

    def evaluate(self, secondary_displacement, primary_displacement, *,
                 _path_lower_bounds=None):
        return self._evaluate(secondary_displacement, primary_displacement,
                              path_lower_bounds=_path_lower_bounds)

    def _evaluate(
        self, secondary_displacement, primary_displacement, *, path_lower_bounds=None
    ):
        # Internal lifecycle optimization only: these bounds must come from
        # accepted geometry and the conservative linear-motion path guard.
        # They are not user-supplied distances or a replacement contact law.
        certified = np.zeros(len(self.nodes), dtype=bool)
        if path_lower_bounds is not None:
            bound = np.asarray(path_lower_bounds)
            if (
                self.geometry_policy != "reviewed_closed_solid"
                or bound.shape != (len(self.nodes),)
                or np.iscomplexobj(bound)
                or not np.isfinite(bound).all()
            ):
                raise ValueError("Path bounds require finite closed-solid candidates.")
            certified = bound > 0
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
        selected = np.arange(len(query))
        zero_indices = []
        zero_tolerance = 0.0
        lower = None
        if self.geometry_policy == "reviewed_closed_solid":
            from ._closed_q1_surface import ClosedQ1Surface

            search = self._closed_search.refitted(xp)
            # Geometry-only cache: no accepted/trial physical state. Every
            # evaluation refits all leaves, including after a failed increment.
            self._closed_search = search
            # Outside the complete current solid box is provably outside the
            # solid. Far candidates remain in the domain, with a distance bound
            # rather than a fabricated closest point or a silently missing pair.
            distance = np.linalg.norm(
                np.maximum(np.maximum(search.lower - query, query - search.upper), 0),
                axis=1,
            )
            pad = (
                256
                * np.finfo(float).eps
                * max(
                    float(np.max(np.abs(xp))), float(np.max(np.abs(query), initial=0))
                )
            )
            lower = np.maximum(0.0, distance - pad)
            if path_lower_bounds is not None:
                lower[certified] = np.maximum(lower[certified], bound[certified])
            selected = np.flatnonzero((lower <= self.search_radius) & ~certified)
            if len(selected):
                geometric = search.search.distance_lower_bounds(query[selected])
                for local in np.flatnonzero(geometric > self.search_radius):
                    index = selected[local]
                    if not search._inside(query[index]):
                        lower[index] = geometric[local]
                selected = np.flatnonzero((lower <= self.search_radius) & ~certified)
            if not len(selected):
                return DeformableContactResponse(
                    np.zeros_like(xs), np.zeros_like(xp), 0.0, None, None, None, lower
                )
        else:
            search = Q1SurfaceSearch(
                xp[self.primary.quadrilaterals], node_ids=self.primary.quadrilaterals,
                execution=self.execution
            )
        # An exterior force-free point needs a positive separation certificate,
        # not an arbitrarily selected closest face. Near a seam, equally close
        # numerical projections may remain unresolved even when the complete
        # solid is provably separated. Preserve that distinction on restart too.
        from ._q1_surface_search import (
            AmbiguousSurfaceProjection, SurfaceOutsideSearchRadius,
        )

        while len(selected):
            try:
                found = search.project(query[selected], maximum_distance=self.search_radius)
                break
            except (AmbiguousSurfaceProjection, SurfaceOutsideSearchRadius) as error:
                if lower is None:
                    raise
                index = selected[error.query_index]
                if isinstance(error, AmbiguousSurfaceProjection) and error.facets:
                    from ._q1_exact_order import witness_within_distance
                    from ._q1_surface_geometry import q1_closest_point

                    tolerance = search.zero_gap_resolution(query)
                    for facet in error.facets:
                        vertices = xp[self.primary.quadrilaterals[facet]]
                        uv = q1_closest_point(vertices, query[index])[0]
                        if witness_within_distance(vertices, query[index], uv, tolerance):
                            # A certified numerical zero band needs no arbitrary
                            # normal. Keep its negative lower bound: it must not
                            # become positive separation for skipping next time.
                            zero_tolerance = tolerance
                            zero_indices.append(index)
                            lower[index] = -tolerance
                            break
                    else:
                        vertices = None
                    if vertices is not None:
                        selected = np.delete(selected, error.query_index)
                        continue
                bound = search.search.distance_lower_bounds(query[index:index + 1])[0]
                if bound <= 0 or search._inside(query[index]):
                    raise
                # A conservative distance bound can be smaller than the search
                # radius even when the actual closest surface is farther away.
                # Only an independent outside classification AND positive
                # separation admit a force-free point. Never enlarge the radius
                # or suppress missing projections for an interior/open surface.
                lower[index] = bound
                selected = np.delete(selected, error.query_index)
        else:
            return DeformableContactResponse(
                np.zeros_like(xs), np.zeros_like(xp), 0.0, None, None, None, lower,
                zero_gap_indices=np.asarray(zero_indices, dtype=np.int64),
                zero_gap_tolerance=zero_tolerance,
            )
        query = query[selected]
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
        point_force = (
            self.weights[selected, None] * response.structural_residual_tractions
        )
        secondary_force = np.zeros_like(xs)
        secondary_force[self.nodes[selected]] = point_force
        primary_force = np.zeros_like(xp)
        nodes = self.primary.quadrilaterals[found["facet_indices"]]
        np.add.at(
            primary_force,
            nodes.ravel(),
            (-found["weights"][:, :, None] * point_force[:, None, :]).reshape(-1, 3),
        )
        if lower is not None:
            lower[selected] = found["separation_lower_bounds"]
        return DeformableContactResponse(
            secondary_force,
            primary_force,
            float(self.weights[selected] @ response.potential_densities),
            projection,
            nodes,
            found["weights"],
            lower,
            selected if len(selected) != len(self.nodes) else None,
            np.asarray(zero_indices, dtype=np.int64),
            zero_tolerance,
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
            "far_candidate_policy": ("positive_closed_solid_distance_bound"
                                     if self.geometry_policy == "reviewed_closed_solid"
                                     else "reject_search_escape"),
            "self_intersection_certificate": False,
        }
