# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Certified one-plane P1 penalty face within the bounded research Operator.

The whole-face geometry proof is borrowed from the reviewed planar branch.
After that proof, affine-gap clipping integrates a quadratic penalty exactly
on the active reference area. No old nearest point, pressure smoothing, or
uncertified fixed-Gauss contact decision enters this path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .closed_polyhedron import ClosedTriangulatedRigidSolid
from .contact_active_partition import affine_active_triangle_quadrature
from .contact_planar_branch_bound import certify_closed_planar_branch


@dataclass(frozen=True, eq=False)
class SmoothFacePenaltyIntegral:
    """Extensive local energy, virtual force and fixed-branch tangent."""

    potential: float
    nodal_residual: np.ndarray
    nodal_tangent: np.ndarray
    branch_areas: tuple[float, float]
    rigid_rotation_reaction_about_origin: np.ndarray

    def __post_init__(self) -> None:
        for name, shape in (
            ("nodal_residual", (3, 3)),
            ("nodal_tangent", (3, 3, 3, 3)),
            ("rigid_rotation_reaction_about_origin", (3,)),
        ):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError("Smooth face produced an invalid local response.")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if (
            not np.isfinite(self.potential) or self.potential < 0.0
            or len(self.branch_areas) != 2
            or not all(np.isfinite(value) and value >= 0.0
                       for value in self.branch_areas)
        ):
            raise ValueError("Smooth face energy or active area is invalid.")


@dataclass(frozen=True, eq=False)
class CertifiedSmoothClosedFacePenalty:
    """One whole-face planar proof paired to its immutable local response."""

    response: SmoothFacePenaltyIntegral
    active_reference_fraction: float
    tool_fingerprint: str
    reference_vertices: np.ndarray
    current_vertices: np.ndarray
    examined_master_facet_count: int

    def __post_init__(self) -> None:
        if (
            not 0.0 < self.active_reference_fraction <= 1.0 + 1e-12
            or type(self.examined_master_facet_count) is not int
            or self.examined_master_facet_count < 1
        ):
            raise ValueError("Smooth face has invalid proof coverage.")
        for name in ("reference_vertices", "current_vertices"):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            if value.shape != (3, 3) or not np.all(np.isfinite(value)):
                raise ValueError("Smooth face needs finite P1 geometry.")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def integrate_certified_smooth_closed_face(
    reference_vertices,
    current_vertices,
    surface: ClosedTriangulatedRigidSolid,
    *,
    penalty: float,
    maximum_master_facets: int = 64,
) -> CertifiedSmoothClosedFacePenalty:
    """Evaluate the exact P1 planar penalty polynomial after whole-face proof.

    A failed or open proof says nothing about contact admissibility and is
    passed to other bounded research branches by the caller. This routine
    never silently treats an unresolved face as inactive.
    """

    reference = np.asarray(reference_vertices, dtype=float)
    current = np.asarray(current_vertices, dtype=float)
    selected_penalty = float(penalty)
    if (
        reference.shape != (3, 3) or current.shape != (3, 3)
        or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(current))
        or not isinstance(surface, ClosedTriangulatedRigidSolid)
        or type(maximum_master_facets) is not int
        or not 2 <= maximum_master_facets <= 64
        or not np.isfinite(selected_penalty) or selected_penalty <= 0.0
    ):
        raise ValueError("Smooth closed-face integral needs bounded P1 geometry.")
    double_area = float(np.linalg.norm(np.cross(
        reference[1] - reference[0], reference[2] - reference[0],
    )))
    if not np.isfinite(double_area) or double_area <= 0.0:
        raise ValueError("Smooth closed-face integral needs positive reference area.")
    bound = certify_closed_planar_branch(
        current[None, :, :], surface,
        maximum_slave_faces=1,
        maximum_master_facets=maximum_master_facets,
    )
    if not bound.certified:
        raise NotImplementedError("The whole face lacks a smooth closed-plane proof.")
    normal = np.asarray(bound.plane_normals[0], dtype=float)
    point = np.asarray(bound.plane_points[0], dtype=float)
    gaps = (current - point) @ normal
    projection = surface.project(current)
    scale = max(
        1.0, float(np.max(np.abs(current))),
        float(np.max(np.abs(surface.facets.vertices))),
    )
    if (
        not np.all(projection.valid)
        or not np.allclose(
            projection.signed_gaps, gaps,
            rtol=0.0, atol=64.0 * np.finfo(float).eps * scale,
        )
    ):
        raise NotImplementedError("Closed-tool signed gap disagrees with proved plane.")
    shapes, unit_weights = affine_active_triangle_quadrature(gaps)
    if not len(unit_weights):
        raise NotImplementedError("Smooth active face has no positive contact area.")
    weights = double_area * unit_weights
    active_area = float(np.sum(weights))
    distances = shapes @ gaps
    force_coefficients = selected_penalty * np.einsum(
        "q,qa,q->a", weights, shapes, distances,
    )
    nodal_residual = force_coefficients[:, None] * normal[None, :]
    nodal_tangent = selected_penalty * np.einsum(
        "q,qa,qb,i,j->aibj", weights, shapes, shapes, normal, normal,
    )
    potential = 0.5 * selected_penalty * float(np.dot(weights, distances**2))
    moment = np.sum(np.cross(current, nodal_residual), axis=0)
    response = SmoothFacePenaltyIntegral(
        potential, nodal_residual, nodal_tangent,
        (active_area, 0.0), moment,
    )
    return CertifiedSmoothClosedFacePenalty(
        response, active_area / (0.5 * double_area),
        surface.geometry_fingerprint, reference, current,
        int(bound.examined_master_facet_count),
    )


__all__ = [
    "SmoothFacePenaltyIntegral",
    "CertifiedSmoothClosedFacePenalty",
    "integrate_certified_smooth_closed_face",
]
