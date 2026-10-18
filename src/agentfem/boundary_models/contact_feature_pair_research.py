# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Research-only area integral for two competing affine nearest features.

The mathematical energy is one half of the squared distance to the *union*
of two fixed affine subspaces, integrated over a reference P1 triangle. It
is not a signed gap, a unilateral contact law, or proof that these are the
nearest *finite* features of a real tool. It isolates the moving feature
boundary and its consistent nodal tangent before any public Step admission.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact_quadratic_region import (
    QuadraticReferenceLevelSet,
    quadratic_level_set_boundary_quadrature,
    quadratic_region_triangle_quadrature,
)


@dataclass(frozen=True, eq=False)
class AffineDistanceFeature:
    """One fixed point, line, or plane with orthonormal tangent columns."""

    anchor: np.ndarray
    tangent_basis: np.ndarray

    def __post_init__(self) -> None:
        anchor = np.asarray(self.anchor, dtype=float).copy()
        basis = np.asarray(self.tangent_basis, dtype=float).copy()
        dimension = anchor.size
        if (
            anchor.shape not in ((2,), (3,))
            or basis.ndim != 2
            or basis.shape[0] != dimension
            or basis.shape[1] >= dimension
            or not np.all(np.isfinite(anchor))
            or not np.all(np.isfinite(basis))
            or not np.allclose(
                basis.T @ basis,
                np.eye(basis.shape[1]),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError("Affine feature needs a finite orthonormal tangent basis.")
        anchor.setflags(write=False)
        basis.setflags(write=False)
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "tangent_basis", basis)

    @property
    def normal_projector(self) -> np.ndarray:
        basis = self.tangent_basis
        return np.eye(len(self.anchor)) - basis @ basis.T


@dataclass(frozen=True, eq=False)
class TwoFeatureDistanceIntegral:
    """Local reference-area result; no unilateral-contact or State claim.

    ``nodal_residual`` is the derivative of potential with respect to body
    displacement. The legacy research names ``rigid_*_reaction`` denote
    force/moment exerted *on the tool*, opposite to the contact force/moment
    on the structure used by AgentFEM's accepted work ledger.
    """

    potential: float
    nodal_residual: np.ndarray
    nodal_tangent: np.ndarray
    bulk_tangent: np.ndarray
    moving_boundary_tangent: np.ndarray
    branch_areas: tuple[float, float]
    inactive_reference_area: float
    boundary_point_count: int
    rigid_translation_reaction: np.ndarray
    rigid_rotation_reaction_about_origin: float | np.ndarray

    @property
    def force_on_structure(self) -> np.ndarray:
        return -self.rigid_translation_reaction

    @property
    def force_on_tool(self) -> np.ndarray:
        return self.rigid_translation_reaction

    @property
    def moment_on_structure_about_origin(self) -> float | np.ndarray:
        return -self.rigid_rotation_reaction_about_origin


def integrate_two_affine_feature_distances(
    reference_vertices,
    current_vertices,
    first: AffineDistanceFeature,
    second: AffineDistanceFeature,
    *,
    stiffness: float = 1.0,
    order: int = 24,
    active_plane_normal=None,
) -> TwoFeatureDistanceIntegral:
    """Integrate the pointwise minimum squared distance and full tangent.

    Each branch's squared distance is quadratic on an affine slave triangle.
    Its equality conic partitions the reference area; the coarea term in the
    residual derivative is essential when the two nearest-point gradients
    differ. A singular or unresolved boundary fails closed. Finite edge and
    facet projection-domain checks belong to a separate geometry owner. If
    ``active_plane_normal`` is supplied, the active side of the *first*
    plane is its negative half-space. This is a mathematical clipped-area
    rule, not a signed-gap or closed-tool certificate. The first branch has
    zero energy and zero force on the clipping line, so its moving activation
    boundary contributes no force or tangent distribution.
    """

    reference = np.asarray(reference_vertices, dtype=float)
    current = np.asarray(current_vertices, dtype=float)
    if (
        reference.shape not in ((3, 2), (3, 3))
        or current.shape != reference.shape
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(current))
        or not isinstance(first, AffineDistanceFeature)
        or not isinstance(second, AffineDistanceFeature)
        or len(first.anchor) != reference.shape[1]
        or len(second.anchor) != reference.shape[1]
    ):
        raise ValueError("Two-feature integral needs compatible finite P1 geometry.")
    if type(order) is not int or not 2 <= order <= 64:
        raise ValueError("Two-feature integration order must be in [2, 64].")
    selected_stiffness = float(stiffness)
    if not np.isfinite(selected_stiffness) or selected_stiffness <= 0.0:
        raise ValueError("Two-feature squared-distance stiffness must be positive.")
    if reference.shape[1] == 2:
        double_area = abs(
            float(
                np.linalg.det(
                    np.column_stack(
                        (
                            reference[1] - reference[0],
                            reference[2] - reference[0],
                        )
                    )
                )
            )
        )
    else:
        double_area = float(
            np.linalg.norm(
                np.cross(
                    reference[1] - reference[0],
                    reference[2] - reference[0],
                )
            )
        )
    if not np.isfinite(double_area) or double_area <= 0.0:
        raise ValueError("Two-feature integral needs a positive reference area.")

    first_projector = first.normal_projector
    second_projector = second.normal_projector
    if active_plane_normal is None:
        polygons = [np.eye(3)]
    else:
        normal = np.asarray(active_plane_normal, dtype=float)
        if (
            normal.shape != (reference.shape[1],)
            or not np.all(np.isfinite(normal))
            or first.tangent_basis.shape[1] != reference.shape[1] - 1
            or not np.isclose(np.linalg.norm(normal), 1.0, rtol=0.0, atol=1e-12)
            or not np.allclose(
                first_projector,
                np.outer(normal, normal),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError("The active half-space must be the first affine plane.")
        signed = (current - first.anchor) @ normal
        coordinate_scale = max(
            float(np.max(np.abs(current))),
            float(np.max(np.abs(first.anchor))),
        )
        local_scale = max(
            float(np.linalg.norm(current[1] - current[0])),
            float(np.linalg.norm(current[2] - current[0])),
        )
        tolerance = max(
            1e-12 * local_scale,
            128.0 * np.finfo(float).eps * coordinate_scale,
        )
        if np.any(np.abs(signed) <= tolerance):
            raise NotImplementedError(
                "An activation boundary passes through a slave vertex."
            )
        if np.all(signed < 0.0):
            polygons = [np.eye(3)]
        elif np.all(signed > 0.0):
            polygons = []
        else:
            polygon = []
            vertices = list(np.eye(3))
            for start, end in zip(vertices, vertices[1:] + vertices[:1], strict=True):
                a, b = float(start @ signed), float(end @ signed)
                if (a < 0.0) != (b < 0.0):
                    polygon.append((b * start - a * end) / (b - a))
                if b < 0.0:
                    polygon.append(end)
            polygons = [np.asarray(polygon)]
    transforms = []
    for polygon in polygons:
        for index in range(1, len(polygon) - 1):
            transform = np.asarray((polygon[0], polygon[index], polygon[index + 1]))
            det = float(
                np.linalg.det(
                    np.column_stack(
                        (
                            transform[1, 1:] - transform[0, 1:],
                            transform[2, 1:] - transform[0, 1:],
                        )
                    )
                )
            )
            if det <= 0.0 or not np.isfinite(det):
                raise ValueError("The active reference polygon is not positive.")
            transforms.append((transform, det))
    if active_plane_normal is not None and 0 < len(transforms):
        active_fraction = sum(det for _, det in transforms)
        if (
            0.0 < active_fraction < 1.0
            and min(active_fraction, 1.0 - active_fraction) <= 1e-10
        ):
            raise NotImplementedError(
                "A near-vertex active-area sliver has unresolved integration precision."
            )
    dimension = reference.shape[1]
    residual = np.zeros((3, dimension))
    bulk = np.zeros((3, dimension, 3, dimension))
    potential = 0.0
    moving = np.zeros_like(bulk)
    areas = np.zeros(2)
    boundary_count = 0
    active_area = 0.5 * double_area * sum(det for _, det in transforms)
    for transform, determinant in transforms:
        local = transform @ current
        origin = local[0]
        u = local[1] - origin
        v = local[2] - origin
        r_first = first_projector @ (origin - first.anchor)
        r_second = second_projector @ (origin - second.anchor)
        difference = first_projector - second_projector
        gradient = r_first - r_second
        boundary = QuadraticReferenceLevelSet(
            a=0.5 * float(v @ difference @ v),
            b0=float(v @ gradient),
            b1=float(u @ difference @ v),
            c0=0.5 * float(r_first @ r_first - r_second @ r_second),
            c1=float(u @ gradient),
            c2=0.5 * float(u @ difference @ u),
        )
        for branch_index, (feature, projector, negative) in enumerate(
            (
                (first, first_projector, True),
                (second, second_projector, False),
            )
        ):
            local_shapes, unit_weights = quadratic_region_triangle_quadrature(
                boundary,
                negative=negative,
                order=order,
            )
            weights = double_area * determinant * unit_weights
            areas[branch_index] += float(weights.sum())
            if not len(weights):
                continue
            shapes = local_shapes @ transform
            distances = (shapes @ current - feature.anchor) @ projector
            potential += (
                0.5
                * selected_stiffness
                * float(np.dot(weights, np.einsum("qi,qi->q", distances, distances)))
            )
            residual += selected_stiffness * np.einsum(
                "q,qa,qi->ai",
                weights,
                shapes,
                distances,
            )
            bulk += selected_stiffness * np.einsum(
                "q,qa,qb,ij->aibj",
                weights,
                shapes,
                shapes,
                projector,
            )
        local_shapes, unit_weights = quadratic_level_set_boundary_quadrature(
            boundary,
            order=order,
        )
        boundary_count += len(unit_weights)
        if len(unit_weights):
            shapes = local_shapes @ transform
            points = shapes @ current
            first_residual = (points - first.anchor) @ first_projector
            second_residual = (points - second.anchor) @ second_projector
            jump = first_residual - second_residual
            moving += -selected_stiffness * np.einsum(
                "q,qa,qb,qi,qj->aibj",
                double_area * determinant * unit_weights,
                shapes,
                shapes,
                jump,
                jump,
            )
    if not np.isclose(sum(areas), active_area, rtol=1e-10, atol=1e-12):
        raise ValueError("Two-feature reference-area partition is incomplete.")
    tool_force = residual.sum(axis=0)
    tool_moment: float | np.ndarray = (
        float(np.sum(current[:, 0] * residual[:, 1] - current[:, 1] * residual[:, 0]))
        if dimension == 2
        else np.sum(np.cross(current, residual), axis=0)
    )
    return TwoFeatureDistanceIntegral(
        potential,
        residual,
        bulk + moving,
        bulk,
        moving,
        (areas[0], areas[1]),
        0.5 * double_area - active_area,
        boundary_count,
        tool_force,
        tool_moment,
    )


__all__ = [
    "AffineDistanceFeature",
    "TwoFeatureDistanceIntegral",
    "integrate_two_affine_feature_distances",
]
