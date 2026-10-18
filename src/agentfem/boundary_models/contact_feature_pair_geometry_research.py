# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Research certificate for two nearest features of a *finite triangle union*.

One finite master triangle must have a single closest face, edge or vertex
Voronoi branch over the entire affine slave triangle. Every other master
triangle is excluded by a center/radius bound, an exact shared feature, or
a positive Bernstein bound on its squared-distance difference. Closed-tool
interiority is a separate, explicitly requested local certificate. Neither
certificate installs a global unilateral-contact Step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .closed_polyhedron import ClosedTriangulatedRigidSolid
from .contact_feature_pair_research import (
    AffineDistanceFeature,
    TwoFeatureDistanceIntegral,
    integrate_two_affine_feature_distances,
)
from .rigid import _closest_point_on_triangle, _triangle_barycentric_coordinates


@dataclass(frozen=True, eq=False)
class CertifiedTwoFacetDistancePair:
    first: AffineDistanceFeature
    second: AffineDistanceFeature
    facet_indices: tuple[int, int]
    feature_kinds: tuple[str, str]
    minimum_competitor_clearance: float | None
    examined_facet_count: int
    quadratic_exclusion_count: int
    equivalent_feature_count: int


@dataclass(frozen=True, eq=False)
class CertifiedClosedTwoFeaturePenalty:
    """One research local Operator, without accepted State or tool work."""

    response: TwoFeatureDistanceIntegral
    geometry: CertifiedTwoFacetDistancePair
    minimum_interior_clearance: float
    tool_fingerprint: str


def integrate_certified_closed_two_feature_penalty(
    reference_vertices,
    current_vertices,
    surface: ClosedTriangulatedRigidSolid,
    selected_indices: tuple[int, int],
    *,
    penalty: float,
    order: int = 24,
) -> CertifiedClosedTwoFeaturePenalty:
    """Integrate a wholly penetrated, two-feature closed-tool P1 face.

    This narrow subset has a physical frictionless reference-area penalty
    potential because the complete slave face is certified inside the closed
    tool, and the pair is certified nearest among all its finite facets.
    No rigid motion, partial active region, material coupling or global Step
    lifecycle is admitted by this local research Operator.
    """

    clearance = certify_wholly_interior_closed_face(current_vertices, surface)
    pair = certify_two_facet_distance_competition(
        current_vertices,
        surface._triangles,
        selected_indices,
    )
    response = integrate_two_affine_feature_distances(
        reference_vertices,
        current_vertices,
        pair.first,
        pair.second,
        stiffness=penalty,
        order=order,
    )
    return CertifiedClosedTwoFeaturePenalty(
        response,
        pair,
        clearance,
        surface.geometry_fingerprint,
    )


def certify_wholly_interior_closed_face(
    slave_vertices,
    surface: ClosedTriangulatedRigidSolid,
) -> float:
    """Certify a whole P1 face is inside a reviewed closed tool.

    One valid winding-classified interior query whose unsigned distance to
    the boundary exceeds the radius enclosing the slave face suffices: that
    ball cannot cross the embedded boundary. An exact medial-axis query may
    be nonunique, so all three vertices and the centroid are tried. This is
    a same-configuration certificate, not an increment-path guarantee.
    """

    slave = np.asarray(slave_vertices, dtype=float)
    if (
        slave.shape != (3, 3)
        or not np.all(np.isfinite(slave))
        or not isinstance(surface, ClosedTriangulatedRigidSolid)
        or surface.sign_method != "winding_reference"
    ):
        raise ValueError(
            "Interior-face proof requires reviewed winding-classified 3D geometry."
        )
    samples = np.vstack((slave, np.mean(slave, axis=0)))
    projection = surface.project(samples)
    world_scale = max(
        float(np.max(np.abs(slave))),
        float(np.max(np.abs(surface.facets.vertices))),
    )
    local_scale = max(
        float(np.max(np.linalg.norm(slave - np.mean(slave, axis=0), axis=1))),
        float(surface.facets.scale),
    )
    margin = max(1e-12 * local_scale, 128.0 * np.finfo(float).eps * world_scale)
    if margin > 1e-4 * local_scale:
        raise ValueError(
            "Interior-face geometry is unresolved at world-coordinate precision."
        )
    clearances = []
    for point, valid, gap in zip(
        samples,
        projection.valid,
        projection.signed_gaps,
        strict=True,
    ):
        if not valid or gap >= 0.0:
            continue
        radius = float(np.max(np.linalg.norm(slave - point, axis=1)))
        if -float(gap) - radius > margin:
            clearances.append(-float(gap) - radius)
    if not clearances:
        raise NotImplementedError(
            "No winding-classified interior ball covers the whole slave face."
        )
    return max(clearances)


def _single_feature_over_slave(slave, master, length_margin):
    edges = (master[1] - master[0], master[2] - master[0])
    normal = np.cross(*edges)
    area_scale = float(np.linalg.norm(normal))
    if area_scale <= length_margin * max(np.linalg.norm(edge) for edge in edges):
        raise ValueError(
            "Two-feature certificate needs nondegenerate master triangles."
        )
    unit_normal = normal / area_scale
    feet = slave - ((slave - master[0]) @ unit_normal)[:, None] * unit_normal
    barycentric = np.asarray(
        [_triangle_barycentric_coordinates(point, master) for point in feet]
    )
    edge_scale = max(float(np.linalg.norm(edge)) for edge in edges)
    if np.min(barycentric) > length_margin / edge_scale:
        basis = np.column_stack(
            (
                edges[0] / np.linalg.norm(edges[0]),
                np.cross(unit_normal, edges[0] / np.linalg.norm(edges[0])),
            )
        )
        signature = ("face", tuple(sorted(tuple(point) for point in master)))
        return AffineDistanceFeature(master[0], basis), "face", signature

    for first_index in range(3):
        second_index = (first_index + 1) % 3
        third_index = (first_index + 2) % 3
        first, second, third = master[[first_index, second_index, third_index]]
        edge = second - first
        length = float(np.linalg.norm(edge))
        direction = edge / length
        transverse = third - first - np.dot(third - first, direction) * direction
        fraction = ((slave - first) @ direction) / length
        if (
            np.min(fraction) > length_margin / length
            and np.max(fraction) < 1.0 - length_margin / length
            and np.max((slave - first) @ transverse)
            < -length_margin * np.linalg.norm(transverse)
        ):
            signature = ("edge", tuple(sorted((tuple(first), tuple(second)))))
            return AffineDistanceFeature(first, direction[:, None]), "edge", signature

    for index in range(3):
        vertex = master[index]
        outgoing = master[[((index + 1) % 3), ((index + 2) % 3)]] - vertex
        if all(
            np.max((slave - vertex) @ edge) < -length_margin * np.linalg.norm(edge)
            for edge in outgoing
        ):
            signature = ("vertex", tuple(vertex))
            return AffineDistanceFeature(vertex, np.empty((3, 0))), "vertex", signature
    raise NotImplementedError(
        "A selected finite triangle changes closest feature over the slave face."
    )


def _quadratic_distance_difference_lower(slave, other, selected):
    """Bernstein lower bound for two fixed-feature squared distances."""

    samples = np.vstack(
        (
            slave,
            0.5 * (slave[0] + slave[1]),
            0.5 * (slave[1] + slave[2]),
            0.5 * (slave[2] + slave[0]),
        )
    )
    other_distance = np.linalg.norm(
        (samples - other.anchor) @ other.normal_projector,
        axis=1,
    )
    selected_distance = np.linalg.norm(
        (samples - selected.anchor) @ selected.normal_projector,
        axis=1,
    )
    values = other_distance**2 - selected_distance**2
    coefficients = (
        *values[:3],
        2.0 * values[3] - 0.5 * (values[0] + values[1]),
        2.0 * values[4] - 0.5 * (values[1] + values[2]),
        2.0 * values[5] - 0.5 * (values[2] + values[0]),
    )
    return float(min(coefficients))


def certify_two_facet_distance_competition(
    slave_vertices,
    master_triangles,
    selected_indices: tuple[int, int],
    *,
    maximum_master_facets: int = 64,
) -> CertifiedTwoFacetDistancePair:
    """Prove only two finite facets may be closest on one current P1 face.

    Strict vertex KKT inequalities place the entire affine slave triangle in
    one convex Voronoi region of each selected triangle. Distance to a fixed
    affine feature is convex, so its maximum on this triangle is bounded by
    its largest vertex value. Distance to any other finite triangle is
    1-Lipschitz; a center/radius bound excludes it everywhere. An inconclusive
    competitor can also be excluded if its *whole-face* fixed-feature KKT
    branch has a positive quadratic Bernstein distance-difference bound, or
    it is the exact same finite feature already selected. Otherwise refuse.
    No sampled nearest-feature assumption is used for the excluded set.
    """

    slave = np.asarray(slave_vertices, dtype=float)
    masters = np.asarray(master_triangles, dtype=float)
    indices = selected_indices
    if (
        slave.shape != (3, 3)
        or masters.ndim != 3
        or masters.shape[1:] != (3, 3)
        or not np.all(np.isfinite(slave))
        or not np.all(np.isfinite(masters))
        or type(maximum_master_facets) is not int
        or maximum_master_facets < 2
        or len(masters) < 2
        or len(masters) > maximum_master_facets
        or type(indices) is not tuple
        or len(indices) != 2
        or any(
            type(index) is not int or not 0 <= index < len(masters) for index in indices
        )
        or indices[0] == indices[1]
    ):
        raise ValueError(
            "Two-facet certificate needs bounded finite 3D triangle geometry."
        )
    local_lengths = [
        np.linalg.norm(slave[1] - slave[0]),
        np.linalg.norm(slave[2] - slave[0]),
    ]
    master_lengths = np.linalg.norm(masters[:, 1:] - masters[:, :1], axis=2)
    local_scale = max(float(np.max(master_lengths)), *local_lengths)
    if local_scale <= 0.0:
        raise ValueError("Two-facet certificate needs nondegenerate local geometry.")
    world_scale = max(float(np.max(np.abs(slave))), float(np.max(np.abs(masters))))
    length_margin = max(
        1e-12 * local_scale,
        128.0 * np.finfo(float).eps * world_scale,
    )
    if length_margin > 1e-4 * local_scale:
        raise ValueError(
            "Two-facet local geometry is unresolved at world-coordinate precision."
        )
    if np.linalg.norm(np.cross(slave[1] - slave[0], slave[2] - slave[0])) <= (
        length_margin * max(local_lengths)
    ):
        raise ValueError("Two-facet certificate needs a nondegenerate slave face.")
    master_areas = np.linalg.norm(
        np.cross(
            masters[:, 1] - masters[:, 0],
            masters[:, 2] - masters[:, 0],
        ),
        axis=1,
    )
    if np.any(master_areas <= length_margin * np.max(master_lengths, axis=1)):
        raise ValueError("Two-facet certificate needs nondegenerate master faces.")
    first, first_kind, first_signature = _single_feature_over_slave(
        slave,
        masters[indices[0]],
        length_margin,
    )
    second, second_kind, second_signature = _single_feature_over_slave(
        slave,
        masters[indices[1]],
        length_margin,
    )
    if first_signature == second_signature:
        raise NotImplementedError(
            "Selected facets name the same finite geometric feature."
        )
    selected_upper = min(
        max(
            float(np.linalg.norm((vertex - feature.anchor) @ feature.normal_projector))
            for vertex in slave
        )
        for feature in (first, second)
    )
    center = np.mean(slave, axis=0)
    radius = float(np.max(np.linalg.norm(slave - center, axis=1)))
    clearance = []
    quadratic_count = 0
    equivalent_count = 0
    for index, triangle in enumerate(masters):
        if index in indices:
            continue
        foot = _closest_point_on_triangle(center, *triangle)
        margin = float(np.linalg.norm(center - foot)) - radius - selected_upper
        if margin > length_margin:
            clearance.append(margin)
            continue
        try:
            other, _, signature = _single_feature_over_slave(
                slave,
                triangle,
                length_margin,
            )
        except NotImplementedError:
            raise NotImplementedError(
                "An additional finite master facet changes feature within the slave face."
            ) from None
        if signature in (first_signature, second_signature):
            equivalent_count += 1
            continue
        squared_margin = length_margin * local_scale
        if any(
            _quadratic_distance_difference_lower(slave, other, selected)
            > squared_margin
            for selected in (first, second)
        ):
            quadratic_count += 1
            continue
        raise NotImplementedError(
            "An additional finite master facet cannot be excluded over the slave face."
        )
    return CertifiedTwoFacetDistancePair(
        first,
        second,
        indices,
        (first_kind, second_kind),
        min(clearance) if clearance else None,
        len(masters),
        quadratic_count,
        equivalent_count,
    )


__all__ = [
    "CertifiedClosedTwoFeaturePenalty",
    "CertifiedTwoFacetDistancePair",
    "certify_two_facet_distance_competition",
    "certify_wholly_interior_closed_face",
    "integrate_certified_closed_two_feature_penalty",
]
