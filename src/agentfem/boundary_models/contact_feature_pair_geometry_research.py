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
from .contact_separation_bound import (
    ContactSeparationBound,
    certify_current_triangle_separation,
)
from .contact_smooth_face_research import (
    CertifiedSmoothClosedFacePenalty,
    integrate_certified_smooth_closed_face,
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
    affine_dominance_count: int
    box_exclusion_count: int


@dataclass(frozen=True, eq=False)
class CertifiedClosedTwoFeaturePenalty:
    """One research local Operator, without accepted State or tool work."""

    response: TwoFeatureDistanceIntegral
    geometry: CertifiedTwoFacetDistancePair
    minimum_interior_clearance: float
    tool_fingerprint: str
    reference_vertices: np.ndarray
    current_vertices: np.ndarray

    def __post_init__(self) -> None:
        for name in ("reference_vertices", "current_vertices"):
            values = np.asarray(getattr(self, name), dtype=float).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True, eq=False)
class CertifiedPartialClosedTwoFeaturePenalty:
    """Local partly active penalty; never an accepted global Step."""

    response: TwoFeatureDistanceIntegral
    geometry: CertifiedTwoFacetDistancePair
    active_reference_fraction: float
    tool_fingerprint: str
    reference_vertices: np.ndarray
    current_vertices: np.ndarray

    def __post_init__(self) -> None:
        for name in ("reference_vertices", "current_vertices"):
            values = np.asarray(getattr(self, name), dtype=float).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True, eq=False)
class CertifiedSeparatedClosedFace:
    """Exactly zero penalty on one proved-separated current P1 face."""

    separation: ContactSeparationBound
    tool_fingerprint: str
    reference_vertices: np.ndarray
    current_vertices: np.ndarray

    def __post_init__(self) -> None:
        if (
            not self.separation.certified
            or self.separation.minimum_gap_lower_bound is None
        ):
            raise ValueError("A zero face needs a strict separation certificate.")
        for name in ("reference_vertices", "current_vertices"):
            values = np.asarray(getattr(self, name), dtype=float).copy()
            if values.shape != (3, 3) or not np.all(np.isfinite(values)):
                raise ValueError("A separated face needs finite P1 triangle geometry.")
            if (
                np.linalg.norm(np.cross(values[1] - values[0], values[2] - values[0]))
                <= 0.0
            ):
                raise ValueError("A separated face cannot have zero area.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True, eq=False)
class CertifiedTouchingClosedFace:
    """Zero penalty at a resolved numerical touch on a proved smooth patch.

    The penalty law has zero first derivative at the kink. Its open-side
    tangent (zero) is used for this trial; every subsequent Newton trial must
    be recertified and can take the active branch instead.
    """

    tool_fingerprint: str
    reference_vertices: np.ndarray
    current_vertices: np.ndarray
    maximum_absolute_gap: float
    numerical_gap_tolerance: float

    def __post_init__(self) -> None:
        for name in ("reference_vertices", "current_vertices"):
            values = np.asarray(getattr(self, name), dtype=float).copy()
            if values.shape != (3, 3) or not np.all(np.isfinite(values)):
                raise ValueError("A touching face needs finite P1 triangle geometry.")
            if np.linalg.norm(np.cross(values[1] - values[0], values[2] - values[0])) <= 0.0:
                raise ValueError("A touching face cannot have zero area.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if (
            not np.isfinite(self.maximum_absolute_gap)
            or not np.isfinite(self.numerical_gap_tolerance)
            or not 0.0 <= self.maximum_absolute_gap <= self.numerical_gap_tolerance
        ):
            raise ValueError("A touching face exceeds its numerical gap tolerance.")


def certify_touching_closed_contact_face(
    reference_vertices,
    current_vertices,
    surface: ClosedTriangulatedRigidSolid,
) -> CertifiedTouchingClosedFace:
    """Admit only a roundoff-scale zero gap on one proved smooth patch.

    This does not round a physically resolvable gap to zero. The reviewed
    whole-face planar bound must cover the triangle and exclude all other
    finite master features first. The numerical convention is recorded on
    the trial; it is not an exact-arithmetic contact theorem.
    """

    from .contact_planar_branch_bound import certify_closed_planar_branch

    reference = np.asarray(reference_vertices, dtype=float)
    current = np.asarray(current_vertices, dtype=float)
    if (
        reference.shape != (3, 3) or current.shape != (3, 3)
        or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(current))
        or not isinstance(surface, ClosedTriangulatedRigidSolid)
    ):
        raise ValueError("Touching contact proof needs finite P1 faces and a closed tool.")
    bound = certify_closed_planar_branch(current[None, :, :], surface)
    if not bound.certified:
        raise NotImplementedError("A zero-gap face has no complete smooth-plane proof.")
    normal = bound.plane_normals[0]
    origin = bound.plane_points[0]
    gaps = (current - origin) @ normal
    world_scale = max(
        1.0, float(np.max(np.abs(current))),
        float(np.max(np.abs(surface.facets.vertices))),
    )
    tolerance = 8.0 * np.finfo(float).eps * world_scale
    face_scale = max(
        float(np.linalg.norm(current[1] - current[0])),
        float(np.linalg.norm(current[2] - current[0])),
    )
    if tolerance > 1e-8 * face_scale:
        raise ValueError("Touching geometry is unresolved at world-coordinate precision.")
    maximum_gap = float(np.max(np.abs(gaps)))
    if maximum_gap > tolerance:
        raise NotImplementedError("A zero-gap face is not numerically coplanar.")
    projection = surface.project(current)
    if not np.all(projection.valid) or not np.all(projection.signed_gaps == 0.0):
        raise NotImplementedError("A zero-gap face is not on the closed boundary.")
    return CertifiedTouchingClosedFace(
        surface.geometry_fingerprint, reference, current,
        maximum_gap, tolerance,
    )


def certify_separated_closed_contact_face(
    reference_vertices,
    current_vertices,
    surface: ClosedTriangulatedRigidSolid,
    *,
    maximum_depth: int = 7,
    maximum_samples: int = 4096,
) -> CertifiedSeparatedClosedFace:
    """Return a zero local Operator only if the *whole* face is separated.

    This is a current-configuration proof, not an accepted-time path proof.
    A later trial must rerun it; a failed proof cannot become an inactive face.
    """

    reference = np.asarray(reference_vertices, dtype=float)
    current = np.asarray(current_vertices, dtype=float)
    if (
        reference.shape != (3, 3)
        or current.shape != (3, 3)
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(current))
        or not isinstance(surface, ClosedTriangulatedRigidSolid)
    ):
        raise ValueError("Zero contact proof needs finite P1 faces and a closed tool.")
    bound = certify_current_triangle_separation(
        current[None, :, :],
        surface,
        maximum_depth=maximum_depth,
        maximum_samples=maximum_samples,
    )
    if not bound.certified:
        raise NotImplementedError(
            "The whole current face is not proved separated; it cannot be skipped."
        )
    return CertifiedSeparatedClosedFace(
        bound, surface.geometry_fingerprint, reference, current
    )


def integrate_discovered_closed_contact_face(
    reference_vertices,
    current_vertices,
    surface: ClosedTriangulatedRigidSolid,
    *,
    penalty: float,
    order: int = 24,
    maximum_master_facets: int = 64,
) -> (
    CertifiedSeparatedClosedFace
    | CertifiedTouchingClosedFace
    | CertifiedSmoothClosedFacePenalty
    | CertifiedClosedTwoFeaturePenalty
    | CertifiedPartialClosedTwoFeaturePenalty
):
    """Certify one complete P1 face without user-selected master IDs.

    Search samples propose finite-facet pairs; only a whole-region proof may
    accept one. The admitted subset is exactly: strict separation, exact
    smooth touching, one certified smooth active plane, one
    transverse activation plane with two features, or a wholly interior
    two-feature face. All other geometry fails closed.
    """

    reference = np.asarray(reference_vertices, dtype=float)
    slave = np.asarray(current_vertices, dtype=float)
    count = (
        len(surface._triangles)
        if isinstance(surface, ClosedTriangulatedRigidSolid)
        else 0
    )
    if (
        reference.shape != (3, 3)
        or slave.shape != (3, 3)
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(slave))
        or type(maximum_master_facets) is not int
        or not 2 <= count <= maximum_master_facets <= 64
        or type(order) is not int
        or not 2 <= order <= 64
        or not np.isfinite(float(penalty))
        or float(penalty) <= 0.0
    ):
        raise ValueError("Closed-face discovery needs a bounded P1 contact problem.")
    try:
        return certify_separated_closed_contact_face(reference, slave, surface)
    except NotImplementedError:
        pass
    try:
        return certify_touching_closed_contact_face(reference, slave, surface)
    except NotImplementedError:
        pass
    try:
        return integrate_certified_smooth_closed_face(
            reference, slave, surface, penalty=penalty,
            maximum_master_facets=maximum_master_facets,
        )
    except NotImplementedError:
        pass
    try:
        return integrate_discovered_partial_closed_two_feature_penalty(
            reference,
            slave,
            surface,
            penalty=penalty,
            order=order,
            maximum_master_facets=maximum_master_facets,
        )
    except NotImplementedError:
        pass
    certify_wholly_interior_closed_face(slave, surface)
    samples = np.vstack(
        (slave, np.mean(slave, axis=0), 0.5 * (slave + np.roll(slave, -1, axis=0)))
    )
    projected = surface.project(samples)
    seeds = {
        surface._facet_slots[int(identity)]
        for identity, valid in zip(projected.entity_ids, projected.valid, strict=True)
        if valid
    }
    # Samples prioritize discovery only. A valid narrow feature may occupy
    # none of those points; the bounded exhaustive tail still requires the
    # complete two-feature certificate before accepting any face.
    first_candidates = tuple(sorted(seeds)) + tuple(
        index for index in range(count) if index not in seeds
    )
    for first_index in first_candidates:
        for second_index in range(count):
            if second_index == first_index:
                continue
            try:
                return integrate_certified_closed_two_feature_penalty(
                    reference,
                    slave,
                    surface,
                    (first_index, second_index),
                    penalty=penalty,
                    order=order,
                )
            except NotImplementedError:
                continue
    raise NotImplementedError(
        "No complete two-feature wholly active closed-tool proof was found."
    )


def integrate_discovered_partial_closed_two_feature_penalty(
    reference_vertices,
    current_vertices,
    surface: ClosedTriangulatedRigidSolid,
    *,
    penalty: float,
    order: int = 24,
    maximum_master_facets: int = 64,
) -> CertifiedPartialClosedTwoFeaturePenalty:
    """Discover candidate facets, then *prove* the bounded local contact law.

    Nearest-point samples are only a cheap proposal source. The returned
    integral is admitted solely by the complete active-polygon, finite-
    feature and competitor certificates. Failure to discover the right pair
    is a conservative refusal, never a sampled-Gauss fallback.
    """

    reference = np.asarray(reference_vertices, dtype=float)
    slave = np.asarray(current_vertices, dtype=float)
    selected_penalty = float(penalty)
    if (
        slave.shape != (3, 3)
        or reference.shape != (3, 3)
        or not np.all(np.isfinite(slave))
        or not np.all(np.isfinite(reference))
        or not np.isfinite(selected_penalty)
        or selected_penalty <= 0.0
        or type(order) is not int
        or not 2 <= order <= 64
        or not isinstance(surface, ClosedTriangulatedRigidSolid)
        or type(maximum_master_facets) is not int
        or not 2 <= len(surface._triangles) <= maximum_master_facets <= 64
    ):
        raise ValueError("Automatic local contact discovery needs bounded 3D facets.")
    proposal = surface.project(slave)
    seeds = set()
    if proposal.entity_ids is not None:
        seeds = {
            surface._facet_slots[int(identity)]
            for identity, valid in zip(
                proposal.entity_ids,
                proposal.valid,
                strict=True,
            )
            if valid
        }
    candidates = tuple(sorted(seeds)) + tuple(
        index for index in range(len(surface._triangles)) if index not in seeds
    )
    for first_index in candidates:
        master = surface._triangles[first_index]
        normal = surface.facets.facet_normals[first_index]
        values = (slave - master[0]) @ normal
        if not np.min(values) < 0.0 or not np.max(values) > 0.0:
            continue
        try:
            active, _ = certify_partial_closed_face(slave, surface, first_index)
        except NotImplementedError:
            continue
        points = active @ slave
        sampled = np.vstack(
            (
                points,
                np.mean(points, axis=0),
                0.5 * (points + np.roll(points, -1, axis=0)),
            )
        )
        proposed = surface.project(sampled)
        second_seeds = {
            surface._facet_slots[int(identity)]
            for identity, valid in zip(
                proposed.entity_ids,
                proposed.valid,
                strict=True,
            )
            if valid and surface._facet_slots[int(identity)] != first_index
        }
        # A narrow active polygon can contain a second feature without any
        # proposal sample selecting it. The samples determine search order,
        # not admissibility: within the declared 64-facet research budget,
        # fall back to every remaining finite facet and let the complete
        # closest-feature/sign certificate accept or refuse the face.
        second_candidates = tuple(sorted(second_seeds)) + tuple(
            index for index in range(len(surface._triangles))
            if index != first_index and index not in second_seeds
        )
        for second_index in second_candidates:
            try:
                return integrate_certified_partial_closed_two_feature_penalty(
                    reference,
                    slave,
                    surface,
                    (first_index, second_index),
                    penalty=selected_penalty,
                    order=order,
                )
            except NotImplementedError:
                continue
    raise NotImplementedError(
        "No complete two-feature partially active closed-tool proof was found."
    )


def integrate_certified_partial_closed_two_feature_penalty(
    reference_vertices,
    current_vertices,
    surface: ClosedTriangulatedRigidSolid,
    selected_indices: tuple[int, int],
    *,
    penalty: float,
    order: int = 24,
) -> CertifiedPartialClosedTwoFeaturePenalty:
    """One transversely cut, single-plane activity region of a closed tool.

    Both resulting open polygons must have a constant, winding-verified sign.
    Every other tool facet plane must be strictly separated from their open
    interiors, except facets coplanar with the selected activation plane.
    The nearest finite feature pair is then certified on the active polygon;
    the first feature must be the activation face. Equalities on its cut edge
    are allowed only where a finite edge begins at that exact plane.
    """

    slave = np.asarray(current_vertices, dtype=float)
    active, normal = certify_partial_closed_face(slave, surface, selected_indices[0])
    pair = certify_two_facet_distance_competition(
        active @ slave,
        surface._triangles,
        selected_indices,
        active_boundary=(surface._triangles[selected_indices[0]][0], normal),
    )
    if pair.feature_kinds[0] != "face":
        raise NotImplementedError("The activation facet must remain a face feature.")
    response = integrate_two_affine_feature_distances(
        reference_vertices,
        current_vertices,
        pair.first,
        pair.second,
        stiffness=penalty,
        order=order,
        active_plane_normal=normal,
    )
    reference = np.asarray(reference_vertices, dtype=float)
    double_area = float(
        np.linalg.norm(
            np.cross(reference[1] - reference[0], reference[2] - reference[0])
        )
    )
    fraction = sum(response.branch_areas) / (0.5 * double_area)
    return CertifiedPartialClosedTwoFeaturePenalty(
        response,
        pair,
        fraction,
        surface.geometry_fingerprint,
        reference,
        slave,
    )


def certify_partial_closed_face(
    slave_vertices,
    surface: ClosedTriangulatedRigidSolid,
    boundary_facet_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Certify inside/outside of a P1 face cut by one tool facet plane.

    Returns barycentric vertices of the negative, active polygon and the
    outward normal. A plane-side test is conservative: it may refuse a valid
    contact, but cannot infer a constant winding sign across a crossing.
    """

    slave = np.asarray(slave_vertices, dtype=float)
    if (
        slave.shape != (3, 3)
        or not np.all(np.isfinite(slave))
        or not isinstance(surface, ClosedTriangulatedRigidSolid)
        or surface.sign_method != "winding_reference"
        or type(boundary_facet_index) is not int
        or not 0 <= boundary_facet_index < len(surface._triangles)
    ):
        raise ValueError("Partial closed-face proof needs reviewed 3D geometry.")
    master = surface._triangles[boundary_facet_index]
    normal = np.asarray(surface.facets.facet_normals[boundary_facet_index])
    local_scale = max(
        float(np.max(np.linalg.norm(slave - slave[0], axis=1))),
        float(surface.facets.scale),
    )
    world_scale = max(
        float(np.max(np.abs(slave))),
        float(np.max(np.abs(surface.facets.vertices))),
    )
    margin = max(1e-12 * local_scale, 128.0 * np.finfo(float).eps * world_scale)
    if margin > 1e-4 * local_scale:
        raise ValueError("Partial closed-face proof loses world-coordinate precision.")
    values = (slave - master[0]) @ normal
    if (
        np.any(np.abs(values) <= margin)
        or not np.min(values) < -margin
        or not np.max(values) > margin
    ):
        raise NotImplementedError(
            "A single transverse activation cut is not certified."
        )
    unit = list(np.eye(3))

    def clip(signed):
        result = []
        for start, end in zip(unit, unit[1:] + unit[:1], strict=True):
            a, b = float(start @ signed), float(end @ signed)
            if (a < 0.0) != (b < 0.0):
                result.append((b * start - a * end) / (b - a))
            if b < 0.0:
                result.append(end)
        return np.asarray(result)

    active, inactive = clip(values), clip(-values)
    if len(active) < 3 or len(inactive) < 3:
        raise NotImplementedError("A positive-area activity split is required.")
    plane_anchor = master[0]
    for polygon, sign in ((active, -1), (inactive, 1)):
        points = polygon @ slave
        center = np.mean(points, axis=0)
        query = surface.project(center)
        if not query.all_valid or sign * float(query.signed_gaps[0]) <= margin:
            raise NotImplementedError(
                "The cut polygon lacks the required winding sign."
            )
        for triangle in surface._triangles:
            facet_normal = np.cross(
                triangle[1] - triangle[0], triangle[2] - triangle[0]
            )
            facet_normal /= np.linalg.norm(facet_normal)
            distances = (points - triangle[0]) @ facet_normal
            coplanar = (
                abs(abs(float(facet_normal @ normal)) - 1.0) <= 1e-12
                and abs(float((triangle[0] - plane_anchor) @ normal)) <= margin
            )
            if coplanar:
                if np.max(np.abs((points - plane_anchor) @ normal)) <= margin:
                    raise NotImplementedError(
                        "A cut polygon lies on the tool boundary."
                    )
                continue
            if not (np.min(distances) > margin or np.max(distances) < -margin):
                raise NotImplementedError("Another tool facet may cross a cut polygon.")
    cut_points = np.vstack((active @ slave, inactive @ slave))
    cut_points = cut_points[np.abs((cut_points - master[0]) @ normal) <= margin]
    if len(cut_points) < 2:
        raise NotImplementedError("Activation intersection is unresolved.")
    for point in cut_points:
        barycentric = _triangle_barycentric_coordinates(point, master)
        if np.min(barycentric) <= margin / local_scale:
            raise NotImplementedError(
                "Activation cut leaves the selected finite facet."
            )
    return active, normal


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
        np.asarray(reference_vertices, dtype=float),
        np.asarray(current_vertices, dtype=float),
    )


def certify_wholly_interior_closed_face(
    slave_vertices,
    surface: ClosedTriangulatedRigidSolid,
) -> float:
    """Certify a whole P1 face is inside a reviewed closed tool.

    One valid winding-classified interior query whose unsigned distance to
    the boundary exceeds the radius enclosing the slave face suffices: that
    ball cannot cross the embedded boundary. An exact medial-axis query may
    be nonunique, so all three vertices and the centroid are tried first.
    Otherwise, bounded longest-edge subdivisions must each fit in a proved
    interior ball. This is a same-configuration certificate, not an
    increment-path guarantee.
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
    if any(
        not valid or float(gap) >= -margin
        for valid, gap in zip(
            projection.valid[:3], projection.signed_gaps[:3], strict=True
        )
    ):
        raise NotImplementedError(
            "A slave vertex lacks a strict interior ball certificate."
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
    if clearances:
        return max(clearances)

    pending = [slave]
    proved = []
    for depth in range(9):
        centers = np.asarray([np.mean(face, axis=0) for face in pending])
        evaluated = surface.project(centers)
        remaining = []
        for face, center, valid, gap in zip(
            pending, centers, evaluated.valid, evaluated.signed_gaps, strict=True
        ):
            radius = float(np.max(np.linalg.norm(face - center, axis=1)))
            clearance = -float(gap) - radius
            if valid and gap < 0.0 and clearance > margin:
                proved.append(clearance)
                continue
            if depth == 8:
                raise NotImplementedError(
                    "No bounded winding-classified interior ball cover "
                    "proves the whole slave face."
                )
            edges = np.asarray((
                np.linalg.norm(face[1] - face[0]),
                np.linalg.norm(face[2] - face[1]),
                np.linalg.norm(face[0] - face[2]),
            ))
            first = int(np.argmax(edges))
            second = (first + 1) % 3
            other = (first + 2) % 3
            midpoint = 0.5 * (face[first] + face[second])
            remaining.extend((
                np.asarray((face[first], midpoint, face[other])),
                np.asarray((midpoint, face[second], face[other])),
            ))
        if not remaining:
            return min(proved)
        if len(remaining) > 512:
            raise NotImplementedError(
                "No bounded winding-classified interior ball cover "
                "proves the whole slave face."
            )
        pending = remaining
    raise AssertionError("Interior ball subdivision depth was not bounded.")


def _single_feature_over_slave(slave, master, length_margin, active_boundary=None):
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
        boundary_admissible = False
        if active_boundary is not None:
            anchor, outward = active_boundary
            on_cut = np.abs((slave - anchor) @ outward) <= length_margin
            near_start = fraction <= length_margin / length
            boundary_admissible = (
                np.any(near_start)
                and np.all(on_cut[near_start])
                and np.min(fraction) >= -length_margin / length
                and abs(float((first - anchor) @ outward)) <= length_margin
                and float(direction @ outward) < -length_margin / length
            )
        if (
            (np.min(fraction) > length_margin / length or boundary_admissible)
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
    lower = []
    for index in range(1, len(slave) - 1):
        triangle = slave[[0, index, index + 1]]
        samples = np.vstack(
            (
                triangle,
                0.5 * (triangle[0] + triangle[1]),
                0.5 * (triangle[1] + triangle[2]),
                0.5 * (triangle[2] + triangle[0]),
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
        lower.append(min(coefficients))
    return float(min(lower))


def _affine_feature_dominates_facet(
    slave, triangle, feature, selected_master, length_margin
):
    """Sufficient whole-polygon bound against a convex competitor facet.

    Let A be the selected affine point/line/plane. For p on the slave and q
    on a competitor triangle, split both into components parallel and normal
    to A. If their normal offsets have a nonpositive dot product, then
    ``|p-q|² >= distance(p,A)²``. The dot product is bilinear, so checking
    every pair of polygon/triangle vertices covers both convex hulls.
    Zeros are admitted only when one offset vanishes exactly at the feature,
    or a vertex is *exactly shared* with a selected master vertex that defines
    the feature. The latter avoids treating a rotated edge endpoint's
    order-roundoff projector residual as a third physical branch. Equality
    there has the same closest point and force as A.
    """

    projector = feature.normal_projector
    slave_offsets = (slave - feature.anchor) @ projector
    facet_offsets = (triangle - feature.anchor) @ projector
    dots = slave_offsets @ facet_offsets.T
    zero_slave = np.all(slave_offsets == 0.0, axis=1)
    zero_facet = np.all(facet_offsets == 0.0, axis=1)
    master_offsets = (selected_master - feature.anchor) @ projector
    defining = np.linalg.norm(master_offsets, axis=1) <= length_margin
    for index, vertex in enumerate(triangle):
        if any(
            np.array_equal(vertex, selected_master[master_index])
            for master_index in np.flatnonzero(defining)
        ):
            zero_facet[index] = True
    scale = max(
        float(np.max(np.linalg.norm(slave_offsets, axis=1))),
        float(np.max(np.linalg.norm(facet_offsets, axis=1))),
        length_margin,
    )
    strict = dots < -(length_margin * scale)
    return bool(np.all(strict | zero_slave[:, None] | zero_facet[None, :]))


def certify_two_facet_distance_competition(
    slave_vertices,
    master_triangles,
    selected_indices: tuple[int, int],
    *,
    maximum_master_facets: int = 64,
    active_boundary: tuple[np.ndarray, np.ndarray] | None = None,
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
        slave.ndim != 2
        or slave.shape[1:] != (3,)
        or not 3 <= len(slave) <= 4
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
    local_lengths = [np.linalg.norm(vertex - slave[0]) for vertex in slave[1:]]
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
    polygon_normal = np.cross(slave[1] - slave[0], slave[2] - slave[0])
    normal_length = float(np.linalg.norm(polygon_normal))
    if normal_length <= length_margin * max(local_lengths):
        raise ValueError("Two-facet certificate needs a nondegenerate slave face.")
    if len(slave) == 4:
        unit_normal = polygon_normal / normal_length
        if abs(float((slave[3] - slave[0]) @ unit_normal)) > length_margin:
            raise ValueError("The bounded slave polygon must be planar.")
        for index in range(4):
            start, middle, end = slave[[index, (index + 1) % 4, (index + 2) % 4]]
            turn = float(np.cross(middle - start, end - middle) @ unit_normal)
            if turn <= length_margin * max(local_lengths):
                raise ValueError(
                    "The bounded slave polygon must be convex and ordered."
                )
    if active_boundary is not None:
        if type(active_boundary) is not tuple or len(active_boundary) != 2:
            raise ValueError(
                "Active-boundary proof needs an anchor and outward normal."
            )
        anchor, outward = (np.asarray(item, dtype=float) for item in active_boundary)
        if (
            anchor.shape != (3,)
            or outward.shape != (3,)
            or not np.all(np.isfinite(anchor))
            or not np.all(np.isfinite(outward))
            or not np.isclose(np.linalg.norm(outward), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("Active-boundary proof has invalid 3D plane geometry.")
        active_boundary = (anchor, outward)
    area = sum(
        float(
            np.linalg.norm(
                np.cross(slave[index] - slave[0], slave[index + 1] - slave[0])
            )
        )
        for index in range(1, len(slave) - 1)
    )
    if area <= length_margin * max(local_lengths):
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
        active_boundary,
    )
    second, second_kind, second_signature = _single_feature_over_slave(
        slave,
        masters[indices[1]],
        length_margin,
        active_boundary,
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
    affine_count = 0
    box_count = 0
    slave_lower = np.min(slave, axis=0)
    slave_upper = np.max(slave, axis=0)
    master_lower = np.min(masters, axis=1)
    master_upper = np.max(masters, axis=1)
    box_lower = np.linalg.norm(
        np.maximum(
            np.maximum(master_lower - slave_upper, slave_lower - master_upper),
            0.0,
        ),
        axis=1,
    )
    for index, triangle in enumerate(masters):
        if index in indices:
            continue
        box_margin = float(box_lower[index]) - selected_upper
        if box_margin > length_margin:
            clearance.append(box_margin)
            box_count += 1
            continue
        foot = _closest_point_on_triangle(center, *triangle)
        margin = float(np.linalg.norm(center - foot)) - radius - selected_upper
        if margin > length_margin:
            clearance.append(margin)
            continue
        if any(
            _affine_feature_dominates_facet(
                slave,
                triangle,
                feature,
                masters[selected_index],
                length_margin,
            )
            for feature, selected_index in zip((first, second), indices, strict=True)
        ):
            affine_count += 1
            continue
        try:
            other, _, signature = _single_feature_over_slave(
                slave,
                triangle,
                length_margin,
                active_boundary,
            )
        except NotImplementedError:
            raise NotImplementedError(
                "An additional finite master facet changes feature within the "
                f"slave face (facet {index})."
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
        affine_count,
        box_count,
    )


__all__ = [
    "CertifiedClosedTwoFeaturePenalty",
    "CertifiedPartialClosedTwoFeaturePenalty",
    "CertifiedSeparatedClosedFace",
    "CertifiedTouchingClosedFace",
    "CertifiedTwoFacetDistancePair",
    "certify_separated_closed_contact_face",
    "certify_touching_closed_contact_face",
    "integrate_discovered_closed_contact_face",
    "certify_two_facet_distance_competition",
    "certify_partial_closed_face",
    "certify_wholly_interior_closed_face",
    "integrate_certified_closed_two_feature_penalty",
    "integrate_certified_partial_closed_two_feature_penalty",
    "integrate_discovered_partial_closed_two_feature_penalty",
]
