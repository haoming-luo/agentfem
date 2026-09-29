# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Solver-neutral rigid geometry, projection, and prescribed kinematics.

The objects in this module describe geometry and its observable projection
semantics.  They deliberately do not own UFL lowering, contact enforcement,
or search infrastructure.  A contact backend may consume a reviewed surface
contract without making the surface itself depend on one FEM backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib

import numpy as np


def _readonly_array(value, *, dtype=float) -> np.ndarray:
    selected = np.array(value, dtype=dtype, copy=True)
    selected.setflags(write=False)
    return selected


def _finite_nonnegative(value, *, name: str) -> float:
    selected = float(value)
    if not np.isfinite(selected) or selected < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return selected


def _sha256_geometry(kind: str, *values) -> str:
    digest = hashlib.sha256()
    digest.update(str(kind).encode("utf-8"))
    digest.update(b"\0")
    for value in values:
        selected = np.asarray(value)
        digest.update(str(selected.dtype).encode("ascii"))
        digest.update(np.asarray(selected.shape, dtype="<i8").tobytes())
        digest.update(selected.tobytes(order="C"))
    return digest.hexdigest()


def _finite_vector(value, *, name: str, dimension: int | None = None) -> np.ndarray:
    selected = np.asarray(value, dtype=float).reshape(-1)
    if selected.size not in {2, 3} or not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must be one finite two- or three-dimensional vector.")
    if dimension is not None and selected.size != int(dimension):
        raise ValueError(
            f"{name} must match dimension {dimension}; got {selected.size}."
        )
    return selected.copy()


@dataclass(frozen=True, eq=False)
class SurfaceProjection:
    """Immutable closest-point evidence produced by one rigid surface.

    ``signed_gaps`` follow one convention throughout AgentFEM: positive is in
    the admissible half-space, zero is on the surface, and negative denotes
    penetration.  ``valid`` is explicit so a future tessellated or trimmed
    surface cannot silently return a plausible point when projection failed.
    Entity identifiers are optional for analytical surfaces and required by a
    discrete search backend that needs stable facet identity.
    """

    surface_name: str
    surface_kind: str
    query_points: object
    closest_points: object
    normals: object
    signed_gaps: object
    valid: object
    status_codes: object
    method: str
    entity_ids: object | None = None
    geometry_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not str(self.surface_name).strip():
            raise ValueError("Surface projection requires a surface name.")
        if not str(self.surface_kind).strip():
            raise ValueError("Surface projection requires a surface kind.")
        if not str(self.method).strip():
            raise ValueError("Surface projection requires a method.")

        query = np.asarray(self.query_points, dtype=float)
        closest = np.asarray(self.closest_points, dtype=float)
        normals = np.asarray(self.normals, dtype=float)
        gaps = np.asarray(self.signed_gaps, dtype=float).reshape(-1)
        valid = np.asarray(self.valid, dtype=bool).reshape(-1)
        status_codes = np.asarray(self.status_codes, dtype=str).reshape(-1)
        if query.ndim != 2 or query.shape[1] not in {2, 3}:
            raise ValueError("Surface query points must have shape (count, 2 or 3).")
        if closest.shape != query.shape or normals.shape != query.shape:
            raise ValueError(
                "Closest points and normals must match the query-point shape."
            )
        count = int(query.shape[0])
        if (
            gaps.shape != (count,)
            or valid.shape != (count,)
            or status_codes.shape != (count,)
        ):
            raise ValueError(
                "Signed gaps, validity flags, and status codes must contain "
                "one value per point."
            )
        if np.any(status_codes == ""):
            raise ValueError("Surface projection status codes must be non-empty.")
        if np.any(valid & (status_codes != "ok")):
            raise ValueError("Valid surface projections must use status code 'ok'.")
        if np.any((~valid) & (status_codes == "ok")):
            raise ValueError("Invalid surface projections require a failure code.")
        if not np.all(np.isfinite(query)):
            raise ValueError("Surface query points must be finite.")
        if np.any(valid) and not (
            np.all(np.isfinite(closest[valid]))
            and np.all(np.isfinite(normals[valid]))
            and np.all(np.isfinite(gaps[valid]))
        ):
            raise ValueError("Valid surface projection values must be finite.")
        if np.any(valid):
            normal_norms = np.linalg.norm(normals[valid], axis=1)
            if not np.allclose(normal_norms, 1.0, rtol=0.0, atol=1.0e-12):
                raise ValueError("Surface projection normals must be unit vectors.")

        entity_ids = None
        if self.entity_ids is not None:
            entity_ids = np.asarray(self.entity_ids, dtype=np.int64).reshape(-1)
            if entity_ids.shape != (count,):
                raise ValueError(
                    "Surface entity identifiers must contain one value per point."
                )
            if np.any(entity_ids[valid] < 0):
                raise ValueError(
                    "Valid discrete projections require non-negative entity IDs."
                )
            if np.any(entity_ids[~valid] != -1):
                raise ValueError(
                    "Invalid discrete projections must use entity ID -1."
                )
        fingerprint = self.geometry_fingerprint
        if fingerprint is not None:
            fingerprint = str(fingerprint).lower()
            if len(fingerprint) != 64 or any(
                character not in "0123456789abcdef" for character in fingerprint
            ):
                raise ValueError(
                    "Surface geometry fingerprint must be one SHA-256 hex digest."
                )
        if entity_ids is not None and fingerprint is None:
            raise ValueError(
                "Discrete surface entity IDs require a geometry fingerprint."
            )

        object.__setattr__(self, "surface_name", str(self.surface_name))
        object.__setattr__(self, "surface_kind", str(self.surface_kind))
        object.__setattr__(self, "method", str(self.method))
        object.__setattr__(self, "query_points", _readonly_array(query))
        object.__setattr__(self, "closest_points", _readonly_array(closest))
        object.__setattr__(self, "normals", _readonly_array(normals))
        object.__setattr__(self, "signed_gaps", _readonly_array(gaps))
        object.__setattr__(self, "valid", _readonly_array(valid, dtype=bool))
        object.__setattr__(
            self,
            "status_codes",
            _readonly_array(status_codes, dtype=str),
        )
        object.__setattr__(
            self,
            "entity_ids",
            None if entity_ids is None else _readonly_array(entity_ids, dtype=np.int64),
        )
        object.__setattr__(self, "geometry_fingerprint", fingerprint)

    @property
    def dimension(self) -> int:
        return int(self.query_points.shape[1])

    @property
    def point_count(self) -> int:
        return int(self.query_points.shape[0])

    @property
    def all_valid(self) -> bool:
        return bool(np.all(self.valid))

    def summary(self) -> dict[str, object]:
        valid_count = int(np.count_nonzero(self.valid))
        status_counts = {
            str(code): int(np.count_nonzero(self.status_codes == code))
            for code in sorted(set(self.status_codes.tolist()))
        }
        return {
            "surface_name": self.surface_name,
            "surface_kind": self.surface_kind,
            "dimension": self.dimension,
            "point_count": self.point_count,
            "valid_count": valid_count,
            "invalid_count": self.point_count - valid_count,
            "all_valid": self.all_valid,
            "status_counts": status_counts,
            "method": self.method,
            "geometry_fingerprint": self.geometry_fingerprint,
            "signed_gap_convention": "positive_admissible_negative_penetration",
            "entity_identity": (
                "not_applicable" if self.entity_ids is None else "provided"
            ),
        }


class RigidSurface(ABC):
    """Solver-neutral contract for auditable rigid-surface projection."""

    name: str

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the ambient geometric dimension."""

    @abstractmethod
    def project(
        self,
        points,
        *,
        motion: PrescribedRigidMotion | None = None,
        factor: float = 0.0,
        maximum_distance: float | None = None,
    ) -> SurfaceProjection:
        """Project physical points and return explicit validity evidence."""

    @abstractmethod
    def summary(self) -> dict[str, object]:
        """Return stable, JSON-compatible geometry semantics."""


@dataclass(frozen=True)
class PrescribedRigidMotion:
    """Normalized prescribed rigid-body translation and rotation.

    The end-of-step translation and rotation are scaled by the nonlinear
    Procedure's accepted load coordinate. Two-dimensional rotation is one
    counter-clockwise angle; three-dimensional rotation is an axis-angle
    vector. Values are expressed in radians.
    """

    translation: object
    rotation: object = 0.0
    reference_point: object | None = None
    name: str = "prescribed_rigid_motion"

    def __post_init__(self) -> None:
        translation = _finite_vector(self.translation, name="Rigid translation")
        dimension = int(translation.size)
        if dimension == 2:
            rotation = np.asarray(self.rotation, dtype=float).reshape(-1)
            if rotation.size != 1 or not np.all(np.isfinite(rotation)):
                raise ValueError("Two-dimensional rigid rotation must be one angle.")
        else:
            rotation = np.asarray(self.rotation, dtype=float).reshape(-1)
            if rotation.size == 1 and float(rotation[0]) == 0.0:
                rotation = np.zeros(3, dtype=float)
            if rotation.size != 3 or not np.all(np.isfinite(rotation)):
                raise ValueError(
                    "Three-dimensional rigid rotation must be one axis-angle vector."
                )
        reference = (
            np.zeros(dimension, dtype=float)
            if self.reference_point is None
            else _finite_vector(
                self.reference_point,
                name="Rigid-motion reference point",
                dimension=dimension,
            )
        )
        if not str(self.name).strip():
            raise ValueError("Prescribed rigid motion requires a name.")
        object.__setattr__(self, "translation", tuple(float(v) for v in translation))
        object.__setattr__(self, "rotation", tuple(float(v) for v in rotation))
        object.__setattr__(
            self,
            "reference_point",
            tuple(float(v) for v in reference),
        )

    @property
    def dimension(self) -> int:
        return len(self.translation)

    @property
    def generalized_size(self) -> int:
        return self.dimension + (1 if self.dimension == 2 else 3)

    def rotation_matrix(self, factor: float) -> np.ndarray:
        selected = float(factor)
        if not np.isfinite(selected) or not 0.0 <= selected <= 1.0 + 1.0e-12:
            raise ValueError("Rigid-motion factor must lie in [0, 1].")
        if self.dimension == 2:
            angle = selected * float(self.rotation[0])
            cosine = float(np.cos(angle))
            sine = float(np.sin(angle))
            return np.asarray(((cosine, -sine), (sine, cosine)), dtype=float)
        vector = selected * np.asarray(self.rotation, dtype=float)
        angle = float(np.linalg.norm(vector))
        if angle <= np.finfo(float).eps:
            return np.eye(3, dtype=float)
        axis = vector / angle
        skew = np.asarray(
            (
                (0.0, -axis[2], axis[1]),
                (axis[2], 0.0, -axis[0]),
                (-axis[1], axis[0], 0.0),
            ),
            dtype=float,
        )
        return (
            np.eye(3, dtype=float)
            + np.sin(angle) * skew
            + (1.0 - np.cos(angle)) * (skew @ skew)
        )

    def state(self, factor: float) -> dict[str, np.ndarray | float]:
        selected = float(factor)
        rotation_matrix = self.rotation_matrix(selected)
        translation = selected * np.asarray(self.translation, dtype=float)
        rotation = selected * np.asarray(self.rotation, dtype=float)
        reference = np.asarray(self.reference_point, dtype=float) + translation
        return {
            "factor": selected,
            "translation": translation,
            "rotation": rotation,
            "rotation_matrix": rotation_matrix,
            "reference_point": reference,
        }

    def generalized_coordinate(self, factor: float) -> np.ndarray:
        state = self.state(factor)
        return np.concatenate((state["translation"], state["rotation"]))

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "prescribed_rigid_motion",
            "dimension": self.dimension,
            "translation": self.translation,
            "rotation": self.rotation,
            "rotation_convention": (
                "counter_clockwise_angle"
                if self.dimension == 2
                else "axis_angle_vector"
            ),
            "rotation_unit": "radian",
            "reference_point": self.reference_point,
            "path": "normalized_proportional",
        }


@dataclass(frozen=True)
class RigidPlaneSurface(RigidSurface):
    """Analytical rigid plane with one explicit point and unit normal."""

    point: object
    normal: object
    name: str = "rigid_plane"

    def __post_init__(self) -> None:
        point = _finite_vector(self.point, name="Rigid-plane point")
        normal = _finite_vector(
            self.normal,
            name="Rigid-plane normal",
            dimension=point.size,
        )
        norm = float(np.linalg.norm(normal))
        if not np.isclose(norm, 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "Rigid-plane normal must be a unit vector; "
                f"norm={norm:.16g}."
            )
        if not str(self.name).strip():
            raise ValueError("Rigid plane requires a name.")
        object.__setattr__(self, "point", tuple(float(v) for v in point))
        object.__setattr__(self, "normal", tuple(float(v) for v in normal))

    @property
    def dimension(self) -> int:
        return len(self.point)

    @property
    def geometry_fingerprint(self) -> str:
        return _sha256_geometry(
            "rigid_plane_surface",
            np.asarray(self.point, dtype="<f8"),
            np.asarray(self.normal, dtype="<f8"),
        )

    def transformed(
        self,
        motion: PrescribedRigidMotion | None,
        factor: float,
    ) -> dict[str, np.ndarray | float]:
        if motion is None:
            return {
                "factor": float(factor),
                "point": np.asarray(self.point, dtype=float),
                "normal": np.asarray(self.normal, dtype=float),
                "reference_point": np.asarray(self.point, dtype=float),
                "translation": np.zeros(self.dimension, dtype=float),
                "rotation": np.zeros(1 if self.dimension == 2 else 3, dtype=float),
            }
        if motion.dimension != self.dimension:
            raise ValueError("Rigid surface and motion dimensions must match.")
        state = motion.state(factor)
        center = np.asarray(motion.reference_point, dtype=float)
        rotation = state["rotation_matrix"]
        point = (
            state["reference_point"]
            + rotation @ (np.asarray(self.point, dtype=float) - center)
        )
        normal = rotation @ np.asarray(self.normal, dtype=float)
        return {**state, "point": point, "normal": normal}

    def project(
        self,
        points,
        *,
        motion: PrescribedRigidMotion | None = None,
        factor: float = 0.0,
        maximum_distance: float | None = None,
    ) -> SurfaceProjection:
        """Return the exact orthogonal projection onto the current plane."""

        query = np.asarray(points, dtype=float)
        if query.ndim == 1:
            query = query.reshape((1, -1))
        if query.ndim != 2 or query.shape[1] != self.dimension:
            raise ValueError(
                "Rigid-plane projection points must have shape "
                f"(count, {self.dimension})."
            )
        if not np.all(np.isfinite(query)):
            raise ValueError("Rigid-plane projection points must be finite.")
        state = self.transformed(motion, factor)
        point = np.asarray(state["point"], dtype=float)
        normal = np.asarray(state["normal"], dtype=float)
        gaps = (query - point) @ normal
        closest = query - gaps[:, None] * normal[None, :]
        normals = np.broadcast_to(normal, query.shape).copy()
        if maximum_distance is None:
            valid = np.ones(query.shape[0], dtype=bool)
        else:
            distance = _finite_nonnegative(
                maximum_distance,
                name="Maximum projection distance",
            )
            valid = np.abs(gaps) <= distance
            closest[~valid] = np.nan
            normals[~valid] = np.nan
            gaps[~valid] = np.nan
        status_codes = np.where(valid, "ok", "no_candidate")
        return SurfaceProjection(
            surface_name=self.name,
            surface_kind="rigid_plane_surface",
            query_points=query,
            closest_points=closest,
            normals=normals,
            signed_gaps=gaps,
            valid=valid,
            status_codes=status_codes,
            method="exact_orthogonal_projection",
            geometry_fingerprint=self.geometry_fingerprint,
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "rigid_plane_surface",
            "dimension": self.dimension,
            "point": self.point,
            "normal": self.normal,
            "geometry_fingerprint": self.geometry_fingerprint,
            "representation": "analytical",
            "projection": "exact_orthogonal",
            "signed_gap_convention": "positive_admissible_negative_penetration",
            "entity_identity": "not_applicable",
        }


def _closest_point_on_triangle(
    point: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
) -> np.ndarray:
    """Return the closest point on one non-degenerate triangle.

    This is the reviewed exhaustive reference kernel used to validate future
    accelerated search backends. It follows the Voronoi-region construction
    from *Real-Time Collision Detection* rather than projecting and clipping
    barycentric coordinates heuristically.
    """

    edge_ab = second - first
    edge_ac = third - first
    offset_ap = point - first
    d1 = float(np.dot(edge_ab, offset_ap))
    d2 = float(np.dot(edge_ac, offset_ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return first

    offset_bp = point - second
    d3 = float(np.dot(edge_ab, offset_bp))
    d4 = float(np.dot(edge_ac, offset_bp))
    if d3 >= 0.0 and d4 <= d3:
        return second

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        coordinate = d1 / (d1 - d3)
        return first + coordinate * edge_ab

    offset_cp = point - third
    d5 = float(np.dot(edge_ab, offset_cp))
    d6 = float(np.dot(edge_ac, offset_cp))
    if d6 >= 0.0 and d5 <= d6:
        return third

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        coordinate = d2 / (d2 - d6)
        return first + coordinate * edge_ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        edge_bc = third - second
        coordinate = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return second + coordinate * edge_bc

    denominator = 1.0 / (va + vb + vc)
    coordinate_b = vb * denominator
    coordinate_c = vc * denominator
    return first + edge_ab * coordinate_b + edge_ac * coordinate_c


@dataclass(frozen=True, eq=False)
class TriangulatedRigidSurface(RigidSurface):
    """Reviewed three-dimensional oriented triangle surface.

    The built-in projector is intentionally an exhaustive reference algorithm,
    not the future parallel broad-phase implementation. It establishes the
    geometry, identity, tolerance, and failure contracts that an accelerated
    DOLFINx BVH adapter must reproduce.
    """

    vertices: object
    triangles: object
    facet_ids: object | None = None
    tolerance: float | None = None
    ambiguity_tolerance: float | None = None
    name: str = "triangulated_rigid_surface"
    _facet_normals: np.ndarray = field(init=False, repr=False, compare=False)
    _facet_twice_areas: np.ndarray = field(init=False, repr=False, compare=False)
    _scale: float = field(init=False, repr=False, compare=False)
    _geometry_fingerprint: str = field(init=False, repr=False, compare=False)
    _topology_summary: dict[str, object] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("Triangulated rigid surface requires a name.")
        vertices = np.asarray(self.vertices, dtype=float)
        triangles_raw = np.asarray(self.triangles)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
            raise ValueError("Surface vertices must have shape (count >= 3, 3).")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("Surface vertices must be finite.")
        if np.unique(vertices, axis=0).shape[0] != vertices.shape[0]:
            raise ValueError(
                "Surface contains duplicate vertices; weld the reviewed "
                "triangle topology before creating a rigid surface."
            )
        if triangles_raw.ndim != 2 or triangles_raw.shape[1] != 3:
            raise ValueError("Surface triangles must have shape (count, 3).")
        if triangles_raw.shape[0] < 1:
            raise ValueError("Triangulated rigid surface requires at least one facet.")
        if not np.issubdtype(triangles_raw.dtype, np.integer):
            try:
                triangles = triangles_raw.astype(np.int64)
            except (TypeError, ValueError) as exc:
                raise ValueError("Surface triangle indices must be integers.") from exc
            if not np.array_equal(triangles_raw, triangles):
                raise ValueError("Surface triangle indices must be integers.")
        else:
            triangles = triangles_raw.astype(np.int64, copy=True)
        if np.any(triangles < 0) or np.any(triangles >= vertices.shape[0]):
            raise ValueError("Surface triangle index lies outside the vertex table.")
        if np.unique(triangles).size != vertices.shape[0]:
            raise ValueError("Every surface vertex must belong to at least one triangle.")
        if np.any(
            (triangles[:, 0] == triangles[:, 1])
            | (triangles[:, 1] == triangles[:, 2])
            | (triangles[:, 2] == triangles[:, 0])
        ):
            raise ValueError("Surface triangles must reference three distinct vertices.")

        lower = np.min(vertices, axis=0)
        upper = np.max(vertices, axis=0)
        scale = float(np.linalg.norm(upper - lower))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Triangulated surface must have non-zero geometric scale.")
        tolerance = (
            max(scale * 1.0e-12, np.finfo(float).eps * scale * 1024.0)
            if self.tolerance is None
            else _finite_nonnegative(self.tolerance, name="Surface tolerance")
        )
        if tolerance <= 0.0:
            raise ValueError("Surface tolerance must be positive.")
        ambiguity_tolerance = (
            tolerance
            if self.ambiguity_tolerance is None
            else _finite_nonnegative(
                self.ambiguity_tolerance,
                name="Projection ambiguity tolerance",
            )
        )
        if ambiguity_tolerance <= 0.0:
            raise ValueError("Projection ambiguity tolerance must be positive.")

        first = vertices[triangles[:, 0]]
        second = vertices[triangles[:, 1]]
        third = vertices[triangles[:, 2]]
        area_vectors = np.cross(second - first, third - first)
        twice_areas = np.linalg.norm(area_vectors, axis=1)
        if np.any(twice_areas <= tolerance * scale):
            indices = np.flatnonzero(twice_areas <= tolerance * scale).tolist()
            raise ValueError(
                "Triangulated surface contains degenerate facets at local indices "
                f"{indices}."
            )
        normals = area_vectors / twice_areas[:, None]

        canonical_indices = np.sort(triangles, axis=1)
        if np.unique(canonical_indices, axis=0).shape[0] != triangles.shape[0]:
            raise ValueError("Triangulated surface contains duplicate facets.")
        geometric_facets = []
        for facet in (vertices[item] for item in triangles):
            ordered = facet[np.lexsort((facet[:, 2], facet[:, 1], facet[:, 0]))]
            geometric_facets.append(tuple(ordered.reshape(-1).tolist()))
        if len(set(geometric_facets)) != len(geometric_facets):
            raise ValueError(
                "Triangulated surface contains geometrically duplicate facets."
            )

        if self.facet_ids is None:
            facet_ids = np.arange(triangles.shape[0], dtype=np.int64)
            identity_source = "generated_local_index"
        else:
            facet_ids_raw = np.asarray(self.facet_ids).reshape(-1)
            try:
                facet_ids = facet_ids_raw.astype(np.int64)
            except (TypeError, ValueError) as exc:
                raise ValueError("Facet IDs must be integers.") from exc
            if not np.array_equal(facet_ids_raw, facet_ids):
                raise ValueError("Facet IDs must be integers.")
            if facet_ids.shape != (triangles.shape[0],):
                raise ValueError("Facet IDs must contain one value per triangle.")
            if np.any(facet_ids < 0) or np.unique(facet_ids).size != facet_ids.size:
                raise ValueError("Facet IDs must be unique non-negative integers.")
            identity_source = "user_provided"

        edge_uses: dict[tuple[int, int], list[int]] = {}
        for triangle in triangles:
            for start, end in (
                (int(triangle[0]), int(triangle[1])),
                (int(triangle[1]), int(triangle[2])),
                (int(triangle[2]), int(triangle[0])),
            ):
                key = (min(start, end), max(start, end))
                direction = 1 if (start, end) == key else -1
                edge_uses.setdefault(key, []).append(direction)
        non_manifold = [edge for edge, uses in edge_uses.items() if len(uses) > 2]
        if non_manifold:
            raise ValueError(
                "Triangulated surface contains non-manifold edges; first edge "
                f"is {non_manifold[0]}."
            )
        orientation_conflicts = [
            edge
            for edge, uses in edge_uses.items()
            if len(uses) == 2 and uses[0] == uses[1]
        ]
        if orientation_conflicts:
            raise ValueError(
                "Adjacent surface facets have inconsistent orientation; first "
                f"edge is {orientation_conflicts[0]}."
            )
        boundary_edge_count = sum(len(uses) == 1 for uses in edge_uses.values())

        fingerprint = _sha256_geometry(
            "triangulated_rigid_surface",
            np.asarray(vertices, dtype="<f8"),
            np.asarray(triangles, dtype="<i8"),
            np.asarray(facet_ids, dtype="<i8"),
            np.asarray((tolerance, ambiguity_tolerance), dtype="<f8"),
        )

        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "vertices", _readonly_array(vertices))
        object.__setattr__(self, "triangles", _readonly_array(triangles, dtype=np.int64))
        object.__setattr__(self, "facet_ids", _readonly_array(facet_ids, dtype=np.int64))
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "ambiguity_tolerance", ambiguity_tolerance)
        object.__setattr__(self, "_facet_normals", _readonly_array(normals))
        object.__setattr__(self, "_facet_twice_areas", _readonly_array(twice_areas))
        object.__setattr__(self, "_scale", scale)
        object.__setattr__(self, "_geometry_fingerprint", fingerprint)
        object.__setattr__(
            self,
            "_topology_summary",
            {
                "edge_count": len(edge_uses),
                "boundary_edge_count": boundary_edge_count,
                "closed": boundary_edge_count == 0,
                "manifold": True,
                "orientation_consistent": True,
                "facet_identity": identity_source,
            },
        )

    @property
    def dimension(self) -> int:
        return 3

    @property
    def geometry_fingerprint(self) -> str:
        return self._geometry_fingerprint

    def _reference_projection(
        self,
        query: np.ndarray,
        maximum_distance: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = int(query.shape[0])
        closest = np.full((count, 3), np.nan, dtype=float)
        normals = np.full((count, 3), np.nan, dtype=float)
        gaps = np.full(count, np.nan, dtype=float)
        valid = np.zeros(count, dtype=bool)
        statuses = np.full(count, "no_candidate", dtype="<U20")
        entity_ids = np.full(count, -1, dtype=np.int64)
        maximum_squared = (
            float("inf")
            if maximum_distance is None
            else _finite_nonnegative(
                maximum_distance,
                name="Maximum projection distance",
            )
            ** 2
        )
        ambiguity_squared = float(self.ambiguity_tolerance) ** 2
        triangle_points = self.vertices[self.triangles]
        for point_index, point in enumerate(query):
            candidates = np.asarray(
                [
                    _closest_point_on_triangle(point, *triangle)
                    for triangle in triangle_points
                ],
                dtype=float,
            )
            squared_distances = np.sum((candidates - point) ** 2, axis=1)
            minimum = float(np.min(squared_distances))
            if minimum > maximum_squared:
                continue
            tied = np.flatnonzero(squared_distances <= minimum + ambiguity_squared)
            selected = int(
                tied[np.argmin(np.asarray(self.facet_ids, dtype=np.int64)[tied])]
            )
            if tied.size > 1:
                tied_points = candidates[tied]
                tied_normals = self._facet_normals[tied]
                same_point = np.all(
                    np.linalg.norm(tied_points - candidates[selected], axis=1)
                    <= self.ambiguity_tolerance
                )
                same_normal = np.all(
                    np.linalg.norm(tied_normals - self._facet_normals[selected], axis=1)
                    <= self.ambiguity_tolerance / self._scale
                )
                if not (same_point and same_normal):
                    statuses[point_index] = "ambiguous_projection"
                    continue
            closest[point_index] = candidates[selected]
            normals[point_index] = self._facet_normals[selected]
            gaps[point_index] = float(
                np.dot(point - candidates[selected], self._facet_normals[selected])
            )
            valid[point_index] = True
            statuses[point_index] = "ok"
            entity_ids[point_index] = int(self.facet_ids[selected])
        return closest, normals, gaps, valid, statuses, entity_ids

    def project(
        self,
        points,
        *,
        motion: PrescribedRigidMotion | None = None,
        factor: float = 0.0,
        maximum_distance: float | None = None,
    ) -> SurfaceProjection:
        query = np.asarray(points, dtype=float)
        if query.ndim == 1:
            query = query.reshape((1, -1))
        if query.ndim != 2 or query.shape[1] != 3:
            raise ValueError(
                "Triangulated-surface projection points must have shape (count, 3)."
            )
        if not np.all(np.isfinite(query)):
            raise ValueError("Triangulated-surface projection points must be finite.")

        if motion is None:
            query_reference = query
            rotation = np.eye(3, dtype=float)
            reference = np.zeros(3, dtype=float)
            translated_reference = reference
        else:
            if motion.dimension != 3:
                raise ValueError("Triangulated surface motion must be three-dimensional.")
            state = motion.state(factor)
            rotation = np.asarray(state["rotation_matrix"], dtype=float)
            reference = np.asarray(motion.reference_point, dtype=float)
            translated_reference = np.asarray(state["reference_point"], dtype=float)
            query_reference = reference + (query - translated_reference) @ rotation

        closest, normals, gaps, valid, statuses, entity_ids = (
            self._reference_projection(query_reference, maximum_distance)
        )
        if motion is not None:
            closest[valid] = (
                translated_reference
                + (closest[valid] - reference) @ rotation.T
            )
            normals[valid] = normals[valid] @ rotation.T
        return SurfaceProjection(
            surface_name=self.name,
            surface_kind="triangulated_rigid_surface",
            query_points=query,
            closest_points=closest,
            normals=normals,
            signed_gaps=gaps,
            valid=valid,
            status_codes=statuses,
            method="exhaustive_triangle_closest_point_reference",
            entity_ids=entity_ids,
            geometry_fingerprint=self.geometry_fingerprint,
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "triangulated_rigid_surface",
            "dimension": 3,
            "representation": "oriented_triangle_surface",
            "vertex_count": int(self.vertices.shape[0]),
            "facet_count": int(self.triangles.shape[0]),
            "geometry_fingerprint": self.geometry_fingerprint,
            "scale": self._scale,
            "tolerance": self.tolerance,
            "ambiguity_tolerance": self.ambiguity_tolerance,
            "minimum_facet_area": 0.5 * float(np.min(self._facet_twice_areas)),
            "maximum_facet_area": 0.5 * float(np.max(self._facet_twice_areas)),
            "projection": "exhaustive_reference",
            "signed_gap_convention": "positive_admissible_negative_penetration",
            **self._topology_summary,
        }


def rigid_plane(
    *,
    point,
    normal,
    name: str = "rigid_plane",
) -> RigidPlaneSurface:
    """Describe one analytical rigid plane independently of a contact law."""

    return RigidPlaneSurface(point=point, normal=normal, name=name)


def triangulated_rigid_surface(
    *,
    vertices,
    triangles,
    facet_ids=None,
    tolerance: float | None = None,
    ambiguity_tolerance: float | None = None,
    name: str = "triangulated_rigid_surface",
) -> TriangulatedRigidSurface:
    """Create one reviewed oriented triangle surface for rigid projection."""

    return TriangulatedRigidSurface(
        vertices=vertices,
        triangles=triangles,
        facet_ids=facet_ids,
        tolerance=tolerance,
        ambiguity_tolerance=ambiguity_tolerance,
        name=name,
    )


def prescribed_rigid_motion(
    *,
    translation,
    rotation=0.0,
    reference_point=None,
    name: str = "prescribed_rigid_motion",
) -> PrescribedRigidMotion:
    """Describe one normalized proportional rigid translation and rotation."""

    return PrescribedRigidMotion(
        translation=translation,
        rotation=rotation,
        reference_point=reference_point,
        name=name,
    )


__all__ = [
    "PrescribedRigidMotion",
    "RigidPlaneSurface",
    "RigidSurface",
    "SurfaceProjection",
    "TriangulatedRigidSurface",
    "prescribed_rigid_motion",
    "rigid_plane",
    "triangulated_rigid_surface",
]
