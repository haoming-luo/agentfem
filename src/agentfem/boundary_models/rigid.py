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
from dataclasses import dataclass

import numpy as np


def _readonly_array(value, *, dtype=float) -> np.ndarray:
    selected = np.array(value, dtype=dtype, copy=True)
    selected.setflags(write=False)
    return selected


def _finite_vector(value, *, name: str, dimension: int | None = None) -> np.ndarray:
    selected = np.asarray(value, dtype=float).reshape(-1)
    if selected.size not in {2, 3} or not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must be one finite two- or three-dimensional vector.")
    if dimension is not None and selected.size != int(dimension):
        raise ValueError(
            f"{name} must match dimension {dimension}; got {selected.size}."
        )
    return selected.copy()


@dataclass(frozen=True)
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
        return SurfaceProjection(
            surface_name=self.name,
            surface_kind="rigid_plane_surface",
            query_points=query,
            closest_points=closest,
            normals=normals,
            signed_gaps=gaps,
            valid=np.ones(query.shape[0], dtype=bool),
            status_codes=np.full(query.shape[0], "ok", dtype="<U2"),
            method="exact_orthogonal_projection",
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "rigid_plane_surface",
            "dimension": self.dimension,
            "point": self.point,
            "normal": self.normal,
            "representation": "analytical",
            "projection": "exact_orthogonal",
            "signed_gap_convention": "positive_admissible_negative_penetration",
            "entity_identity": "not_applicable",
        }


def rigid_plane(
    *,
    point,
    normal,
    name: str = "rigid_plane",
) -> RigidPlaneSurface:
    """Describe one analytical rigid plane independently of a contact law."""

    return RigidPlaneSurface(point=point, normal=normal, name=name)


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
    "prescribed_rigid_motion",
    "rigid_plane",
]
