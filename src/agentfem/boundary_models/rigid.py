# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Solver-neutral analytical rigid geometry and prescribed kinematics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
class RigidPlaneSurface:
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

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "rigid_plane_surface",
            "dimension": self.dimension,
            "point": self.point,
            "normal": self.normal,
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
    "prescribed_rigid_motion",
    "rigid_plane",
]
