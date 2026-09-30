# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Solver-neutral rigid-body assets for contact models.

A rigid body owns geometry and kinematics, not search, contact enforcement, or
time integration.  The first public contract is deliberately restricted to a
fixed body or a body following one prescribed motion schedule.  Free rigid-body
dynamics requires mass, inertia, generalized State, and a Procedure and is not
silently represented by this object.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .contact_work import PrescribedRigidMotionSchedule
from .rigid import RigidSurface


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class RigidBody:
    """One rigid contact body with fixed or prescribed kinematics.

    ``surface`` is the scientific geometry asset.  A backend may construct a
    serial or distributed search accelerator for that surface independently;
    keeping search outside this object prevents rank-local implementation
    details from contaminating the body's scientific identity.
    """

    surface: RigidSurface
    motion_schedule: PrescribedRigidMotionSchedule | None = None
    reference_point: object | None = None
    name: str = "rigid_body"

    def __post_init__(self) -> None:
        if not isinstance(self.surface, RigidSurface):
            raise TypeError("RigidBody.surface must implement RigidSurface.")
        schedule = self.motion_schedule
        if schedule is not None:
            if not isinstance(schedule, PrescribedRigidMotionSchedule):
                raise TypeError(
                    "RigidBody.motion_schedule must be PrescribedRigidMotionSchedule."
                )
            if schedule.motion.dimension != self.surface.dimension:
                raise ValueError("Rigid-body surface and motion dimensions must match.")
        if not str(self.name).strip():
            raise ValueError("RigidBody requires a name.")
        reference = self.reference_point
        if reference is not None:
            reference = np.asarray(reference, dtype=float).reshape(-1)
            if reference.shape != (self.surface.dimension,) or not np.all(
                np.isfinite(reference)
            ):
                raise ValueError(
                    "RigidBody.reference_point must be one finite surface-dimension "
                    "vector."
                )
            reference = tuple(float(value) for value in reference)
        if schedule is not None:
            scheduled_reference = tuple(
                float(value) for value in schedule.motion.reference_point
            )
            if reference is None:
                reference = scheduled_reference
            elif reference != scheduled_reference:
                raise ValueError(
                    "RigidBody reference_point must match the prescribed motion "
                    "reference point."
                )
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "reference_point", reference)

    @property
    def dimension(self) -> int:
        return self.surface.dimension

    @property
    def kinematic_mode(self) -> str:
        return "fixed" if self.motion_schedule is None else "prescribed"

    @property
    def scientific_identity(self) -> str:
        payload = {
            "name": self.name,
            "surface": self.surface.summary(),
            "motion_schedule": (
                None
                if self.motion_schedule is None
                else self.motion_schedule.summary()
            ),
            "reference_point": self.reference_point,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "rigid_body",
            "dimension": self.dimension,
            "kinematic_mode": self.kinematic_mode,
            "surface": self.surface.summary(),
            "motion_schedule": (
                None
                if self.motion_schedule is None
                else self.motion_schedule.summary()
            ),
            "reference_point": self.reference_point,
            "scientific_identity": self.scientific_identity,
            "free_body_dynamics": False,
        }


def rigid_body(
    surface: RigidSurface,
    *,
    motion_schedule: PrescribedRigidMotionSchedule | None = None,
    reference_point=None,
    name: str = "rigid_body",
) -> RigidBody:
    """Create a fixed or prescribed-kinematics rigid-body asset."""

    return RigidBody(
        surface=surface,
        motion_schedule=motion_schedule,
        reference_point=reference_point,
        name=name,
    )


__all__ = ["RigidBody", "rigid_body"]
