# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Solver-neutral local response for frictionless penalty contact.

This module owns the constitutive part of contact at already projected surface
points.  Geometry and candidate search remain in :mod:`agentfem.boundary_models`
surface/search objects, while finite-element trace integration remains the
responsibility of a backend Operator.  Keeping those boundaries explicit
prevents a point response from being mistaken for a complete contact element.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rigid import SurfaceProjection, _readonly_array


def _penalty_values(value, point_count: int) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.ndim == 0:
        selected = np.full(point_count, float(selected), dtype=float)
    else:
        selected = selected.reshape(-1)
        if selected.shape != (point_count,):
            raise ValueError(
                "Contact penalty must be scalar or contain one value per point."
            )
    if not np.all(np.isfinite(selected)) or np.any(selected <= 0.0):
        raise ValueError("Contact penalty values must be finite and positive.")
    return selected


@dataclass(frozen=True, eq=False)
class FrictionlessPenaltyContactResponse:
    """Immutable pointwise response to one reviewed surface projection.

    The normal points from the obstacle into the admissible half-space and a
    negative signed gap denotes penetration.  ``structural_residual_tractions``
    are the first variation of the penalty potential with respect to the
    deformable point position. ``surface_generalized_tractions`` are the first
    variation with respect to a rigid translation of the projected surface.
    Their negatives are the physical contact forces on the structure and on
    the obstacle, respectively.

    All quantities are pointwise.  A finite-element backend must still provide
    trace interpolation, quadrature weights, assembly, and a geometry-consistent
    linearization before claiming a complete implicit contact Operator.
    """

    projection: SurfaceProjection
    penalties: object
    penetration: object
    active: object
    pressures: object
    potential_densities: object
    structural_residual_tractions: object
    surface_generalized_tractions: object
    invalid_policy: str

    def __post_init__(self) -> None:
        if not isinstance(self.projection, SurfaceProjection):
            raise TypeError("Contact response requires SurfaceProjection evidence.")
        count = self.projection.point_count
        dimension = self.projection.dimension
        vectors = {
            "structural_residual_tractions": self.structural_residual_tractions,
            "surface_generalized_tractions": self.surface_generalized_tractions,
        }
        scalars = {
            "penalties": self.penalties,
            "penetration": self.penetration,
            "pressures": self.pressures,
            "potential_densities": self.potential_densities,
        }
        for name, value in scalars.items():
            selected = np.asarray(value, dtype=float)
            if selected.shape != (count,) or not np.all(np.isfinite(selected)):
                raise ValueError(
                    f"{name} must contain one finite value per projected point."
                )
            object.__setattr__(self, name, _readonly_array(selected))
        for name, value in vectors.items():
            selected = np.asarray(value, dtype=float)
            if selected.shape != (count, dimension) or not np.all(
                np.isfinite(selected)
            ):
                raise ValueError(
                    f"{name} must contain one finite vector per projected point."
                )
            object.__setattr__(self, name, _readonly_array(selected))
        active = np.asarray(self.active, dtype=bool)
        if active.shape != (count,):
            raise ValueError("Contact active flags must contain one value per point.")
        object.__setattr__(self, "active", _readonly_array(active, dtype=bool))
        if self.invalid_policy not in {"reject", "inactive"}:
            raise ValueError("Contact invalid_policy must be 'reject' or 'inactive'.")
        if (
            np.any(self.penalties <= 0.0)
            or np.any(self.penetration < 0.0)
            or np.any(self.pressures < 0.0)
            or np.any(self.potential_densities < 0.0)
        ):
            raise ValueError("Contact response scalars violate penalty-law bounds.")
        expected_active = self.projection.valid & (self.penetration > 0.0)
        if not np.array_equal(self.active, expected_active):
            raise ValueError("Contact active flags are inconsistent with penetration.")
        if not np.allclose(
            self.pressures,
            self.penalties * self.penetration,
            rtol=64.0 * np.finfo(float).eps,
            atol=0.0,
        ):
            raise ValueError("Contact pressures are inconsistent with the penalty law.")
        if not np.allclose(
            self.potential_densities,
            0.5 * self.penalties * self.penetration**2,
            rtol=64.0 * np.finfo(float).eps,
            atol=0.0,
        ):
            raise ValueError(
                "Contact potential densities are inconsistent with the law."
            )
        if not np.allclose(
            self.structural_residual_tractions,
            -self.surface_generalized_tractions,
            rtol=64.0 * np.finfo(float).eps,
            atol=0.0,
        ):
            raise ValueError("Contact action and reaction tractions do not balance.")
        expected_surface = np.zeros((count, dimension), dtype=float)
        valid = self.projection.valid
        expected_surface[valid] = (
            self.pressures[valid, None] * self.projection.normals[valid]
        )
        if not np.allclose(
            self.surface_generalized_tractions,
            expected_surface,
            rtol=64.0 * np.finfo(float).eps,
            atol=0.0,
        ):
            raise ValueError(
                "Contact tractions are inconsistent with pressure and surface normal."
            )

    @property
    def point_count(self) -> int:
        return self.projection.point_count

    @property
    def dimension(self) -> int:
        return self.projection.dimension

    @property
    def contact_tractions_on_structure(self) -> np.ndarray:
        """Return pointwise physical traction exerted by the obstacle."""

        return -self.structural_residual_tractions

    @property
    def contact_tractions_on_surface(self) -> np.ndarray:
        """Return the equal-and-opposite pointwise traction on the obstacle."""

        return -self.surface_generalized_tractions

    def summary(self) -> dict[str, object]:
        return {
            "kind": "frictionless_penalty_contact_response",
            "point_count": self.point_count,
            "active_count": int(np.count_nonzero(self.active)),
            "invalid_count": int(np.count_nonzero(~self.projection.valid)),
            "invalid_policy": self.invalid_policy,
            "signed_gap_convention": ("positive_admissible_negative_penetration"),
            "response_level": "pointwise_unintegrated",
            "potential": "0.5 * penalty * penetration^2",
            "linearization": "not_provided",
            "friction": "none",
        }


@dataclass(frozen=True, eq=False)
class FrictionlessPenaltyContactLaw:
    """Evaluate a conservative one-sided penalty law on projected points."""

    penalty: object
    invalid_policy: str = "reject"
    name: str = "frictionless_penalty_contact"

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("Contact law requires a name.")
        if self.invalid_policy not in {"reject", "inactive"}:
            raise ValueError("Contact invalid_policy must be 'reject' or 'inactive'.")
        raw = np.asarray(self.penalty, dtype=float)
        if raw.ndim > 1:
            raise ValueError("Contact penalty must be scalar or one-dimensional.")
        if not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
            raise ValueError("Contact penalty values must be finite and positive.")
        object.__setattr__(
            self,
            "penalty",
            float(raw) if raw.ndim == 0 else _readonly_array(raw.reshape(-1)),
        )

    def evaluate(
        self,
        projection: SurfaceProjection,
    ) -> FrictionlessPenaltyContactResponse:
        """Return the local conservative response for exact projection evidence."""

        if not isinstance(projection, SurfaceProjection):
            raise TypeError("Contact law requires SurfaceProjection evidence.")
        if self.invalid_policy == "reject" and not projection.all_valid:
            raise ValueError(
                "Contact law refuses invalid projection evidence; "
                f"status_counts={projection.summary()['status_counts']}."
            )

        penalties = _penalty_values(self.penalty, projection.point_count)
        valid = np.asarray(projection.valid, dtype=bool)
        gaps = np.asarray(projection.signed_gaps, dtype=float)
        penetration = np.zeros(projection.point_count, dtype=float)
        penetration[valid] = np.maximum(-gaps[valid], 0.0)
        active = valid & (penetration > 0.0)
        pressures = penalties * penetration
        potential_densities = 0.5 * penalties * penetration**2
        surface_generalized = np.zeros(
            (projection.point_count, projection.dimension),
            dtype=float,
        )
        surface_generalized[valid] = pressures[valid, None] * projection.normals[valid]
        structural_residual = -surface_generalized

        return FrictionlessPenaltyContactResponse(
            projection=projection,
            penalties=penalties,
            penetration=penetration,
            active=active,
            pressures=pressures,
            potential_densities=potential_densities,
            structural_residual_tractions=structural_residual,
            surface_generalized_tractions=surface_generalized,
            invalid_policy=self.invalid_policy,
        )

    def summary(self) -> dict[str, object]:
        penalty = np.asarray(self.penalty, dtype=float)
        return {
            "name": self.name,
            "kind": "frictionless_penalty_contact_law",
            "penalty": (float(penalty) if penalty.ndim == 0 else penalty.tolist()),
            "invalid_policy": self.invalid_policy,
            "potential": "0.5 * penalty * max(-signed_gap, 0)^2",
            "friction": "none",
            "integration": "backend_required",
            "linearization": "backend_required",
        }


def frictionless_penalty_contact_law(
    penalty,
    *,
    invalid_policy: str = "reject",
    name: str = "frictionless_penalty_contact",
) -> FrictionlessPenaltyContactLaw:
    """Create the geometry-neutral local frictionless contact law."""

    return FrictionlessPenaltyContactLaw(
        penalty=penalty,
        invalid_policy=invalid_policy,
        name=name,
    )


__all__ = [
    "FrictionlessPenaltyContactLaw",
    "FrictionlessPenaltyContactResponse",
    "frictionless_penalty_contact_law",
]
