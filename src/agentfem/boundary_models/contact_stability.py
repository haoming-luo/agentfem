# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Solver-neutral explicit penalty-contact stability screening.

The functions in this module do not advance an explicit Procedure.  They
bound the largest eigenvalue added by a penalty-contact trace after symmetric
mass scaling.  A backend owns ghost accumulation and the global maximum; this
module owns the finite-element interpolation and numerical meaning of the
bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np

from .contact_trace import ContactTrace
from .rigid import _readonly_array


_METHOD = "mass_scaled_contact_gershgorin_upper_bound"
_MASS_COMPATIBILITY = {"function_space_identity", "shape_only"}


def _point_penalties(value, point_count: int, *, name: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.ndim == 0:
        selected = np.full(point_count, float(selected), dtype=float)
    else:
        selected = selected.reshape(-1)
        if selected.shape != (point_count,):
            raise ValueError(f"{name} must be scalar or contain one value per point.")
    if not np.all(np.isfinite(selected)) or np.any(selected <= 0.0):
        raise ValueError(f"{name} values must be finite and positive.")
    return selected


@dataclass(frozen=True)
class ContactStabilityEstimate:
    """Auditable screening estimate for explicit penalty contact.

    ``spectral_radius_upper_bound`` bounds the contact contribution to the
    largest eigenvalue of the mass-scaled semi-discrete system.  ``selected``
    is the central-difference limit after applying ``safety_factor``.
    """

    selected: float
    unsafed_limit: float
    spectral_radius_upper_bound: float
    safety_factor: float
    normal_penalty_maximum: float
    tangential_penalty_maximum: float | None
    friction_coefficient: float
    point_count: int
    mass_compatibility: str = "shape_only"
    method: str = _METHOD

    def __post_init__(self) -> None:
        selected = float(self.selected)
        unsafed = float(self.unsafed_limit)
        spectral = float(self.spectral_radius_upper_bound)
        safety = float(self.safety_factor)
        normal = float(self.normal_penalty_maximum)
        tangential = self.tangential_penalty_maximum
        friction = float(self.friction_coefficient)
        if tangential is not None:
            tangential = float(tangential)
        if not all(
            isfinite(value) and value > 0.0
            for value in (selected, unsafed, spectral, normal)
        ):
            raise ValueError("Contact stability values must be finite and positive.")
        if not isfinite(safety) or not 0.0 < safety <= 1.0:
            raise ValueError(
                "Contact stability safety_factor must satisfy 0 < value <= 1."
            )
        if tangential is not None and (not isfinite(tangential) or tangential <= 0.0):
            raise ValueError("Tangential penalty maximum must be finite and positive.")
        if not isfinite(friction) or friction < 0.0:
            raise ValueError("Friction coefficient must be finite and non-negative.")
        if int(self.point_count) < 1:
            raise ValueError(
                "Contact stability requires at least one integration point."
            )
        mass_compatibility = str(self.mass_compatibility)
        if mass_compatibility not in _MASS_COMPATIBILITY:
            raise ValueError(
                "Contact stability mass_compatibility must be "
                "'function_space_identity' or 'shape_only'."
            )
        if str(self.method) != _METHOD:
            raise ValueError(f"Unsupported contact stability method {self.method!r}.")
        expected_unsafed = 2.0 / sqrt(spectral)
        tolerance = 256.0 * np.finfo(float).eps
        if not np.isclose(unsafed, expected_unsafed, rtol=tolerance, atol=0.0):
            raise ValueError(
                "Contact stability limit is inconsistent with its spectral bound."
            )
        if not np.isclose(selected, safety * unsafed, rtol=tolerance, atol=0.0):
            raise ValueError(
                "Selected contact limit is inconsistent with its safety factor."
            )
        object.__setattr__(self, "selected", selected)
        object.__setattr__(self, "unsafed_limit", unsafed)
        object.__setattr__(self, "spectral_radius_upper_bound", spectral)
        object.__setattr__(self, "safety_factor", safety)
        object.__setattr__(self, "normal_penalty_maximum", normal)
        object.__setattr__(self, "tangential_penalty_maximum", tangential)
        object.__setattr__(self, "friction_coefficient", friction)
        object.__setattr__(self, "point_count", int(self.point_count))
        object.__setattr__(self, "mass_compatibility", mass_compatibility)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "explicit_contact_stability_estimate",
            "selected": self.selected,
            "unsafed_limit": self.unsafed_limit,
            "spectral_radius_upper_bound": self.spectral_radius_upper_bound,
            "safety_factor": self.safety_factor,
            "normal_penalty_maximum": self.normal_penalty_maximum,
            "tangential_penalty_maximum": self.tangential_penalty_maximum,
            "friction_coefficient": self.friction_coefficient,
            "point_count": self.point_count,
            "mass_compatibility": self.mass_compatibility,
            "method": self.method,
            "scope": "contact_penalty_contribution_only",
            "linearization_scope": "fixed_or_piecewise_planar_normal",
            "status": "conservative_screening_bound",
        }


@dataclass(frozen=True)
class CombinedExplicitStabilityEstimate:
    """Conservative composition of non-contact and contact spectral bounds."""

    selected: float
    unsafed_limit: float
    noncontact_unsafed_limit: float
    noncontact_spectral_radius_upper_bound: float
    contact_spectral_radius_upper_bound: float
    total_spectral_radius_upper_bound: float
    safety_factor: float

    def __post_init__(self) -> None:
        values = (
            self.selected,
            self.unsafed_limit,
            self.noncontact_unsafed_limit,
            self.noncontact_spectral_radius_upper_bound,
            self.contact_spectral_radius_upper_bound,
            self.total_spectral_radius_upper_bound,
            self.safety_factor,
        )
        if not all(isfinite(float(value)) and float(value) > 0.0 for value in values):
            raise ValueError("Combined stability values must be finite and positive.")
        if float(self.safety_factor) > 1.0:
            raise ValueError("Combined stability safety_factor must not exceed one.")
        tolerance = 256.0 * np.finfo(float).eps
        expected_total = float(self.noncontact_spectral_radius_upper_bound) + float(
            self.contact_spectral_radius_upper_bound
        )
        if not np.isclose(
            float(self.total_spectral_radius_upper_bound),
            expected_total,
            rtol=tolerance,
            atol=0.0,
        ):
            raise ValueError("Combined stability spectral bound is inconsistent.")
        expected_unsafed = 2.0 / sqrt(expected_total)
        if not np.isclose(
            float(self.unsafed_limit),
            expected_unsafed,
            rtol=tolerance,
            atol=0.0,
        ):
            raise ValueError("Combined stability time limit is inconsistent.")
        if not np.isclose(
            float(self.selected),
            float(self.safety_factor) * expected_unsafed,
            rtol=tolerance,
            atol=0.0,
        ):
            raise ValueError("Selected combined stability limit is inconsistent.")

    def summary(self) -> dict[str, object]:
        return {
            "kind": "combined_explicit_stability_estimate",
            "selected": float(self.selected),
            "unsafed_limit": float(self.unsafed_limit),
            "noncontact_unsafed_limit": float(self.noncontact_unsafed_limit),
            "noncontact_spectral_radius_upper_bound": float(
                self.noncontact_spectral_radius_upper_bound
            ),
            "contact_spectral_radius_upper_bound": float(
                self.contact_spectral_radius_upper_bound
            ),
            "total_spectral_radius_upper_bound": float(
                self.total_spectral_radius_upper_bound
            ),
            "safety_factor": float(self.safety_factor),
            "composition": "additive_spectral_upper_bounds",
            "status": "conservative_screening_bound",
        }


def contact_penalty_local_row_sums(
    trace: ContactTrace,
    nodal_mass,
    *,
    dimension: int,
    normal_penalty,
    tangential_penalty=None,
    friction_coefficient: float = 0.0,
) -> np.ndarray:
    """Return local-plus-ghost rows of a mass-scaled contact bound.

    For every contact point, the fixed-normal generalized tangent is bounded
    by the normal penalty plus, when friction is active, the tangential
    penalty and Coulomb pressure-cap coupling. Replacing those contributions
    by an isotropic upper operator and applying the triangle inequality yields
    a block absolute row-sum bound for
    ``M**(-1/2) K_contact M**(-1/2)``.  The backend must reverse-add ghost rows
    to their owners before taking a global maximum.
    """

    if not isinstance(trace, ContactTrace):
        raise TypeError("Contact stability screening requires ContactTrace.")
    selected_dimension = int(dimension)
    if selected_dimension not in {2, 3}:
        raise ValueError("Contact stability dimension must be two or three.")
    masses = np.asarray(nodal_mass, dtype=float)
    if masses.ndim != 2 or masses.shape[1] != selected_dimension:
        raise ValueError("nodal_mass must have shape (number_of_nodes, dimension).")
    if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
        raise ValueError("Nodal masses must be finite and non-negative.")
    if trace.node_ids.size and int(np.max(trace.node_ids)) >= masses.shape[0]:
        raise ValueError("Contact trace references a node outside nodal_mass.")
    referenced = np.unique(trace.node_ids)
    if referenced.size and np.any(masses[referenced] <= 0.0):
        raise ValueError("Every contact degree of freedom must have positive mass.")

    normal = _point_penalties(
        normal_penalty,
        trace.point_count,
        name="Normal contact penalty",
    )
    friction = float(friction_coefficient)
    if not isfinite(friction) or friction < 0.0:
        raise ValueError("friction_coefficient must be finite and non-negative.")
    if tangential_penalty is None or friction == 0.0:
        stiffness = normal
    else:
        tangential = _point_penalties(
            tangential_penalty,
            trace.point_count,
            name="Tangential contact penalty",
        )
        # A generalized Coulomb tangent is bounded by the normal penalty,
        # elastic tangential penalty, and pressure-cap coupling mu * k_n.
        stiffness = normal + tangential + friction * normal

    rows = np.zeros_like(masses)
    for point in range(trace.point_count):
        node_ids = trace.node_ids[point]
        interpolation = np.abs(trace.shape_values[point])
        # The smallest component mass bounds the norm of each diagonal
        # inverse-square-root mass block. This retains cross-component contact
        # coupling without paying an artificial factor equal to the dimension.
        mass_scaled = interpolation / np.sqrt(np.min(masses[node_ids], axis=1))
        contribution = (
            trace.weights[point] * stiffness[point] * mass_scaled * np.sum(mass_scaled)
        )
        np.add.at(rows, node_ids, contribution[:, None])
    return _readonly_array(rows)


def contact_stability_estimate_from_bound(
    spectral_radius_upper_bound: float,
    *,
    safety_factor: float,
    normal_penalty_maximum: float,
    tangential_penalty_maximum: float | None,
    friction_coefficient: float = 0.0,
    point_count: int,
    mass_compatibility: str = "shape_only",
) -> ContactStabilityEstimate:
    """Create a checked estimate from an MPI-global spectral bound."""

    spectral = float(spectral_radius_upper_bound)
    factor = float(safety_factor)
    if not isfinite(spectral) or spectral <= 0.0:
        raise ValueError("Contact spectral-radius bound must be finite and positive.")
    if not isfinite(factor) or not 0.0 < factor <= 1.0:
        raise ValueError("safety_factor must satisfy 0 < value <= 1.")
    unsafed = 2.0 / sqrt(spectral)
    return ContactStabilityEstimate(
        selected=factor * unsafed,
        unsafed_limit=unsafed,
        spectral_radius_upper_bound=spectral,
        safety_factor=factor,
        normal_penalty_maximum=normal_penalty_maximum,
        tangential_penalty_maximum=tangential_penalty_maximum,
        friction_coefficient=friction_coefficient,
        point_count=point_count,
        mass_compatibility=mass_compatibility,
    )


def combine_explicit_stability_bounds(
    contact: ContactStabilityEstimate,
    *,
    noncontact_unsafed_limit: float,
    safety_factor: float | None = None,
) -> CombinedExplicitStabilityEstimate:
    """Compose additive non-contact and contact bounds before selecting ``dt``.

    Taking the minimum of separately derived body and contact time limits is
    not generally conservative because their stiffness operators add.  This
    helper converts the unsafed non-contact limit back to a spectral upper
    bound, adds the contact contribution, and only then applies the safety
    factor. The non-contact limit may already represent body, material, or
    other interface contributions.
    """

    if not isinstance(contact, ContactStabilityEstimate):
        raise TypeError("contact must be ContactStabilityEstimate.")
    noncontact_limit = float(noncontact_unsafed_limit)
    if not isfinite(noncontact_limit) or noncontact_limit <= 0.0:
        raise ValueError("noncontact_unsafed_limit must be finite and positive.")
    factor = contact.safety_factor if safety_factor is None else float(safety_factor)
    if not isfinite(factor) or not 0.0 < factor <= 1.0:
        raise ValueError("safety_factor must satisfy 0 < value <= 1.")
    noncontact_spectral = 4.0 / (noncontact_limit * noncontact_limit)
    total_spectral = noncontact_spectral + contact.spectral_radius_upper_bound
    unsafed = 2.0 / sqrt(total_spectral)
    return CombinedExplicitStabilityEstimate(
        selected=factor * unsafed,
        unsafed_limit=unsafed,
        noncontact_unsafed_limit=noncontact_limit,
        noncontact_spectral_radius_upper_bound=noncontact_spectral,
        contact_spectral_radius_upper_bound=contact.spectral_radius_upper_bound,
        total_spectral_radius_upper_bound=total_spectral,
        safety_factor=factor,
    )


__all__ = [
    "CombinedExplicitStabilityEstimate",
    "ContactStabilityEstimate",
    "combine_explicit_stability_bounds",
    "contact_penalty_local_row_sums",
    "contact_stability_estimate_from_bound",
]
