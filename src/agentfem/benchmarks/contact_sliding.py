# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""External-reference contracts for finite-sliding rigid contact."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_ABAQUS_REFERENCE = (
    "https://docs.software.vt.edu/abaqusv2025/English/"
    "SIMACAEVERRefMap/simaver-c-finslcontdefrigid.htm"
)
_ABAQUS_EXPLICIT_INPUT = (
    "https://docs.software.vt.edu/abaqusv2025/English/"
    "SIMAINPRefResources/cpair_beam3d_xpl.inp"
)


@dataclass(frozen=True)
class FiniteSlidingContactReference:
    """Public two-stage contact protocol and its declared invariants."""

    identifier: str
    reference: str
    input_deck: str
    young: float
    poisson: float
    density: float
    friction_coefficient: float
    normal_load: float
    sliding_displacement: float
    protocol: tuple[str, ...]
    invariants: tuple[str, ...]

    def summary(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "reference": self.reference,
            "input_deck": self.input_deck,
            "young": self.young,
            "poisson": self.poisson,
            "density": self.density,
            "friction_coefficient": self.friction_coefficient,
            "normal_load": self.normal_load,
            "sliding_displacement": self.sliding_displacement,
            "protocol": list(self.protocol),
            "invariants": list(self.invariants),
        }


@dataclass(frozen=True)
class FiniteSlidingContactAssessment:
    """Force, friction, search, and energy evidence for one sliding state."""

    reference: FiniteSlidingContactReference
    relative_normal_balance_error: float
    relative_coulomb_cap_error: float
    relative_action_reaction_error: float
    active_point_count: int
    sliding_point_count: int
    invalid_point_count: int
    facet_crossing_count: int
    friction_dissipation: float
    force_tolerance: float
    require_facet_crossing: bool
    protocol_mismatches: tuple[str, ...]

    @property
    def failures(self) -> tuple[str, ...]:
        """Return stable, machine-readable reasons for rejection."""

        failures: list[str] = []
        if self.relative_normal_balance_error > self.force_tolerance:
            failures.append("normal_force_balance")
        if self.relative_coulomb_cap_error > self.force_tolerance:
            failures.append("coulomb_cap")
        if self.relative_action_reaction_error > self.force_tolerance:
            failures.append("action_reaction")
        if self.active_point_count <= 0:
            failures.append("no_active_contact")
        if self.sliding_point_count <= 0:
            failures.append("no_sliding_contact")
        if self.sliding_point_count > self.active_point_count:
            failures.append("sliding_count_exceeds_active_count")
        if self.invalid_point_count != 0:
            failures.append("invalid_projection")
        if self.friction_dissipation < 0.0:
            failures.append("negative_friction_dissipation")
        if self.require_facet_crossing and self.facet_crossing_count <= 0:
            failures.append("no_facet_crossing")
        failures.extend(
            f"protocol_parameter:{name}" for name in self.protocol_mismatches
        )
        return tuple(failures)

    @property
    def acceptable(self) -> bool:
        return not self.failures

    def summary(self) -> dict[str, object]:
        return {
            "kind": "finite_sliding_contact_assessment",
            "reference": self.reference.identifier,
            "acceptable": self.acceptable,
            "relative_normal_balance_error": self.relative_normal_balance_error,
            "relative_coulomb_cap_error": self.relative_coulomb_cap_error,
            "relative_action_reaction_error": self.relative_action_reaction_error,
            "active_point_count": self.active_point_count,
            "sliding_point_count": self.sliding_point_count,
            "invalid_point_count": self.invalid_point_count,
            "facet_crossing_count": self.facet_crossing_count,
            "friction_dissipation": self.friction_dissipation,
            "force_tolerance": self.force_tolerance,
            "require_facet_crossing": self.require_facet_crossing,
            "protocol_parameter_match": not self.protocol_mismatches,
            "protocol_mismatches": list(self.protocol_mismatches),
            "failures": list(self.failures),
            "status": "accepted" if self.acceptable else "failed",
        }


def abaqus_explicit_finite_sliding_reference() -> FiniteSlidingContactReference:
    """Return the public Abaqus/Explicit B31 finite-sliding protocol.

    The reference is used as an external protocol, not as permission to treat
    a different AgentFEM discretization as a pointwise Abaqus reproduction.
    Abaqus first establishes frictionless normal contact under a 500-unit
    distributed beam load, then redefines the pair with ``mu=0.3`` and applies
    0.1 units of tangential displacement.
    """

    return FiniteSlidingContactReference(
        identifier="abaqus_explicit_deformable_rigid_finite_sliding_b31",
        reference=_ABAQUS_REFERENCE,
        input_deck=_ABAQUS_EXPLICIT_INPUT,
        young=30.0e6,
        poisson=0.3,
        density=0.284,
        friction_coefficient=0.3,
        normal_load=500.0,
        sliding_displacement=0.1,
        protocol=(
            "establish frictionless normal contact",
            "commit the accepted contact state",
            "activate penalty Coulomb friction",
            "apply prescribed tangential sliding",
        ),
        invariants=(
            "normal contact force balances the applied normal load",
            "sliding traction reaches the Coulomb cap",
            "contact action and reaction balance",
            "frictional dissipation is nonnegative",
        ),
    )


def assess_finite_sliding_contact(
    *,
    contact_force_on_structure,
    contact_force_on_surface,
    admissible_normal,
    young: float,
    poisson: float,
    density: float,
    applied_normal_load: float,
    friction_coefficient: float,
    sliding_displacement: float,
    active_point_count: int,
    sliding_point_count: int,
    invalid_point_count: int,
    friction_dissipation: float,
    facet_crossing_count: int = 0,
    require_facet_crossing: bool = False,
    force_tolerance: float = 1.0e-8,
    reference: FiniteSlidingContactReference | None = None,
) -> FiniteSlidingContactAssessment:
    """Assess external-protocol invariants from accepted contact evidence."""

    selected_reference = reference or abaqus_explicit_finite_sliding_reference()
    structure = np.asarray(contact_force_on_structure, dtype=float)
    surface = np.asarray(contact_force_on_surface, dtype=float)
    normal = np.asarray(admissible_normal, dtype=float)
    if structure.ndim != 1 or structure.shape not in {(2,), (3,)}:
        raise ValueError("Contact force must be one 2D or 3D vector.")
    if surface.shape != structure.shape or normal.shape != structure.shape:
        raise ValueError("Contact forces and normal must have the same dimension.")
    if not all(np.all(np.isfinite(value)) for value in (structure, surface, normal)):
        raise ValueError("Contact forces and normal must be finite.")
    normal_length = float(np.linalg.norm(normal))
    if not np.isclose(normal_length, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("Contact benchmark normal must be a unit vector.")
    applied = float(applied_normal_load)
    coefficient = float(friction_coefficient)
    selected_young = float(young)
    selected_poisson = float(poisson)
    selected_density = float(density)
    selected_sliding = float(sliding_displacement)
    tolerance = float(force_tolerance)
    dissipation = float(friction_dissipation)
    if not np.isfinite(applied) or applied <= 0.0:
        raise ValueError("applied_normal_load must be finite and positive.")
    if not np.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("friction_coefficient must be finite and positive.")
    if not np.isfinite(selected_young) or selected_young <= 0.0:
        raise ValueError("young must be finite and positive.")
    if not np.isfinite(selected_poisson) or not -1.0 < selected_poisson < 0.5:
        raise ValueError("poisson must be finite and lie in (-1, 0.5).")
    if not np.isfinite(selected_density) or selected_density <= 0.0:
        raise ValueError("density must be finite and positive.")
    if not np.isfinite(selected_sliding) or selected_sliding <= 0.0:
        raise ValueError("sliding_displacement must be finite and positive.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("force_tolerance must be finite and positive.")
    if not np.isfinite(dissipation):
        raise ValueError("friction_dissipation must be finite.")
    raw_counts = (
        active_point_count,
        sliding_point_count,
        invalid_point_count,
        facet_crossing_count,
    )
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw_counts
    ):
        raise TypeError("Contact benchmark counts must be integers.")
    counts = tuple(int(value) for value in raw_counts)
    if any(value < 0 for value in counts):
        raise ValueError("Contact benchmark counts cannot be negative.")
    if counts[1] > counts[0]:
        raise ValueError("sliding_point_count cannot exceed active_point_count.")

    signed_normal = float(np.dot(structure, normal))
    normal_force = abs(signed_normal)
    tangential_force = float(
        np.linalg.norm(structure - signed_normal * normal)
    )
    coulomb_force = coefficient * normal_force
    normal_scale = max(applied, np.finfo(float).tiny)
    friction_scale = max(coulomb_force, np.finfo(float).tiny)
    reaction_scale = max(float(np.linalg.norm(structure)), applied)
    protocol_values = {
        "young": (selected_young, selected_reference.young),
        "poisson": (selected_poisson, selected_reference.poisson),
        "density": (selected_density, selected_reference.density),
        "normal_load": (applied, selected_reference.normal_load),
        "friction_coefficient": (
            coefficient,
            selected_reference.friction_coefficient,
        ),
        "sliding_displacement": (
            selected_sliding,
            selected_reference.sliding_displacement,
        ),
    }
    protocol_mismatches = tuple(
        name
        for name, (observed, expected) in protocol_values.items()
        if not np.isclose(observed, expected, rtol=1.0e-12, atol=0.0)
    )
    return FiniteSlidingContactAssessment(
        reference=selected_reference,
        relative_normal_balance_error=abs(normal_force - applied) / normal_scale,
        relative_coulomb_cap_error=abs(tangential_force - coulomb_force)
        / friction_scale,
        relative_action_reaction_error=float(np.linalg.norm(structure + surface))
        / reaction_scale,
        active_point_count=counts[0],
        sliding_point_count=counts[1],
        invalid_point_count=counts[2],
        facet_crossing_count=counts[3],
        friction_dissipation=dissipation,
        force_tolerance=tolerance,
        require_facet_crossing=bool(require_facet_crossing),
        protocol_mismatches=protocol_mismatches,
    )


__all__ = [
    "FiniteSlidingContactAssessment",
    "FiniteSlidingContactReference",
    "abaqus_explicit_finite_sliding_reference",
    "assess_finite_sliding_contact",
]
