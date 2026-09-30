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


@dataclass(frozen=True)
class FiniteSlidingSolidBridge:
    """Accepted evidence from the public protocol on a 3D solid bridge.

    This is deliberately a protocol bridge rather than an elementwise
    reproduction of the public Abaqus B31 model.  It exercises AgentFEM's
    ordinary finite-strain Explicit Procedure with a deformable tetrahedral
    solid, actual applied normal resultant, staged contact State, prescribed
    rigid motion, and Coulomb sliding.
    """

    reference: FiniteSlidingContactReference
    assessment: FiniteSlidingContactAssessment
    preload_transfer: dict[str, object]
    preload_relative_normal_balance_error: float
    preload_relative_energy_error: float
    sliding_relative_energy_error: float
    normal_time_increment: float
    sliding_time_increment: float
    preload_steps: int
    sliding_steps: int
    cells: tuple[int, int, int]
    surface_representation: str = "triangulated_piecewise_planar"
    preload_force_tolerance: float = 1.0e-6
    energy_tolerance: float = 1.0e-3

    @property
    def failures(self) -> tuple[str, ...]:
        failures = [f"contact:{item}" for item in self.assessment.failures]
        if not bool(self.preload_transfer.get("equilibrium_accepted", False)):
            failures.append("preload_transfer_not_equilibrated")
        if self.preload_relative_normal_balance_error > self.preload_force_tolerance:
            failures.append("preload_normal_force_balance")
        if self.preload_relative_energy_error > self.energy_tolerance:
            failures.append("preload_energy_balance")
        if self.sliding_relative_energy_error > self.energy_tolerance:
            failures.append("sliding_energy_balance")
        return tuple(failures)

    @property
    def acceptable(self) -> bool:
        return not self.failures

    def summary(self) -> dict[str, object]:
        return {
            "kind": "finite_sliding_solid_protocol_bridge",
            "status": "accepted" if self.acceptable else "failed",
            "acceptable": self.acceptable,
            "external_reference": self.reference.identifier,
            "comparison_level": "public_protocol_bridge_not_b31_reproduction",
            "discretization": "three_dimensional_tetrahedral_cg1_solid",
            "surface_representation": self.surface_representation,
            "cells": list(self.cells),
            "preload_steps": self.preload_steps,
            "sliding_steps": self.sliding_steps,
            "normal_time_increment": self.normal_time_increment,
            "sliding_time_increment": self.sliding_time_increment,
            "preload_transfer": dict(self.preload_transfer),
            "preload_relative_normal_balance_error": (
                self.preload_relative_normal_balance_error
            ),
            "preload_relative_energy_error": self.preload_relative_energy_error,
            "sliding_relative_energy_error": self.sliding_relative_energy_error,
            "preload_force_tolerance": self.preload_force_tolerance,
            "energy_tolerance": self.energy_tolerance,
            "contact_assessment": self.assessment.summary(),
            "failures": list(self.failures),
            "promotion_boundary": (
                "component_and_protocol evidence only; exact B31 reproduction, "
                "mesh/time refinement, and independent curved-tool evidence "
                "remain external promotion gates"
            ),
        }


@dataclass(frozen=True)
class FiniteSlidingSolidRefinement:
    """Separated time- and space-refinement evidence for the solid bridge.

    The three runs keep physical stage durations fixed.  The second run halves
    both stage time increments on the coarse mesh; the third retains those
    increments while refining the in-plane solid mesh.  This is deliberately
    called refinement evidence rather than an observed-order claim.
    """

    coarse: FiniteSlidingSolidBridge
    time_refined: FiniteSlidingSolidBridge
    space_refined: FiniteSlidingSolidBridge
    dissipation_tolerance: float = 1.0e-4
    energy_growth_tolerance: float = 0.05

    @staticmethod
    def _relative_change(first: float, second: float) -> float:
        scale = max(abs(float(first)), abs(float(second)), np.finfo(float).tiny)
        return abs(float(second) - float(first)) / scale

    @property
    def time_relative_dissipation_change(self) -> float:
        return self._relative_change(
            self.coarse.assessment.friction_dissipation,
            self.time_refined.assessment.friction_dissipation,
        )

    @property
    def space_relative_dissipation_change(self) -> float:
        return self._relative_change(
            self.time_refined.assessment.friction_dissipation,
            self.space_refined.assessment.friction_dissipation,
        )

    @property
    def failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        for name, bridge in (
            ("coarse", self.coarse),
            ("time_refined", self.time_refined),
            ("space_refined", self.space_refined),
        ):
            if not bridge.acceptable:
                failures.append(f"{name}_bridge_failed")
            if bridge.assessment.facet_crossing_count <= 0:
                failures.append(f"{name}_lacks_facet_crossing")
        if self.time_refined.normal_time_increment >= self.coarse.normal_time_increment:
            failures.append("preload_time_increment_not_refined")
        if self.time_refined.sliding_time_increment >= self.coarse.sliding_time_increment:
            failures.append("sliding_time_increment_not_refined")
        if self.space_refined.cells == self.time_refined.cells:
            failures.append("space_mesh_not_refined")
        duration_pairs = (
            (
                self.coarse.normal_time_increment * self.coarse.preload_steps,
                self.time_refined.normal_time_increment
                * self.time_refined.preload_steps,
                "preload_time_duration_changed",
            ),
            (
                self.coarse.sliding_time_increment * self.coarse.sliding_steps,
                self.time_refined.sliding_time_increment
                * self.time_refined.sliding_steps,
                "sliding_time_duration_changed",
            ),
            (
                self.time_refined.normal_time_increment
                * self.time_refined.preload_steps,
                self.space_refined.normal_time_increment
                * self.space_refined.preload_steps,
                "preload_space_duration_changed",
            ),
            (
                self.time_refined.sliding_time_increment
                * self.time_refined.sliding_steps,
                self.space_refined.sliding_time_increment
                * self.space_refined.sliding_steps,
                "sliding_space_duration_changed",
            ),
        )
        for first, second, failure in duration_pairs:
            if not np.isclose(first, second, rtol=1.0e-12, atol=0.0):
                failures.append(failure)
        if self.time_relative_dissipation_change > self.dissipation_tolerance:
            failures.append("time_refined_dissipation_changed")
        if self.space_relative_dissipation_change > self.dissipation_tolerance:
            failures.append("space_refined_dissipation_changed")
        allowed_energy = 1.0 + float(self.energy_growth_tolerance)
        if (
            self.time_refined.sliding_relative_energy_error
            > allowed_energy * self.coarse.sliding_relative_energy_error
        ):
            failures.append("time_refined_energy_error_grew")
        if (
            self.space_refined.sliding_relative_energy_error
            > allowed_energy * self.time_refined.sliding_relative_energy_error
        ):
            failures.append("space_refined_energy_error_grew")
        return tuple(failures)

    @property
    def acceptable(self) -> bool:
        return not self.failures

    def summary(self) -> dict[str, object]:
        return {
            "kind": "finite_sliding_solid_protocol_refinement",
            "status": "accepted" if self.acceptable else "failed",
            "acceptable": self.acceptable,
            "scope": "separated_time_and_space_refinement_not_observed_order",
            "time_relative_dissipation_change": (
                self.time_relative_dissipation_change
            ),
            "space_relative_dissipation_change": (
                self.space_relative_dissipation_change
            ),
            "dissipation_tolerance": self.dissipation_tolerance,
            "energy_growth_tolerance": self.energy_growth_tolerance,
            "coarse": self.coarse.summary(),
            "time_refined": self.time_refined.summary(),
            "space_refined": self.space_refined.summary(),
            "failures": list(self.failures),
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


def finite_sliding_solid_protocol_bridge(
    *,
    cells=(1, 1, 1),
    preload_steps: int = 400,
    sliding_steps: int = 750,
    preload_ramp_steps: int = 50,
    penalty_factor: float = 20.0,
    mass_damping: float = 2.0e4,
    preload_stability_scale: float = 1.0,
    sliding_stability_scale: float = 0.8,
    preload_dt: float | None = None,
    sliding_dt: float | None = None,
    comm=None,
) -> FiniteSlidingSolidBridge:
    """Run a real two-stage solid-contact bridge to the public protocol.

    The first stage ramps the declared 500-unit surface resultant into a
    frictionless rigid plane and reaches an accepted equilibrium.  The second
    stage keeps that load, transfers the accepted configuration atomically,
    activates ``mu=0.3``, and moves the rigid plane by 0.1 units.  A second
    Model shares the same physical displacement field so time-local load and
    tool schedules restart at zero without resetting the accepted structure.
    """

    from mpi4py import MPI

    from agentfem import (
        amplitudes,
        boundary_models,
        constitutive,
        constraints,
        fields,
        mesh,
        models,
        studies,
    )

    selected_comm = MPI.COMM_SELF if comm is None else comm
    selected_cells = tuple(int(value) for value in cells)
    if len(selected_cells) != 3 or any(value < 1 for value in selected_cells):
        raise ValueError("Solid bridge cells must contain three positive integers.")
    selected_preload_steps = int(preload_steps)
    selected_sliding_steps = int(sliding_steps)
    selected_ramp_steps = int(preload_ramp_steps)
    if min(selected_preload_steps, selected_sliding_steps, selected_ramp_steps) < 1:
        raise ValueError("Solid bridge step counts must be positive.")
    if selected_ramp_steps >= selected_preload_steps:
        raise ValueError("Preload ramp must end before the preload stage.")
    selected_penalty_factor = float(penalty_factor)
    selected_damping = float(mass_damping)
    selected_preload_scale = float(preload_stability_scale)
    selected_stability_scale = float(sliding_stability_scale)
    selected_preload_dt = None if preload_dt is None else float(preload_dt)
    selected_sliding_dt = None if sliding_dt is None else float(sliding_dt)
    if not np.isfinite(selected_penalty_factor) or selected_penalty_factor <= 0.0:
        raise ValueError("penalty_factor must be finite and positive.")
    if not np.isfinite(selected_damping) or selected_damping < 0.0:
        raise ValueError("mass_damping must be finite and nonnegative.")
    if not 0.0 < selected_preload_scale <= 1.0:
        raise ValueError("preload_stability_scale must lie in (0, 1].")
    if not 0.0 < selected_stability_scale <= 1.0:
        raise ValueError("sliding_stability_scale must lie in (0, 1].")
    if selected_preload_dt is not None and (
        not np.isfinite(selected_preload_dt) or selected_preload_dt <= 0.0
    ):
        raise ValueError("preload_dt must be finite and positive when provided.")
    if selected_sliding_dt is not None and (
        not np.isfinite(selected_sliding_dt) or selected_sliding_dt <= 0.0
    ):
        raise ValueError("sliding_dt must be finite and positive when provided.")

    reference = abaqus_explicit_finite_sliding_reference()
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.1),
        selected_cells,
        comm=selected_comm,
        cell_type="tetrahedron",
    )
    bottom = mesh.boundary(
        domain,
        lambda x: np.isclose(x[2], 0.0),
        name="contact_slave",
        tag=41,
    )
    top = mesh.boundary(
        domain,
        lambda x: np.isclose(x[2], 0.1),
        name="normal_load",
        tag=42,
    )
    x_symmetry = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 0.0),
        name="x_symmetry",
        tag=43,
    )
    y_symmetry = mesh.boundary(
        domain,
        lambda x: np.isclose(x[1], 0.0),
        name="y_symmetry",
        tag=44,
    )
    surface = _triangulated_protocol_plane(boundary_models)
    penalty = selected_penalty_factor * reference.young

    preload_model = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"),
        mesh=domain,
        name="finite_sliding_normal_preload",
    )
    displacement = preload_model.field(fields.displacement(domain))
    preload_material = preload_model.material(
        constitutive.neo_hookean(
            young=reference.young,
            poisson=reference.poisson,
            density=reference.density,
        )
    )
    preload_model.constraint(
        constraints.component_dirichlet(
            displacement, 0, on=x_symmetry, value=0.0, name="x_symmetry"
        )
    )
    preload_model.constraint(
        constraints.component_dirichlet(
            displacement, 1, on=y_symmetry, value=0.0, name="y_symmetry"
        )
    )
    fixed_body = boundary_models.rigid_body(surface, name="fixed_plane")
    normal_pair = boundary_models.rigid_contact_pair(
        bottom,
        fixed_body,
        penalty=penalty,
        name="normal_contact",
    )
    stability_probe = preload_model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=preload_material,
        contact_pairs=(normal_pair,),
        steps=1,
        progress=False,
        name="normal_stability_probe",
    )
    normal_dt = (
        selected_preload_scale * float(stability_probe.dt)
        if selected_preload_dt is None
        else selected_preload_dt
    )
    preload_model.surface_force(
        (0.0, 0.0, -reference.normal_load),
        on=top,
        amplitude=amplitudes.smooth_step(
            end_time=selected_ramp_steps * normal_dt,
            name="normal_load_ramp",
        ),
        name="normal_load",
    )
    preload = preload_model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=preload_material,
        contact_pairs=(normal_pair,),
        dt=normal_dt,
        steps=selected_preload_steps,
        mass_damping=selected_damping,
        history_every=max(1, selected_preload_steps // 20),
        progress=False,
        name="normal_preload",
    )
    preload.run()
    normal_residual = _single_contact_residual(preload.residual)
    normal_evidence = normal_residual.accepted_evidence
    preload_projection = normal_residual.lifecycle.state.accepted
    normal_force = abs(float(np.dot(normal_evidence.contact_force_on_structure, (0, 0, 1))))
    preload_balance = abs(normal_force - reference.normal_load) / reference.normal_load

    sliding_model = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"),
        mesh=domain,
        name="finite_sliding_stage",
    )
    sliding_displacement = sliding_model.field(displacement)
    sliding_material = sliding_model.material(
        constitutive.neo_hookean(
            young=reference.young,
            poisson=reference.poisson,
            density=reference.density,
        )
    )
    sliding_model.constraint(
        constraints.component_dirichlet(
            sliding_displacement,
            0,
            on=x_symmetry,
            value=0.0,
            name="x_symmetry",
        )
    )
    sliding_model.constraint(
        constraints.component_dirichlet(
            sliding_displacement,
            1,
            on=y_symmetry,
            value=0.0,
            name="y_symmetry",
        )
    )
    sliding_model.surface_force(
        (0.0, 0.0, -reference.normal_load),
        on=top,
        name="held_normal_load",
    )

    def sliding_pair(end_time: float, *, name: str):
        schedule = boundary_models.prescribed_rigid_motion_schedule(
            boundary_models.prescribed_rigid_motion(
                translation=(reference.sliding_displacement, 0.0, 0.0),
                name="public_tangential_slide",
            ),
            end_time=end_time,
            name="public_tangential_slide_schedule",
        )
        body = boundary_models.rigid_body(
            surface,
            motion_schedule=schedule,
            name="sliding_plane",
        )
        return boundary_models.rigid_contact_pair(
            bottom,
            body,
            penalty=penalty,
            friction_coefficient=reference.friction_coefficient,
            tangential_penalty=penalty,
            name=name,
        )

    sliding_probe_pair = sliding_pair(1.0, name="sliding_stability_contact")
    sliding_probe = sliding_model.finite_strain_explicit_dynamics_step(
        target=sliding_displacement,
        material=sliding_material,
        contact_pairs=(sliding_probe_pair,),
        steps=1,
        progress=False,
        name="sliding_stability_probe",
    )
    selected_stage_dt = (
        selected_stability_scale * float(sliding_probe.dt)
        if selected_sliding_dt is None
        else selected_sliding_dt
    )
    active_pair = sliding_pair(
        selected_sliding_steps * selected_stage_dt,
        name="finite_sliding_contact",
    )
    sliding = sliding_model.finite_strain_explicit_dynamics_step(
        target=sliding_displacement,
        material=sliding_material,
        contact_pairs=(active_pair,),
        dt=selected_stage_dt,
        steps=selected_sliding_steps,
        mass_damping=selected_damping,
        history_every=max(1, selected_sliding_steps // 20),
        progress=False,
        name="finite_sliding",
    )
    transfer = sliding.initialize_from_preload(
        displacement.value,
        source_step=preload,
        force_tolerance=max(1.0e-4, reference.normal_load * 1.0e-8),
    )
    sliding.run()
    sliding_residual = _single_contact_residual(sliding.residual)
    sliding_evidence = sliding_residual.accepted_evidence
    sliding_projection = sliding_residual.lifecycle.state.accepted
    facet_crossings = _global_facet_crossing_count(
        preload_projection,
        sliding_projection,
        selected_comm,
    )
    assessment = assess_finite_sliding_contact(
        contact_force_on_structure=sliding_evidence.contact_force_on_structure,
        contact_force_on_surface=sliding_evidence.contact_force_on_surface,
        admissible_normal=(0.0, 0.0, 1.0),
        young=reference.young,
        poisson=reference.poisson,
        density=reference.density,
        applied_normal_load=reference.normal_load,
        friction_coefficient=reference.friction_coefficient,
        sliding_displacement=reference.sliding_displacement,
        active_point_count=sliding_evidence.active_point_count,
        sliding_point_count=sliding_evidence.sliding_point_count,
        invalid_point_count=sliding_evidence.invalid_point_count,
        friction_dissipation=sliding_evidence.friction_dissipation,
        facet_crossing_count=facet_crossings,
        require_facet_crossing=True,
        force_tolerance=1.0e-3,
        reference=reference,
    )
    return FiniteSlidingSolidBridge(
        reference=reference,
        assessment=assessment,
        preload_transfer=transfer.summary(),
        preload_relative_normal_balance_error=preload_balance,
        preload_relative_energy_error=float(
            preload.history_records[-1]["relative_energy_balance_error"]
        ),
        sliding_relative_energy_error=float(
            sliding.history_records[-1]["relative_energy_balance_error"]
        ),
        normal_time_increment=normal_dt,
        sliding_time_increment=selected_stage_dt,
        preload_steps=selected_preload_steps,
        sliding_steps=selected_sliding_steps,
        cells=selected_cells,
    )


def finite_sliding_solid_protocol_refinement(
    *,
    coarse_cells=(1, 1, 1),
    refined_cells=(2, 2, 1),
    comm=None,
) -> FiniteSlidingSolidRefinement:
    """Run a separated two-axis refinement check at fixed stage durations."""

    coarse = finite_sliding_solid_protocol_bridge(
        cells=coarse_cells,
        comm=comm,
    )
    time_refined = finite_sliding_solid_protocol_bridge(
        cells=coarse_cells,
        preload_steps=2 * coarse.preload_steps,
        sliding_steps=2 * coarse.sliding_steps,
        preload_ramp_steps=100,
        preload_stability_scale=0.5,
        sliding_stability_scale=0.4,
        comm=comm,
    )
    space_refined = finite_sliding_solid_protocol_bridge(
        cells=refined_cells,
        preload_steps=time_refined.preload_steps,
        sliding_steps=time_refined.sliding_steps,
        preload_ramp_steps=100,
        preload_dt=time_refined.normal_time_increment,
        sliding_dt=time_refined.sliding_time_increment,
        comm=comm,
    )
    return FiniteSlidingSolidRefinement(
        coarse=coarse,
        time_refined=time_refined,
        space_refined=space_refined,
    )


def _single_contact_residual(residual):
    """Return the single contact Operator below transparent wrappers."""

    selected = residual
    while not hasattr(selected, "accepted_evidence"):
        selected = getattr(selected, "base", None)
        if selected is None:
            raise TypeError("Benchmark residual contains no explicit contact Operator.")
    return selected


def _triangulated_protocol_plane(boundary_models):
    """Return a bounded coplanar tool whose stable facets must be crossed.

    The x-breaks lie between each slave point's initial and final coordinates
    in the moving tool frame.  The bridge therefore proves that closest-point
    identity is updated during finite sliding rather than merely exercising a
    triangulated surface without leaving the original facet.
    """

    # The CG1 tetrahedral boundary trace is integrated at triangle centroids
    # (x = 1/6, 1/3, 2/3, 5/6 on this mesh), not only at corner nodes.  Breaks
    # at 1/4 and 3/4 are crossed by the 1/3 and 5/6 traces under a 0.1 slide.
    x_coordinates = (-0.3, 0.25, 0.75, 1.3)
    y_coordinates = (-0.3, 0.5, 1.3)
    vertices = np.asarray(
        [(x, y, 0.0) for y in y_coordinates for x in x_coordinates],
        dtype=float,
    )
    column_count = len(x_coordinates)
    triangles: list[tuple[int, int, int]] = []
    for row in range(len(y_coordinates) - 1):
        for column in range(column_count - 1):
            lower_left = row * column_count + column
            lower_right = lower_left + 1
            upper_left = lower_left + column_count
            upper_right = upper_left + 1
            triangles.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    return boundary_models.triangulated_rigid_surface(
        vertices=vertices,
        triangles=np.asarray(triangles, dtype=np.int64),
        facet_ids=np.arange(100, 100 + len(triangles), dtype=np.int64),
        name="public_protocol_triangulated_plane",
    )


def _global_facet_crossing_count(initial, final, comm) -> int:
    """Count point-keyed accepted master-facet changes across all ranks."""

    if initial is None or final is None:
        raise RuntimeError("Finite-sliding bridge lacks accepted projection State.")
    if not np.array_equal(initial.point_ids, final.point_ids):
        raise RuntimeError("Finite-sliding bridge contact point identities differ.")
    initial_entities = initial.projection.entity_ids
    final_entities = final.projection.entity_ids
    if initial_entities is None or final_entities is None:
        raise RuntimeError("Triangulated finite sliding lacks stable facet identity.")
    local = int(np.count_nonzero(initial_entities != final_entities))
    return int(comm.allreduce(local))


__all__ = [
    "FiniteSlidingContactAssessment",
    "FiniteSlidingContactReference",
    "FiniteSlidingSolidBridge",
    "FiniteSlidingSolidRefinement",
    "abaqus_explicit_finite_sliding_reference",
    "assess_finite_sliding_contact",
    "finite_sliding_solid_protocol_bridge",
    "finite_sliding_solid_protocol_refinement",
]
