# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Assembled mixed-mode-bending verification providers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .mixed_mode import (
    DelaminationBenchmarkSpec,
    DelaminationEnergyReleaseCurve,
    mmb_beam_energy_release_curve,
)


@dataclass(frozen=True)
class MMBFiniteElementPoint:
    """One rigid-lever controlled solution of a discretized MMB specimen."""

    crack_length: float
    effective_crack_length: float
    load: float
    displacement: float
    compliance: float
    control_residual: float
    element_size: float
    elements_per_arm: int
    newton_iterations: int
    residual_norm: float
    vcct_mode_i_energy_release_rate: float
    vcct_mode_ii_energy_release_rate: float
    vcct_total_energy_release_rate: float
    vcct_mode_i_fraction: float
    bulk_material: dict[str, object] | None = None

    def summary(self) -> dict[str, object]:
        return {"kind": "mmb_finite_element_point", **self.__dict__}


@dataclass(frozen=True)
class MMBFiniteElementCurve:
    """Rigid-lever MMB compliance and energy-release evidence."""

    specification: DelaminationBenchmarkSpec
    points: tuple[MMBFiniteElementPoint, ...]
    energy_release: DelaminationEnergyReleaseCurve
    lever_length: float
    source: str
    poisson: float | None = None
    assumption: str | None = None
    interface_stiffness: float | None = None
    bulk_material: dict[str, object] | None = None

    @property
    def element_size(self) -> float:
        return max(point.element_size for point in self.points)

    @property
    def identity_sha256(self) -> str:
        encoded = json.dumps(
            self.summary(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.mmb-finite-element-curve.v1",
            "specification": self.specification.summary(),
            "source": self.source,
            "lever_length": self.lever_length,
            "model": {
                "poisson": self.poisson,
                "assumption": self.assumption,
                "interface_stiffness": self.interface_stiffness,
                "bulk_material": self.bulk_material,
            },
            "points": [point.summary() for point in self.points],
            "energy_release": self.energy_release.summary(),
            "discretization": {
                "cell": "Q1 quadrilateral",
                "interface": "paired zero-thickness line facets",
                "fixture": "exact scalar rigid-lever kinematic control",
                "control": "q=(c/L)u_Ay-((c+L)/L)u_By",
                "reaction": "work-conjugate scalar Lagrange multiplier",
                "precrack": "fully failed interface facets",
            },
            "mode_partition": (
                "independent two-dimensional VCCT; Reeder--Crews is comparison only"
            ),
        }


@dataclass(frozen=True)
class MMBComplianceCertificate:
    """Assembled MMB compliance comparison with a Reeder--Crews oracle."""

    curve_identity_sha256: str
    reference_identity_sha256: str
    compliance_relative_l2_error: float
    compliance_relative_tolerance: float
    maximum_control_residual: float
    control_residual_tolerance: float
    maximum_newton_residual: float
    newton_residual_tolerance: float
    accepted: bool

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.mmb-compliance-certificate.v1",
            **self.__dict__,
            "scope": "assembled elastic MMB fixture compliance",
            "excludes": (
                "cohesive propagation",
                "external experimental curve agreement",
                "independent finite-element mode partition",
            ),
        }


@dataclass(frozen=True)
class MMBModePartitionCertificate:
    """Independent VCCT comparison with the Reeder--Crews beam partition."""

    curve_identity_sha256: str
    reference_identity_sha256: str
    vcct_compliance_energy_relative_l2_error: float
    energy_closure_relative_tolerance: float
    mode_i_fraction_maximum_error: float
    mode_i_fraction_absolute_tolerance: float
    accepted: bool

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.mmb-mode-partition-certificate.v1",
            **self.__dict__,
            "computed_method": "two-dimensional virtual crack closure technique",
            "energy_closure": "VCCT total G versus assembled compliance derivative",
            "reference_method": "Reeder--Crews classical simple-beam partition",
            "scope": "elastic, straight, equal-arm MMB specimen",
            "excludes": (
                "cohesive propagation mode partition",
                "dissimilar-arm interface oscillation",
                "three-dimensional free-edge effects",
            ),
        }


@dataclass(frozen=True)
class MMBCohesivePropagationPoint:
    """One accepted rigid-lever MMB cohesive increment."""

    increment: int
    displacement: float
    reaction: float
    bulk_strain_energy: float
    cohesive_stored_energy: float
    cohesive_dissipation: float
    external_work: float
    energy_balance_error: float
    relative_energy_balance_error: float
    maximum_damage: float
    damaged_length: float
    failed_length: float
    process_zone_length: float
    damage_weighted_mode_i_fraction: float
    beam_mode_i_fraction: float
    newton_iterations: int
    residual_norm: float
    control_residual: float
    accepted_subincrements: int = 1
    cutbacks: int = 0

    def summary(self) -> dict[str, object]:
        return {"kind": "mmb_cohesive_propagation_point", **self.__dict__}


@dataclass(frozen=True)
class MMBCohesivePropagationCurve:
    """Accepted mixed-mode cohesive path and work--energy evidence."""

    specification: DelaminationBenchmarkSpec
    points: tuple[MMBCohesivePropagationPoint, ...]
    lever_length: float
    element_size: float
    process_zone_elements: float
    law: dict[str, object]
    source: str
    poisson: float
    assumption: str
    bulk_material: dict[str, object] | None = None

    @property
    def identity_sha256(self) -> str:
        encoded = json.dumps(
            self.summary(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def peak_reaction(self) -> float:
        return max(point.reaction for point in self.points)

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.mmb-cohesive-propagation-curve.v1",
            "specification": self.specification.summary(),
            "source": self.source,
            "lever_length": self.lever_length,
            "element_size": self.element_size,
            "process_zone_elements": self.process_zone_elements,
            "law": dict(self.law),
            "model": {
                "poisson": self.poisson,
                "assumption": self.assumption,
                "bulk_material": self.bulk_material,
            },
            "points": [point.summary() for point in self.points],
            "evidence_scope": (
                "rigid-lever displacement-controlled mixed-mode cohesive propagation"
            ),
            "mode_partition": {
                "local": "damage-weighted cohesive energy-mode fraction",
                "reference": "Reeder--Crews beam partition",
            },
        }


@dataclass(frozen=True)
class MMBCohesivePropagationCertificate:
    """Fail-closed certificate for one assembled mixed-mode propagation path."""

    curve_identity_sha256: str
    propagation_observed: bool
    maximum_relative_energy_error: float
    energy_relative_tolerance: float
    process_zone_elements: float
    required_process_zone_elements: float
    maximum_control_residual: float
    control_residual_tolerance: float
    maximum_newton_residual: float
    newton_residual_tolerance: float
    final_mode_i_fraction: float
    beam_mode_i_fraction: float
    mode_i_fraction_absolute_error: float
    accepted: bool

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.mmb-cohesive-propagation-certificate.v1",
            **self.__dict__,
            "scope": "assembled MMB propagation, energy and local mode evidence",
            "mode_partition_status": (
                "diagnostic only: local cohesive separation mix is not a global "
                "energy-release-rate partition"
            ),
            "excludes": (
                "external experimental curve agreement",
                "independent finite-element global mode partition",
                "mesh-converged peak load",
                "unstable branches skipped by displacement control",
            ),
        }


def certify_mmb_compliance(
    curve: MMBFiniteElementCurve,
    *,
    compliance_relative_tolerance: float,
    control_residual_tolerance: float = 1.0e-10,
    newton_residual_tolerance: float = 1.0e-8,
) -> MMBComplianceCertificate:
    """Compare one assembled fixture curve with its declared beam oracle."""

    if not isinstance(curve, MMBFiniteElementCurve):
        raise TypeError("MMB compliance certification requires an assembled curve.")
    cracks = np.asarray(
        [point.effective_crack_length for point in curve.points], dtype=float
    )
    loads = np.asarray([point.load for point in curve.points], dtype=float)
    computed = np.asarray([point.compliance for point in curve.points], dtype=float)
    reference = mmb_beam_energy_release_curve(
        curve.specification,
        crack_length=cracks,
        load=loads,
        lever_length=curve.lever_length,
    )
    scale = float(np.linalg.norm(reference.compliance))
    error = float(np.linalg.norm(computed - reference.compliance) / scale)
    control_error = max(abs(point.control_residual) for point in curve.points)
    residual = max(point.residual_norm for point in curve.points)
    limits = (
        float(compliance_relative_tolerance),
        float(control_residual_tolerance),
        float(newton_residual_tolerance),
    )
    if any(not np.isfinite(value) or value < 0.0 for value in limits):
        raise ValueError("MMB certificate tolerances must be finite and nonnegative.")
    return MMBComplianceCertificate(
        curve_identity_sha256=curve.identity_sha256,
        reference_identity_sha256=reference.identity_sha256,
        compliance_relative_l2_error=error,
        compliance_relative_tolerance=limits[0],
        maximum_control_residual=float(control_error),
        control_residual_tolerance=limits[1],
        maximum_newton_residual=float(residual),
        newton_residual_tolerance=limits[2],
        accepted=(
            error <= limits[0]
            and control_error <= limits[1]
            and residual <= limits[2]
        ),
    )


def certify_mmb_mode_partition(
    curve: MMBFiniteElementCurve,
    *,
    energy_closure_relative_tolerance: float = 0.08,
    mode_i_fraction_absolute_tolerance: float = 0.08,
) -> MMBModePartitionCertificate:
    """Compare independently recovered VCCT channels with beam theory.

    The assembled result is obtained from crack-tip nodal forces and the
    relative displacement one element behind the tip.  Reeder--Crews is used
    only after the solve as an external analytical comparison; it is not used
    to populate the computed ``G_I`` or ``G_II`` channels.
    """

    if not isinstance(curve, MMBFiniteElementCurve):
        raise TypeError("MMB mode-partition certification requires an assembled curve.")
    limits = (
        float(energy_closure_relative_tolerance),
        float(mode_i_fraction_absolute_tolerance),
    )
    if any(not np.isfinite(value) or value < 0.0 for value in limits):
        raise ValueError("MMB mode-partition tolerances must be finite and nonnegative.")
    reference = mmb_beam_energy_release_curve(
        curve.specification,
        crack_length=[point.effective_crack_length for point in curve.points],
        load=[point.load for point in curve.points],
        lever_length=curve.lever_length,
    )
    predicted = curve.energy_release
    crack = np.asarray(
        [point.effective_crack_length for point in curve.points], dtype=float
    )
    loads = np.asarray([point.load for point in curve.points], dtype=float)
    compliance = np.asarray([point.compliance for point in curve.points], dtype=float)
    compliance_energy = (
        loads**2
        * np.gradient(compliance, crack, edge_order=2)
        / (2.0 * curve.specification.width)
    )
    total_scale = float(np.linalg.norm(compliance_energy))
    total_error = float(
        np.linalg.norm(
            predicted.total_energy_release_rate
            - compliance_energy
        )
        / max(total_scale, np.finfo(float).eps)
    )
    reference_fraction = (
        reference.mode_i_energy_release_rate
        / reference.total_energy_release_rate
    )
    predicted_fraction = (
        predicted.mode_i_energy_release_rate
        / predicted.total_energy_release_rate
    )
    fraction_error = float(np.max(np.abs(predicted_fraction - reference_fraction)))
    return MMBModePartitionCertificate(
        curve_identity_sha256=curve.identity_sha256,
        reference_identity_sha256=reference.identity_sha256,
        vcct_compliance_energy_relative_l2_error=total_error,
        energy_closure_relative_tolerance=limits[0],
        mode_i_fraction_maximum_error=fraction_error,
        mode_i_fraction_absolute_tolerance=limits[1],
        accepted=bool(total_error <= limits[0] and fraction_error <= limits[1]),
    )


def mmb_finite_element_curve(
    spec: DelaminationBenchmarkSpec,
    *,
    crack_length,
    control_displacement: float,
    lever_length: float,
    elements_along: int,
    elements_per_arm: int,
    poisson: float = 0.3,
    assumption: str = "plane_stress",
    interface_stiffness: float | None = None,
    bulk_material=None,
    solver_options=None,
) -> MMBFiniteElementCurve:
    """Solve elastic precracked MMB points with an exact rigid-lever control."""

    if not isinstance(spec, DelaminationBenchmarkSpec) or spec.kind != "mmb":
        raise ValueError("mmb_finite_element_curve requires an MMB specification.")
    cracks = np.asarray(crack_length, dtype=float).reshape(-1)
    control_value = float(control_displacement)
    lever = float(lever_length)
    length = 2.0 * float(spec.half_span)
    nx = int(elements_along)
    ny = int(elements_per_arm)
    if cracks.size < 3 or np.any(cracks <= 0.0) or np.any(np.diff(cracks) <= 0.0):
        raise ValueError("MMB finite-element evidence needs three increasing cracks.")
    if (
        not np.isfinite(control_value)
        or control_value <= 0.0
        or not np.isfinite(lever)
        or lever < float(spec.half_span) / 3.0
        or cracks[-1] >= length
        or nx < 4
        or nx % 2
        or ny < 1
    ):
        raise ValueError("MMB control, lever, geometry or mesh counts are invalid.")
    dx = length / nx
    indices = np.rint(cracks / dx).astype(int)
    effective = indices * dx
    tolerance = 128.0 * np.finfo(float).eps * max(length, 1.0)
    if np.any(np.abs(effective - cracks) > tolerance):
        raise ValueError("Every MMB crack length must align with the axial mesh.")
    selected_assumption = str(assumption).strip().lower().replace("-", "_")
    if selected_assumption not in {"plane_stress", "plane_strain"}:
        raise ValueError("MMB assumption must be plane_stress or plane_strain.")
    stiffness = (
        1.0e6 * spec.elastic_modulus / spec.arm_thickness
        if interface_stiffness is None
        else float(interface_stiffness)
    )
    if not np.isfinite(stiffness) or stiffness <= 0.0:
        raise ValueError("interface_stiffness must be finite and positive.")
    points = tuple(
        _mmb_point(
            spec,
            crack_index=int(index),
            control_displacement=control_value,
            lever_length=lever,
            elements_along=nx,
            elements_per_arm=ny,
            poisson=float(poisson),
            assumption=selected_assumption,
            interface_stiffness=stiffness,
            bulk_material=bulk_material,
            solver_options=solver_options,
        )
        for index in indices
    )
    source = (
        "AgentFEM assembled rigid-lever MMB Q1/cohesive curve; "
        f"{nx}x{2 * ny} bulk cells"
    )
    energy = DelaminationEnergyReleaseCurve(
        crack_length=np.asarray(
            [point.effective_crack_length for point in points], dtype=float
        ),
        compliance=np.asarray([point.compliance for point in points], dtype=float),
        total_energy_release_rate=np.asarray(
            [point.vcct_total_energy_release_rate for point in points], dtype=float
        ),
        mode_i_energy_release_rate=np.asarray(
            [point.vcct_mode_i_energy_release_rate for point in points], dtype=float
        ),
        mode_ii_energy_release_rate=np.asarray(
            [point.vcct_mode_ii_energy_release_rate for point in points], dtype=float
        ),
        source=(source + "; independent two-dimensional VCCT mode partition"),
    )
    return MMBFiniteElementCurve(
        specification=spec,
        points=points,
        energy_release=energy,
        lever_length=lever,
        source=source,
        poisson=float(poisson),
        assumption=selected_assumption,
        interface_stiffness=stiffness,
        bulk_material=points[0].bulk_material,
    )


def certify_mmb_cohesive_propagation(
    curve: MMBCohesivePropagationCurve,
    *,
    energy_relative_tolerance: float = 0.03,
    required_process_zone_elements: float = 3.0,
    control_residual_tolerance: float = 1.0e-10,
    newton_residual_tolerance: float = 1.0e-7,
) -> MMBCohesivePropagationCertificate:
    """Certify mechanism evidence without calling it external validation."""

    if not isinstance(curve, MMBCohesivePropagationCurve):
        raise TypeError("MMB propagation certification requires an assembled curve.")
    limits = tuple(
        float(value)
        for value in (
            energy_relative_tolerance,
            required_process_zone_elements,
            control_residual_tolerance,
            newton_residual_tolerance,
        )
    )
    if any(not np.isfinite(value) or value < 0.0 for value in limits):
        raise ValueError("MMB propagation tolerances must be finite and nonnegative.")
    points = curve.points
    initial_damaged = min(point.damaged_length for point in points)
    final = points[-1]
    propagation = bool(
        final.damaged_length > initial_damaged + 0.5 * curve.element_size
        and final.maximum_damage >= 0.95
    )
    energy_error = max(point.relative_energy_balance_error for point in points)
    control_error = max(abs(point.control_residual) for point in points)
    newton_error = max(point.residual_norm for point in points)
    mode_error = abs(
        final.damage_weighted_mode_i_fraction - final.beam_mode_i_fraction
    )
    accepted = bool(
        propagation
        and energy_error <= limits[0]
        and curve.process_zone_elements >= limits[1]
        and control_error <= limits[2]
        and newton_error <= limits[3]
    )
    return MMBCohesivePropagationCertificate(
        curve_identity_sha256=curve.identity_sha256,
        propagation_observed=propagation,
        maximum_relative_energy_error=float(energy_error),
        energy_relative_tolerance=limits[0],
        process_zone_elements=float(curve.process_zone_elements),
        required_process_zone_elements=limits[1],
        maximum_control_residual=float(control_error),
        control_residual_tolerance=limits[2],
        maximum_newton_residual=float(newton_error),
        newton_residual_tolerance=limits[3],
        final_mode_i_fraction=float(final.damage_weighted_mode_i_fraction),
        beam_mode_i_fraction=float(final.beam_mode_i_fraction),
        mode_i_fraction_absolute_error=float(mode_error),
        accepted=accepted,
    )


def mmb_cohesive_propagation_curve(
    spec: DelaminationBenchmarkSpec,
    *,
    precrack_length: float,
    displacement,
    lever_length: float,
    normal_strength: float,
    shear_strength: float,
    normal_fracture_energy: float,
    shear_fracture_energy: float,
    normal_stiffness: float,
    tangential_stiffness: float,
    elements_along: int,
    elements_per_arm: int,
    poisson: float = 0.3,
    assumption: str = "plane_stress",
    bulk_material=None,
    interaction: str = "bk",
    interaction_exponent: float = 1.45,
    solver_options=None,
    minimum_displacement_increment: float | None = None,
    maximum_cutbacks: int = 12,
) -> MMBCohesivePropagationCurve:
    """Run an irreversible MMB path with exact work-conjugate control.

    Cohesive history is committed only after global equilibrium converges.
    Failed target increments are bisected transactionally. The scalar lever
    reaction and prescribed load-point coordinate are an exact conjugate pair,
    so their trapezoidal product is the external-work ledger.
    """

    from mpi4py import MPI

    from agentfem import interfaces, solvers, steps

    if not isinstance(spec, DelaminationBenchmarkSpec) or spec.kind != "mmb":
        raise ValueError("mmb_cohesive_propagation_curve requires an MMB specification.")
    if MPI.COMM_WORLD.size != 1:
        raise NotImplementedError("The MMB propagation certificate is currently serial.")
    targets = np.asarray(displacement, dtype=float).reshape(-1)
    if (
        targets.size < 3
        or not np.isclose(targets[0], 0.0)
        or np.any(np.diff(targets) <= 0.0)
        or not np.all(np.isfinite(targets))
    ):
        raise ValueError("MMB displacement must start at zero and increase twice.")
    lever = float(lever_length)
    span = float(spec.half_span)
    length = 2.0 * span
    if not np.isfinite(lever) or lever < span / 3.0:
        raise ValueError("MMB lever_length must be finite and at least half_span/3.")
    nx = int(elements_along)
    ny = int(elements_per_arm)
    precrack = float(precrack_length)
    if nx < 4 or nx % 2 or ny < 1 or precrack <= 0.0 or precrack >= length:
        raise ValueError("MMB propagation geometry and mesh counts are invalid.")
    dx = length / nx
    crack_index = int(round(precrack / dx))
    represented_crack = crack_index * dx
    tolerance = 128.0 * np.finfo(float).eps * max(length, 1.0)
    if abs(represented_crack - precrack) > tolerance:
        raise ValueError("The MMB precrack must align with the axial mesh.")
    selected_assumption = str(assumption).strip().lower().replace("-", "_")
    if selected_assumption not in {"plane_stress", "plane_strain"}:
        raise ValueError("MMB assumption must be plane_stress or plane_strain.")
    cutback_limit = int(maximum_cutbacks)
    if cutback_limit < 0:
        raise ValueError("maximum_cutbacks must be nonnegative.")
    minimum_increment = (
        float(minimum_displacement_increment)
        if minimum_displacement_increment is not None
        else float(np.min(np.diff(targets)) / 256.0)
    )
    if not np.isfinite(minimum_increment) or minimum_increment <= 0.0:
        raise ValueError("minimum_displacement_increment must be finite and positive.")
    law = interfaces.mixed_mode_bilinear_cohesive(
        normal_strength=float(normal_strength),
        shear_strength=float(shear_strength),
        normal_fracture_energy=float(normal_fracture_energy),
        shear_fracture_energy=float(shear_fracture_energy),
        normal_stiffness=float(normal_stiffness),
        tangential_stiffness=float(tangential_stiffness),
        interaction=interaction,
        interaction_exponent=float(interaction_exponent),
    )
    fixture = _build_mmb_fixture(
        spec,
        crack_index=crack_index,
        lever_length=lever,
        elements_along=nx,
        elements_per_arm=ny,
        poisson=float(poisson),
        assumption=selected_assumption,
        law=law,
        bulk_material=bulk_material,
        solver_options=solver_options
        or solvers.newton(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-11,
            maximum_iterations=40,
            line_search=None,
            linear_solver=solvers.direct_solver(),
        ),
    )
    equilibrium = fixture["equilibrium"]
    cohesive = fixture["cohesive"]
    residual = fixture["residual"]
    control = fixture["control"]
    displacement_field = fixture["displacement"]
    initial_mask = fixture["initial_mask"]
    records = []
    previous_displacement = 0.0
    previous_reaction = 0.0
    external_work = 0.0
    initial_cohesive_dissipation = None

    def accept_trial(imposed):
        nonlocal previous_displacement
        nonlocal previous_reaction
        nonlocal external_work
        nonlocal initial_cohesive_dissipation

        try:
            solved = equilibrium(load=float(imposed), branch="monotonic", cycle=0)
        except RuntimeError:
            if equilibrium.last_info is None or equilibrium.last_info.converged:
                raise
            return None
        if not solved["converged"]:
            return None
        response = cohesive.begin()
        reaction = abs(float(solved["reaction"]))
        external_work += 0.5 * (previous_reaction + reaction) * (
            float(imposed) - previous_displacement
        )
        raw_dissipation = float(response.dissipated_energy)
        if initial_cohesive_dissipation is None:
            initial_cohesive_dissipation = raw_dissipation
        dissipated = raw_dissipation - initial_cohesive_dissipation
        damage_at_points = np.asarray(response.damage, dtype=float)
        damage = np.max(damage_at_points, axis=1)
        damaged_mask = damage > 1.0e-10
        failed_mask = damage >= 1.0 - 1.0e-8
        process_mask = damaged_mask & ~failed_mask
        active_points = np.broadcast_to((~initial_mask)[:, None], damage_at_points.shape)
        weights = np.where(active_points, damage_at_points, 0.0)
        weight_sum = float(np.sum(weights))
        mode_ii_fraction = (
            float(np.sum(weights * np.asarray(response.mode_mixity)) / weight_sum)
            if weight_sum > np.finfo(float).eps
            else 0.0
        )
        residual.commit()
        previous_displacement = float(imposed)
        previous_reaction = reaction
        return {
            "solved": solved,
            "reaction": reaction,
            "bulk_energy": float(solved["bulk_strain_energy"]),
            "stored": float(response.stored_energy),
            "dissipated": dissipated,
            "damage": damage,
            "damaged_length": float(np.count_nonzero(damaged_mask) * dx),
            "failed_length": float(np.count_nonzero(failed_mask) * dx),
            "process_length": float(np.count_nonzero(process_mask) * dx),
            "mode_i_fraction": float(1.0 - mode_ii_fraction),
        }

    beam_mode_i = float(
        12.0 * (3.0 * lever - span) ** 2
        / (
            12.0 * (3.0 * lever - span) ** 2
            + 9.0 * (lever + span) ** 2
        )
    )
    for increment, imposed in enumerate(targets):
        advance = steps.advance_monotonic_targets(
            (float(imposed),),
            try_accept=accept_trial,
            initial_coordinate=previous_displacement,
            minimum_increment=minimum_increment,
            maximum_cutbacks=cutback_limit,
            coordinate_name="MMB rigid-lever displacement",
            failure_message=lambda: (
                equilibrium.last_info.message
                if equilibrium.last_info is not None
                else "unknown nonlinear failure"
            ),
        )[0]
        accepted_trials = advance.accepted_values
        accepted = advance.final
        accounted = accepted["bulk_energy"] + accepted["stored"] + accepted["dissipated"]
        balance = external_work - accounted
        scale = max(abs(external_work), abs(accounted), np.finfo(float).eps)
        records.append(
            MMBCohesivePropagationPoint(
                increment=increment,
                displacement=float(imposed),
                reaction=accepted["reaction"],
                bulk_strain_energy=accepted["bulk_energy"],
                cohesive_stored_energy=accepted["stored"],
                cohesive_dissipation=accepted["dissipated"],
                external_work=external_work,
                energy_balance_error=balance,
                relative_energy_balance_error=float(abs(balance) / scale),
                maximum_damage=float(np.max(accepted["damage"][~initial_mask], initial=0.0)),
                damaged_length=max(accepted["damaged_length"], represented_crack),
                failed_length=max(accepted["failed_length"], represented_crack),
                process_zone_length=accepted["process_length"],
                damage_weighted_mode_i_fraction=accepted["mode_i_fraction"],
                beam_mode_i_fraction=beam_mode_i,
                newton_iterations=sum(
                    int(item["solved"]["iterations"]) for item in accepted_trials
                ),
                residual_norm=float(equilibrium.last_info.residual_norm),
                control_residual=float(control.coordinate(displacement_field) - imposed),
                accepted_subincrements=len(accepted_trials),
                cutbacks=advance.subdivisions,
            )
        )
    characteristic = min(
        float(spec.elastic_modulus)
        * float(normal_fracture_energy)
        / float(normal_strength) ** 2,
        float(spec.elastic_modulus)
        * float(shear_fracture_energy)
        / float(shear_strength) ** 2,
    )
    return MMBCohesivePropagationCurve(
        specification=spec,
        points=tuple(records),
        lever_length=lever,
        element_size=float(fixture["element_size"]),
        process_zone_elements=float(characteristic / dx),
        law=law.summary(),
        source=(
            "AgentFEM assembled rigid-lever MMB cohesive path; "
            f"{nx}x{2 * ny} Q1 bulk cells"
        ),
        poisson=float(poisson),
        assumption=selected_assumption,
        bulk_material=fixture["material_summary"],
    )


def _mmb_point(
    spec,
    *,
    crack_index,
    control_displacement,
    lever_length,
    elements_along,
    elements_per_arm,
    poisson,
    assumption,
    interface_stiffness,
    bulk_material,
    solver_options,
) -> MMBFiniteElementPoint:
    from agentfem import interfaces, solvers

    law = interfaces.mixed_mode_bilinear_cohesive(
        normal_strength=interface_stiffness,
        shear_strength=interface_stiffness,
        normal_fracture_energy=interface_stiffness,
        shear_fracture_energy=interface_stiffness,
        normal_stiffness=interface_stiffness,
        tangential_stiffness=interface_stiffness,
        interaction="bk",
    )
    fixture = _build_mmb_fixture(
        spec,
        crack_index=crack_index,
        lever_length=lever_length,
        elements_along=elements_along,
        elements_per_arm=elements_per_arm,
        poisson=poisson,
        assumption=assumption,
        law=law,
        bulk_material=bulk_material,
        solver_options=solver_options
        or solvers.newton(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-11,
            maximum_iterations=30,
            line_search=None,
            linear_solver=solvers.direct_solver(),
        ),
    )
    solved = fixture["equilibrium"](
        load=float(control_displacement), branch="monotonic", cycle=0
    )
    if not solved["converged"]:
        raise RuntimeError("MMB finite-element equilibrium did not converge.")
    load = abs(float(solved["reaction"]))
    if not np.isfinite(load) or load <= np.finfo(float).eps:
        raise RuntimeError("MMB generalized reaction is not positive and finite.")
    vcct = _mmb_virtual_crack_closure(fixture)
    fixture["cohesive"].commit()
    return MMBFiniteElementPoint(
        crack_length=fixture["represented_crack"],
        effective_crack_length=fixture["represented_crack"],
        load=load,
        displacement=float(control_displacement),
        compliance=float(control_displacement / load),
        control_residual=float(
            fixture["control"].coordinate(fixture["displacement"])
            - control_displacement
        ),
        element_size=fixture["element_size"],
        elements_per_arm=int(elements_per_arm),
        newton_iterations=int(solved["iterations"]),
        residual_norm=float(fixture["equilibrium"].last_info.residual_norm),
        vcct_mode_i_energy_release_rate=vcct["mode_i"],
        vcct_mode_ii_energy_release_rate=vcct["mode_ii"],
        vcct_total_energy_release_rate=vcct["total"],
        vcct_mode_i_fraction=vcct["mode_i_fraction"],
        bulk_material=fixture["material_summary"],
    )


def _mmb_virtual_crack_closure(fixture) -> dict[str, float]:
    """Recover crack-tip ``G_I`` and ``G_II`` without a beam partition.

    The implementation follows the standard one-element virtual crack closure
    construction: the interface force at the crack-tip node is paired with
    the relative displacement of the duplicated nodes one element behind the
    tip.  The paired-facet topology, rather than coordinate coincidence, owns
    side identity throughout the calculation.
    """

    cohesive = fixture["cohesive"]
    response = cohesive.begin()
    topology = cohesive.assembler.topology
    mask = np.asarray(fixture["initial_mask"], dtype=bool)
    midpoints = np.mean(
        fixture["split"].coordinates[topology.negative_nodes], axis=1
    )
    cracked = np.flatnonzero(mask)
    intact = np.flatnonzero(~mask)
    if cracked.size == 0 or intact.size == 0:
        raise RuntimeError("VCCT requires both a precrack and an intact ligament.")
    cracked_facet = int(cracked[np.argmax(midpoints[cracked, 0])])
    intact_facet = int(intact[np.argmin(midpoints[intact, 0])])
    negative_common = np.intersect1d(
        topology.negative_nodes[cracked_facet],
        topology.negative_nodes[intact_facet],
    )
    positive_common = np.intersect1d(
        topology.positive_nodes[cracked_facet],
        topology.positive_nodes[intact_facet],
    )
    if negative_common.size != 1 or positive_common.size != 1:
        raise RuntimeError("VCCT could not identify the paired crack-tip nodes.")
    negative_tip = int(negative_common[0])
    positive_tip = int(positive_common[0])
    negative_behind = int(
        next(
            node
            for node in topology.negative_nodes[cracked_facet]
            if int(node) != negative_tip
        )
    )
    positive_behind = int(
        next(
            node
            for node in topology.positive_nodes[cracked_facet]
            if int(node) != positive_tip
        )
    )
    values = cohesive.displacement.x.array.reshape((-1, cohesive.block_size))
    nodal = values[cohesive.node_to_block_dof]
    jump = nodal[positive_behind] - nodal[negative_behind]
    force = 0.5 * (
        response.internal_force[positive_tip]
        - response.internal_force[negative_tip]
    )
    normal = np.asarray(topology.normals[intact_facet], dtype=float)
    tangent = np.asarray((normal[1], -normal[0]), dtype=float)
    normal_work = float(np.dot(force, normal) * np.dot(jump, normal))
    tangential_work = float(np.dot(force, tangent) * np.dot(jump, tangent))
    denominator = float(2.0 * fixture["width"] * fixture["dx"])
    mode_i = abs(normal_work) / denominator
    mode_ii = abs(tangential_work) / denominator
    total = mode_i + mode_ii
    if not np.isfinite(total) or total <= np.finfo(float).eps:
        raise RuntimeError("VCCT recovered a nonpositive or nonfinite total G.")
    return {
        "mode_i": float(mode_i),
        "mode_ii": float(mode_ii),
        "total": float(total),
        "mode_i_fraction": float(mode_i / total),
    }


def _build_mmb_fixture(
    spec,
    *,
    crack_index,
    lever_length,
    elements_along,
    elements_per_arm,
    poisson,
    assumption,
    law,
    bulk_material,
    solver_options,
):
    import ufl
    from mpi4py import MPI

    from agentfem import (
        constitutive,
        constraints,
        fields,
        fracture,
        interfaces,
        operators,
        studies,
    )

    if MPI.COMM_WORLD.size != 1:
        raise NotImplementedError("The initial MMB structural provider is serial.")
    span = float(spec.half_span)
    length = 2.0 * span
    h = float(spec.arm_thickness)
    nx = int(elements_along)
    ny = int(elements_per_arm)
    x_values = np.linspace(0.0, length, nx + 1)
    y_values = np.linspace(-h, h, 2 * ny + 1)
    coordinates = np.asarray([(x, y) for y in y_values for x in x_values])

    def node(i, j):
        return j * (nx + 1) + i

    cells = np.asarray(
        [
            [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]
            for j in range(2 * ny)
            for i in range(nx)
        ],
        dtype=np.int64,
    )
    interface = np.asarray(
        [[node(i, ny), node(i + 1, ny)] for i in range(nx)], dtype=np.int64
    )
    upper_cells = np.arange(ny * nx, 2 * ny * nx, dtype=np.int64)
    split = interfaces.split_conforming_line_interface(
        coordinates, cells, interface, positive_cells=upper_cells
    )
    domain = interfaces.create_dolfinx_split_mesh(
        split, comm=MPI.COMM_SELF, cell_type="quadrilateral"
    )
    displacement = fields.displacement(domain)
    material = bulk_material or constitutive.isotropic_elastic(
        young=spec.elastic_modulus,
        poisson=poisson,
        density=1.0,
        name="MMB isotropic elastic",
    )
    study = studies.static_solid(dimension=2, assumption=assumption)
    internal = spec.width * operators.internal_force_vector(
        displacement.value,
        displacement.test,
        material,
        study=study,
        measure=ufl.Measure("dx", domain=domain),
    )
    bulk = operators.residual_operator(
        internal.expression,
        name="R_MMB_bulk",
        family="small_strain_linear_elasticity",
    )
    tangent = operators.linearize(bulk, displacement)
    cohesive = fracture.cohesive_force(
        split, displacement, law, normal_hint=(0.0, 1.0), thickness=spec.width
    )
    midpoints = np.mean(
        split.coordinates[cohesive.assembler.topology.negative_nodes], axis=1
    )
    represented_crack = float(crack_index * length / nx)
    precrack = midpoints[:, 0] < represented_crack
    if int(np.count_nonzero(precrack)) != int(crack_index):
        raise RuntimeError("MMB precrack facets do not match the requested mesh tip.")
    cohesive.initialize_precrack(precrack)
    residual = fracture.FiniteStrainCohesiveResidual(bulk, cohesive)

    left_support = lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], -h)
    right_support = lambda x: np.isclose(x[0], length) & np.isclose(x[1], -h)
    bcs = (
        constraints.component_dirichlet(
            displacement, 1, on=left_support, value=0.0, name="MMB left roller"
        ).bc,
        constraints.component_dirichlet(
            displacement, 0, on=left_support, value=0.0, name="MMB horizontal datum"
        ).bc,
        constraints.component_dirichlet(
            displacement, 1, on=right_support, value=0.0, name="MMB right roller"
        ).bc,
    )
    control = constraints.linear_kinematic_control(
        displacement,
        (
            constraints.point_kinematic_term(
                (0.0, h),
                component=1,
                coefficient=lever_length / span,
                name="upper crack-mouth contact",
            ),
            constraints.point_kinematic_term(
                (span, h),
                component=1,
                coefficient=-(lever_length + span) / span,
                name="upper mid-span contact",
            ),
        ),
        name="MMB rigid-lever load-point displacement",
        unit="length",
    )
    equilibrium = fracture.FiniteStrainCohesiveKinematicEquilibrium(
        residual,
        tangent,
        displacement,
        control=control,
        bcs=bcs,
        solver_options=solver_options,
        bulk_strain_energy=lambda function: 0.5
        * operators.dual_product(internal, function),
    )
    return {
        "cohesive": cohesive,
        "control": control,
        "displacement": displacement,
        "element_size": max(length / nx, h / ny),
        "equilibrium": equilibrium,
        "initial_mask": precrack,
        "internal": internal,
        "law": law,
        "material_summary": _material_manifest(material),
        "represented_crack": represented_crack,
        "residual": residual,
        "split": split,
        "width": float(spec.width),
        "dx": length / nx,
    }


def _material_manifest(material) -> dict[str, object]:
    """Return a JSON-safe public identity for an injected bulk material."""

    if hasattr(material, "as_dict") and callable(material.as_dict):
        return dict(material.as_dict())
    if hasattr(material, "summary") and callable(material.summary):
        summary = material.summary()
        if isinstance(summary, dict):
            return dict(summary)
        return {"type": type(material).__name__, "summary": str(summary)}
    raise TypeError(
        "MMB bulk_material must provide as_dict() or summary() evidence."
    )


__all__ = (
    "MMBComplianceCertificate",
    "MMBCohesivePropagationCertificate",
    "MMBCohesivePropagationCurve",
    "MMBCohesivePropagationPoint",
    "MMBFiniteElementCurve",
    "MMBFiniteElementPoint",
    "MMBModePartitionCertificate",
    "certify_mmb_compliance",
    "certify_mmb_cohesive_propagation",
    "certify_mmb_mode_partition",
    "mmb_cohesive_propagation_curve",
    "mmb_finite_element_curve",
)
