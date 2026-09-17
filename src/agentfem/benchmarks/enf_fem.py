# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Executable end-notched-flexure finite-element benchmark provider.

The provider owns the assembled Mode-II structural problem.  Analytical beam
theory remains a separately identified oracle in :mod:`mixed_mode`; the two
are compared only by an explicit convergence certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .mixed_mode import (
    DelaminationBenchmarkSpec,
    DelaminationEnergyReleaseCurve,
    compliance_energy_release_curve,
    enf_beam_compliance,
)


@dataclass(frozen=True)
class ENFFiniteElementPoint:
    """One displacement-controlled solution of an ENF specimen."""

    crack_length: float
    effective_crack_length: float
    load: float
    displacement: float
    compliance: float
    element_size: float
    elements_per_arm: int
    newton_iterations: int
    residual_norm: float

    def summary(self) -> dict[str, object]:
        return {"kind": "enf_finite_element_point", **self.__dict__}


@dataclass(frozen=True)
class ENFFiniteElementCurve:
    """Assembled ENF compliance and pure Mode-II energy-release evidence."""

    specification: DelaminationBenchmarkSpec
    points: tuple[ENFFiniteElementPoint, ...]
    energy_release: DelaminationEnergyReleaseCurve
    source: str
    poisson: float
    assumption: str
    interface_stiffness: float

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
            "schema": "agentfem.enf-finite-element-curve.v1",
            "specification": self.specification.summary(),
            "source": self.source,
            "model": {
                "poisson": self.poisson,
                "assumption": self.assumption,
                "interface_stiffness": self.interface_stiffness,
            },
            "points": [point.summary() for point in self.points],
            "energy_release": self.energy_release.summary(),
            "discretization": {
                "cell": "Q1 quadrilateral",
                "interface": "paired zero-thickness line facets",
                "kinematics": "small-strain plane stress or plane strain",
                "loading": "three-point bending with prescribed mid-span motion",
                "precrack": "fully failed interface facets",
            },
        }


@dataclass(frozen=True)
class ENFComplianceConvergenceCertificate:
    """Three-level ENF compliance evidence against a declared beam oracle."""

    reference_source: str
    curve_identity_sha256: tuple[str, ...]
    element_sizes: tuple[float, ...]
    relative_errors_to_reference: tuple[float, ...]
    successive_relative_changes: tuple[float, ...]
    maximum_residual_norms: tuple[float, ...]
    observed_order: float | None
    asymptotic_trend: bool
    reference_relative_tolerance: float
    refinement_relative_tolerance: float
    residual_tolerance: float
    accepted: bool

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.enf-compliance-convergence-certificate.v1",
            **self.__dict__,
            "scope": "precracked elastic ENF structural compliance",
            "excludes": (
                "cohesive-zone evolution",
                "Mode-II propagation",
                "experimental material calibration",
            ),
        }


@dataclass(frozen=True)
class ENFFiniteElementConvergenceStudy:
    """Three-or-more-level assembled ENF compliance study."""

    specification: DelaminationBenchmarkSpec
    curves: tuple[ENFFiniteElementCurve, ...]
    certificate: ENFComplianceConvergenceCertificate

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.enf-finite-element-convergence-study.v1",
            "specification": self.specification.summary(),
            "curves": [curve.summary() for curve in self.curves],
            "certificate": self.certificate.summary(),
        }


@dataclass(frozen=True)
class ENFCohesivePropagationPoint:
    """One accepted displacement-controlled ENF propagation increment."""

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
    damage_weighted_mode_ii_fraction: float
    newton_iterations: int
    residual_norm: float
    accepted_subincrements: int = 1
    cutbacks: int = 0

    def summary(self) -> dict[str, object]:
        return {"kind": "enf_cohesive_propagation_point", **self.__dict__}


@dataclass(frozen=True)
class ENFCohesivePropagationCurve:
    """Accepted ENF cohesive evolution and its work--energy evidence."""

    specification: DelaminationBenchmarkSpec
    points: tuple[ENFCohesivePropagationPoint, ...]
    element_size: float
    process_zone_elements: float
    law: dict[str, object]
    source: str
    poisson: float | None = None
    assumption: str | None = None

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
            "schema": "agentfem.enf-cohesive-propagation-curve.v1",
            "specification": self.specification.summary(),
            "source": self.source,
            "element_size": self.element_size,
            "process_zone_elements": self.process_zone_elements,
            "law": dict(self.law),
            "model": {"poisson": self.poisson, "assumption": self.assumption},
            "points": [point.summary() for point in self.points],
            "evidence_scope": (
                "displacement-controlled monotonic Mode-II cohesive propagation"
            ),
        }


@dataclass(frozen=True)
class ENFCohesivePropagationCertificate:
    """Three-level ENF propagation, energy and Mode-II certificate."""

    curve_identity_sha256: tuple[str, ...]
    element_sizes: tuple[float, ...]
    peak_reactions: tuple[float, ...]
    final_damaged_lengths: tuple[float, ...]
    final_failed_lengths: tuple[float, ...]
    maximum_relative_energy_errors: tuple[float, ...]
    process_zone_elements: tuple[float, ...]
    final_mode_ii_fractions: tuple[float, ...]
    peak_reaction_changes: tuple[float, ...]
    damaged_length_changes: tuple[float, ...]
    failed_length_changes: tuple[float, ...]
    refinement_relative_tolerance: float
    energy_relative_tolerance: float
    required_process_zone_elements: float
    required_mode_ii_fraction: float
    propagation_observed: bool
    accepted: bool

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.enf-cohesive-propagation-certificate.v1",
            **self.__dict__,
            "scope": "Mode-II ENF cohesive propagation",
            "excludes": (
                "mixed-mode MMB",
                "experimental material calibration",
                "unstable branches skipped by displacement control",
            ),
        }


@dataclass(frozen=True)
class ENFCohesivePropagationStudy:
    """Three-or-more-level assembled ENF cohesive propagation study."""

    specification: DelaminationBenchmarkSpec
    curves: tuple[ENFCohesivePropagationCurve, ...]
    certificate: ENFCohesivePropagationCertificate

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.enf-cohesive-propagation-study.v1",
            "specification": self.specification.summary(),
            "curves": [curve.summary() for curve in self.curves],
            "certificate": self.certificate.summary(),
        }


def certify_enf_compliance_convergence(
    spec: DelaminationBenchmarkSpec,
    curves,
    *,
    reference_relative_tolerance: float,
    refinement_relative_tolerance: float,
    residual_tolerance: float = 1.0e-8,
) -> ENFComplianceConvergenceCertificate:
    """Compare refined assembled ENF compliance with simple-beam theory."""

    if not isinstance(spec, DelaminationBenchmarkSpec) or spec.kind != "enf":
        raise ValueError("ENF compliance convergence requires an ENF specification.")
    selected = tuple(curves)
    if len(selected) < 3 or not all(
        isinstance(curve, ENFFiniteElementCurve) for curve in selected
    ):
        raise ValueError("ENF compliance convergence requires at least three curves.")
    coordinates = np.asarray(
        [point.effective_crack_length for point in selected[0].points], dtype=float
    )
    if coordinates.size < 3:
        raise ValueError("Every ENF convergence curve needs at least three cracks.")
    reference = np.asarray(enf_beam_compliance(spec, coordinates), dtype=float)
    scale = float(np.linalg.norm(reference))
    compliance = []
    sizes = []
    residuals = []
    for curve in selected:
        if curve.specification != spec:
            raise ValueError("Every ENF curve must use the same specification.")
        curve_coordinates = np.asarray(
            [point.effective_crack_length for point in curve.points], dtype=float
        )
        if curve_coordinates.shape != coordinates.shape or not np.allclose(
            curve_coordinates, coordinates, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("Every ENF curve must use identical crack coordinates.")
        compliance.append(np.asarray([point.compliance for point in curve.points]))
        sizes.append(float(curve.element_size))
        residuals.append(max(point.residual_norm for point in curve.points))
    model_contracts = {
        (curve.poisson, curve.assumption, curve.interface_stiffness)
        for curve in selected
    }
    if len(model_contracts) != 1:
        raise ValueError("Every ENF curve must use the same discrete model contract.")
    if any(left <= right for left, right in zip(sizes[:-1], sizes[1:])):
        raise ValueError("ENF element sizes must decrease from coarse to fine.")
    errors = tuple(
        float(np.linalg.norm(values - reference) / scale) for values in compliance
    )
    changes = tuple(
        float(
            np.linalg.norm(right - left)
            / max(np.linalg.norm(right), np.finfo(float).eps)
        )
        for left, right in zip(compliance[:-1], compliance[1:])
    )
    observed_order = None
    ratios = tuple(left / right for left, right in zip(sizes[:-1], sizes[1:]))
    if (
        len(changes) >= 2
        and max(ratios) / min(ratios) <= 1.05
        and changes[-1] > np.finfo(float).eps
        and changes[-2] > 0.0
    ):
        observed_order = float(np.log(changes[-2] / changes[-1]) / np.log(ratios[-1]))
    asymptotic = bool(
        all(right < left for left, right in zip(changes[:-1], changes[1:]))
    )
    reference_limit = float(reference_relative_tolerance)
    refinement_limit = float(refinement_relative_tolerance)
    residual_limit = float(residual_tolerance)
    if any(
        not np.isfinite(value) or value < 0.0
        for value in (reference_limit, refinement_limit, residual_limit)
    ):
        raise ValueError("ENF convergence tolerances must be finite and nonnegative.")
    return ENFComplianceConvergenceCertificate(
        reference_source=f"{spec.source}; classical simple-beam compliance",
        curve_identity_sha256=tuple(curve.identity_sha256 for curve in selected),
        element_sizes=tuple(sizes),
        relative_errors_to_reference=errors,
        successive_relative_changes=changes,
        maximum_residual_norms=tuple(float(value) for value in residuals),
        observed_order=observed_order,
        asymptotic_trend=asymptotic,
        reference_relative_tolerance=reference_limit,
        refinement_relative_tolerance=refinement_limit,
        residual_tolerance=residual_limit,
        accepted=(
            asymptotic
            and errors[-1] <= reference_limit
            and changes[-1] <= refinement_limit
            and max(residuals) <= residual_limit
        ),
    )


def enf_finite_element_convergence(
    spec: DelaminationBenchmarkSpec,
    *,
    crack_length,
    control_displacement: float,
    mesh_levels,
    poisson: float = 0.3,
    assumption: str = "plane_stress",
    interface_stiffness: float | None = None,
    solver_options=None,
    reference_relative_tolerance: float = 0.10,
    refinement_relative_tolerance: float = 0.05,
    residual_tolerance: float = 1.0e-8,
) -> ENFFiniteElementConvergenceStudy:
    """Run and certify three or more assembled ENF mesh levels."""

    levels = tuple((int(nx), int(ny)) for nx, ny in mesh_levels)
    if len(levels) < 3 or any(nx < 4 or ny < 1 for nx, ny in levels):
        raise ValueError("ENF convergence needs at least three valid mesh levels.")
    if any(
        right[0] <= left[0] or right[1] < left[1]
        for left, right in zip(levels[:-1], levels[1:])
    ):
        raise ValueError("ENF mesh levels must refine the structural mesh.")
    curves = tuple(
        enf_finite_element_curve(
            spec,
            crack_length=crack_length,
            control_displacement=control_displacement,
            elements_along=nx,
            elements_per_arm=ny,
            poisson=poisson,
            assumption=assumption,
            interface_stiffness=interface_stiffness,
            solver_options=solver_options,
        )
        for nx, ny in levels
    )
    certificate = certify_enf_compliance_convergence(
        spec,
        curves,
        reference_relative_tolerance=reference_relative_tolerance,
        refinement_relative_tolerance=refinement_relative_tolerance,
        residual_tolerance=residual_tolerance,
    )
    return ENFFiniteElementConvergenceStudy(spec, curves, certificate)


def enf_finite_element_curve(
    spec: DelaminationBenchmarkSpec,
    *,
    crack_length,
    control_displacement: float,
    elements_along: int,
    elements_per_arm: int,
    poisson: float = 0.3,
    assumption: str = "plane_stress",
    interface_stiffness: float | None = None,
    solver_options=None,
) -> ENFFiniteElementCurve:
    """Solve a precracked ENF family under three-point bending.

    The specimen spans ``2 * half_span``.  Its bottom corners are simple
    supports and a vertical motion is prescribed at the top mid-span node.
    The split mid-plane is traction free behind the crack tip and tied in both
    normal and tangential directions ahead of it.  Crack tips must align with
    the uniform mesh so geometry error cannot be hidden in a comparison.
    """

    if not isinstance(spec, DelaminationBenchmarkSpec) or spec.kind != "enf":
        raise ValueError("enf_finite_element_curve requires an ENF specification.")
    lengths = np.asarray(crack_length, dtype=float).reshape(-1)
    if lengths.size < 3 or np.any(np.diff(lengths) <= 0.0):
        raise ValueError("ENF finite-element evidence needs three increasing cracks.")
    motion = float(control_displacement)
    if not np.isfinite(motion) or motion <= 0.0:
        raise ValueError("control_displacement must be finite and positive.")
    length = 2.0 * float(spec.half_span)
    nx = int(elements_along)
    ny = int(elements_per_arm)
    if length <= float(lengths[-1]) or nx < 4 or nx % 2 or ny < 1:
        raise ValueError("ENF mesh needs an even axial count and valid geometry.")
    dx = length / nx
    indices = np.rint(lengths / dx).astype(int)
    represented = indices * dx
    tolerance = 128.0 * np.finfo(float).eps * max(length, 1.0)
    if np.any(np.abs(represented - lengths) > tolerance):
        raise ValueError("Every ENF crack length must align with the axial mesh.")
    selected_assumption = str(assumption).strip().lower().replace("-", "_")
    if selected_assumption not in {"plane_stress", "plane_strain"}:
        raise ValueError("ENF assumption must be plane_stress or plane_strain.")
    stiffness = (
        250.0 * spec.elastic_modulus / spec.arm_thickness
        if interface_stiffness is None
        else float(interface_stiffness)
    )
    if not np.isfinite(stiffness) or stiffness <= 0.0:
        raise ValueError("interface_stiffness must be finite and positive.")
    points = tuple(
        _enf_point(
            spec,
            crack_index=int(index),
            control_displacement=motion,
            elements_along=nx,
            elements_per_arm=ny,
            poisson=float(poisson),
            assumption=selected_assumption,
            interface_stiffness=stiffness,
            solver_options=solver_options,
        )
        for index in indices
    )
    source = f"AgentFEM assembled ENF Q1/cohesive curve; {nx}x{2 * ny} bulk cells"
    energy = compliance_energy_release_curve(
        spec,
        crack_length=[point.effective_crack_length for point in points],
        load=[point.load for point in points],
        compliance=[point.compliance for point in points],
        source=source,
    )
    return ENFFiniteElementCurve(
        specification=spec,
        points=points,
        energy_release=energy,
        source=source,
        poisson=float(poisson),
        assumption=selected_assumption,
        interface_stiffness=stiffness,
    )


def certify_enf_cohesive_propagation(
    spec: DelaminationBenchmarkSpec,
    curves,
    *,
    refinement_relative_tolerance: float = 0.10,
    energy_relative_tolerance: float = 0.03,
    required_process_zone_elements: float = 3.0,
    required_mode_ii_fraction: float = 0.90,
) -> ENFCohesivePropagationCertificate:
    """Certify refined ENF propagation without accepting damage images alone."""

    if not isinstance(spec, DelaminationBenchmarkSpec) or spec.kind != "enf":
        raise ValueError("ENF propagation certification requires an ENF specification.")
    selected = tuple(curves)
    if len(selected) < 3 or not all(
        isinstance(curve, ENFCohesivePropagationCurve) for curve in selected
    ):
        raise ValueError("ENF propagation certification requires at least three curves.")
    sizes = tuple(float(curve.element_size) for curve in selected)
    if any(left <= right for left, right in zip(sizes[:-1], sizes[1:])):
        raise ValueError("ENF propagation meshes must be ordered coarse to fine.")
    reference_displacement = np.asarray(
        [point.displacement for point in selected[0].points], dtype=float
    )
    reference_law = selected[0].law
    reference_model = (selected[0].poisson, selected[0].assumption)
    for curve in selected:
        if (
            curve.specification != spec
            or curve.law != reference_law
            or (curve.poisson, curve.assumption) != reference_model
        ):
            raise ValueError("Every ENF propagation curve must share one model and law.")
        displacement = np.asarray(
            [point.displacement for point in curve.points], dtype=float
        )
        if displacement.shape != reference_displacement.shape or not np.allclose(
            displacement, reference_displacement, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                "Every ENF propagation curve needs identical displacements."
            )
    peak = tuple(float(curve.peak_reaction) for curve in selected)
    damaged = tuple(float(curve.points[-1].damaged_length) for curve in selected)
    failed = tuple(float(curve.points[-1].failed_length) for curve in selected)
    energy = tuple(
        max(point.relative_energy_balance_error for point in curve.points)
        for curve in selected
    )
    zone = tuple(float(curve.process_zone_elements) for curve in selected)
    mode_ii = tuple(
        float(curve.points[-1].damage_weighted_mode_ii_fraction)
        for curve in selected
    )

    def changes(values):
        return tuple(
            abs(right - left) / max(abs(right), np.finfo(float).eps)
            for left, right in zip(values[:-1], values[1:])
        )

    peak_changes = changes(peak)
    damaged_changes = changes(damaged)
    failed_changes = changes(failed)
    limits = (
        float(refinement_relative_tolerance),
        float(energy_relative_tolerance),
        float(required_process_zone_elements),
        float(required_mode_ii_fraction),
    )
    if any(not np.isfinite(value) or value < 0.0 for value in limits):
        raise ValueError("ENF propagation tolerances must be finite and nonnegative.")
    refinement_limit, energy_limit, zone_limit, mode_ii_limit = limits
    if mode_ii_limit > 1.0:
        raise ValueError("required_mode_ii_fraction must not exceed one.")
    initial_damaged = min(point.damaged_length for point in selected[-1].points)
    propagation = bool(
        damaged[-1] > initial_damaged + 0.5 * sizes[-1]
        and selected[-1].points[-1].maximum_damage >= 0.95
    )
    accepted = bool(
        propagation
        and peak_changes[-1] <= refinement_limit
        and damaged_changes[-1] <= refinement_limit
        and failed_changes[-1] <= refinement_limit
        and max(energy) <= energy_limit
        and min(zone) >= zone_limit
        and min(mode_ii) >= mode_ii_limit
    )
    return ENFCohesivePropagationCertificate(
        curve_identity_sha256=tuple(curve.identity_sha256 for curve in selected),
        element_sizes=sizes,
        peak_reactions=peak,
        final_damaged_lengths=damaged,
        final_failed_lengths=failed,
        maximum_relative_energy_errors=energy,
        process_zone_elements=zone,
        final_mode_ii_fractions=mode_ii,
        peak_reaction_changes=peak_changes,
        damaged_length_changes=damaged_changes,
        failed_length_changes=failed_changes,
        refinement_relative_tolerance=refinement_limit,
        energy_relative_tolerance=energy_limit,
        required_process_zone_elements=zone_limit,
        required_mode_ii_fraction=mode_ii_limit,
        propagation_observed=propagation,
        accepted=accepted,
    )


def enf_cohesive_propagation_convergence(
    spec: DelaminationBenchmarkSpec,
    *,
    precrack_length: float,
    displacement,
    normal_strength: float,
    shear_strength: float,
    normal_fracture_energy: float,
    shear_fracture_energy: float,
    normal_stiffness: float,
    tangential_stiffness: float,
    mesh_levels,
    poisson: float = 0.3,
    assumption: str = "plane_stress",
    interaction: str = "bk",
    interaction_exponent: float = 1.45,
    solver_options=None,
    minimum_displacement_increment: float | None = None,
    maximum_cutbacks: int = 12,
    refinement_relative_tolerance: float = 0.10,
    energy_relative_tolerance: float = 0.03,
    required_process_zone_elements: float = 3.0,
    required_mode_ii_fraction: float = 0.90,
) -> ENFCohesivePropagationStudy:
    """Execute and certify three or more ENF cohesive propagation levels."""

    levels = tuple((int(nx), int(ny)) for nx, ny in mesh_levels)
    if len(levels) < 3 or any(nx < 4 or nx % 2 or ny < 1 for nx, ny in levels):
        raise ValueError("ENF propagation convergence needs three valid levels.")
    if any(
        right[0] <= left[0] or right[1] < left[1]
        for left, right in zip(levels[:-1], levels[1:])
    ):
        raise ValueError("ENF propagation levels must refine the bulk mesh.")
    curves = tuple(
        enf_cohesive_propagation_curve(
            spec,
            precrack_length=precrack_length,
            displacement=displacement,
            normal_strength=normal_strength,
            shear_strength=shear_strength,
            normal_fracture_energy=normal_fracture_energy,
            shear_fracture_energy=shear_fracture_energy,
            normal_stiffness=normal_stiffness,
            tangential_stiffness=tangential_stiffness,
            elements_along=nx,
            elements_per_arm=ny,
            poisson=poisson,
            assumption=assumption,
            interaction=interaction,
            interaction_exponent=interaction_exponent,
            solver_options=solver_options,
            minimum_displacement_increment=minimum_displacement_increment,
            maximum_cutbacks=maximum_cutbacks,
        )
        for nx, ny in levels
    )
    certificate = certify_enf_cohesive_propagation(
        spec,
        curves,
        refinement_relative_tolerance=refinement_relative_tolerance,
        energy_relative_tolerance=energy_relative_tolerance,
        required_process_zone_elements=required_process_zone_elements,
        required_mode_ii_fraction=required_mode_ii_fraction,
    )
    return ENFCohesivePropagationStudy(spec, curves, certificate)


def enf_cohesive_propagation_curve(
    spec: DelaminationBenchmarkSpec,
    *,
    precrack_length: float,
    displacement,
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
    interaction: str = "bk",
    interaction_exponent: float = 1.45,
    solver_options=None,
    minimum_displacement_increment: float | None = None,
    maximum_cutbacks: int = 12,
) -> ENFCohesivePropagationCurve:
    """Run an irreversible ENF path with explicit Mode-II evidence.

    The returned curve is numerical mechanism evidence, not an external
    material validation. Cohesive history is committed only after global
    convergence; failed increments are transactionally bisected.
    """

    import ufl
    from mpi4py import MPI

    from agentfem import (
        constitutive,
        constraints,
        fields,
        fracture,
        interfaces,
        operators,
        results,
        solvers,
        steps,
        studies,
    )

    if not isinstance(spec, DelaminationBenchmarkSpec) or spec.kind != "enf":
        raise ValueError("enf_cohesive_propagation_curve requires an ENF specification.")
    if MPI.COMM_WORLD.size != 1:
        raise NotImplementedError("The ENF propagation certificate is currently serial.")
    displacements = np.asarray(displacement, dtype=float).reshape(-1)
    if (
        displacements.size < 3
        or not np.isclose(displacements[0], 0.0)
        or np.any(np.diff(displacements) <= 0.0)
        or not np.all(np.isfinite(displacements))
    ):
        raise ValueError("ENF displacement must start at zero and increase twice.")
    cutback_limit = int(maximum_cutbacks)
    if cutback_limit < 0:
        raise ValueError("maximum_cutbacks must be nonnegative.")
    minimum_increment = (
        float(minimum_displacement_increment)
        if minimum_displacement_increment is not None
        else float(np.min(np.diff(displacements)) / 256.0)
    )
    if not np.isfinite(minimum_increment) or minimum_increment <= 0.0:
        raise ValueError(
            "minimum_displacement_increment must be finite and positive."
        )
    length = 2.0 * float(spec.half_span)
    precrack = float(precrack_length)
    nx = int(elements_along)
    ny = int(elements_per_arm)
    if length <= precrack or precrack <= 0.0 or nx < 4 or nx % 2 or ny < 1:
        raise ValueError("ENF propagation geometry and mesh counts are invalid.")
    dx = length / nx
    crack_index = int(round(precrack / dx))
    represented_crack = crack_index * dx
    tolerance = 128.0 * np.finfo(float).eps * max(length, 1.0)
    if abs(represented_crack - precrack) > tolerance:
        raise ValueError("The ENF precrack must align with the axial mesh.")
    selected_assumption = str(assumption).strip().lower().replace("-", "_")
    if selected_assumption not in {"plane_stress", "plane_strain"}:
        raise ValueError("ENF assumption must be plane_stress or plane_strain.")

    h = float(spec.arm_thickness)
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
    interface_facets = np.asarray(
        [[node(i, ny), node(i + 1, ny)] for i in range(nx)], dtype=np.int64
    )
    upper_cells = np.arange(ny * nx, 2 * ny * nx, dtype=np.int64)
    split = interfaces.split_conforming_line_interface(
        coordinates, cells, interface_facets, positive_cells=upper_cells
    )
    domain = interfaces.create_dolfinx_split_mesh(
        split, comm=MPI.COMM_SELF, cell_type="quadrilateral"
    )
    displacement_field = fields.displacement(domain)
    study = studies.static_solid(dimension=2, assumption=selected_assumption)
    material = constitutive.isotropic_elastic(
        young=spec.elastic_modulus,
        poisson=float(poisson),
        density=1.0,
        name="ENF isotropic elastic",
    )
    internal = spec.width * operators.internal_force_vector(
        displacement_field.value,
        displacement_field.test,
        material,
        study=study,
        measure=ufl.Measure("dx", domain=domain),
    )
    bulk = operators.residual_operator(
        internal.expression,
        name="R_ENF_propagation_bulk",
        family="small_strain_linear_elasticity",
    )
    tangent = operators.linearize(bulk, displacement_field)
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
    cohesive = fracture.cohesive_force(
        split,
        displacement_field,
        law,
        normal_hint=(0.0, 1.0),
        thickness=spec.width,
    )
    topology = cohesive.assembler.topology
    facet_midpoints = np.mean(split.coordinates[topology.negative_nodes], axis=1)
    initial_mask = facet_midpoints[:, 0] < represented_crack
    if int(np.count_nonzero(initial_mask)) != crack_index:
        raise RuntimeError("ENF precrack facets do not match the represented tip.")
    cohesive.initialize_precrack(initial_mask)
    residual = fracture.FiniteStrainCohesiveResidual(bulk, cohesive)

    left_support = lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], -h)
    right_support = lambda x: np.isclose(x[0], length) & np.isclose(x[1], -h)
    load_point = lambda x: np.isclose(x[0], spec.half_span) & np.isclose(x[1], h)
    motion = constraints.component_dirichlet(
        displacement_field,
        1,
        on=load_point,
        value=0.0,
        name="ENF mid-span motion",
    )
    bcs = (
        constraints.component_dirichlet(
            displacement_field, 1, on=left_support, value=0.0,
            name="ENF left roller",
        ).bc,
        constraints.component_dirichlet(
            displacement_field, 0, on=left_support, value=0.0,
            name="ENF horizontal datum",
        ).bc,
        constraints.component_dirichlet(
            displacement_field, 1, on=right_support, value=0.0,
            name="ENF right roller",
        ).bc,
        motion.bc,
    )

    def set_motion(value):
        motion.value.value = -float(value)

    equilibrium = fracture.FiniteStrainCohesiveEquilibrium(
        residual,
        tangent,
        displacement_field,
        bcs=bcs,
        set_load=set_motion,
        solver_options=solver_options
        or solvers.newton(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-11,
            maximum_iterations=40,
            line_search=None,
            linear_solver=solvers.direct_solver(),
        ),
        reaction=lambda _function: results.reaction_resultant(
            residual, on=load_point, component=1
        ),
        bulk_strain_energy=lambda function: 0.5
        * operators.dual_product(internal, function),
    )
    records = []
    previous_displacement = 0.0
    previous_reaction = 0.0
    external_work = 0.0
    initial_failed_length = represented_crack
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
        bulk_energy = float(solved["bulk_strain_energy"])
        stored = float(response.stored_energy)
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
            "bulk_energy": bulk_energy,
            "stored": stored,
            "dissipated": dissipated,
            "damage": damage,
            "damaged_length": float(np.count_nonzero(damaged_mask) * dx),
            "failed_length": float(np.count_nonzero(failed_mask) * dx),
            "process_length": float(np.count_nonzero(process_mask) * dx),
            "mode_ii_fraction": mode_ii_fraction,
        }

    for increment, imposed in enumerate(displacements):
        advance = steps.advance_monotonic_targets(
            (float(imposed),),
            try_accept=accept_trial,
            initial_coordinate=previous_displacement,
            minimum_increment=minimum_increment,
            maximum_cutbacks=cutback_limit,
            coordinate_name="ENF mid-span displacement",
            failure_message=lambda: (
                equilibrium.last_info.message
                if equilibrium.last_info is not None
                else "unknown nonlinear failure"
            ),
        )[0]
        accepted_trials = advance.accepted_values
        accepted = advance.final
        accounted = (
            accepted["bulk_energy"]
            + accepted["stored"]
            + accepted["dissipated"]
        )
        balance = external_work - accounted
        scale = max(abs(external_work), abs(accounted), np.finfo(float).eps)
        records.append(
            ENFCohesivePropagationPoint(
                increment=increment,
                displacement=float(imposed),
                reaction=accepted["reaction"],
                bulk_strain_energy=accepted["bulk_energy"],
                cohesive_stored_energy=accepted["stored"],
                cohesive_dissipation=accepted["dissipated"],
                external_work=external_work,
                energy_balance_error=balance,
                relative_energy_balance_error=float(abs(balance) / scale),
                maximum_damage=float(
                    np.max(accepted["damage"][~initial_mask], initial=0.0)
                ),
                damaged_length=max(
                    accepted["damaged_length"], initial_failed_length
                ),
                failed_length=max(accepted["failed_length"], initial_failed_length),
                process_zone_length=accepted["process_length"],
                damage_weighted_mode_ii_fraction=accepted["mode_ii_fraction"],
                newton_iterations=sum(
                    int(item["solved"]["iterations"]) for item in accepted_trials
                ),
                residual_norm=float(equilibrium.last_info.residual_norm),
                accepted_subincrements=len(accepted_trials),
                cutbacks=advance.subdivisions,
            )
        )
    mode_ii_length = (
        float(spec.elastic_modulus)
        * float(shear_fracture_energy)
        / float(shear_strength) ** 2
    )
    return ENFCohesivePropagationCurve(
        specification=spec,
        points=tuple(records),
        element_size=max(dx, h / ny),
        process_zone_elements=float(mode_ii_length / dx),
        law=law.summary(),
        source=(
            "AgentFEM assembled displacement-controlled ENF cohesive path; "
            f"{nx}x{2 * ny} Q1 bulk cells"
        ),
        poisson=float(poisson),
        assumption=selected_assumption,
    )


def _enf_point(
    spec,
    *,
    crack_index,
    control_displacement,
    elements_along,
    elements_per_arm,
    poisson,
    assumption,
    interface_stiffness,
    solver_options,
) -> ENFFiniteElementPoint:
    import ufl
    from mpi4py import MPI

    from agentfem import (
        constitutive,
        constraints,
        fields,
        fracture,
        interfaces,
        operators,
        results,
        solvers,
        studies,
    )

    if MPI.COMM_WORLD.size != 1:
        raise NotImplementedError("The initial ENF structural provider is serial.")
    length = 2.0 * float(spec.half_span)
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
    material = constitutive.isotropic_elastic(
        young=spec.elastic_modulus,
        poisson=poisson,
        density=1.0,
        name="ENF isotropic elastic",
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
        name="R_ENF_bulk",
        family="small_strain_linear_elasticity",
    )
    tangent = operators.linearize(bulk, displacement)
    law = interfaces.mixed_mode_bilinear_cohesive(
        normal_strength=interface_stiffness,
        shear_strength=interface_stiffness,
        normal_fracture_energy=interface_stiffness,
        shear_fracture_energy=interface_stiffness,
        normal_stiffness=interface_stiffness,
        tangential_stiffness=interface_stiffness,
        interaction="bk",
    )
    cohesive = fracture.cohesive_force(
        split, displacement, law, normal_hint=(0.0, 1.0), thickness=spec.width
    )
    midpoints = np.mean(
        split.coordinates[cohesive.assembler.topology.negative_nodes], axis=1
    )
    represented_crack = float(crack_index * length / nx)
    precrack = midpoints[:, 0] < represented_crack
    if int(np.count_nonzero(precrack)) != int(crack_index):
        raise RuntimeError("ENF precrack facets do not match the requested mesh tip.")
    cohesive.initialize_precrack(precrack)
    residual = fracture.FiniteStrainCohesiveResidual(bulk, cohesive)

    left_support = lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], -h)
    right_support = lambda x: np.isclose(x[0], length) & np.isclose(x[1], -h)
    load_point = lambda x: np.isclose(x[0], spec.half_span) & np.isclose(x[1], h)
    motion = constraints.component_dirichlet(
        displacement,
        1,
        on=load_point,
        value=0.0,
        name="ENF mid-span motion",
    )
    bcs = (
        constraints.component_dirichlet(
            displacement, 1, on=left_support, value=0.0, name="ENF left roller"
        ).bc,
        constraints.component_dirichlet(
            displacement, 0, on=left_support, value=0.0, name="ENF horizontal datum"
        ).bc,
        constraints.component_dirichlet(
            displacement, 1, on=right_support, value=0.0, name="ENF right roller"
        ).bc,
        motion.bc,
    )

    def set_motion(value):
        motion.value.value = -float(value)

    equilibrium = fracture.FiniteStrainCohesiveEquilibrium(
        residual,
        tangent,
        displacement,
        bcs=bcs,
        set_load=set_motion,
        solver_options=solver_options
        or solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-10,
            maximum_iterations=40,
            line_search=None,
            linear_solver=solvers.direct_solver(),
        ),
        reaction=lambda _function: results.reaction_resultant(
            residual, on=load_point, component=1
        ),
    )
    solved = equilibrium(load=control_displacement, branch="monotonic", cycle=0)
    if not solved["converged"]:
        raise RuntimeError("ENF finite-element equilibrium did not converge.")
    load = abs(float(solved["reaction"]))
    if not np.isfinite(load) or load <= np.finfo(float).eps:
        raise RuntimeError("ENF reaction load is not positive and finite.")
    return ENFFiniteElementPoint(
        crack_length=represented_crack,
        effective_crack_length=represented_crack,
        load=load,
        displacement=float(control_displacement),
        compliance=float(control_displacement / load),
        element_size=max(length / nx, h / ny),
        elements_per_arm=ny,
        newton_iterations=int(solved["iterations"]),
        residual_norm=float(equilibrium.last_info.residual_norm),
    )


__all__ = [
    "ENFCohesivePropagationCertificate",
    "ENFCohesivePropagationCurve",
    "ENFCohesivePropagationPoint",
    "ENFCohesivePropagationStudy",
    "ENFComplianceConvergenceCertificate",
    "ENFFiniteElementConvergenceStudy",
    "ENFFiniteElementCurve",
    "ENFFiniteElementPoint",
    "certify_enf_cohesive_propagation",
    "certify_enf_compliance_convergence",
    "enf_cohesive_propagation_convergence",
    "enf_cohesive_propagation_curve",
    "enf_finite_element_convergence",
    "enf_finite_element_curve",
]
