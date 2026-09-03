"""Executable finite-element delamination benchmark providers.

These providers are deliberately separate from the analytical oracles in
``mixed_mode``.  A result produced here comes from an assembled bulk and
zero-thickness interface problem; it is never relabelled beam theory.
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
    dcb_beam_compliance,
)


@dataclass(frozen=True)
class DCBFiniteElementPoint:
    """One load--opening solution for a discretized DCB specimen."""

    crack_length: float
    effective_crack_length: float
    load: float
    opening: float
    compliance: float
    element_size: float
    elements_per_arm: int
    newton_iterations: int
    residual_norm: float

    def summary(self) -> dict[str, object]:
        return {
            "kind": "dcb_finite_element_point",
            **self.__dict__,
        }


@dataclass(frozen=True)
class DCBFiniteElementCurve:
    """Structure-level DCB compliance and energy-release evidence."""

    specification: DelaminationBenchmarkSpec
    points: tuple[DCBFiniteElementPoint, ...]
    energy_release: DelaminationEnergyReleaseCurve
    source: str
    poisson: float | None = None
    assumption: str | None = None
    interface_stiffness: float | None = None

    @property
    def element_size(self) -> float:
        return max(point.element_size for point in self.points)

    @property
    def identity_sha256(self) -> str:
        """Stable identity of the mesh-level structural evidence."""

        encoded = json.dumps(
            self.summary(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.dcb-finite-element-curve.v1",
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
                "precrack": "fully failed interface facets",
            },
        }


@dataclass(frozen=True)
class DCBComplianceConvergenceCertificate:
    """Spatial-convergence evidence for a precracked, elastic DCB model.

    This certificate deliberately covers structural compliance only.  It does
    not certify cohesive-zone evolution, fracture energy or crack growth.
    Those require the separate delamination propagation certificate.
    """

    reference_source: str
    curve_identity_sha256: tuple[str, ...]
    element_sizes: tuple[float, ...]
    relative_errors_to_reference: tuple[float, ...]
    successive_relative_changes: tuple[float, ...]
    maximum_residual_norms: tuple[float, ...]
    observed_order: float | None
    asymptotic_trend: bool
    reference_errors_nonincreasing: bool
    reference_relative_tolerance: float
    refinement_relative_tolerance: float
    residual_tolerance: float
    accepted: bool

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.dcb-compliance-convergence-certificate.v1",
            **self.__dict__,
            "scope": "precracked elastic structural compliance",
            "excludes": (
                "cohesive-zone evolution",
                "fracture-energy calibration",
                "crack propagation",
            ),
        }


@dataclass(frozen=True)
class DCBFiniteElementConvergenceStudy:
    """Three-or-more-level assembled DCB compliance study."""

    specification: DelaminationBenchmarkSpec
    curves: tuple[DCBFiniteElementCurve, ...]
    certificate: DCBComplianceConvergenceCertificate

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.dcb-finite-element-convergence-study.v1",
            "specification": self.specification.summary(),
            "curves": [curve.summary() for curve in self.curves],
            "certificate": self.certificate.summary(),
        }


@dataclass(frozen=True)
class DCBCohesivePropagationPoint:
    """One accepted displacement-controlled DCB propagation increment."""

    increment: int
    opening: float
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
    newton_iterations: int
    residual_norm: float
    accepted_subincrements: int = 1
    cutbacks: int = 0

    def summary(self) -> dict[str, object]:
        return {"kind": "dcb_cohesive_propagation_point", **self.__dict__}


@dataclass(frozen=True)
class DCBCohesivePropagationCurve:
    """Accepted DCB cohesive evolution and its work--energy evidence."""

    specification: DelaminationBenchmarkSpec
    points: tuple[DCBCohesivePropagationPoint, ...]
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
            "schema": "agentfem.dcb-cohesive-propagation-curve.v1",
            "specification": self.specification.summary(),
            "source": self.source,
            "element_size": self.element_size,
            "process_zone_elements": self.process_zone_elements,
            "law": dict(self.law),
            "model": {
                "poisson": self.poisson,
                "assumption": self.assumption,
            },
            "points": [point.summary() for point in self.points],
            "evidence_scope": (
                "displacement-controlled monotonic Mode-I cohesive propagation"
            ),
        }


@dataclass(frozen=True)
class DCBCohesivePropagationCertificate:
    """Three-level DCB propagation, energy and process-zone certificate."""

    curve_identity_sha256: tuple[str, ...]
    element_sizes: tuple[float, ...]
    peak_reactions: tuple[float, ...]
    final_damaged_lengths: tuple[float, ...]
    final_failed_lengths: tuple[float, ...]
    maximum_relative_energy_errors: tuple[float, ...]
    process_zone_elements: tuple[float, ...]
    peak_reaction_changes: tuple[float, ...]
    damaged_length_changes: tuple[float, ...]
    failed_length_changes: tuple[float, ...]
    refinement_relative_tolerance: float
    energy_relative_tolerance: float
    required_process_zone_elements: float
    propagation_observed: bool
    accepted: bool

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.dcb-cohesive-propagation-certificate.v1",
            **self.__dict__,
            "scope": "Mode-I DCB cohesive propagation",
            "excludes": (
                "Mode-II ENF",
                "mixed-mode MMB",
                "experimental material calibration",
            ),
        }


@dataclass(frozen=True)
class DCBCohesivePropagationStudy:
    """Three-or-more-level assembled DCB cohesive propagation study."""

    specification: DelaminationBenchmarkSpec
    curves: tuple[DCBCohesivePropagationCurve, ...]
    certificate: DCBCohesivePropagationCertificate

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.dcb-cohesive-propagation-study.v1",
            "specification": self.specification.summary(),
            "curves": [curve.summary() for curve in self.curves],
            "certificate": self.certificate.summary(),
        }


def certify_dcb_compliance_convergence(
    spec: DelaminationBenchmarkSpec,
    curves,
    *,
    reference_relative_tolerance: float,
    refinement_relative_tolerance: float,
    residual_tolerance: float = 1.0e-8,
) -> DCBComplianceConvergenceCertificate:
    """Certify assembled DCB compliance against a declared beam oracle.

    The curves must describe successively refined meshes on identical crack
    coordinates.  The oracle is useful for a thin-beam verification rung but
    remains explicitly identified as analytical rather than experimental.
    Acceptance requires a decreasing inter-mesh change; it does not require
    every 2D elasticity mesh to approach the lower-dimensional beam oracle
    monotonically.
    """

    if not isinstance(spec, DelaminationBenchmarkSpec) or spec.kind != "dcb":
        raise ValueError("DCB compliance convergence requires a DCB specification.")
    selected = tuple(curves)
    if len(selected) < 3 or not all(
        isinstance(curve, DCBFiniteElementCurve) for curve in selected
    ):
        raise ValueError("DCB compliance convergence requires at least three FE curves.")
    coordinates = np.asarray(
        [point.effective_crack_length for point in selected[0].points], dtype=float
    )
    if coordinates.size < 3:
        raise ValueError("Every DCB convergence curve needs at least three cracks.")
    reference = np.asarray(dcb_beam_compliance(spec, coordinates), dtype=float)
    reference_scale = float(np.linalg.norm(reference))
    if reference_scale <= np.finfo(float).eps:
        raise ValueError("The DCB analytical compliance reference is zero.")

    compliance = []
    sizes = []
    residuals = []
    for curve in selected:
        if curve.specification != spec:
            raise ValueError("Every DCB curve must use the same specification.")
        curve_coordinates = np.asarray(
            [point.effective_crack_length for point in curve.points], dtype=float
        )
        if curve_coordinates.shape != coordinates.shape or not np.allclose(
            curve_coordinates, coordinates, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("Every DCB curve must use identical crack coordinates.")
        compliance.append(np.asarray([point.compliance for point in curve.points]))
        sizes.append(float(curve.element_size))
        residuals.append(max(point.residual_norm for point in curve.points))
    model_contracts = {
        (curve.poisson, curve.assumption, curve.interface_stiffness)
        for curve in selected
    }
    if len(model_contracts) != 1:
        raise ValueError("Every DCB curve must use the same discrete model contract.")
    if any(left <= right for left, right in zip(sizes[:-1], sizes[1:])):
        raise ValueError("DCB element sizes must decrease from coarse to fine.")

    errors = tuple(
        float(np.linalg.norm(values - reference) / reference_scale)
        for values in compliance
    )
    changes = tuple(
        float(np.linalg.norm(right - left) / max(np.linalg.norm(right), np.finfo(float).eps))
        for left, right in zip(compliance[:-1], compliance[1:])
    )
    refinement_ratios = tuple(
        left / right for left, right in zip(sizes[:-1], sizes[1:])
    )
    uniform_refinement = bool(
        max(refinement_ratios) / min(refinement_ratios) <= 1.05
    )
    observed_order = None
    if (
        uniform_refinement
        and len(changes) >= 2
        and changes[-1] > np.finfo(float).eps
    ):
        ratio = refinement_ratios[-1]
        if changes[-2] > 0.0:
            observed_order = float(np.log(changes[-2] / changes[-1]) / np.log(ratio))
    asymptotic = bool(
        all(right < left for left, right in zip(changes[:-1], changes[1:]))
    )
    reference_errors_nonincreasing = bool(
        all(right <= left for left, right in zip(errors[:-1], errors[1:]))
    )
    reference_limit = float(reference_relative_tolerance)
    refinement_limit = float(refinement_relative_tolerance)
    residual_limit = float(residual_tolerance)
    if any(
        not np.isfinite(value) or value < 0.0
        for value in (reference_limit, refinement_limit, residual_limit)
    ):
        raise ValueError("DCB convergence tolerances must be finite and nonnegative.")
    return DCBComplianceConvergenceCertificate(
        reference_source=f"{spec.source}; classical simple-beam compliance",
        curve_identity_sha256=tuple(curve.identity_sha256 for curve in selected),
        element_sizes=tuple(sizes),
        relative_errors_to_reference=errors,
        successive_relative_changes=changes,
        maximum_residual_norms=tuple(float(value) for value in residuals),
        observed_order=observed_order,
        asymptotic_trend=asymptotic,
        reference_errors_nonincreasing=reference_errors_nonincreasing,
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


def dcb_finite_element_convergence(
    spec: DelaminationBenchmarkSpec,
    *,
    crack_length,
    load: float,
    specimen_length: float,
    mesh_levels,
    poisson: float = 0.3,
    assumption: str = "plane_stress",
    interface_stiffness: float | None = None,
    solver_options=None,
    reference_relative_tolerance: float = 0.10,
    refinement_relative_tolerance: float = 0.05,
    residual_tolerance: float = 1.0e-8,
) -> DCBFiniteElementConvergenceStudy:
    """Run and certify three or more assembled DCB mesh levels."""

    levels = tuple((int(nx), int(ny)) for nx, ny in mesh_levels)
    if len(levels) < 3 or any(nx < 4 or ny < 1 for nx, ny in levels):
        raise ValueError("DCB convergence needs at least three valid mesh levels.")
    if any(
        right[0] <= left[0] or right[1] < left[1]
        for left, right in zip(levels[:-1], levels[1:])
    ):
        raise ValueError("DCB mesh levels must refine axial and arm discretization.")
    curves = tuple(
        dcb_finite_element_curve(
            spec,
            crack_length=crack_length,
            load=load,
            specimen_length=specimen_length,
            elements_along=nx,
            elements_per_arm=ny,
            poisson=poisson,
            assumption=assumption,
            interface_stiffness=interface_stiffness,
            solver_options=solver_options,
        )
        for nx, ny in levels
    )
    certificate = certify_dcb_compliance_convergence(
        spec,
        curves,
        reference_relative_tolerance=reference_relative_tolerance,
        refinement_relative_tolerance=refinement_relative_tolerance,
        residual_tolerance=residual_tolerance,
    )
    return DCBFiniteElementConvergenceStudy(spec, curves, certificate)


def dcb_finite_element_curve(
    spec: DelaminationBenchmarkSpec,
    *,
    crack_length,
    load: float,
    specimen_length: float,
    elements_along: int,
    elements_per_arm: int,
    poisson: float = 0.3,
    assumption: str = "plane_stress",
    interface_stiffness: float | None = None,
    solver_options=None,
) -> DCBFiniteElementCurve:
    """Solve a linear-elastic DCB family with one fixed-path interface.

    The interface is geometrically split over the complete specimen. Facets
    behind each requested crack tip are initialized as a precrack; facets
    ahead of it remain an intact, high-stiffness cohesive tie.  The function
    returns independently assembled compliance points and derives ``G_I``
    from their finite-element compliance curve.

    This provider is an executable verification rung, not an ASTM material
    test reduction.  Crack tips must align with the uniform axial mesh so the
    requested and represented geometries cannot silently differ.
    """

    if not isinstance(spec, DelaminationBenchmarkSpec) or spec.kind != "dcb":
        raise ValueError("dcb_finite_element_curve requires a DCB specification.")
    lengths = np.asarray(crack_length, dtype=float).reshape(-1)
    if lengths.size < 3 or np.any(np.diff(lengths) <= 0.0):
        raise ValueError("DCB finite-element evidence needs three increasing cracks.")
    selected_load = float(load)
    length = float(specimen_length)
    nx = int(elements_along)
    ny = int(elements_per_arm)
    if (
        not np.isfinite(selected_load)
        or selected_load <= 0.0
        or not np.isfinite(length)
        or length <= float(lengths[-1])
        or nx < 4
        or ny < 1
    ):
        raise ValueError("DCB load, specimen length and mesh counts are invalid.")
    dx = length / nx
    mesh_indices = np.rint(lengths / dx).astype(int)
    effective = mesh_indices * dx
    tolerance = 128.0 * np.finfo(float).eps * max(length, 1.0)
    if np.any(np.abs(effective - lengths) > tolerance):
        raise ValueError(
            "Every DCB crack length must align with the uniform axial mesh."
        )
    selected_assumption = str(assumption).strip().lower().replace("-", "_")
    if selected_assumption not in {"plane_stress", "plane_strain"}:
        raise ValueError("DCB assumption must be plane_stress or plane_strain.")
    stiffness = (
        1.0e6 * spec.elastic_modulus / spec.arm_thickness
        if interface_stiffness is None
        else float(interface_stiffness)
    )
    if not np.isfinite(stiffness) or stiffness <= 0.0:
        raise ValueError("interface_stiffness must be finite and positive.")

    points = tuple(
        _dcb_point(
            spec,
            crack_index=int(index),
            load=selected_load,
            specimen_length=length,
            elements_along=nx,
            elements_per_arm=ny,
            poisson=float(poisson),
            assumption=selected_assumption,
            interface_stiffness=stiffness,
            solver_options=solver_options,
        )
        for index in mesh_indices
    )
    source = f"AgentFEM assembled DCB Q1/cohesive curve; {nx}x{2 * ny} bulk cells"
    energy = compliance_energy_release_curve(
        spec,
        crack_length=[point.effective_crack_length for point in points],
        load=[point.load for point in points],
        compliance=[point.compliance for point in points],
        source=source,
    )
    return DCBFiniteElementCurve(
        specification=spec,
        points=points,
        energy_release=energy,
        source=source,
        poisson=float(poisson),
        assumption=selected_assumption,
        interface_stiffness=stiffness,
    )


def certify_dcb_cohesive_propagation(
    spec: DelaminationBenchmarkSpec,
    curves,
    *,
    refinement_relative_tolerance: float = 0.10,
    energy_relative_tolerance: float = 0.03,
    required_process_zone_elements: float = 3.0,
) -> DCBCohesivePropagationCertificate:
    """Certify refined DCB propagation without conflating local-law tests.

    Every curve must use the same specimen, cohesive law and imposed opening
    coordinates.  Acceptance requires actual growth beyond the initialized
    precrack, bounded work--energy error, adequate cohesive-zone resolution,
    and stable peak load and final failed length on the finest two meshes.
    """

    if not isinstance(spec, DelaminationBenchmarkSpec) or spec.kind != "dcb":
        raise ValueError("DCB propagation certification requires a DCB specification.")
    selected = tuple(curves)
    if len(selected) < 3 or not all(
        isinstance(curve, DCBCohesivePropagationCurve) for curve in selected
    ):
        raise ValueError("DCB propagation certification requires at least three curves.")
    sizes = tuple(float(curve.element_size) for curve in selected)
    if any(left <= right for left, right in zip(sizes[:-1], sizes[1:])):
        raise ValueError("DCB propagation meshes must be ordered coarse to fine.")
    reference_opening = np.asarray(
        [point.opening for point in selected[0].points], dtype=float
    )
    reference_law = selected[0].law
    reference_model = (selected[0].poisson, selected[0].assumption)
    for curve in selected:
        if (
            curve.specification != spec
            or curve.law != reference_law
            or (curve.poisson, curve.assumption) != reference_model
        ):
            raise ValueError("Every DCB propagation curve must share one model and law.")
        opening = np.asarray([point.opening for point in curve.points], dtype=float)
        if opening.shape != reference_opening.shape or not np.allclose(
            opening, reference_opening, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("Every DCB propagation curve needs identical openings.")
    peak = tuple(float(curve.peak_reaction) for curve in selected)
    damaged = tuple(float(curve.points[-1].damaged_length) for curve in selected)
    failed = tuple(float(curve.points[-1].failed_length) for curve in selected)
    energy = tuple(
        max(point.relative_energy_balance_error for point in curve.points)
        for curve in selected
    )
    zone = tuple(float(curve.process_zone_elements) for curve in selected)

    def changes(values):
        return tuple(
            abs(right - left) / max(abs(right), np.finfo(float).eps)
            for left, right in zip(values[:-1], values[1:])
        )

    peak_changes = changes(peak)
    damaged_changes = changes(damaged)
    failed_changes = changes(failed)
    refinement_limit = float(refinement_relative_tolerance)
    energy_limit = float(energy_relative_tolerance)
    zone_limit = float(required_process_zone_elements)
    if any(
        not np.isfinite(value) or value < 0.0
        for value in (refinement_limit, energy_limit, zone_limit)
    ):
        raise ValueError("DCB propagation tolerances must be finite and nonnegative.")
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
    )
    return DCBCohesivePropagationCertificate(
        curve_identity_sha256=tuple(curve.identity_sha256 for curve in selected),
        element_sizes=sizes,
        peak_reactions=peak,
        final_damaged_lengths=damaged,
        final_failed_lengths=failed,
        maximum_relative_energy_errors=energy,
        process_zone_elements=zone,
        peak_reaction_changes=peak_changes,
        damaged_length_changes=damaged_changes,
        failed_length_changes=failed_changes,
        refinement_relative_tolerance=refinement_limit,
        energy_relative_tolerance=energy_limit,
        required_process_zone_elements=zone_limit,
        propagation_observed=propagation,
        accepted=accepted,
    )


def dcb_cohesive_propagation_convergence(
    spec: DelaminationBenchmarkSpec,
    *,
    precrack_length: float,
    specimen_length: float,
    opening,
    strength: float,
    fracture_energy: float,
    initial_stiffness: float,
    mesh_levels,
    poisson: float = 0.3,
    assumption: str = "plane_stress",
    solver_options=None,
    minimum_opening_increment: float | None = None,
    maximum_cutbacks: int = 12,
    refinement_relative_tolerance: float = 0.10,
    energy_relative_tolerance: float = 0.03,
    required_process_zone_elements: float = 3.0,
) -> DCBCohesivePropagationStudy:
    """Execute and certify three or more DCB cohesive propagation levels."""

    levels = tuple((int(nx), int(ny)) for nx, ny in mesh_levels)
    if len(levels) < 3 or any(nx < 4 or ny < 1 for nx, ny in levels):
        raise ValueError("DCB propagation convergence needs three valid levels.")
    if any(
        right[0] <= left[0] or right[1] < left[1]
        for left, right in zip(levels[:-1], levels[1:])
    ):
        raise ValueError("DCB propagation levels must refine the bulk mesh.")
    curves = tuple(
        dcb_cohesive_propagation_curve(
            spec,
            precrack_length=precrack_length,
            specimen_length=specimen_length,
            opening=opening,
            strength=strength,
            fracture_energy=fracture_energy,
            initial_stiffness=initial_stiffness,
            elements_along=nx,
            elements_per_arm=ny,
            poisson=poisson,
            assumption=assumption,
            solver_options=solver_options,
            minimum_opening_increment=minimum_opening_increment,
            maximum_cutbacks=maximum_cutbacks,
        )
        for nx, ny in levels
    )
    certificate = certify_dcb_cohesive_propagation(
        spec,
        curves,
        refinement_relative_tolerance=refinement_relative_tolerance,
        energy_relative_tolerance=energy_relative_tolerance,
        required_process_zone_elements=required_process_zone_elements,
    )
    return DCBCohesivePropagationStudy(spec, curves, certificate)


def dcb_cohesive_propagation_curve(
    spec: DelaminationBenchmarkSpec,
    *,
    precrack_length: float,
    specimen_length: float,
    opening,
    strength: float,
    fracture_energy: float,
    initial_stiffness: float,
    elements_along: int,
    elements_per_arm: int,
    poisson: float = 0.3,
    assumption: str = "plane_stress",
    solver_options=None,
    minimum_opening_increment: float | None = None,
    maximum_cutbacks: int = 12,
) -> DCBCohesivePropagationCurve:
    """Run a displacement-controlled, irreversible Mode-I DCB path.

    The opening coordinate is the relative vertical displacement of the two
    crack-mouth arms. Cohesive state is committed only after global Newton
    convergence. Failed requested increments are bisected into internal
    subincrements without changing the requested output coordinates. The
    returned path keeps reaction, interface state, continuation evidence and
    a work--energy ledger together so crack growth cannot be certified from a
    damage image alone.
    """

    import ufl
    from dolfinx import fem
    from dolfinx import mesh as dolfinx_mesh
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

    if not isinstance(spec, DelaminationBenchmarkSpec) or spec.kind != "dcb":
        raise ValueError("dcb_cohesive_propagation_curve requires a DCB specification.")
    if MPI.COMM_WORLD.size != 1:
        raise NotImplementedError("The DCB propagation certificate is currently serial.")
    openings = np.asarray(opening, dtype=float).reshape(-1)
    if (
        openings.size < 3
        or not np.isclose(openings[0], 0.0)
        or np.any(np.diff(openings) <= 0.0)
        or not np.all(np.isfinite(openings))
    ):
        raise ValueError("DCB opening must start at zero and increase at least twice.")
    cutback_limit = int(maximum_cutbacks)
    if cutback_limit < 0:
        raise ValueError("maximum_cutbacks must be nonnegative.")
    requested_differences = np.diff(openings)
    minimum_increment = (
        float(minimum_opening_increment)
        if minimum_opening_increment is not None
        else float(np.min(requested_differences) / 256.0)
    )
    if not np.isfinite(minimum_increment) or minimum_increment <= 0.0:
        raise ValueError("minimum_opening_increment must be finite and positive.")
    length = float(specimen_length)
    precrack = float(precrack_length)
    nx = int(elements_along)
    ny_arm = int(elements_per_arm)
    if length <= precrack or precrack <= 0.0 or nx < 4 or ny_arm < 1:
        raise ValueError("DCB geometry and mesh counts are invalid.")
    dx = length / nx
    crack_index = int(round(precrack / dx))
    represented_crack = crack_index * dx
    tolerance = 128.0 * np.finfo(float).eps * max(length, 1.0)
    if abs(represented_crack - precrack) > tolerance:
        raise ValueError("The DCB precrack must align with the axial mesh.")
    selected_assumption = str(assumption).strip().lower().replace("-", "_")
    if selected_assumption not in {"plane_stress", "plane_strain"}:
        raise ValueError("DCB assumption must be plane_stress or plane_strain.")

    h = float(spec.arm_thickness)
    x_values = np.linspace(0.0, length, nx + 1)
    y_values = np.linspace(-h, h, 2 * ny_arm + 1)
    coordinates = np.asarray([(x, y) for y in y_values for x in x_values])

    def node(i, j):
        return j * (nx + 1) + i

    cells = np.asarray(
        [
            [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]
            for j in range(2 * ny_arm)
            for i in range(nx)
        ],
        dtype=np.int64,
    )
    mid = ny_arm
    interface_facets = np.asarray(
        [[node(i, mid), node(i + 1, mid)] for i in range(nx)], dtype=np.int64
    )
    upper_cells = np.arange(ny_arm * nx, 2 * ny_arm * nx, dtype=np.int64)
    split = interfaces.split_conforming_line_interface(
        coordinates, cells, interface_facets, positive_cells=upper_cells
    )
    domain = interfaces.create_dolfinx_split_mesh(
        split, comm=MPI.COMM_SELF, cell_type="quadrilateral"
    )
    displacement = fields.displacement(domain)
    study = studies.static_solid(dimension=2, assumption=selected_assumption)
    material = constitutive.isotropic_elastic(
        young=spec.elastic_modulus,
        poisson=float(poisson),
        density=1.0,
        name="DCB isotropic elastic",
    )
    measure = ufl.Measure("dx", domain=domain)
    internal = spec.width * operators.internal_force_vector(
        displacement.value,
        displacement.test,
        material,
        study=study,
        measure=measure,
    )
    bulk = operators.residual_operator(
        internal.expression,
        name="R_DCB_bulk",
        family="small_strain_linear_elasticity",
    )
    tangent = operators.linearize(bulk, displacement)
    law = interfaces.bilinear_cohesive(
        strength=float(strength),
        fracture_energy=float(fracture_energy),
        initial_stiffness=float(initial_stiffness),
    )
    cohesive = fracture.cohesive_force(
        split,
        displacement,
        law,
        normal_hint=(0.0, 1.0),
        tangential="tie",
        tangential_stiffness=float(initial_stiffness),
        thickness=spec.width,
    )
    topology = cohesive.assembler.topology
    facet_midpoints = np.mean(split.coordinates[topology.negative_nodes], axis=1)
    initial_mask = facet_midpoints[:, 0] < represented_crack
    if int(np.count_nonzero(initial_mask)) != crack_index:
        raise RuntimeError("DCB precrack facets do not match the represented tip.")
    cohesive.initialize_precrack(initial_mask)
    residual = fracture.FiniteStrainCohesiveResidual(bulk, cohesive)

    upper_motion = constraints.component_dirichlet(
        displacement,
        1,
        on=lambda x: np.isclose(x[0], 0.0) & (x[1] > 1.0e-12),
        value=0.0,
        name="upper crack-mouth opening",
    )
    lower_motion = constraints.component_dirichlet(
        displacement,
        1,
        on=lambda x: np.isclose(x[0], 0.0) & (x[1] < -1.0e-12),
        value=0.0,
        name="lower crack-mouth opening",
    )
    right_bottom = lambda x: np.isclose(x[0], length) & np.isclose(x[1], -h)
    right_top = lambda x: np.isclose(x[0], length) & np.isclose(x[1], h)
    bcs = (
        upper_motion.bc,
        lower_motion.bc,
        constraints.component_dirichlet(
            displacement, 0, on=right_bottom, value=0.0,
            name="remove rigid x translation",
        ).bc,
        constraints.component_dirichlet(
            displacement, 0, on=right_top, value=0.0,
            name="remove rigid rotation",
        ).bc,
    )

    def set_opening(value):
        upper_motion.value.value = 0.5 * float(value)
        lower_motion.value.value = -0.5 * float(value)

    equilibrium = fracture.FiniteStrainCohesiveEquilibrium(
        residual,
        tangent,
        displacement,
        bcs=bcs,
        set_load=set_opening,
        solver_options=solver_options
        or solvers.newton(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-11,
            maximum_iterations=30,
            # A local cohesive tangent may be negative on the softening
            # branch. Requiring monotonic residual decrease can reject a valid
            # displacement-controlled Newton correction, so this structural
            # benchmark uses full Newton and lets convergence/energy evidence
            # decide acceptance.
            line_search=None,
            linear_solver=solvers.direct_solver(),
        ),
        reaction=lambda _function: results.reaction_resultant(
            residual,
            on=lambda x: np.isclose(x[0], 0.0) & (x[1] > 1.0e-12),
            component=1,
        ),
        bulk_strain_energy=lambda function: 0.5
        * operators.dual_product(internal, function),
    )
    records = []
    previous_opening = 0.0
    previous_reaction = 0.0
    external_work = 0.0
    initial_failed_length = represented_crack
    initial_cohesive_dissipation = None

    def accept_trial(imposed):
        nonlocal previous_opening
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
            float(imposed) - previous_opening
        )
        bulk_energy = float(solved["bulk_strain_energy"])
        stored = float(response.stored_energy)
        raw_dissipation = float(response.dissipated_energy)
        if initial_cohesive_dissipation is None:
            initial_cohesive_dissipation = raw_dissipation
        dissipated = raw_dissipation - initial_cohesive_dissipation
        damage = np.max(np.asarray(response.damage, dtype=float), axis=1)
        damaged_mask = damage > 1.0e-10
        failed_mask = damage >= 1.0 - 1.0e-8
        process_mask = damaged_mask & ~failed_mask
        damaged_length = float(np.count_nonzero(damaged_mask) * dx)
        failed_length = float(np.count_nonzero(failed_mask) * dx)
        process_length = float(np.count_nonzero(process_mask) * dx)
        residual.commit()
        previous_opening = float(imposed)
        previous_reaction = reaction
        return {
            "solved": solved,
            "response": response,
            "reaction": reaction,
            "bulk_energy": bulk_energy,
            "stored": stored,
            "dissipated": dissipated,
            "damage": damage,
            "damaged_length": damaged_length,
            "failed_length": failed_length,
            "process_length": process_length,
        }

    def advance_to(target, *, depth, evidence):
        start = previous_opening
        accepted = accept_trial(target)
        if accepted is not None:
            evidence.append(accepted)
            return accepted
        interval = float(target) - start
        if depth >= cutback_limit or 0.5 * interval < minimum_increment:
            message = (
                equilibrium.last_info.message
                if equilibrium.last_info is not None
                else "unknown nonlinear failure"
            )
            raise RuntimeError(
                "DCB cohesive continuation could not reach opening "
                f"{float(target):.12g}; last accepted opening {start:.12g}; "
                f"cutback depth {depth}; {message}."
            )
        midpoint = start + 0.5 * interval
        advance_to(midpoint, depth=depth + 1, evidence=evidence)
        return advance_to(target, depth=depth + 1, evidence=evidence)

    for increment, imposed in enumerate(openings):
        accepted_trials = []
        accepted = advance_to(float(imposed), depth=0, evidence=accepted_trials)
        solved = accepted["solved"]
        reaction = accepted["reaction"]
        bulk_energy = accepted["bulk_energy"]
        stored = accepted["stored"]
        dissipated = accepted["dissipated"]
        damage = accepted["damage"]
        damaged_length = accepted["damaged_length"]
        failed_length = accepted["failed_length"]
        process_length = accepted["process_length"]
        accounted = bulk_energy + stored + dissipated
        balance = external_work - accounted
        scale = max(abs(external_work), abs(accounted), np.finfo(float).eps)
        records.append(
            DCBCohesivePropagationPoint(
                increment=increment,
                opening=float(imposed),
                reaction=reaction,
                bulk_strain_energy=bulk_energy,
                cohesive_stored_energy=stored,
                cohesive_dissipation=dissipated,
                external_work=external_work,
                energy_balance_error=balance,
                relative_energy_balance_error=float(abs(balance) / scale),
                maximum_damage=float(
                    np.max(damage[~initial_mask], initial=0.0)
                ),
                damaged_length=max(damaged_length, initial_failed_length),
                failed_length=max(failed_length, initial_failed_length),
                process_zone_length=process_length,
                newton_iterations=sum(
                    int(item["solved"]["iterations"])
                    for item in accepted_trials
                ),
                residual_norm=float(equilibrium.last_info.residual_norm),
                accepted_subincrements=len(accepted_trials),
                cutbacks=max(len(accepted_trials) - 1, 0),
            )
        )
    characteristic = law.characteristic_length(spec.elastic_modulus)
    return DCBCohesivePropagationCurve(
        specification=spec,
        points=tuple(records),
        element_size=max(dx, h / ny_arm),
        process_zone_elements=float(characteristic / dx),
        law=law.summary(),
        source=(
            "AgentFEM assembled displacement-controlled DCB cohesive path; "
            f"{nx}x{2 * ny_arm} Q1 bulk cells"
        ),
        poisson=float(poisson),
        assumption=selected_assumption,
    )


def _dcb_point(
    spec,
    *,
    crack_index,
    load,
    specimen_length,
    elements_along,
    elements_per_arm,
    poisson,
    assumption,
    interface_stiffness,
    solver_options,
) -> DCBFiniteElementPoint:
    # Heavy numerical imports remain local so benchmark metadata and analytical
    # oracles stay usable in lightweight documentation environments.
    import ufl
    from dolfinx import fem
    from dolfinx import mesh as dolfinx_mesh
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
        raise NotImplementedError(
            "The initial DCB finite-element verification provider is serial; "
            "distributed cohesive assembly is verified through its separate gate."
        )
    h = float(spec.arm_thickness)
    nx = int(elements_along)
    ny_arm = int(elements_per_arm)
    x_values = np.linspace(0.0, float(specimen_length), nx + 1)
    y_values = np.linspace(-h, h, 2 * ny_arm + 1)
    points = np.asarray(
        [(x, y) for y in y_values for x in x_values],
        dtype=float,
    )

    def node(i, j):
        return j * (nx + 1) + i

    cells = np.asarray(
        [
            [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]
            for j in range(2 * ny_arm)
            for i in range(nx)
        ],
        dtype=np.int64,
    )
    mid = ny_arm
    interface_facets = np.asarray(
        [[node(i, mid), node(i + 1, mid)] for i in range(nx)],
        dtype=np.int64,
    )
    upper_cells = np.arange(ny_arm * nx, 2 * ny_arm * nx, dtype=np.int64)
    split = interfaces.split_conforming_line_interface(
        points,
        cells,
        interface_facets,
        positive_cells=upper_cells,
    )
    domain = interfaces.create_dolfinx_split_mesh(
        split,
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    displacement = fields.displacement(domain)
    study = studies.static_solid(dimension=2, assumption=assumption)
    material = constitutive.isotropic_elastic(
        young=spec.elastic_modulus,
        poisson=poisson,
        density=1.0,
        name="DCB isotropic elastic",
    )
    bulk_measure = ufl.Measure("dx", domain=domain)
    internal = spec.width * operators.internal_force_vector(
        displacement.value,
        displacement.test,
        material,
        study=study,
        measure=bulk_measure,
    )

    facet_dimension = domain.topology.dim - 1
    upper_left = dolfinx_mesh.locate_entities_boundary(
        domain,
        facet_dimension,
        lambda x: np.isclose(x[0], 0.0) & (x[1] >= -1.0e-12),
    )
    lower_left = dolfinx_mesh.locate_entities_boundary(
        domain,
        facet_dimension,
        lambda x: np.isclose(x[0], 0.0) & (x[1] <= 1.0e-12),
    )
    tagged_facets = np.concatenate((upper_left, lower_left))
    tagged_values = np.concatenate(
        (
            np.full(upper_left.size, 1, dtype=np.int32),
            np.full(lower_left.size, 2, dtype=np.int32),
        )
    )
    if np.unique(tagged_facets).size != tagged_facets.size:
        raise RuntimeError("DCB crack-mouth facet sets must not overlap.")
    order = np.argsort(tagged_facets)
    tags = dolfinx_mesh.meshtags(
        domain,
        facet_dimension,
        tagged_facets[order],
        tagged_values[order],
    )
    ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
    load_parameter = fem.Constant(domain, 0.0)
    traction = load_parameter / h
    external = operators.OperatorForm(
        name="DCB crack-mouth load",
        expression=(
            traction * displacement.test[1] * ds(1)
            - traction * displacement.test[1] * ds(2)
        ),
        kind="surface_force",
        role="vector",
        family="dcb_mode_i",
    )
    bulk = operators.residual_operator(
        (internal - external).expression,
        name="R_DCB_bulk",
        family="small_strain_linear_elasticity",
    )
    tangent = operators.linearize(bulk, displacement)
    law = interfaces.bilinear_cohesive(
        strength=interface_stiffness,
        fracture_energy=interface_stiffness,
        initial_stiffness=interface_stiffness,
    )
    cohesive = fracture.cohesive_force(
        split,
        displacement,
        law,
        normal_hint=(0.0, 1.0),
        tangential="tie",
        tangential_stiffness=interface_stiffness,
        thickness=spec.width,
    )
    topology = cohesive.assembler.topology
    facet_midpoints = np.mean(
        split.coordinates[topology.negative_nodes],
        axis=1,
    )
    represented_crack = float(crack_index * specimen_length / nx)
    precrack = facet_midpoints[:, 0] < represented_crack
    if int(np.count_nonzero(precrack)) != int(crack_index):
        raise RuntimeError("DCB precrack facets do not match the requested mesh tip.")
    cohesive.initialize_precrack(precrack)
    residual = fracture.FiniteStrainCohesiveResidual(bulk, cohesive)
    right_bottom = lambda x: np.isclose(x[0], specimen_length) & np.isclose(x[1], -h)
    right_top = lambda x: np.isclose(x[0], specimen_length) & np.isclose(x[1], h)
    bcs = (
        constraints.component_dirichlet(
            displacement,
            0,
            on=right_bottom,
            value=0.0,
            name="remove rigid x translation",
        ).bc,
        constraints.component_dirichlet(
            displacement,
            1,
            on=right_bottom,
            value=0.0,
            name="remove rigid y motion",
        ).bc,
        constraints.component_dirichlet(
            displacement,
            0,
            on=right_top,
            value=0.0,
            name="remove rigid rotation",
        ).bc,
    )
    dof_coordinates = displacement.space.tabulate_dof_coordinates()

    def opening(function):
        values = function.x.array.reshape((-1, 2))
        left = np.isclose(dof_coordinates[:, 0], 0.0)
        upper = left & (dof_coordinates[:, 1] > 1.0e-12)
        lower = left & (dof_coordinates[:, 1] < -1.0e-12)
        return float(np.mean(values[upper, 1]) - np.mean(values[lower, 1]))

    equilibrium = fracture.FiniteStrainCohesiveEquilibrium(
        residual,
        tangent,
        displacement,
        bcs=bcs,
        load_parameter=load_parameter,
        solver_options=solver_options
        or solvers.newton(
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
            maximum_iterations=12,
            linear_solver=solvers.direct_solver(),
        ),
        control_displacement=opening,
        reaction=lambda _function: results.reaction_resultant(
            residual,
            on=lambda x: np.isclose(x[0], 0.0) & (x[1] > 1.0e-12),
            component=1,
        ),
    )
    solved = equilibrium(load=load, branch="monotonic", cycle=0)
    if not solved["converged"]:
        raise RuntimeError("DCB finite-element equilibrium did not converge.")
    opening_value = float(solved["control_displacement"])
    if not np.isfinite(opening_value) or opening_value <= 0.0:
        raise RuntimeError("DCB finite-element opening is not positive and finite.")
    return DCBFiniteElementPoint(
        crack_length=represented_crack,
        effective_crack_length=represented_crack,
        load=float(load),
        opening=opening_value,
        compliance=opening_value / float(load),
        element_size=max(specimen_length / nx, h / ny_arm),
        elements_per_arm=ny_arm,
        newton_iterations=int(solved["iterations"]),
        residual_norm=float(equilibrium.last_info.residual_norm),
    )


__all__ = [
    "DCBCohesivePropagationCertificate",
    "DCBCohesivePropagationCurve",
    "DCBCohesivePropagationPoint",
    "DCBCohesivePropagationStudy",
    "DCBComplianceConvergenceCertificate",
    "DCBFiniteElementCurve",
    "DCBFiniteElementConvergenceStudy",
    "DCBFiniteElementPoint",
    "certify_dcb_cohesive_propagation",
    "certify_dcb_compliance_convergence",
    "dcb_cohesive_propagation_convergence",
    "dcb_cohesive_propagation_curve",
    "dcb_finite_element_convergence",
    "dcb_finite_element_curve",
]
