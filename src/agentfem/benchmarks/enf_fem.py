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
    "ENFComplianceConvergenceCertificate",
    "ENFFiniteElementConvergenceStudy",
    "ENFFiniteElementCurve",
    "ENFFiniteElementPoint",
    "certify_enf_compliance_convergence",
    "enf_finite_element_convergence",
    "enf_finite_element_curve",
]
