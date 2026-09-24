# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""External structure-level ratcheting evidence for 316 stainless steel.

The benchmark reproduces the public SIMULIA shouldered axisymmetric specimen.
Geometry, material parameters, load ratios and the averaging location come from
the public example.  The experimental response is available only as Figure 4,
so digitized values remain explicitly uncertain and are never presented as an
exact vendor table.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from importlib import resources
from pathlib import Path

import numpy as np

from .cyclic_plasticity import (
    _ABAQUS_RATCHETING_URL,
    _abaqus_316_ratcheting_material,
    _abaqus_316_unsymmetric_stress_path,
)


_ABAQUS_RATCHETING_INPUT_URL = (
    "https://docs.software.vt.edu/abaqusv2025/English/"
    "SIMAINPRefResources/ratch_axi_unsymcyclic_2.inp"
)
_ABAQUS_RATCHETING_INPUT_SHA256 = (
    "a0d5ed2a64bbc9062887faa2d80f3a2c8f2c1737f37bddcf2da5fccda328be8b"
)


@dataclass(frozen=True)
class DigitizedRatchetingCurve:
    """Traceable points read from a published raster figure."""

    cycle: tuple[int, ...]
    maximum_axial_strain: tuple[float, ...]
    absolute_uncertainty: float

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "agentfem.digitized-ratcheting-curve.v1",
            "source": _ABAQUS_RATCHETING_URL,
            "figure": "Figure 4: maximum axial strain versus number of cycles",
            "curve": "experiment",
            "cycle": list(self.cycle),
            "maximum_axial_strain": list(self.maximum_axial_strain),
            "strain_unit": "1",
            "absolute_uncertainty": self.absolute_uncertainty,
            "uncertainty_basis": (
                "Raster-axis calibration, curve thickness and marker overlap; "
                "the source publishes no numerical table."
            ),
        }


@dataclass(frozen=True)
class ShoulderedRatchetingAssessment:
    """Structure-level comparison without overstating plotted evidence."""

    cycle_count: int
    backstress_count: int
    mesh_size: float
    cell_count: int
    maximum_absolute_curve_error: float
    root_mean_square_curve_error: float
    allowed_absolute_curve_error: float
    final_residual_norm: float
    residual_tolerance: float
    source_input_verified: bool
    accepted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "agentfem.shouldered-ratcheting-assessment.v1",
            "benchmark": "simulia_316_shouldered_axisymmetric_ratcheting",
            "source": _ABAQUS_RATCHETING_URL,
            "source_input": _ABAQUS_RATCHETING_INPUT_URL,
            "evidence_level": "external_structure_digitized_figure",
            "reference_curve": "digitized_experimental_figure_with_uncertainty",
            "cycle_count": self.cycle_count,
            "backstress_count": self.backstress_count,
            "mesh_size": self.mesh_size,
            "cell_count": self.cell_count,
            "maximum_absolute_curve_error": self.maximum_absolute_curve_error,
            "root_mean_square_curve_error": self.root_mean_square_curve_error,
            "allowed_absolute_curve_error": self.allowed_absolute_curve_error,
            "final_residual_norm": self.final_residual_norm,
            "residual_tolerance": self.residual_tolerance,
            "source_input_verified": self.source_input_verified,
            "accepted": self.accepted,
            "convergence_required_for_promotion": True,
            "promotion_eligible": False,
            "claim_boundary": (
                "Comparison to a digitized raster curve, not an exact vendor "
                "response table. The acceptance tolerance is AgentFEM-defined."
            ),
        }


@dataclass(frozen=True)
class ShoulderedRatchetingConvergence:
    """Independent spatial and path-integration convergence certificate."""

    mesh_sizes: tuple[float, ...]
    mesh_final_strains: tuple[float, ...]
    refinements: tuple[int, ...]
    refinement_final_strains: tuple[float, ...]
    spatial_relative_change: float
    refinement_relative_change: float
    relative_tolerance: float
    accepted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "agentfem.shouldered-ratcheting-convergence.v1",
            "benchmark": "simulia_316_shouldered_axisymmetric_ratcheting",
            "mesh_sizes": list(self.mesh_sizes),
            "mesh_final_strains": list(self.mesh_final_strains),
            "refinements": list(self.refinements),
            "refinement_final_strains": list(self.refinement_final_strains),
            "spatial_relative_change": self.spatial_relative_change,
            "refinement_relative_change": self.refinement_relative_change,
            "relative_tolerance": self.relative_tolerance,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class ShoulderedRatchetingAccuracy:
    """Spatial and accuracy-driven path-integration certificate."""

    cycle_count: int
    full_reference_curve_covered: bool
    mesh_sizes: tuple[float, ...]
    mesh_final_strains: tuple[float, ...]
    maximum_inelastic_increments: tuple[float, ...]
    adaptive_final_strains: tuple[float, ...]
    accepted_increment_counts: tuple[int, ...]
    rejected_attempt_counts: tuple[int, ...]
    spatial_relative_change: float
    path_relative_change: float
    relative_tolerance: float
    terminal_cases_accepted: bool
    accepted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "agentfem.shouldered-ratcheting-accuracy.v1",
            "benchmark": "simulia_316_shouldered_axisymmetric_ratcheting",
            "path_control": "maximum_equivalent_plastic_strain_increment",
            "mandatory_points": "every_published_load_peak_and_reversal",
            "cycle_count": self.cycle_count,
            "full_reference_curve_covered": self.full_reference_curve_covered,
            "mesh_sizes": list(self.mesh_sizes),
            "mesh_final_strains": list(self.mesh_final_strains),
            "maximum_inelastic_increments": list(self.maximum_inelastic_increments),
            "adaptive_final_strains": list(self.adaptive_final_strains),
            "accepted_increment_counts": list(self.accepted_increment_counts),
            "rejected_attempt_counts": list(self.rejected_attempt_counts),
            "spatial_relative_change": self.spatial_relative_change,
            "path_relative_change": self.path_relative_change,
            "relative_tolerance": self.relative_tolerance,
            "terminal_cases_accepted": self.terminal_cases_accepted,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class ShoulderedRatchetingFullReference:
    """One fine-path run covering every point on the public 100-cycle curve."""

    cycle_count: int
    mesh_size: float
    maximum_inelastic_increment: float
    accepted_increment_count: int
    rejected_attempt_count: int
    mandatory_coordinate_count: int
    all_mandatory_coordinates_reached: bool
    maximum_absolute_curve_error: float
    final_residual_norm: float
    accepted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "agentfem.shouldered-ratcheting-full-reference.v1",
            "benchmark": "simulia_316_shouldered_axisymmetric_ratcheting",
            "reference": "digitized_public_figure_with_uncertainty",
            "claim_boundary": (
                "Full published cycle range from a digitized raster curve; "
                "not an exact vendor response table or an automatic maturity promotion."
            ),
            "cycle_count": self.cycle_count,
            "full_reference_curve_covered": self.cycle_count >= 100,
            "mesh_size": self.mesh_size,
            "path_control": "maximum_equivalent_plastic_strain_increment",
            "maximum_inelastic_increment": self.maximum_inelastic_increment,
            "accepted_increment_count": self.accepted_increment_count,
            "rejected_attempt_count": self.rejected_attempt_count,
            "mandatory_coordinate_count": self.mandatory_coordinate_count,
            "all_mandatory_coordinates_reached": (
                self.all_mandatory_coordinates_reached
            ),
            "maximum_absolute_curve_error": self.maximum_absolute_curve_error,
            "final_residual_norm": self.final_residual_norm,
            "accepted": self.accepted,
            "promotion_eligible": False,
        }


def simulia_316_experimental_ratcheting_curve() -> DigitizedRatchetingCurve:
    """Return auditable experimental points digitized from public Figure 4.

    Strain values are dimensionless.  A conservative 0.025 percentage-point
    absolute uncertainty covers the raster resolution, line thickness and the
    overlap of experimental and simulation markers.
    """

    asset = resources.files("agentfem.knowledge.external_data").joinpath(
        "simulia_316_ratcheting_figure4.csv"
    )
    with asset.open("r", encoding="utf-8", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    uncertainty = tuple(float(row["absolute_uncertainty"]) for row in rows)
    if not rows or len(set(uncertainty)) != 1:
        raise ValueError("Digitized ratcheting data require one shared uncertainty.")
    return DigitizedRatchetingCurve(
        cycle=tuple(int(row["cycle"]) for row in rows),
        maximum_axial_strain=tuple(float(row["maximum_axial_strain"]) for row in rows),
        absolute_uncertainty=uncertainty[0],
    )


def verify_simulia_ratcheting_input(path: str | Path) -> bool:
    """Verify the exact public two-backstress Abaqus input deck by SHA-256."""

    selected = Path(path)
    digest = sha256(selected.read_bytes()).hexdigest()
    return digest == _ABAQUS_RATCHETING_INPUT_SHA256


def _shouldered_radius(z):
    """Outer-radius profile from public Figure 1, in millimetres."""

    selected = np.asarray(z, dtype=float)
    radius = np.empty_like(selected)
    radius[selected <= 8.0] = 9.0
    mask = (selected > 8.0) & (selected <= 16.0)
    radius[mask] = 5.0
    mask = (selected > 16.0) & (selected < 31.4)
    radius[mask] = 63.0 - np.sqrt(60.0**2 - (selected[mask] - 31.4) ** 2)
    mask = (selected >= 31.4) & (selected <= 43.4)
    radius[mask] = 3.0
    mask = (selected > 43.4) & (selected < 58.8)
    radius[mask] = 63.0 - np.sqrt(60.0**2 - (selected[mask] - 43.4) ** 2)
    mask = (selected >= 58.8) & (selected < 66.8)
    radius[mask] = 5.0
    radius[selected >= 66.8] = 9.0
    return radius


def simulia_316_shouldered_specimen_mesh(
    *,
    mesh_size: float = 1.25,
    comm=None,
):
    """Mesh the published meridian without redistributing a vendor input deck."""

    from mpi4py import MPI

    from .. import mesh

    if mesh_size <= 0.0:
        raise ValueError("mesh_size must be positive.")
    selected_comm = MPI.COMM_WORLD if comm is None else comm
    gmsh = mesh.require_gmsh()
    initialized_here = not gmsh.isInitialized()
    if initialized_here:
        gmsh.initialize()
    model_rank = 0
    try:
        if selected_comm.rank == model_rank:
            gmsh.clear()
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.model.add("simulia_316_shouldered_ratcheting")

            lower_z = np.linspace(16.0, 31.4, 13)
            upper_z = np.linspace(43.4, 58.8, 13)[1:]
            outer_z = np.concatenate(
                (
                    (0.0, 8.0, 8.0, 16.0),
                    lower_z[1:],
                    (43.4,),
                    upper_z,
                    (66.8, 66.8, 74.8),
                )
            )
            outer_r = np.concatenate(
                (
                    (9.0, 9.0, 5.0, 5.0),
                    _shouldered_radius(lower_z[1:]),
                    (3.0,),
                    _shouldered_radius(upper_z),
                    (5.0, 9.0, 9.0),
                )
            )
            polygon = [(0.0, 0.0), *zip(outer_r, outer_z), (0.0, 74.8)]
            points = [
                gmsh.model.geo.addPoint(float(r), float(z), 0.0, float(mesh_size))
                for r, z in polygon
            ]
            lines = [
                gmsh.model.geo.addLine(points[index], points[(index + 1) % len(points)])
                for index in range(len(points))
            ]
            loop = gmsh.model.geo.addCurveLoop(lines)
            surface = gmsh.model.geo.addPlaneSurface([loop])
            gmsh.model.geo.synchronize()
            gmsh.model.addPhysicalGroup(2, [surface], 1)
            gmsh.model.setPhysicalName(2, 1, "specimen")
            gmsh.model.mesh.generate(2)
        imported = mesh.import_gmsh_model(
            gmsh.model,
            selected_comm,
            model_rank=model_rank,
            gdim=2,
        )
    finally:
        if selected_comm.rank == model_rank and gmsh.isInitialized():
            gmsh.clear()
        if initialized_here and gmsh.isInitialized():
            gmsh.finalize()
    return imported.domain


def simulia_316_shouldered_ratcheting_benchmark(
    *,
    cycle_count: int = 5,
    refinement: int = 4,
    mesh_size: float = 3.5,
    backstress_count: int = 2,
    allowed_absolute_curve_error: float = 1.25e-3,
    residual_tolerance: float = 1.0e-7,
    maximum_inelastic_increment: float | None = None,
    source_input: str | Path | None = None,
    progress: bool = False,
):
    """Run the public shouldered-specimen ratcheting comparison.

    The finite-element model uses the published axisymmetric geometry, 316
    steel calibration, nominal top pressure, bottom axial restraint and center
    averaging definition. ``source_input`` is optional: when supplied, its hash
    must match the public SIMULIA two-backstress input deck.
    """

    from mpi4py import MPI

    from .. import amplitudes, constraints, fields, mesh, models, results, solvers
    from .. import steps, studies
    from ..constitutive import elasticity

    if int(cycle_count) != cycle_count or int(cycle_count) < 5:
        raise ValueError("cycle_count must be an integer of at least five.")
    if int(backstress_count) != backstress_count or int(backstress_count) not in {1, 2}:
        raise ValueError("backstress_count must be one or two.")
    if allowed_absolute_curve_error <= 0.0 or residual_tolerance <= 0.0:
        raise ValueError("Benchmark tolerances must be positive.")
    if maximum_inelastic_increment is not None and (
        not np.isfinite(maximum_inelastic_increment)
        or maximum_inelastic_increment <= 0.0
    ):
        raise ValueError("maximum_inelastic_increment must be finite and positive.")

    source_verified = False
    if source_input is not None:
        if int(backstress_count) != 2:
            raise ValueError(
                "The verified public source_input is the two-backstress deck."
            )
        source_verified = verify_simulia_ratcheting_input(source_input)
        if not source_verified:
            raise ValueError("source_input does not match the published SIMULIA deck.")

    domain = simulia_316_shouldered_specimen_mesh(
        mesh_size=mesh_size,
        comm=MPI.COMM_WORLD,
    )
    study = studies.nonlinear_static(
        physics="solid_mechanics",
        dimension=2,
        assumption="axisymmetric",
    )
    model = models.create(
        study=study,
        mesh=domain,
        name="simulia_316_shouldered_axisymmetric_ratcheting",
    )
    displacement = model.field(fields.displacement(domain, degree=1))
    bottom = mesh.face(domain, axis="y", value=0.0, name="bottom", tag=1)
    top = mesh.face(domain, axis="y", value=74.8, name="top", tag=2)
    axis = mesh.face(domain, axis="x", value=0.0, name="axis", tag=3)
    # Select the same 3 mm center band by cell centroid. ``locate_entities``
    # requires every vertex of a cell to satisfy a predicate and therefore
    # makes a physical averaging window disappear on a coarse mesh.
    from dolfinx import mesh as dolfinx_mesh

    topological_dimension = domain.topology.dim
    local_cells = np.arange(
        domain.topology.index_map(topological_dimension).size_local,
        dtype=np.int32,
    )
    cell_midpoints = dolfinx_mesh.compute_midpoints(
        domain,
        topological_dimension,
        local_cells,
    )
    center_cells = local_cells[
        (cell_midpoints[:, 1] >= 35.9) & (cell_midpoints[:, 1] <= 38.9)
    ]
    global_center_cell_count = domain.comm.allreduce(
        int(center_cells.size),
        op=MPI.SUM,
    )
    if global_center_cell_count == 0:
        raise ValueError(
            "The mesh does not resolve the 3 mm center averaging band; "
            "reduce mesh_size."
        )
    center_tags = mesh.mark_cells(domain, center_cells, 1)
    center = mesh.cell_region(
        domain,
        center_tags,
        tag=1,
        name="center_gauge",
    )
    model.fix(displacement, on=bottom, component=1, value=0.0)
    model.constraint(constraints.axisymmetric_axis(displacement, on=axis))
    material = model.material(
        _abaqus_316_ratcheting_material(backstress_count=int(backstress_count))
    )

    top_area = np.pi * 9.0**2
    model.surface_force((0.0, 11.0 * top_area), on=top)
    path = _abaqus_316_unsymmetric_stress_path(
        cycle_count=int(cycle_count),
        refinement=int(refinement),
    )
    normalized_coordinate = path.coordinate / path.coordinate[-1]
    amplitude = amplitudes.tabular(
        normalized_coordinate,
        path.stress[:, 0, 0] / 100.0,
        name="simulia_nominal_pressure_amplitude",
    )
    if maximum_inelastic_increment is None:
        increment_control = steps.at(*normalized_coordinate[1:])
    else:
        segment_sizes = np.diff(normalized_coordinate)
        smallest_segment = float(np.min(segment_sizes))
        largest_segment = float(np.max(segment_sizes))
        increment_control = steps.automatic(
            initial=smallest_segment,
            minimum=max(smallest_segment / 65_536.0, 1.0e-10),
            maximum=largest_segment,
            max_increments=max(10_000, len(normalized_coordinate) * 128),
            max_cutbacks=20,
            cutback_factor=0.5,
            growth_factor=1.25,
            fast_iterations=4,
            slow_iterations=10,
            maximum_inelastic_increment=float(maximum_inelastic_increment),
        )
    step = model.step(
        target=displacement,
        material=material,
        constraints=model.constraints,
        amplitude=amplitude,
        incrementation=increment_control,
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=35,
            line_search="backtracking",
        ),
        progress=progress,
        name="simulia_316_shouldered_axisymmetric_ratcheting",
    )

    maximum_indices = set(
        int(index) for index in np.flatnonzero(np.isclose(path.stress[:, 0, 0], 240.0))
    )
    strain = elasticity.strain(displacement.value, study=study)
    maximum_strain = []
    for index, target in enumerate(normalized_coordinate[1:], start=1):
        step.solve(until=float(target))
        if index in maximum_indices:
            maximum_strain.append(
                float(results.region_average(strain[2, 2], on=center, study=study))
            )
    maximum_strain = np.asarray(maximum_strain, dtype=float)
    cycle = np.arange(1, maximum_strain.size + 1, dtype=float)

    reference = simulia_316_experimental_ratcheting_curve()
    available = [
        index for index, value in enumerate(reference.cycle) if value <= cycle_count
    ]
    reference_cycle = np.asarray([reference.cycle[index] for index in available])
    reference_strain = np.asarray(
        [reference.maximum_axial_strain[index] for index in available]
    )
    computed_at_reference = maximum_strain[reference_cycle - 1]
    curve_error = computed_at_reference - reference_strain
    maximum_error = float(np.max(np.abs(curve_error)))
    rms_error = float(np.sqrt(np.mean(curve_error**2)))
    final_residual = float(step.accepted_increments[-1].residual_norm)
    cell_count = int(
        domain.comm.allreduce(
            domain.topology.index_map(domain.topology.dim).size_local,
            op=MPI.SUM,
        )
    )
    assessment = ShoulderedRatchetingAssessment(
        cycle_count=int(cycle_count),
        backstress_count=int(backstress_count),
        mesh_size=float(mesh_size),
        cell_count=cell_count,
        maximum_absolute_curve_error=maximum_error,
        root_mean_square_curve_error=rms_error,
        allowed_absolute_curve_error=float(allowed_absolute_curve_error),
        final_residual_norm=final_residual,
        residual_tolerance=float(residual_tolerance),
        source_input_verified=source_verified,
        accepted=bool(
            maximum_error <= allowed_absolute_curve_error
            and final_residual <= residual_tolerance
        ),
    )
    result = step.solve_result()
    result.add_histories(
        cycle,
        {"maximum_center_axial_strain": maximum_strain},
        abscissa_name="cycle",
        abscissa_unit=None,
    )
    result.add_histories(
        reference_cycle.astype(float),
        {
            "digitized_experimental_maximum_axial_strain": reference_strain,
            "computed_at_digitized_cycles": computed_at_reference,
            "computed_minus_digitized": curve_error,
        },
        abscissa_name="digitized_cycle",
        abscissa_unit=None,
    )
    result.metadata["external_benchmark"] = assessment.as_dict()
    result.metadata["external_benchmark"]["digitized_reference"] = reference.as_dict()
    result.metadata["external_benchmark"]["geometry"] = {
        "units": "mm",
        "length": 74.8,
        "grip_diameter": 18.0,
        "gauge_diameter": 6.0,
        "gauge_length": 12.0,
        "fillet_radius": 60.0,
    }
    result.metadata["external_benchmark"]["source_input_sha256"] = (
        _ABAQUS_RATCHETING_INPUT_SHA256
    )
    rejected_attempts = tuple(
        item for item in step.attempted_increments if not item.converged
    )
    result.metadata["external_benchmark"]["path_control"] = {
        "kind": (
            "fixed_nested_refinement"
            if maximum_inelastic_increment is None
            else "automatic_maximum_inelastic_increment"
        ),
        "mandatory_coordinates": normalized_coordinate.tolist(),
        "mandatory_coordinate_count": int(len(normalized_coordinate)),
        "maximum_inelastic_increment": maximum_inelastic_increment,
        "accepted_increment_count": len(step.accepted_increments),
        "rejected_attempt_count": len(rejected_attempts),
        "maximum_accepted_plastic_increment": max(
            (item.maximum_plastic_increment for item in step.accepted_increments),
            default=0.0,
        ),
        "all_mandatory_coordinates_reached": bool(
            abs(step.accepted_load_factor - 1.0) <= 1.0e-12
        ),
    }
    return assessment, result


def certify_simulia_316_shouldered_ratcheting_accuracy(
    *,
    cycle_count: int = 5,
    mesh_sizes: tuple[float, ...] = (3.5, 2.5),
    maximum_inelastic_increments: tuple[float, ...] = (
        4.0e-3,
        2.0e-3,
        1.0e-3,
    ),
    relative_tolerance: float = 0.01,
    progress: bool = False,
):
    """Certify mesh and adaptive constitutive-path accuracy independently.

    Every published load peak and reversal remains a mandatory global target.
    Between those targets the standard J2 procedure rejects and rolls back an
    otherwise converged increment when its largest equivalent-plastic-strain
    increment exceeds the selected limit. Successive limits must be strictly
    decreasing so a non-nested study cannot masquerade as refinement.
    """

    if len(mesh_sizes) < 2 or any(value <= 0.0 for value in mesh_sizes):
        raise ValueError("mesh_sizes must contain at least two positive values.")
    if len(maximum_inelastic_increments) < 2 or any(
        not np.isfinite(value) or value <= 0.0 for value in maximum_inelastic_increments
    ):
        raise ValueError(
            "maximum_inelastic_increments must contain at least two positive values."
        )
    if any(
        right >= left
        for left, right in zip(
            maximum_inelastic_increments,
            maximum_inelastic_increments[1:],
        )
    ):
        raise ValueError("maximum_inelastic_increments must be strictly decreasing.")
    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive.")

    selected_mesh_sizes = tuple(float(value) for value in mesh_sizes)
    selected_limits = tuple(float(value) for value in maximum_inelastic_increments)
    cache = {}

    def run(mesh_size, limit):
        key = (float(mesh_size), float(limit))
        if key not in cache:
            cache[key] = simulia_316_shouldered_ratcheting_benchmark(
                cycle_count=cycle_count,
                mesh_size=key[0],
                refinement=1,
                maximum_inelastic_increment=key[1],
                progress=progress,
            )
        return cache[key]

    tightest_limit = selected_limits[-1]
    coarsest_mesh = selected_mesh_sizes[0]
    mesh_final_strains = tuple(
        float(
            run(mesh_size, tightest_limit)[1]
            .histories["maximum_center_axial_strain"]
            .values[-1]
        )
        for mesh_size in selected_mesh_sizes
    )
    adaptive_final_strains = tuple(
        float(
            run(coarsest_mesh, limit)[1]
            .histories["maximum_center_axial_strain"]
            .values[-1]
        )
        for limit in selected_limits
    )
    path_controls = tuple(
        run(coarsest_mesh, limit)[1].metadata["external_benchmark"]["path_control"]
        for limit in selected_limits
    )
    accepted_counts = tuple(
        int(control["accepted_increment_count"]) for control in path_controls
    )
    rejected_counts = tuple(
        int(control["rejected_attempt_count"]) for control in path_controls
    )
    spatial_change = abs(mesh_final_strains[-1] - mesh_final_strains[-2]) / max(
        abs(mesh_final_strains[-1]),
        1.0e-15,
    )
    path_change = abs(adaptive_final_strains[-1] - adaptive_final_strains[-2]) / max(
        abs(adaptive_final_strains[-1]), 1.0e-15
    )
    terminal_keys = {
        *((mesh_size, tightest_limit) for mesh_size in selected_mesh_sizes[-2:]),
        *((coarsest_mesh, limit) for limit in selected_limits[-2:]),
    }
    terminal_cases_accepted = all(
        run(mesh_size, limit)[0].accepted
        and run(mesh_size, limit)[1].metadata["external_benchmark"]["path_control"][
            "all_mandatory_coordinates_reached"
        ]
        for mesh_size, limit in terminal_keys
    )
    certificate = ShoulderedRatchetingAccuracy(
        cycle_count=int(cycle_count),
        full_reference_curve_covered=bool(
            int(cycle_count) >= max(simulia_316_experimental_ratcheting_curve().cycle)
        ),
        mesh_sizes=selected_mesh_sizes,
        mesh_final_strains=mesh_final_strains,
        maximum_inelastic_increments=selected_limits,
        adaptive_final_strains=adaptive_final_strains,
        accepted_increment_counts=accepted_counts,
        rejected_attempt_counts=rejected_counts,
        spatial_relative_change=float(spatial_change),
        path_relative_change=float(path_change),
        relative_tolerance=float(relative_tolerance),
        terminal_cases_accepted=bool(terminal_cases_accepted),
        accepted=bool(
            terminal_cases_accepted
            and spatial_change <= relative_tolerance
            and path_change <= relative_tolerance
        ),
    )
    for assessment, result in cache.values():
        result.metadata["accuracy_certificate"] = certificate.as_dict()
        result.metadata["external_benchmark"]["convergence_required"] = True
        result.metadata["external_benchmark"]["convergence_accepted"] = (
            certificate.accepted
        )
        result.metadata["external_benchmark"]["promotion_eligible"] = bool(
            certificate.accepted and certificate.full_reference_curve_covered
        )
    return certificate, cache


def certify_simulia_316_shouldered_ratcheting_full_reference(
    *,
    cycle_count: int = 100,
    mesh_size: float = 2.5,
    maximum_inelastic_increment: float = 1.0e-3,
    progress: bool = False,
):
    """Run the fine-path case over the complete published cycle range.

    Spatial and path convergence remain the responsibility of the independent
    accuracy certificate.  This gate answers a different question: whether an
    exact candidate distribution reaches every physical reversal through the
    last digitized cycle and stays inside its declared response and equilibrium
    contracts.
    """

    reference = simulia_316_experimental_ratcheting_curve()
    published_cycle_count = int(max(reference.cycle))
    if int(cycle_count) != cycle_count or int(cycle_count) != published_cycle_count:
        raise ValueError(
            "Full-reference ratcheting evidence requires exactly every published "
            f"cycle ({published_cycle_count})."
        )
    assessment, result = simulia_316_shouldered_ratcheting_benchmark(
        cycle_count=int(cycle_count),
        mesh_size=float(mesh_size),
        refinement=1,
        maximum_inelastic_increment=float(maximum_inelastic_increment),
        progress=progress,
    )
    control = result.metadata["external_benchmark"]["path_control"]
    covered = (
        len(result.histories["maximum_center_axial_strain"].values)
        == published_cycle_count
    )
    accepted = bool(
        covered
        and assessment.accepted
        and control["all_mandatory_coordinates_reached"]
    )
    certificate = ShoulderedRatchetingFullReference(
        cycle_count=int(cycle_count),
        mesh_size=float(mesh_size),
        maximum_inelastic_increment=float(maximum_inelastic_increment),
        accepted_increment_count=int(control["accepted_increment_count"]),
        rejected_attempt_count=int(control["rejected_attempt_count"]),
        mandatory_coordinate_count=int(control["mandatory_coordinate_count"]),
        all_mandatory_coordinates_reached=bool(
            control["all_mandatory_coordinates_reached"]
        ),
        maximum_absolute_curve_error=assessment.maximum_absolute_curve_error,
        final_residual_norm=assessment.final_residual_norm,
        accepted=accepted,
    )
    result.metadata["full_reference_certificate"] = certificate.as_dict()
    result.metadata["external_benchmark"]["promotion_eligible"] = False
    return certificate, {(float(mesh_size), float(maximum_inelastic_increment)): (
        assessment,
        result,
    )}


def certify_simulia_316_shouldered_ratcheting_convergence(
    *,
    cycle_count: int = 5,
    mesh_sizes: tuple[float, ...] = (3.5, 2.5),
    refinements: tuple[int, ...] = (8, 16, 32),
    relative_tolerance: float = 0.01,
    progress: bool = False,
):
    """Certify spatial and material-path refinement independently.

    This is deliberately a release/nightly obligation rather than a per-commit
    test. Repeated ``(mesh_size, refinement)`` cases are evaluated only once.
    """

    if len(mesh_sizes) < 2 or any(value <= 0.0 for value in mesh_sizes):
        raise ValueError("mesh_sizes must contain at least two positive values.")
    if len(refinements) < 2 or any(
        int(value) != value or int(value) < 1 for value in refinements
    ):
        raise ValueError("refinements must contain at least two positive integers.")
    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive.")

    selected_mesh_sizes = tuple(float(value) for value in mesh_sizes)
    selected_refinements = tuple(int(value) for value in refinements)
    cache = {}

    def run(mesh_size, refinement):
        key = (float(mesh_size), int(refinement))
        if key not in cache:
            cache[key] = simulia_316_shouldered_ratcheting_benchmark(
                cycle_count=cycle_count,
                mesh_size=key[0],
                refinement=key[1],
                progress=progress,
            )
        return cache[key]

    finest_refinement = selected_refinements[-1]
    coarsest_mesh = selected_mesh_sizes[0]
    mesh_final_strains = tuple(
        float(
            run(mesh_size, finest_refinement)[1]
            .histories["maximum_center_axial_strain"]
            .values[-1]
        )
        for mesh_size in selected_mesh_sizes
    )
    refinement_final_strains = tuple(
        float(
            run(coarsest_mesh, refinement)[1]
            .histories["maximum_center_axial_strain"]
            .values[-1]
        )
        for refinement in selected_refinements
    )
    spatial_change = abs(mesh_final_strains[-1] - mesh_final_strains[-2]) / max(
        abs(mesh_final_strains[-1]),
        1.0e-15,
    )
    refinement_change = abs(
        refinement_final_strains[-1] - refinement_final_strains[-2]
    ) / max(abs(refinement_final_strains[-1]), 1.0e-15)
    fine_assessment = run(selected_mesh_sizes[-1], finest_refinement)[0]
    certificate = ShoulderedRatchetingConvergence(
        mesh_sizes=selected_mesh_sizes,
        mesh_final_strains=mesh_final_strains,
        refinements=selected_refinements,
        refinement_final_strains=refinement_final_strains,
        spatial_relative_change=float(spatial_change),
        refinement_relative_change=float(refinement_change),
        relative_tolerance=float(relative_tolerance),
        accepted=bool(
            fine_assessment.accepted
            and spatial_change <= relative_tolerance
            and refinement_change <= relative_tolerance
        ),
    )
    for assessment, result in cache.values():
        result.metadata["convergence_certificate"] = certificate.as_dict()
        result.metadata["external_benchmark"]["convergence_required"] = True
        result.metadata["external_benchmark"]["convergence_accepted"] = (
            certificate.accepted
        )
        result.metadata["external_benchmark"]["promotion_eligible"] = bool(
            certificate.accepted
            and int(cycle_count)
            >= max(simulia_316_experimental_ratcheting_curve().cycle)
        )
    return certificate, cache


__all__ = [
    "DigitizedRatchetingCurve",
    "ShoulderedRatchetingAssessment",
    "ShoulderedRatchetingAccuracy",
    "ShoulderedRatchetingConvergence",
    "certify_simulia_316_shouldered_ratcheting_accuracy",
    "certify_simulia_316_shouldered_ratcheting_convergence",
    "simulia_316_experimental_ratcheting_curve",
    "simulia_316_shouldered_ratcheting_benchmark",
    "simulia_316_shouldered_specimen_mesh",
    "verify_simulia_ratcheting_input",
]
