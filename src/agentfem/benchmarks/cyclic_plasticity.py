# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Public cyclic-plasticity material benchmarks.

The OFHC copper paths and comparison values come from the SIMULIA example
``cyclictests``.  AgentFEM reconstructs the calibrated combined-hardening law
and solves the same mixed material control, rather than storing AgentFEM's own
response as the reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import constitutive


_ABAQUS_CYCLIC_URL = (
    "https://docs.software.vt.edu/abaqusv2025/English/"
    "SIMACAEBMKRefMap/simabmk-c-cyclictests.htm"
)
_ABAQUS_RATCHETING_URL = (
    "https://docs.software.vt.edu/abaqusv2025/English/"
    "SIMACAEEXARefMap/simaexa-c-ratchetting.htm"
)


@dataclass(frozen=True)
class CyclicPlasticityBenchmark:
    """Compact acceptance evidence against two published cyclic observables."""

    symmetric_final_peeq: float
    symmetric_reference_peeq: float
    tension_torsion_maximum_normal_stress: float
    tension_torsion_reference_stress: float
    fourth_to_final_cycle_change: float
    relative_tolerance: float
    saturation_tolerance: float
    accepted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "agentfem.cyclic-plasticity-benchmark.v1",
            "benchmark": "abaqus_ofhc_copper_cyclic",
            "source": _ABAQUS_CYCLIC_URL,
            "symmetric_final_peeq": self.symmetric_final_peeq,
            "symmetric_reference_peeq": self.symmetric_reference_peeq,
            "symmetric_relative_error": self.symmetric_relative_error,
            "tension_torsion_maximum_normal_stress": (
                self.tension_torsion_maximum_normal_stress
            ),
            "tension_torsion_reference_stress": (self.tension_torsion_reference_stress),
            "tension_torsion_relative_error": (self.tension_torsion_relative_error),
            "fourth_to_final_cycle_change": self.fourth_to_final_cycle_change,
            "relative_tolerance": self.relative_tolerance,
            "saturation_tolerance": self.saturation_tolerance,
            "accepted": self.accepted,
        }

    @property
    def symmetric_relative_error(self) -> float:
        return abs(self.symmetric_final_peeq - self.symmetric_reference_peeq) / abs(
            self.symmetric_reference_peeq
        )

    @property
    def tension_torsion_relative_error(self) -> float:
        return abs(
            self.tension_torsion_maximum_normal_stress
            - self.tension_torsion_reference_stress
        ) / abs(self.tension_torsion_reference_stress)


@dataclass(frozen=True)
class RatchetingPathComparison:
    """Published-path conformance without inventing a numerical Golden."""

    cycle_count: int
    one_backstress_accumulation: float
    two_backstress_accumulation: float
    two_to_one_accumulation_ratio: float
    one_backstress_last_cycle_increment: float
    two_backstress_last_cycle_increment: float
    maximum_stress_control_residual: float
    residual_tolerance: float
    accepted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "agentfem.ratcheting-path-comparison.v1",
            "benchmark": "abaqus_316_steel_unsymmetric_stress_path",
            "source": _ABAQUS_RATCHETING_URL,
            "evidence_level": "published_path_and_qualitative_trend",
            "reference_curve": "figure_only_no_numerical_table",
            "cycle_count": self.cycle_count,
            "one_backstress_accumulation": self.one_backstress_accumulation,
            "two_backstress_accumulation": self.two_backstress_accumulation,
            "two_to_one_accumulation_ratio": self.two_to_one_accumulation_ratio,
            "one_backstress_last_cycle_increment": (
                self.one_backstress_last_cycle_increment
            ),
            "two_backstress_last_cycle_increment": (
                self.two_backstress_last_cycle_increment
            ),
            "maximum_stress_control_residual": (self.maximum_stress_control_residual),
            "residual_tolerance": self.residual_tolerance,
            "accepted": self.accepted,
            "claim_boundary": (
                "Path/control/trend conformance only; not a numerical "
                "reproduction of the published structure curve."
            ),
        }


@dataclass(frozen=True)
class AxisymmetricRatchetingCrosscheck:
    """Global axisymmetric equilibrium checked against the same local path."""

    cycle_count: int
    maximum_relative_peak_strain_error: float
    final_relative_peak_strain_error: float
    final_residual_norm: float
    relative_tolerance: float
    residual_tolerance: float
    accepted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "agentfem.axisymmetric-ratcheting-crosscheck.v1",
            "benchmark": "axisymmetric_uniform_tube_chaboche_ratcheting",
            "evidence_level": "global_fem_material_point_crosscheck",
            "external_specimen_golden": False,
            "cycle_count": self.cycle_count,
            "maximum_relative_peak_strain_error": (
                self.maximum_relative_peak_strain_error
            ),
            "final_relative_peak_strain_error": (self.final_relative_peak_strain_error),
            "final_residual_norm": self.final_residual_norm,
            "relative_tolerance": self.relative_tolerance,
            "residual_tolerance": self.residual_tolerance,
            "accepted": self.accepted,
        }


def _abaqus_ofhc_copper():
    equivalent_plastic_strain = (
        0.0,
        0.0068,
        0.0340,
        0.0612,
        0.0884,
        0.1156,
        0.1428,
        0.1700,
        0.1972,
        0.2244,
        0.2516,
    )
    yield_stress = (
        13.0,
        13.0,
        29.37,
        50.13,
        63.91,
        73.01,
        80.76,
        86.25,
        90.498,
        93.95,
        96.155,
    )
    isotropic = constitutive.TabulatedIsotropicHardening(
        equivalent_plastic_strain=equivalent_plastic_strain,
        yield_stress=yield_stress,
        extrapolation="constant",
    )
    return constitutive.chaboche(
        young=104_000.0,
        poisson=0.3,
        yield_stress=13.0,
        backstresses=((33_550.0, 701.3),),
        isotropic_hardening=isotropic,
        name="Abaqus OFHC copper combined hardening",
    )


def _uniaxial_stress_free_transverse_path(*, substeps_per_half_cycle: int):
    anchor_coordinate = np.asarray(
        [0.0, 5.0, *np.arange(15.0, 186.0, 10.0)],
        dtype=float,
    )
    anchor_axial_strain = np.asarray(
        [0.0, 0.0075, *([-0.0075, 0.0075] * 9)],
        dtype=float,
    )
    strain = np.zeros((anchor_coordinate.size, 3, 3))
    strain[:, 0, 0] = anchor_axial_strain
    stress = np.zeros_like(strain)
    strain_control = np.zeros((3, 3), dtype=bool)
    strain_control[0, 0] = True
    return constitutive.material_mixed_path(
        anchor_coordinate,
        strain=strain,
        stress=stress,
        strain_control=strain_control,
        name="abaqus_ofhc_symmetric_strain_cycle",
        coordinate_name="load_time",
    ).refine(substeps_per_half_cycle)


def _tension_torsion_path(*, points_per_cycle: int, cycle_count: int = 9):
    ramp = np.linspace(0.0, 1.0, 11)
    phase = np.linspace(
        0.0,
        2.0 * cycle_count,
        points_per_cycle * cycle_count + 1,
    )[1:]
    coordinate = np.concatenate((ramp, 1.0 + phase))
    strain = np.zeros((coordinate.size, 3, 3))
    stress = np.zeros_like(strain)
    strain[: ramp.size, 0, 0] = 0.01 * ramp
    strain[ramp.size :, 0, 0] = 0.01 * np.cos(np.pi * phase)
    engineering_shear = np.sqrt(3.0) * 0.01 * np.sin(np.pi * phase)
    strain[ramp.size :, 0, 1] = 0.5 * engineering_shear
    strain[ramp.size :, 1, 0] = 0.5 * engineering_shear
    strain_control = np.zeros((3, 3), dtype=bool)
    strain_control[0, 0] = True
    strain_control[0, 1] = strain_control[1, 0] = True
    return constitutive.material_mixed_path(
        coordinate,
        strain=strain,
        stress=stress,
        strain_control=strain_control,
        name="abaqus_ofhc_tension_torsion_cycle",
        coordinate_name="load_time",
    )


def abaqus_ofhc_copper_cyclic_benchmark(
    *,
    substeps_per_half_cycle: int = 10,
    points_per_cycle: int = 20,
    relative_tolerance: float = 0.01,
    saturation_tolerance: float = 0.01,
):
    """Run the published symmetric and nonproportional OFHC copper tests.

    Returns the external assessment plus the two common ``SimulationResult``
    objects.  The 1% default gates are AgentFEM acceptance contracts, not
    tolerances claimed by SIMULIA.
    """

    if (
        int(substeps_per_half_cycle) != substeps_per_half_cycle
        or int(substeps_per_half_cycle) < 1
    ):
        raise ValueError("substeps_per_half_cycle must be a positive integer.")
    if int(points_per_cycle) != points_per_cycle or int(points_per_cycle) < 8:
        raise ValueError("points_per_cycle must be an integer of at least 8.")
    if relative_tolerance <= 0.0 or saturation_tolerance <= 0.0:
        raise ValueError("Benchmark tolerances must be positive.")

    material = _abaqus_ofhc_copper()
    symmetric = material.history(
        _uniaxial_stress_free_transverse_path(
            substeps_per_half_cycle=int(substeps_per_half_cycle)
        ),
        name="abaqus_ofhc_symmetric_cyclic",
    ).solve()
    tension_torsion = material.history(
        _tension_torsion_path(points_per_cycle=int(points_per_cycle)),
        name="abaqus_ofhc_tension_torsion",
    ).solve()

    ramp_points = 11
    points_per_cycle = int(points_per_cycle)
    axial = tension_torsion.stress[:, 0, 0]
    cycle_four = float(
        np.max(
            axial[
                ramp_points + 3 * points_per_cycle : ramp_points + 4 * points_per_cycle
            ]
        )
    )
    final_cycle = float(np.max(axial[-points_per_cycle:]))
    saturation_change = abs(final_cycle - cycle_four) / abs(final_cycle)
    symmetric_reference = 0.2367
    normal_stress_reference = 143.1
    symmetric_value = float(symmetric.equivalent_plastic_strain[-1])
    assessment = CyclicPlasticityBenchmark(
        symmetric_final_peeq=symmetric_value,
        symmetric_reference_peeq=symmetric_reference,
        tension_torsion_maximum_normal_stress=final_cycle,
        tension_torsion_reference_stress=normal_stress_reference,
        fourth_to_final_cycle_change=saturation_change,
        relative_tolerance=float(relative_tolerance),
        saturation_tolerance=float(saturation_tolerance),
        accepted=bool(
            abs(symmetric_value - symmetric_reference) / symmetric_reference
            <= relative_tolerance
            and abs(final_cycle - normal_stress_reference) / normal_stress_reference
            <= relative_tolerance
            and saturation_change <= saturation_tolerance
        ),
    )
    results = {
        "symmetric": symmetric.to_result(),
        "tension_torsion": tension_torsion.to_result(),
    }
    for result in results.values():
        result.metadata["external_benchmark"] = assessment.as_dict()
        result.metadata["external_benchmark"]["acceptance_tolerances_are"] = (
            "agentfem_defined"
        )
    return assessment, results


def _abaqus_316_ratcheting_material(*, backstress_count: int):
    if backstress_count == 1:
        backstresses = ((218_500.0, 1956.6),)
    elif backstress_count == 2:
        backstresses = ((2_067.0, 44.7), (246_200.0, 2551.4))
    else:
        raise ValueError("The published comparison declares one or two backstresses.")
    return constitutive.chaboche(
        young=192_000.0,
        poisson=0.3,
        yield_stress=120.0,
        backstresses=backstresses,
        isotropic_saturation=120.0,
        isotropic_rate=13.2,
        name=f"Abaqus 316 steel {backstress_count}-backstress calibration",
    )


def _abaqus_316_unsymmetric_stress_path(*, cycle_count: int, refinement: int):
    if int(cycle_count) != cycle_count or int(cycle_count) < 2:
        raise ValueError("cycle_count must be an integer of at least two.")
    if int(refinement) != refinement or int(refinement) < 1:
        raise ValueError("refinement must be a positive integer.")
    coordinate = [0.0, 0.7143, 1.7143]
    axial_stress = [0.0, 100.0, 240.0]
    for cycle in range(int(cycle_count)):
        minimum_coordinate = 3.7143 + 4.0 * cycle
        coordinate.extend((minimum_coordinate, minimum_coordinate + 1.0))
        axial_stress.extend((-40.0, 100.0))
        if cycle < int(cycle_count) - 1:
            coordinate.append(minimum_coordinate + 2.0)
            axial_stress.append(240.0)
    stress = np.zeros((len(coordinate), 3, 3))
    strain = np.zeros_like(stress)
    stress[:, 0, 0] = axial_stress
    strain_control = np.zeros((3, 3), dtype=bool)
    path = constitutive.material_mixed_path(
        coordinate,
        strain=strain,
        stress=stress,
        strain_control=strain_control,
        name="abaqus_316_unsymmetric_stress_cycle",
        coordinate_name="load_time",
    )
    return path.refine(int(refinement))


def abaqus_316_steel_ratcheting_path_comparison(
    *,
    cycle_count: int = 50,
    refinement: int = 2,
    residual_tolerance: float = 1.0e-7,
):
    """Exercise the published 316-steel asymmetric stress path.

    SIMULIA publishes the specimen result as a graph rather than a numerical
    table.  This function therefore verifies the exact material parameters,
    local stress path and stated one-versus-two-backstress trend, while keeping
    the numerical structure-level comparison as a separate promotion gate.
    """

    if residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be positive.")
    path = _abaqus_316_unsymmetric_stress_path(
        cycle_count=cycle_count,
        refinement=refinement,
    )
    responses = {
        count: _abaqus_316_ratcheting_material(backstress_count=count)
        .history(path, name=f"abaqus_316_ratcheting_{count}_backstress")
        .solve()
        for count in (1, 2)
    }
    maximum_indices = np.flatnonzero(np.isclose(path.stress[:, 0, 0], 240.0))
    if maximum_indices.size < 2:
        raise RuntimeError("Ratcheting path did not preserve repeated maxima.")
    maxima = {
        count: response.strain[maximum_indices, 0, 0]
        for count, response in responses.items()
    }
    accumulation = {
        count: float(values[-1] - values[0]) for count, values in maxima.items()
    }
    last_cycle = {
        count: float(values[-1] - values[-2]) for count, values in maxima.items()
    }
    maximum_residual = max(
        float(
            np.max(
                np.abs(response.stress - path.stress),
                initial=0.0,
            )
        )
        for response in responses.values()
    )
    ratio = accumulation[2] / accumulation[1]
    accepted = bool(
        maximum_residual <= residual_tolerance
        and accumulation[1] > 0.0
        and accumulation[2] > 0.0
        and accumulation[2] < accumulation[1]
        and last_cycle[1] > 0.0
        and last_cycle[2] > 0.0
    )
    assessment = RatchetingPathComparison(
        cycle_count=int(cycle_count),
        one_backstress_accumulation=accumulation[1],
        two_backstress_accumulation=accumulation[2],
        two_to_one_accumulation_ratio=float(ratio),
        one_backstress_last_cycle_increment=last_cycle[1],
        two_backstress_last_cycle_increment=last_cycle[2],
        maximum_stress_control_residual=maximum_residual,
        residual_tolerance=float(residual_tolerance),
        accepted=accepted,
    )
    results = {count: response.to_result() for count, response in responses.items()}
    for result in results.values():
        result.metadata["external_benchmark"] = assessment.as_dict()
    return assessment, results


def axisymmetric_chaboche_ratcheting_crosscheck(
    *,
    cycle_count: int = 3,
    refinement: int = 2,
    radial_cells: int = 1,
    axial_cells: int = 2,
    relative_tolerance: float = 2.0e-3,
    residual_tolerance: float = 1.0e-7,
):
    """Cross-check a global axisymmetric tube against one material point.

    The annular tube carries uniform axial traction, so its gauge strain should
    reproduce the independent stress-controlled constitutive history. This is
    a structural-path regression for lowering, load measure, Newton state and
    result ownership. It is not the shouldered specimen Golden published only
    as a graph in the SIMULIA ratcheting example.
    """

    from mpi4py import MPI

    from .. import fields, mesh, models, results, solvers, steps, studies

    if int(radial_cells) != radial_cells or int(radial_cells) < 1:
        raise ValueError("radial_cells must be a positive integer.")
    if int(axial_cells) != axial_cells or int(axial_cells) < 1:
        raise ValueError("axial_cells must be a positive integer.")
    if relative_tolerance <= 0.0 or residual_tolerance <= 0.0:
        raise ValueError("Cross-check tolerances must be positive.")

    path = _abaqus_316_unsymmetric_stress_path(
        cycle_count=cycle_count,
        refinement=refinement,
    )
    material = _abaqus_316_ratcheting_material(backstress_count=2)
    local = material.history(path, name="axisymmetric_ratcheting_reference").solve()
    maximum_indices = np.flatnonzero(np.isclose(path.stress[:, 0, 0], 240.0))

    inner_radius, outer_radius, height = 1.0, 2.0, 1.0
    domain = mesh.rectangle(
        (inner_radius, 0.0),
        (outer_radius, height),
        (int(radial_cells), int(axial_cells)),
        comm=MPI.COMM_WORLD,
        cell_type="quadrilateral",
    )
    study = studies.nonlinear_static(
        physics="solid_mechanics",
        dimension=2,
        assumption="axisymmetric",
    )
    model = models.create(
        study=study,
        mesh=domain,
        name="axisymmetric_uniform_tube_chaboche_ratcheting",
    )
    displacement = model.field(fields.displacement(domain, degree=1))
    bottom = mesh.face(domain, axis="y", value=0.0, name="bottom", tag=1)
    top = mesh.face(domain, axis="y", value=height, name="top", tag=2)
    model.fix(displacement, on=bottom, component=1, value=0.0)
    selected_material = model.material(material)

    area = np.pi * (outer_radius**2 - inner_radius**2)
    model.surface_force((0.0, 240.0 * area), on=top)
    normalized_coordinate = path.coordinate / path.coordinate[-1]
    load_amplitude = path.stress[:, 0, 0] / 240.0
    from .. import amplitudes

    amplitude = amplitudes.tabular(
        normalized_coordinate,
        load_amplitude,
        name="abaqus_316_unsymmetric_stress_amplitude",
    )
    step = model.step(
        target=displacement,
        material=selected_material,
        constraints=model.constraints,
        amplitude=amplitude,
        incrementation=steps.at(*normalized_coordinate[1:]),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-10,
            maximum_iterations=30,
            line_search="backtracking",
        ),
        progress=False,
        name="axisymmetric_uniform_tube_chaboche_ratcheting",
    )

    peak_set = set(int(index) for index in maximum_indices)
    global_peak_strain = []
    for index, target in enumerate(normalized_coordinate[1:], start=1):
        step.solve(until=float(target))
        if index in peak_set:
            average_top_displacement = results.region_average(
                displacement.value[1],
                on=top,
                study=study,
            )
            global_peak_strain.append(float(average_top_displacement) / height)

    local_peak_strain = local.strain[maximum_indices, 0, 0]
    global_peak_strain = np.asarray(global_peak_strain, dtype=float)
    if global_peak_strain.shape != local_peak_strain.shape:
        raise RuntimeError("Global and local ratcheting peak counts differ.")
    scale = np.maximum(np.abs(local_peak_strain), 1.0e-12)
    relative_errors = np.abs(global_peak_strain - local_peak_strain) / scale
    final_residual = float(step.accepted_increments[-1].residual_norm)
    assessment = AxisymmetricRatchetingCrosscheck(
        cycle_count=int(cycle_count),
        maximum_relative_peak_strain_error=float(np.max(relative_errors)),
        final_relative_peak_strain_error=float(relative_errors[-1]),
        final_residual_norm=final_residual,
        relative_tolerance=float(relative_tolerance),
        residual_tolerance=float(residual_tolerance),
        accepted=bool(
            np.max(relative_errors) <= relative_tolerance
            and final_residual <= residual_tolerance
        ),
    )
    result = step.solve_result()
    result.add_histories(
        np.arange(global_peak_strain.size, dtype=float) + 1.0,
        {
            "global_peak_axial_strain": global_peak_strain,
            "material_point_peak_axial_strain": local_peak_strain,
            "relative_peak_axial_strain_error": relative_errors,
        },
        abscissa_name="cycle",
        abscissa_unit=None,
    )
    result.metadata["structural_crosscheck"] = assessment.as_dict()
    return assessment, result


__all__ = [
    "CyclicPlasticityBenchmark",
    "AxisymmetricRatchetingCrosscheck",
    "RatchetingPathComparison",
    "abaqus_316_steel_ratcheting_path_comparison",
    "abaqus_ofhc_copper_cyclic_benchmark",
    "axisymmetric_chaboche_ratcheting_crosscheck",
]
