"""Executable evidence for the staged dynamic-fracture verification route."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite

import numpy as np
from dolfinx import fem
from mpi4py import MPI
import ufl

from .. import amplitudes
from .. import constitutive
from .. import constraints
from .. import fields
from .. import fracture
from .. import interfaces
from .. import mesh
from .. import models
from .. import problems
from .. import results
from .. import studies


@dataclass(frozen=True)
class WaveArrivalBenchmark:
    """Measured and acoustic-tensor wave speed in reference coordinates."""

    prestrain: float
    predicted_reference_speed: float
    measured_reference_speed: float
    relative_error: float
    first_arrival_time: float
    second_arrival_time: float
    time_increment: float
    cells: int
    maximum_relative_energy_error: float

    def summary(self) -> dict[str, object]:
        return {
            "kind": "finite_strain_wave_arrival_v1",
            "prestrain": self.prestrain,
            "predicted_reference_speed": self.predicted_reference_speed,
            "measured_reference_speed": self.measured_reference_speed,
            "relative_error": self.relative_error,
            "first_arrival_time": self.first_arrival_time,
            "second_arrival_time": self.second_arrival_time,
            "time_increment": self.time_increment,
            "cells": self.cells,
            "maximum_relative_energy_error": self.maximum_relative_energy_error,
            "coordinate_measure": "reference",
        }


@dataclass(frozen=True)
class ThinThreeDimensionalCrossCheck:
    """Plane-stress condensation versus an affine thin-3D FEM patch."""

    axial_stretch: float
    lateral_stretch: float
    thickness_stretch: float
    reference_thickness: float
    cells: tuple[int, int, int]
    jacobian: float
    plane_stress_first_piola: np.ndarray
    thin_3d_first_piola: np.ndarray
    plane_stress_energy_density: float
    thin_3d_energy_density: float
    maximum_relative_stress_error: float
    relative_energy_error: float
    traction_free_stress_ratio: float
    accepted: bool

    def summary(self) -> dict[str, object]:
        return {
            "kind": "plane_stress_thin_3d_neo_hookean_crosscheck",
            "axial_stretch": self.axial_stretch,
            "lateral_stretch": self.lateral_stretch,
            "thickness_stretch": self.thickness_stretch,
            "reference_thickness": self.reference_thickness,
            "cells": self.cells,
            "jacobian": self.jacobian,
            "plane_stress_first_piola": self.plane_stress_first_piola.tolist(),
            "thin_3d_first_piola": self.thin_3d_first_piola.tolist(),
            "plane_stress_energy_density": self.plane_stress_energy_density,
            "thin_3d_energy_density": self.thin_3d_energy_density,
            "maximum_relative_stress_error": self.maximum_relative_stress_error,
            "relative_energy_error": self.relative_energy_error,
            "traction_free_stress_ratio": self.traction_free_stress_ratio,
            "accepted": self.accepted,
            "claim_scope": "homogeneous_affine_geometry_assumption_crosscheck",
        }


@dataclass(frozen=True)
class CohesiveEnergyBenchmark:
    """Energy closure for one uniformly separating cohesive interface."""

    time_increment: float
    increments: int
    prescribed_opening: float
    maximum_damage: float
    fracture_dissipation: float
    declared_fracture_energy: float
    final_relative_energy_error: float
    maximum_relative_energy_error: float

    def summary(self) -> dict[str, object]:
        return {
            "kind": "cohesive_energy_balance_v2",
            "time_increment": self.time_increment,
            "increments": self.increments,
            "prescribed_opening": self.prescribed_opening,
            "maximum_damage": self.maximum_damage,
            "fracture_dissipation": self.fracture_dissipation,
            "declared_fracture_energy": self.declared_fracture_energy,
            "final_relative_energy_error": self.final_relative_energy_error,
            "maximum_relative_energy_error": self.maximum_relative_energy_error,
        }


@dataclass(frozen=True)
class ClassicalCrackBenchmark:
    """Fixed-path Mode-I crack propagation evidence for the V3 guardrail."""

    cells: int
    time_increment: float
    initial_crack_length: float
    final_crack_length: float
    propagated_length: float
    maximum_fitted_speed: float
    rayleigh_wave_speed: float
    speed_ratio: float
    final_relative_energy_error: float
    damping: float
    numerical_damping_dissipation: float
    maximum_simultaneous_failed_fraction: float

    def summary(self) -> dict[str, object]:
        return {
            "kind": "classical_sub_rayleigh_cohesive_crack_v3",
            "cells": self.cells,
            "time_increment": self.time_increment,
            "initial_crack_length": self.initial_crack_length,
            "final_crack_length": self.final_crack_length,
            "propagated_length": self.propagated_length,
            "maximum_fitted_speed": self.maximum_fitted_speed,
            "rayleigh_wave_speed": self.rayleigh_wave_speed,
            "speed_ratio": self.speed_ratio,
            "final_relative_energy_error": self.final_relative_energy_error,
            "damping": self.damping,
            "numerical_damping_dissipation": self.numerical_damping_dissipation,
            "maximum_simultaneous_failed_fraction": (
                self.maximum_simultaneous_failed_fraction
            ),
        }


@dataclass(frozen=True)
class WeakInterfaceTransitionBenchmark:
    """One prestressed thin-sheet case in the JMPS V4 mechanism ladder."""

    label: str
    cells: int
    transverse_cells: int
    axial_strain: float
    strength: float
    fracture_energy: float
    cohesive_length: float
    normalized_cohesive_length: float
    time_increment: float
    propagated_length: float
    maximum_fitted_speed: float
    rayleigh_wave_speed: float
    shear_wave_speed: float
    pressure_wave_speed: float
    rayleigh_speed_ratio: float
    shear_speed_ratio: float
    pressure_speed_ratio: float
    failed_ligament_fraction: float
    maximum_simultaneous_failed_fraction: float
    rapid_failed_fraction: float
    crack_speed_fit_window: int
    crack_speed_fit_length: float
    spall_time_window: float
    maximum_ligament_traction_ratio: float
    regime: str
    final_relative_energy_error: float
    preload_energy_jump: float
    release_acceleration_norm: float
    preload_ligament_traction_ratio: float
    preload_interface_opening: float
    impact_displacement: float
    impact_rise_time: float
    performance: dict[str, object]
    trace: fracture.CohesiveInterfaceTrace | None = None

    def summary(self) -> dict[str, object]:
        return {
            "kind": "jmps_weak_interface_transition_v4_case",
            "label": self.label,
            "cells": self.cells,
            "transverse_cells": self.transverse_cells,
            "kinematics": "finite_strain_plane_stress",
            "loading": (
                "homogeneous_prestrain_then_remote_impact"
                if self.impact_displacement
                else "homogeneous_prestrain_then_precrack_release"
            ),
            "axial_strain": self.axial_strain,
            "strength": self.strength,
            "fracture_energy": self.fracture_energy,
            "cohesive_length": self.cohesive_length,
            "normalized_cohesive_length": self.normalized_cohesive_length,
            "time_increment": self.time_increment,
            "propagated_length": self.propagated_length,
            "maximum_fitted_speed": self.maximum_fitted_speed,
            "rayleigh_wave_speed": self.rayleigh_wave_speed,
            "rayleigh_speed_convention": "unstretched_isotropic_reference",
            "shear_wave_speed": self.shear_wave_speed,
            "shear_speed_convention": "prestrained_acoustic_tensor_reference",
            "pressure_wave_speed": self.pressure_wave_speed,
            "pressure_speed_convention": "prestrained_acoustic_tensor_reference",
            "rayleigh_speed_ratio": self.rayleigh_speed_ratio,
            "shear_speed_ratio": self.shear_speed_ratio,
            "pressure_speed_ratio": self.pressure_speed_ratio,
            "failed_ligament_fraction": self.failed_ligament_fraction,
            "maximum_simultaneous_failed_fraction": (
                self.maximum_simultaneous_failed_fraction
            ),
            "rapid_failed_fraction": self.rapid_failed_fraction,
            "crack_speed_fit_window": self.crack_speed_fit_window,
            "crack_speed_fit_length": self.crack_speed_fit_length,
            "spall_time_window": self.spall_time_window,
            "maximum_ligament_traction_ratio": self.maximum_ligament_traction_ratio,
            "regime": self.regime,
            "final_relative_energy_error": self.final_relative_energy_error,
            "preload_energy_jump": self.preload_energy_jump,
            "release_acceleration_norm": self.release_acceleration_norm,
            "preload_ligament_traction_ratio": self.preload_ligament_traction_ratio,
            "preload_interface_opening": self.preload_interface_opening,
            "impact_displacement": self.impact_displacement,
            "impact_rise_time": self.impact_rise_time,
            "performance": self.performance,
            "trace": None if self.trace is None else self.trace.summary(),
            "maturity": "experimental_v4_mechanism_case",
        }


@dataclass(frozen=True)
class WeakInterfaceTransitionSuite:
    """Auditable crack-like to supershear to spall-like V4 mechanism gate."""

    crack_like: WeakInterfaceTransitionBenchmark
    supershear: WeakInterfaceTransitionBenchmark
    spall_like: WeakInterfaceTransitionBenchmark
    accepted: bool
    acceptance_failures: tuple[str, ...]

    def summary(self) -> dict[str, object]:
        return {
            "kind": "jmps_weak_interface_transition_v4_suite",
            "accepted": self.accepted,
            "acceptance_failures": self.acceptance_failures,
            "cases": {
                "crack_like": self.crack_like.summary(),
                "supershear": self.supershear.summary(),
                "spall_like": self.spall_like.summary(),
            },
            "claim_scope": "JMPS-inspired numerical mechanism benchmark",
            "publication_curve_reproduction": False,
        }


@dataclass(frozen=True)
class WeakInterfaceConvergenceStudy:
    """Two-dimensional mesh and time-step evidence for one V4 mechanism."""

    baseline: WeakInterfaceTransitionBenchmark
    spatial_refined: WeakInterfaceTransitionBenchmark
    spatial_fine: WeakInterfaceTransitionBenchmark
    temporal_refined: WeakInterfaceTransitionBenchmark
    spatial_speed_change: float
    fine_spatial_speed_change: float
    temporal_speed_change: float
    mechanism_preserved: bool
    speed_converged: bool
    accepted: bool
    acceptance_failures: tuple[str, ...]

    def summary(self) -> dict[str, object]:
        return {
            "kind": "jmps_weak_interface_transition_v4_convergence",
            "accepted": self.accepted,
            "acceptance_failures": self.acceptance_failures,
            "spatial_speed_change": self.spatial_speed_change,
            "fine_spatial_speed_change": self.fine_spatial_speed_change,
            "temporal_speed_change": self.temporal_speed_change,
            "mechanism_preserved": self.mechanism_preserved,
            "speed_converged": self.speed_converged,
            "cases": {
                "baseline": self.baseline.summary(),
                "spatial_refined": self.spatial_refined.summary(),
                "spatial_fine": self.spatial_fine.summary(),
                "temporal_refined": self.temporal_refined.summary(),
            },
            "claim_scope": "two-dimensional supershear refinement evidence",
            "publication_curve_reproduction": False,
        }


def plane_stress_thin_3d_crosscheck(
    *,
    axial_stretch: float = 1.12,
    reference_thickness: float = 0.02,
    cells=(2, 2, 1),
    young: float = 1.0e6,
    poisson: float = 0.49,
    density: float = 1000.0,
    tolerance: float = 1.0e-9,
) -> ThinThreeDimensionalCrossCheck:
    """Compare condensed 2D membrane response with a thin 3D FEM patch.

    The three-dimensional cuboid carries the same homogeneous principal
    stretches as the local plane-stress solution.  Its volume-averaged UFL
    first-Piola stress and strain energy must recover the condensed values and
    zero transverse nominal tractions.  This is a geometry/formulation patch
    test, not a thin-3D fracture simulation or a locking study.
    """

    axial = float(axial_stretch)
    thickness = float(reference_thickness)
    selected_cells = tuple(int(value) for value in cells)
    selected_tolerance = float(tolerance)
    if not isfinite(axial) or axial <= 0.0:
        raise ValueError("axial_stretch must be finite and positive.")
    if not isfinite(thickness) or thickness <= 0.0:
        raise ValueError("reference_thickness must be finite and positive.")
    if len(selected_cells) != 3 or any(value <= 0 for value in selected_cells):
        raise ValueError("cells must contain three positive integers.")
    if not isfinite(selected_tolerance) or selected_tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    membrane = constitutive.neo_hookean_plane_stress(
        young=young,
        poisson=poisson,
        density=density,
    )
    bulk = constitutive.neo_hookean(
        young=young,
        poisson=poisson,
        density=density,
    )
    F2 = constitutive.plane_stress_uniaxial_deformation_gradient(axial, membrane)
    thickness_stretch = constitutive.plane_stress_thickness_stretch_value(
        F2,
        membrane,
    )
    F3 = np.diag((F2[0, 0], F2[1, 1], thickness_stretch))
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, thickness),
        selected_cells,
        comm=MPI.COMM_SELF,
        cell_type="hexahedron",
    )
    displacement = fields.displacement(domain)
    gradient = F3 - np.eye(3)
    displacement.value.interpolate(lambda x: gradient @ x[:3])
    displacement.value.x.scatter_forward()
    P = constitutive.hyperelasticity.first_piola(displacement.value, bulk)
    energy = constitutive.hyperelasticity.strain_energy_density(
        displacement.value,
        bulk,
    )
    volume = float(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=domain))))
    average = np.asarray(
        [
            fem.assemble_scalar(fem.form(P[i, j] * ufl.dx(domain=domain))) / volume
            for i in range(3)
            for j in range(3)
        ],
        dtype=float,
    ).reshape((3, 3))
    average_energy = float(
        fem.assemble_scalar(fem.form(energy * ufl.dx(domain=domain))) / volume
    )
    condensed = constitutive.plane_stress_first_piola_value(F2, membrane)
    condensed_energy = constitutive.hyperelasticity.principal_energy_density(
        np.diag(F3),
        bulk,
    )
    scale = max(float(np.max(np.abs(condensed))), float(young) * 1.0e-15)
    stress_error = float(np.max(np.abs(average[:2, :2] - condensed)) / scale)
    energy_scale = max(abs(condensed_energy), float(young) * 1.0e-15)
    energy_error = abs(average_energy - condensed_energy) / energy_scale
    traction_free = average.copy()
    traction_free[1, 1] = 0.0
    traction_free_ratio = float(np.max(np.abs(traction_free)) / scale)
    jacobian = float(np.linalg.det(F3))
    accepted = (
        jacobian > 0.0
        and stress_error <= selected_tolerance
        and energy_error <= selected_tolerance
        and traction_free_ratio <= selected_tolerance
    )
    return ThinThreeDimensionalCrossCheck(
        axial_stretch=axial,
        lateral_stretch=float(F2[0, 0]),
        thickness_stretch=float(thickness_stretch),
        reference_thickness=thickness,
        cells=selected_cells,
        jacobian=jacobian,
        plane_stress_first_piola=condensed,
        thin_3d_first_piola=average,
        plane_stress_energy_density=float(condensed_energy),
        thin_3d_energy_density=average_energy,
        maximum_relative_stress_error=stress_error,
        relative_energy_error=float(energy_error),
        traction_free_stress_ratio=traction_free_ratio,
        accepted=bool(accepted),
    )


def finite_strain_wave_arrival(
    *,
    prestrain: float = 0.0,
    cells: int = 80,
    courant: float = 0.3,
    length: float = 2.0,
    source_position: float = 0.25,
    receiver_positions=(0.75, 1.25),
    pulse_width: float = 0.1,
) -> WaveArrivalBenchmark:
    """Measure a small longitudinal pulse about a held homogeneous stretch.

    The body is one cell thick and its transverse displacement is fixed, so
    the direct wave is the plane-strain longitudinal mode.  Peak times are
    selected only inside non-overlapping acoustic-oracle windows to exclude
    boundary reflections.  Distances and the returned measured speed use the
    reference mesh coordinates, matching the material acoustic tensor.
    """

    selected_prestrain = float(prestrain)
    selected_cells = int(cells)
    if not isfinite(selected_prestrain) or selected_prestrain <= -1.0:
        raise ValueError("prestrain must be finite and greater than -1.")
    if selected_cells < 40:
        raise ValueError("The V1 arrival benchmark requires at least 40 cells.")
    if not 0.0 < float(courant) <= 0.5:
        raise ValueError("courant must lie in (0, 0.5].")
    receivers = tuple(float(value) for value in receiver_positions)
    if len(receivers) != 2 or not source_position < receivers[0] < receivers[1] < length:
        raise ValueError("receiver_positions must contain two ordered interior points.")
    dx = float(length) / selected_cells
    height = dx
    domain = mesh.rectangle(
        (0.0, 0.0),
        (float(length), height),
        (selected_cells, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2,
            assumption="plane_strain",
            method="explicit",
        ),
        mesh=domain,
        name="finite_strain_wave_arrival_v1",
    )
    displacement = model.field(fields.displacement(domain))
    displacement.value.interpolate(
        lambda x: np.vstack(
            (selected_prestrain * x[0], np.zeros_like(x[1]))
        )
    )
    material = model.material(
        constitutive.neo_hookean(
            young=1.0e6,
            poisson=0.25,
            density=1000.0,
        )
    )
    model.constraint(
        constraints.component_dirichlet(
            displacement,
            1,
            on=lambda x: np.ones(x.shape[1], dtype=bool),
            value=0.0,
            name="plane_wave_transverse_hold",
        )
    )
    model.constraint(
        constraints.component_dirichlet(
            displacement,
            0,
            on=lambda x: np.isclose(x[0], 0.0),
            value=0.0,
            name="left_prestrain_hold",
        )
    )
    model.constraint(
        constraints.component_dirichlet(
            displacement,
            0,
            on=lambda x: np.isclose(x[0], length),
            value=selected_prestrain * length,
            name="right_prestrain_hold",
        )
    )
    F = np.diag((1.0 + selected_prestrain, 1.0))
    modes = fracture.incremental_wave_speeds(
        F,
        (1.0, 0.0),
        material,
        direction_configuration="reference",
    )
    predicted = float(modes.reference_speeds[-1])
    dt = float(courant) * dx / predicted
    end_time = (receivers[-1] - source_position + 4.0 * pulse_width) / predicted
    steps = int(ceil(end_time / dt))
    state = problems.second_order_state(displacement)
    velocity_scale = 1.0e-4 * predicted
    state.v.value.interpolate(
        lambda x: np.vstack(
            (
                velocity_scale
                * np.exp(-((x[0] - source_position) / pulse_width) ** 2),
                np.zeros_like(x[1]),
            )
        )
    )
    step = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        state=state,
        dt=dt,
        steps=steps,
        print_every=1,
        progress=False,
        name="wave_arrival",
    )
    times = []
    signals = [[], []]

    def collect(info, current) -> None:
        times.append(float(info.time))
        for index, coordinate in enumerate(receivers):
            value = results.probe(
                current.v,
                at=(coordinate, 0.5 * height),
            )
            signals[index].append(float(np.asarray(value)[0]))

    step.run(progress=collect)
    time_values = np.asarray(times)
    peak_times = []
    half_window = 2.5 * pulse_width / predicted
    for coordinate, signal in zip(receivers, signals, strict=True):
        expected = (coordinate - source_position) / predicted
        mask = np.abs(time_values - expected) <= half_window
        if np.count_nonzero(mask) < 3:
            raise RuntimeError("V1 wave-arrival window contains too few frames.")
        candidates = np.flatnonzero(mask)
        magnitude = np.abs(np.asarray(signal, dtype=float))
        selected = candidates[int(np.argmax(magnitude[mask]))]
        peak_times.append(_quadratic_peak_time(time_values, magnitude, selected))
    delay = peak_times[1] - peak_times[0]
    if delay <= 0.0:
        raise RuntimeError("V1 receiver peaks are not causally ordered.")
    measured = (receivers[1] - receivers[0]) / delay
    return WaveArrivalBenchmark(
        prestrain=selected_prestrain,
        predicted_reference_speed=predicted,
        measured_reference_speed=float(measured),
        relative_error=abs(float(measured) - predicted) / predicted,
        first_arrival_time=peak_times[0],
        second_arrival_time=peak_times[1],
        time_increment=dt,
        cells=selected_cells,
        maximum_relative_energy_error=float(
            max(
                record["relative_energy_balance_error"]
                for record in step.history_records
            )
        ),
    )


def cohesive_energy_balance(
    *,
    dt: float = 1.0e-3,
    loading_time: float = 0.2,
    opening: float = 0.08,
) -> CohesiveEnergyBenchmark:
    """Open one split interface through a smooth prescribed-motion history."""

    selected_dt = float(dt)
    selected_loading_time = float(loading_time)
    selected_opening = float(opening)
    if selected_dt <= 0.0 or selected_loading_time <= 0.0:
        raise ValueError("dt and loading_time must be positive.")
    if selected_opening <= 0.0:
        raise ValueError("opening must be positive.")
    coordinates = np.array(
        [
            [0.0, 0.0], [1.0, 0.0],
            [1.0, 0.5], [0.0, 0.5],
            [1.0, 1.0], [0.0, 1.0],
        ],
        dtype=float,
    )
    split = interfaces.split_conforming_line_interface(
        coordinates,
        np.array([[0, 1, 2, 3], [3, 2, 4, 5]], dtype=int),
        [[3, 2]],
        positive_cells=[1],
    )
    domain = interfaces.create_dolfinx_split_mesh(split)
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2, assumption="plane_strain", method="explicit",
        ),
        mesh=domain,
        name="cohesive_energy_balance_v2",
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.neo_hookean(
            young=1000.0,
            poisson=0.25,
            density=1.0,
        )
    )
    model.constraint(
        constraints.component_dirichlet(
            displacement, 1, on=lambda x: np.isclose(x[1], 0.0), value=0.0,
        )
    )
    model.constraint(
        constraints.component_dirichlet(
            displacement, 0, on=lambda x: np.isclose(x[0], 0.0), value=0.0,
        )
    )

    def smooth_opening(time_value: float) -> float:
        if time_value <= 0.0:
            return 0.0
        if time_value >= selected_loading_time:
            return selected_opening
        phase = np.pi * time_value / selected_loading_time
        return 0.5 * selected_opening * (1.0 - np.cos(phase))

    top = constraints.time_dependent_component_dirichlet(
        displacement,
        1,
        on=lambda x: np.isclose(x[1], 1.0),
        amplitude=smooth_opening,
        name="smooth_interface_opening",
    )
    model.constraint(top)
    law = interfaces.bilinear_cohesive(
        strength=10.0,
        fracture_energy=0.2,
        initial_stiffness=10_000.0,
    )
    cohesive = fracture.mode_i_cohesive_force(
        split,
        displacement,
        law,
        normal_hint=(0.0, 1.0),
    )
    increments = int(ceil(selected_loading_time / selected_dt))
    actual_dt = selected_loading_time / increments
    step = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        cohesive_force=cohesive,
        dt=actual_dt,
        steps=increments,
        progress=False,
        name="cohesive_energy",
    )
    step.run()
    response = step.residual.cohesive.current_response()
    history = step.history_records
    final_work = abs(float(history[-1]["external_work"]))
    significant = [
        record
        for record in history
        if final_work == 0.0
        or abs(float(record["external_work"])) >= 0.01 * final_work
    ]
    return CohesiveEnergyBenchmark(
        time_increment=actual_dt,
        increments=increments,
        prescribed_opening=selected_opening,
        maximum_damage=float(np.max(response.damage)),
        fracture_dissipation=float(history[-1]["cohesive_fracture_dissipation"]),
        declared_fracture_energy=float(law.fracture_energy),
        final_relative_energy_error=float(
            history[-1]["relative_energy_balance_error"]
        ),
        maximum_relative_energy_error=float(
            max(
                record["relative_energy_balance_error"]
                for record in significant
            )
        ),
    )


def classical_cohesive_crack(
    *,
    cells: int = 60,
    length: float = 3.0,
    precrack_length: float = 0.5,
    opening: float = 0.0135,
    loading_time: float = 0.15,
    hold_time: float = 0.15,
    time_step_scale: float = 0.8,
    damping: float = 0.0,
) -> ClassicalCrackBenchmark:
    """Propagate a precracked cohesive strip below the classical limit.

    This is the first V3 guardrail, not a supershear configuration.  A smooth
    remote opening loads a long weak interface with a declared precrack.  The
    crack front is threshold-interpolated and its speed is fitted over an odd
    local window so a single failed facet cannot create a velocity spike.
    """

    selected_cells = int(cells)
    if selected_cells < 30:
        raise ValueError("The V3 strip requires at least 30 interface cells.")
    if not 0.0 < float(time_step_scale) <= 1.0:
        raise ValueError("time_step_scale must lie in (0, 1].")
    if not isfinite(float(damping)) or float(damping) < 0.0:
        raise ValueError("damping must be finite and nonnegative.")
    x = np.linspace(0.0, float(length), selected_cells + 1)
    coordinates = np.vstack(
        (
            np.column_stack((x, np.zeros_like(x))),
            np.column_stack((x, np.full_like(x, 0.5))),
            np.column_stack((x, np.ones_like(x))),
        )
    )
    bottom = []
    top = []
    row = selected_cells + 1
    for index in range(selected_cells):
        bottom.append((index, index + 1, row + index + 1, row + index))
        top.append(
            (
                row + index,
                row + index + 1,
                2 * row + index + 1,
                2 * row + index,
            )
        )
    cells_array = np.asarray((*bottom, *top), dtype=int)
    interface_facets = np.asarray(
        [(row + index, row + index + 1) for index in range(selected_cells)],
        dtype=int,
    )
    split = interfaces.split_conforming_line_interface(
        coordinates,
        cells_array,
        interface_facets,
        positive_cells=np.arange(selected_cells, 2 * selected_cells),
    )
    domain = interfaces.create_dolfinx_split_mesh(split)
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2, assumption="plane_strain", method="explicit",
        ),
        mesh=domain,
        name="classical_cohesive_crack_v3",
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.neo_hookean(
            young=1000.0,
            poisson=0.25,
            density=1.0,
        )
    )
    model.constraint(
        constraints.component_dirichlet(
            displacement, 1, on=lambda p: np.isclose(p[1], 0.0), value=0.0,
        )
    )
    model.constraint(
        constraints.component_dirichlet(
            displacement,
            0,
            on=lambda p: np.isclose(p[0], 0.0) & np.isclose(p[1], 0.0),
            value=0.0,
        )
    )

    def smooth_remote_opening(time_value: float) -> float:
        if time_value <= 0.0:
            return 0.0
        if time_value >= loading_time:
            return float(opening)
        phase = np.pi * time_value / float(loading_time)
        return 0.5 * float(opening) * (1.0 - np.cos(phase))

    model.constraint(
        constraints.time_dependent_component_dirichlet(
            displacement,
            1,
            on=lambda p: np.isclose(p[1], 1.0),
            amplitude=smooth_remote_opening,
            name="smooth_remote_opening",
        )
    )
    law = interfaces.bilinear_cohesive(
        strength=10.0,
        fracture_energy=0.1,
        initial_stiffness=10_000.0,
    )
    cohesive = fracture.mode_i_cohesive_force(
        split,
        displacement,
        law,
        normal_hint=(0.0, 1.0),
    )
    facet_centers = np.mean(split.coordinates[split.negative_facets, 0], axis=1)
    precracked = np.flatnonzero(facet_centers <= float(precrack_length))
    cohesive.assembler.initialize_precrack(precracked)
    provisional = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        cohesive_force=cohesive,
        dt="auto",
        steps=1,
        progress=False,
        mass_damping=damping,
        name="v3_stability_probe",
    )
    dt = float(time_step_scale) * provisional.dt
    total_time = float(loading_time) + float(hold_time)
    increments = int(ceil(total_time / dt))
    dt = total_time / increments
    step = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        cohesive_force=cohesive,
        dt=dt,
        steps=increments,
        print_every=1,
        progress=False,
        mass_damping=damping,
        name="classical_crack",
    )
    cohesive_residual = step.residual.base if damping else step.residual
    times = []
    damage_frames = []

    def collect(info, _state) -> None:
        response = cohesive_residual.cohesive.current_response()
        times.append(float(info.time))
        damage_frames.append(np.max(response.damage, axis=1))

    step.run(progress=collect)
    crack = fracture.crack_tip_history(
        times,
        facet_centers,
        damage_frames,
        threshold=0.95,
        fit_window=7,
    )
    finite = crack.speed[np.isfinite(crack.speed)]
    maximum_speed = 0.0 if finite.size == 0 else float(np.max(finite))
    initial_length = float(crack.position[0])
    final_length = float(crack.position[-1])
    wave = fracture.isotropic_reference_wave_speeds(material)
    failed = np.asarray(damage_frames) >= 0.95
    newly_failed = np.count_nonzero(
        failed[1:] & ~failed[:-1], axis=1
    )
    maximum_simultaneous = (
        0.0
        if newly_failed.size == 0
        else float(np.max(newly_failed) / selected_cells)
    )
    return ClassicalCrackBenchmark(
        cells=selected_cells,
        time_increment=dt,
        initial_crack_length=initial_length,
        final_crack_length=final_length,
        propagated_length=max(0.0, final_length - initial_length),
        maximum_fitted_speed=maximum_speed,
        rayleigh_wave_speed=float(wave.rayleigh),
        speed_ratio=maximum_speed / float(wave.rayleigh),
        final_relative_energy_error=float(
            step.history_records[-1]["relative_energy_balance_error"]
        ),
        damping=float(damping),
        numerical_damping_dissipation=float(
            step.history_records[-1].get("numerical_damping_dissipation", 0.0)
        ),
        maximum_simultaneous_failed_fraction=maximum_simultaneous,
    )


def prestressed_weak_interface_separation(
    *,
    label: str = "v4_candidate",
    cells: int = 60,
    transverse_cells: int = 2,
    length: float = 3.0,
    height: float = 1.0,
    precrack_length: float = 0.5,
    axial_strain: float = 0.12,
    strength: float = 10.0,
    fracture_energy: float = 0.1,
    initial_stiffness: float = 10_000.0,
    young: float = 1000.0,
    poisson: float = 0.49,
    density: float = 1.0,
    total_time: float = 0.2,
    time_step_scale: float = 0.8,
    damping: float = 0.0,
    history_every: int = 1,
    impact_displacement: float = 0.0,
    impact_rise_time: float | None = None,
    speed_fit_length: float | None = None,
    retain_trace: bool = False,
) -> WeakInterfaceTransitionBenchmark:
    """Drive a precrack through a prestressed plane-stress weak interface.

    This is the reusable V4 mechanism case.  The body is first placed in a
    homogeneous uniaxial plane-stress state.  The precrack is then declared
    fully separated.  Its release alone, or a smooth remote impact prescribed
    after release, drives the dynamics. Crack speed is measured in reference
    coordinates and compared with the prestrained acoustic-tensor bulk modes.
    """

    selected_cells = int(cells)
    if selected_cells < 30:
        raise ValueError("The V4 strip requires at least 30 interface cells.")
    selected_transverse_cells = int(transverse_cells)
    if (
        selected_transverse_cells < 2
        or selected_transverse_cells % 2 != 0
    ):
        raise ValueError(
            "transverse_cells must be an even integer of at least two so the "
            "weak interface lies on a complete mesh row."
        )
    selected_length = float(length)
    selected_height = float(height)
    selected_precrack = float(precrack_length)
    selected_strain = float(axial_strain)
    selected_time = float(total_time)
    if not 0.0 < selected_precrack < selected_length:
        raise ValueError("precrack_length must lie inside the interface.")
    if selected_height <= 0.0 or selected_length <= 0.0:
        raise ValueError("length and height must be positive.")
    if selected_strain <= 0.0:
        raise ValueError("V4 axial_strain must be positive.")
    if selected_time <= 0.0:
        raise ValueError("total_time must be positive.")
    if not 0.0 < float(time_step_scale) <= 1.0:
        raise ValueError("time_step_scale must lie in (0, 1].")
    if not isfinite(float(damping)) or float(damping) < 0.0:
        raise ValueError("damping must be finite and nonnegative.")
    selected_impact = float(impact_displacement)
    if not isfinite(selected_impact) or selected_impact < 0.0:
        raise ValueError("impact_displacement must be finite and nonnegative.")
    selected_rise = (
        0.0 if selected_impact == 0.0 and impact_rise_time is None
        else 0.1 * selected_time if impact_rise_time is None
        else float(impact_rise_time)
    )
    if selected_impact > 0.0 and not 0.0 < selected_rise <= selected_time:
        raise ValueError("impact_rise_time must lie in (0, total_time].")
    interface_spacing = selected_length / selected_cells
    selected_fit_length = (
        3.0 * interface_spacing
        if speed_fit_length is None
        else float(speed_fit_length)
    )
    minimum_fit_length = 3.0 * interface_spacing
    fit_tolerance = 32.0 * np.finfo(float).eps * max(
        1.0,
        selected_length,
    )
    if (
        not isfinite(selected_fit_length)
        or selected_fit_length + fit_tolerance < minimum_fit_length
        or selected_fit_length > selected_length - selected_precrack
    ):
        raise ValueError(
            "speed_fit_length must be finite, span at least three interface "
            "facets, and remain inside the intact ligament."
        )

    x = np.linspace(0.0, selected_length, selected_cells + 1)
    y = np.linspace(0.0, selected_height, selected_transverse_cells + 1)
    coordinates = np.asarray(
        [(x_value, y_value) for y_value in y for x_value in x],
        dtype=float,
    )
    row = selected_cells + 1
    cells_array = np.asarray(
        [
            (
                y_index * row + x_index,
                y_index * row + x_index + 1,
                (y_index + 1) * row + x_index + 1,
                (y_index + 1) * row + x_index,
            )
            for y_index in range(selected_transverse_cells)
            for x_index in range(selected_cells)
        ],
        dtype=int,
    )
    interface_row = selected_transverse_cells // 2
    interface_facets = np.asarray(
        [
            (interface_row * row + index, interface_row * row + index + 1)
            for index in range(selected_cells)
        ],
        dtype=int,
    )
    first_positive_cell = interface_row * selected_cells
    split = interfaces.split_conforming_line_interface(
        coordinates,
        cells_array,
        interface_facets,
        positive_cells=np.arange(first_positive_cell, len(cells_array)),
    )
    domain = interfaces.create_dolfinx_split_mesh(split)
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2,
            assumption="plane_stress",
            method="explicit",
        ),
        mesh=domain,
        name=f"prestressed_weak_interface_{label}",
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.neo_hookean_plane_stress(
            young=float(young),
            poisson=float(poisson),
            density=float(density),
        )
    )
    deformation_gradient, preload_opening, preload_traction = (
        _equilibrated_plane_stress_interface_preload(
            total_axial_strain=selected_strain,
            height=selected_height,
            interface_stiffness=float(initial_stiffness),
            material=material,
        )
    )
    if preload_traction >= float(strength):
        raise ValueError(
            "The requested homogeneous preload already reaches the cohesive "
            f"strength ({preload_traction:.6g} >= {float(strength):.6g}). "
            "Increase strength or reduce axial_strain before dynamic release."
        )
    model.constraint(
        constraints.component_dirichlet(
            displacement,
            0,
            on=lambda point: np.isclose(point[0], 0.0),
            value=0.0,
            name="preload_lateral_origin",
        )
    )
    model.constraint(
        constraints.component_dirichlet(
            displacement,
            1,
            on=lambda point: np.isclose(point[1], 0.0),
            value=0.0,
            name="preload_bottom_hold",
        )
    )
    preload_top = (
        (deformation_gradient[1, 1] - 1.0) * selected_height
        + preload_opening
    )
    if selected_impact:
        top_constraint = constraints.time_dependent_component_dirichlet(
            displacement,
            1,
            on=lambda point: np.isclose(point[1], selected_height),
            amplitude=amplitudes.smooth_step(
                preload_top,
                preload_top + selected_impact,
                start_time=0.0,
                end_time=selected_rise,
                name="remote_impact_motion",
            ),
            name="remote_impact_top",
        )
        top_constraint.update(0.0)
    else:
        top_constraint = constraints.component_dirichlet(
            displacement,
            1,
            on=lambda point: np.isclose(point[1], selected_height),
            value=preload_top,
            name="preload_top_hold",
        )
    model.constraint(top_constraint)
    law = interfaces.bilinear_cohesive(
        strength=float(strength),
        fracture_energy=float(fracture_energy),
        initial_stiffness=float(initial_stiffness),
    )
    cohesive = fracture.mode_i_cohesive_force(
        split,
        displacement,
        law,
        normal_hint=(0.0, 1.0),
    )
    nodal_displacement = np.column_stack(
        (
            (deformation_gradient[0, 0] - 1.0) * split.coordinates[:, 0],
            (deformation_gradient[1, 1] - 1.0) * split.coordinates[:, 1],
        )
    )
    positive_nodes = np.unique(split.cells[split.positive_cells].reshape(-1))
    nodal_displacement[positive_nodes, 1] += preload_opening
    displacement_blocks = displacement.value.x.array.reshape((-1, 2))
    displacement_blocks[cohesive.node_to_block_dof] = nodal_displacement
    displacement.value.x.scatter_forward()
    facet_centers = np.mean(split.coordinates[split.negative_facets, 0], axis=1)
    precracked = facet_centers <= selected_precrack
    cohesive.assembler.initialize_precrack(np.flatnonzero(precracked))
    provisional = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        cohesive_force=cohesive,
        dt="auto",
        steps=1,
        progress=False,
        mass_damping=damping,
        name=f"{label}_stability_probe",
    )
    dt = float(time_step_scale) * provisional.dt
    increments = int(ceil(selected_time / dt))
    dt = selected_time / increments
    step = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        cohesive_force=cohesive,
        dt=dt,
        steps=increments,
        print_every=1,
        history_every=history_every,
        progress=False,
        mass_damping=damping,
        name=f"{label}_dynamic_release",
    )
    initial_energy_values = step.history_monitor.energy.evaluate(
        displacement=displacement,
        velocity=step.state.v,
    )
    source_energy = float(
        initial_energy_values.get(
            "accounted_internal_kinetic_energy",
            initial_energy_values.get("total_mechanical_energy"),
        )
    )
    transfer = step.initialize_from_preload(
        displacement,
        source_step=f"{label}_homogeneous_preload",
        mode="release",
        source_energy=source_energy,
    )
    cohesive_residual = step.residual.base if damping else step.residual
    initial_response = cohesive_residual.cohesive.current_response()
    initial_point_response = (
        cohesive.assembler.material_point_response(initial_response)
        if retain_trace
        else None
    )
    times = [0.0]
    damage_frames = [np.max(initial_response.damage, axis=1)]
    opening_frames = (
        [np.mean(initial_response.opening, axis=1)] if retain_trace else None
    )
    traction_frames = (
        [np.mean(initial_response.traction, axis=1)] if retain_trace else None
    )
    dissipation_frames = (
        [
            np.mean(
                initial_point_response.dissipated_energy.reshape((-1, 2)),
                axis=1,
            )
        ]
        if retain_trace
        else None
    )
    traction_ratios = [
        float(np.max(np.abs(initial_response.traction[~precracked])) / law.strength)
        if np.any(~precracked)
        else 0.0
    ]

    def collect(info, _state) -> None:
        response = cohesive_residual.cohesive.current_response()
        times.append(float(info.time))
        damage_frames.append(np.max(response.damage, axis=1))
        if retain_trace:
            point_response = cohesive.assembler.material_point_response(response)
            opening_frames.append(np.mean(response.opening, axis=1))
            traction_frames.append(np.mean(response.traction, axis=1))
            dissipation_frames.append(
                np.mean(
                    point_response.dissipated_energy.reshape((-1, 2)),
                    axis=1,
                )
            )
        traction_ratios.append(
            float(np.max(np.abs(response.traction[~precracked])) / law.strength)
            if np.any(~precracked)
            else 0.0
        )

    step.run(progress=collect)
    modes = fracture.incremental_wave_speeds(
        deformation_gradient,
        (1.0, 0.0),
        material,
        direction_configuration="reference",
    )
    shear_speed = float(modes.reference_speeds[0])
    pressure_speed = float(modes.reference_speeds[-1])
    minimum_fit_time = selected_fit_length / shear_speed
    fit_window = max(7, int(ceil(minimum_fit_time / dt)) + 1)
    if fit_window % 2 == 0:
        fit_window += 1
    crack = fracture.crack_tip_history(
        times,
        facet_centers,
        damage_frames,
        threshold=0.95,
        fit_window=fit_window,
    )
    finite = crack.speed[np.isfinite(crack.speed)]
    maximum_speed = 0.0 if finite.size == 0 else float(np.max(finite))
    initial_length = float(crack.position[0])
    final_length = float(crack.position[-1])
    rayleigh_speed = float(
        fracture.isotropic_reference_wave_speeds(material).rayleigh
    )
    failed = np.asarray(damage_frames) >= 0.95
    ligament = ~precracked
    ligament_count = max(1, int(np.count_nonzero(ligament)))
    newly_failed = np.count_nonzero(
        (failed[1:] & ~failed[:-1])[:, ligament],
        axis=1,
    )
    simultaneous = (
        0.0
        if newly_failed.size == 0
        else float(np.max(newly_failed) / ligament_count)
    )
    failed_fraction = float(np.count_nonzero(failed[-1, ligament]) / ligament_count)
    first_failure_time = np.full(selected_cells, np.inf, dtype=float)
    for frame_index, time_value in enumerate(times):
        newly_observed = failed[frame_index] & np.isinf(first_failure_time)
        first_failure_time[newly_observed] = float(time_value)
    spall_time_window = selected_height / shear_speed
    finite_failure_times = first_failure_time[ligament]
    finite_failure_times = finite_failure_times[np.isfinite(finite_failure_times)]
    rapid_failed_fraction = 0.0
    for start in finite_failure_times:
        count = np.count_nonzero(
            (finite_failure_times >= start)
            & (finite_failure_times <= start + spall_time_window)
        )
        rapid_failed_fraction = max(
            rapid_failed_fraction,
            float(count / ligament_count),
        )
    maximum_traction_ratio = float(max(traction_ratios))
    regime = fracture.separation_regime(
        crack_speed=maximum_speed,
        rayleigh_wave_speed=rayleigh_speed,
        shear_wave_speed=shear_speed,
        failed_fraction=failed_fraction,
        simultaneous_failed_fraction=simultaneous,
        rapid_failed_fraction=rapid_failed_fraction,
        ligament_traction_ratio=maximum_traction_ratio,
        pressure_wave_speed=pressure_speed,
    )
    cohesive_length = law.characteristic_length(float(young))
    trace = None
    if retain_trace:
        trace = fracture.CohesiveInterfaceTrace(
            time=np.asarray(times),
            path_coordinate=facet_centers,
            opening=np.asarray(opening_frames),
            traction=np.asarray(traction_frames),
            damage=np.asarray(damage_frames),
            dissipated_energy_density=np.asarray(dissipation_frames),
            metadata={
                "benchmark": "prestressed_weak_interface_separation",
                "label": str(label),
                "spatial_configuration": "reference",
                "path_coordinate": "interface_facet_center_x",
                "facet_reduction": {
                    "opening": "quadrature_mean",
                    "traction": "quadrature_mean",
                    "damage": "quadrature_maximum",
                    "dissipated_energy_density": "quadrature_mean",
                },
                "strength": float(strength),
                "fracture_energy": float(fracture_energy),
                "precrack_length": selected_precrack,
            },
        )
    return WeakInterfaceTransitionBenchmark(
        label=str(label),
        cells=selected_cells,
        transverse_cells=selected_transverse_cells,
        axial_strain=selected_strain,
        strength=float(strength),
        fracture_energy=float(fracture_energy),
        cohesive_length=cohesive_length,
        normalized_cohesive_length=cohesive_length / selected_height,
        time_increment=dt,
        propagated_length=max(0.0, final_length - initial_length),
        maximum_fitted_speed=maximum_speed,
        rayleigh_wave_speed=rayleigh_speed,
        shear_wave_speed=shear_speed,
        pressure_wave_speed=pressure_speed,
        rayleigh_speed_ratio=maximum_speed / rayleigh_speed,
        shear_speed_ratio=maximum_speed / shear_speed,
        pressure_speed_ratio=maximum_speed / pressure_speed,
        failed_ligament_fraction=failed_fraction,
        maximum_simultaneous_failed_fraction=simultaneous,
        rapid_failed_fraction=rapid_failed_fraction,
        crack_speed_fit_window=fit_window,
        crack_speed_fit_length=selected_fit_length,
        spall_time_window=spall_time_window,
        maximum_ligament_traction_ratio=maximum_traction_ratio,
        regime=regime,
        final_relative_energy_error=float(
            step.history_records[-1]["relative_energy_balance_error"]
        ),
        preload_energy_jump=(
            0.0
            if transfer.relative_energy_jump is None
            else float(transfer.relative_energy_jump)
        ),
        release_acceleration_norm=float(transfer.acceleration_norm),
        preload_ligament_traction_ratio=preload_traction / float(strength),
        preload_interface_opening=preload_opening,
        impact_displacement=selected_impact,
        impact_rise_time=selected_rise,
        performance=step.performance.summary(),
        trace=trace,
    )


def jmps_weak_interface_transition_v4(
    *,
    cells: int = 30,
    total_time: float = 0.1,
    history_every: int = 5,
) -> WeakInterfaceTransitionSuite:
    """Run the first fixed, executable JMPS-inspired V4 mechanism ladder.

    The three cases deliberately separate a crack-like release, a remote
    impact that produces a resolved intersonic front, and weak-interface
    parameters that produce distributed spall-like separation. This is a
    numerical mechanism gate, not a fit to unpublished author parameters.
    """

    common = {
        "cells": int(cells),
        "total_time": float(total_time),
        "axial_strain": 0.12,
        "initial_stiffness": 1.0e5,
        "history_every": int(history_every),
    }
    crack_like = prestressed_weak_interface_separation(
        label="crack_like",
        strength=150.0,
        fracture_energy=1.0,
        **common,
    )
    supershear = prestressed_weak_interface_separation(
        label="supershear",
        strength=150.0,
        fracture_energy=1.0,
        impact_displacement=0.04,
        impact_rise_time=0.015,
        **common,
    )
    spall_like = prestressed_weak_interface_separation(
        label="spall_like",
        strength=115.0,
        fracture_energy=2.0,
        impact_displacement=0.04,
        impact_rise_time=0.015,
        **common,
    )
    failures: list[str] = []
    if crack_like.regime != "sub_rayleigh_crack_like":
        failures.append("The release-only endpoint is not crack-like/sub-Rayleigh.")
    if supershear.regime != "supershear":
        failures.append("The impact endpoint does not contain a resolved supershear front.")
    if not 1.05 < supershear.shear_speed_ratio < 0.95 * (
        supershear.pressure_wave_speed / supershear.shear_wave_speed
    ):
        failures.append("The supershear front is not cleanly between c_s and c_d.")
    if supershear.maximum_simultaneous_failed_fraction >= 0.1:
        failures.append("The supershear endpoint fails too many facets simultaneously.")
    if spall_like.regime != "spall_like":
        failures.append("The weak-interface endpoint is not classified as spall-like.")
    if spall_like.rapid_failed_fraction < 0.8:
        failures.append("The spall-like endpoint lacks rapid distributed separation.")
    for result in (crack_like, supershear, spall_like):
        if result.final_relative_energy_error > 0.005:
            failures.append(f"{result.label} exceeds the 0.5-percent energy gate.")
        if result.preload_energy_jump > 1.0e-12:
            failures.append(f"{result.label} introduces a preload-transfer energy jump.")
    return WeakInterfaceTransitionSuite(
        crack_like=crack_like,
        supershear=supershear,
        spall_like=spall_like,
        accepted=not failures,
        acceptance_failures=tuple(failures),
    )


def jmps_weak_interface_convergence_v4(
    *,
    history_every: int = 20,
    spatial_speed_tolerance: float = 0.10,
    temporal_speed_tolerance: float = 0.02,
) -> WeakInterfaceConvergenceStudy:
    """Run the opt-in two-dimensional V4 supershear convergence contract.

    Unlike the inexpensive two-cell-thick mechanism ladder, this study uses
    near-isotropic quadrilateral meshes through the sheet height.  The impact
    amplitude is intentionally lower: the stronger screening impact becomes
    distributed spall as the transverse discretization is resolved.  The
    contract distinguishes preservation of the physical regime from numerical
    convergence of the reported propagation speed.
    """

    spatial_tolerance = float(spatial_speed_tolerance)
    temporal_tolerance = float(temporal_speed_tolerance)
    if not 0.0 < spatial_tolerance < 1.0:
        raise ValueError("spatial_speed_tolerance must lie in (0, 1).")
    if not 0.0 < temporal_tolerance < 1.0:
        raise ValueError("temporal_speed_tolerance must lie in (0, 1).")
    selected_history_every = int(history_every)
    if selected_history_every <= 0:
        raise ValueError("history_every must be positive.")
    common = {
        "total_time": 0.1,
        "axial_strain": 0.12,
        "strength": 150.0,
        "fracture_energy": 1.0,
        "initial_stiffness": 1.0e5,
        "history_every": selected_history_every,
        "impact_displacement": 0.01,
        "impact_rise_time": 0.015,
        # Hold the physical observation scale fixed across meshes.  A
        # mesh-dependent three-facet window would shrink under refinement and
        # compare different maximum-speed observables.
        "speed_fit_length": 0.3,
    }
    baseline = prestressed_weak_interface_separation(
        label="v4_convergence_baseline",
        cells=30,
        transverse_cells=10,
        time_step_scale=0.8,
        **common,
    )
    spatial = prestressed_weak_interface_separation(
        label="v4_convergence_spatial",
        cells=40,
        transverse_cells=14,
        time_step_scale=0.8,
        **common,
    )
    spatial_fine = prestressed_weak_interface_separation(
        label="v4_convergence_spatial_fine",
        cells=60,
        transverse_cells=20,
        time_step_scale=0.8,
        **common,
    )
    temporal = prestressed_weak_interface_separation(
        label="v4_convergence_temporal",
        cells=30,
        transverse_cells=10,
        time_step_scale=0.4,
        **common,
    )
    spatial_change = abs(
        spatial.maximum_fitted_speed - baseline.maximum_fitted_speed
    ) / max(abs(spatial.maximum_fitted_speed), np.finfo(float).eps)
    temporal_change = abs(
        temporal.maximum_fitted_speed - baseline.maximum_fitted_speed
    ) / max(abs(temporal.maximum_fitted_speed), np.finfo(float).eps)
    fine_spatial_change = abs(
        spatial_fine.maximum_fitted_speed - spatial.maximum_fitted_speed
    ) / max(abs(spatial_fine.maximum_fitted_speed), np.finfo(float).eps)
    failures: list[str] = []
    mechanism_failures: list[str] = []
    speed_failures: list[str] = []
    for result in (baseline, spatial, spatial_fine, temporal):
        if result.regime != "supershear":
            mechanism_failures.append(
                f"{result.label} does not preserve supershear."
            )
        if result.maximum_simultaneous_failed_fraction >= 0.1:
            mechanism_failures.append(
                f"{result.label} is not a resolved contiguous front."
            )
        if result.final_relative_energy_error >= 0.005:
            mechanism_failures.append(
                f"{result.label} exceeds the 0.5-percent energy gate."
            )
    failures.extend(mechanism_failures)
    if spatial_change >= spatial_tolerance:
        speed_failures.append(
            "Spatial refinement changes fitted speed by "
            f"{spatial_change:.3%}, above {spatial_tolerance:.3%}."
        )
    if temporal_change >= temporal_tolerance:
        speed_failures.append(
            "Time-step refinement changes fitted speed by "
            f"{temporal_change:.3%}, above {temporal_tolerance:.3%}."
        )
    if fine_spatial_change >= spatial_tolerance:
        speed_failures.append(
            "Second spatial refinement changes fitted speed by "
            f"{fine_spatial_change:.3%}, above {spatial_tolerance:.3%}."
        )
    if fine_spatial_change >= spatial_change:
        speed_failures.append(
            "Successive spatial fitted-speed changes do not decrease "
            f"({spatial_change:.3%} then {fine_spatial_change:.3%}); "
            "the reported maximum speed is not yet asymptotically converged."
        )
    failures.extend(speed_failures)
    return WeakInterfaceConvergenceStudy(
        baseline=baseline,
        spatial_refined=spatial,
        spatial_fine=spatial_fine,
        temporal_refined=temporal,
        spatial_speed_change=spatial_change,
        fine_spatial_speed_change=fine_spatial_change,
        temporal_speed_change=temporal_change,
        mechanism_preserved=not mechanism_failures,
        speed_converged=not speed_failures,
        accepted=not failures,
        acceptance_failures=tuple(failures),
    )


def _equilibrated_plane_stress_interface_preload(
    *,
    total_axial_strain: float,
    height: float,
    interface_stiffness: float,
    material,
):
    """Solve homogeneous bulk/interface compatibility before crack release."""

    extension = float(total_axial_strain) * float(height)
    stiffness = float(interface_stiffness)
    if stiffness <= 0.0:
        raise ValueError("interface_stiffness must be positive.")

    def state(axial_stretch: float):
        gradient = constitutive.plane_stress_uniaxial_deformation_gradient(
            axial_stretch,
            material,
        )
        traction = float(
            constitutive.plane_stress_first_piola_value(gradient, material)[1, 1]
        )
        opening = traction / stiffness
        residual = (axial_stretch - 1.0) * float(height) + opening - extension
        return residual, gradient, opening, traction

    lower = 1.0
    upper = 1.0 + float(total_axial_strain)
    lower_value = state(lower)[0]
    upper_value = state(upper)[0]
    if lower_value > 0.0 or upper_value < 0.0:
        raise RuntimeError("Could not bracket the homogeneous cohesive preload state.")
    selected = None
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        candidate = state(middle)
        selected = candidate
        if abs(candidate[0]) <= 1.0e-13 * max(1.0, abs(extension)):
            break
        if candidate[0] < 0.0:
            lower = middle
        else:
            upper = middle
    if selected is None:
        raise RuntimeError("The cohesive preload solve did not start.")
    _, gradient, opening, traction = selected
    return gradient, float(opening), float(traction)


def _quadratic_peak_time(times, magnitude, index: int) -> float:
    """Refine a sampled local maximum without changing the arrival signal."""

    if index <= 0 or index >= len(times) - 1:
        return float(times[index])
    x = np.asarray(times[index - 1:index + 2], dtype=float)
    y = np.asarray(magnitude[index - 1:index + 2], dtype=float)
    coefficients = np.polyfit(x - x[1], y, 2)
    if coefficients[0] >= 0.0 or abs(coefficients[0]) <= np.finfo(float).eps:
        return float(times[index])
    offset = -coefficients[1] / (2.0 * coefficients[0])
    if abs(offset) > x[2] - x[1]:
        return float(times[index])
    return float(x[1] + offset)


__all__ = [
    "CohesiveEnergyBenchmark",
    "ClassicalCrackBenchmark",
    "ThinThreeDimensionalCrossCheck",
    "WaveArrivalBenchmark",
    "WeakInterfaceTransitionBenchmark",
    "WeakInterfaceTransitionSuite",
    "WeakInterfaceConvergenceStudy",
    "cohesive_energy_balance",
    "classical_cohesive_crack",
    "finite_strain_wave_arrival",
    "jmps_weak_interface_transition_v4",
    "jmps_weak_interface_convergence_v4",
    "plane_stress_thin_3d_crosscheck",
    "prestressed_weak_interface_separation",
]
