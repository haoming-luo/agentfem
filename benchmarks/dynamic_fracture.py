"""Executable evidence for the staged dynamic-fracture verification route."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite

import numpy as np
from mpi4py import MPI

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
    "WaveArrivalBenchmark",
    "cohesive_energy_balance",
    "classical_cohesive_crack",
    "finite_strain_wave_arrival",
]
