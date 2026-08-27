"""Finite-strain field output and periodic-cell homogenization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc

from ..constitutive import hyperelasticity
from .field_catalog import resolve_field_variables
from .quantities import integral


@dataclass(frozen=True)
class HomogenizedFrame:
    """Macroscopic response reconstructed from one periodic-cell state."""

    load_factor: float
    deformation_gradient: np.ndarray
    green_lagrange_strain: np.ndarray
    logarithmic_strain: np.ndarray
    first_piola_stress: np.ndarray
    cauchy_stress: np.ndarray
    deformation_jacobian: float
    strain_energy_density: float
    solid_reference_fraction: float
    solid_current_fraction: float
    stress_consistency_error: float

    def as_dict(self) -> dict[str, object]:
        return {
            "load_factor": self.load_factor,
            "deformation_gradient": self.deformation_gradient.tolist(),
            "green_lagrange_strain": self.green_lagrange_strain.tolist(),
            "logarithmic_strain": self.logarithmic_strain.tolist(),
            "first_piola_stress": self.first_piola_stress.tolist(),
            "cauchy_stress": self.cauchy_stress.tolist(),
            "deformation_jacobian": self.deformation_jacobian,
            "strain_energy_density": self.strain_energy_density,
            "solid_reference_fraction": self.solid_reference_fraction,
            "solid_current_fraction": self.solid_current_fraction,
            "stress_consistency_error": self.stress_consistency_error,
        }


@dataclass(frozen=True)
class StressStateInvariants:
    """Three-dimensional Cauchy-stress invariants with explicit validity.

    Triaxiality and normalized Lode parameter are undefined for a vanishing
    deviatoric stress.  They are represented by ``None`` rather than silently
    inserting a numerical value into a scientific history.
    """

    mean_stress: float
    von_mises_stress: float
    third_deviatoric_invariant: float
    triaxiality: float | None
    normalized_lode_parameter: float | None
    deviatoric_state_defined: bool
    symmetry_error: float

    def as_dict(self) -> dict[str, object]:
        return {
            "mean_stress": self.mean_stress,
            "von_mises_stress": self.von_mises_stress,
            "third_deviatoric_invariant": self.third_deviatoric_invariant,
            "triaxiality": self.triaxiality,
            "normalized_lode_parameter": self.normalized_lode_parameter,
            "deviatoric_state_defined": self.deviatoric_state_defined,
            "symmetry_error": self.symmetry_error,
            "triaxiality_convention": "mean_cauchy_stress / von_mises_stress",
            "lode_convention": "1 - 2/pi * acos(27*J3/(2*q^3))",
        }


@dataclass(frozen=True)
class HillMandelIncrement:
    """Finite-strain macrohomogeneity evidence over one accepted increment."""

    start_load_factor: float
    load_factor: float
    microscopic_work_density: float
    macroscopic_work_density: float
    residual: float
    relative_error: float

    def as_dict(self) -> dict[str, float]:
        return {
            "start_load_factor": self.start_load_factor,
            "load_factor": self.load_factor,
            "microscopic_work_density": self.microscopic_work_density,
            "macroscopic_work_density": self.macroscopic_work_density,
            "residual": self.residual,
            "relative_error": self.relative_error,
        }


@dataclass
class PeriodicCellHistoryRecorder:
    """Collect lightweight RVE evidence at every accepted increment.

    Only macroscopic tensors and one preceding microscopic state are retained.
    Spatial field output may therefore remain sparse without losing the load
    history or macrohomogeneity audit.
    """

    properties: object
    constraint: object
    frames: list[HomogenizedFrame] = field(default_factory=list)
    hill_mandel: list[HillMandelIncrement] = field(default_factory=list)
    increment_info: list[object | None] = field(default_factory=list)
    _previous_snapshot: object | None = None

    def reset(self, snapshot) -> None:
        self.frames.clear()
        self.hill_mandel.clear()
        self.increment_info.clear()
        frame = _homogenize_snapshot(snapshot, self.properties, self.constraint)
        self.frames.append(frame)
        self.increment_info.append(None)
        self._previous_snapshot = snapshot

    def accept(self, snapshot) -> None:
        if self._previous_snapshot is None:
            self.reset(snapshot)
            return
        current = _homogenize_snapshot(snapshot, self.properties, self.constraint)
        self.hill_mandel.append(
            hill_mandel_increment(
                self._previous_snapshot,
                snapshot,
                self.properties,
                constraint=self.constraint,
                start_frame=self.frames[-1],
                frame=current,
            )
        )
        self.frames.append(current)
        self.increment_info.append(getattr(snapshot, "solve_info", None))
        self._previous_snapshot = snapshot

    def summary(self) -> dict[str, object]:
        return {
            "kind": "periodic_cell_accepted_increment_history",
            "frame_count": len(self.frames),
            "increment_count": len(self.hill_mandel),
            "convergence_record_count": sum(
                item is not None for item in self.increment_info
            ),
            "spatial_output_independent": True,
            "hill_mandel_measure": (
                "trapezoidal first-Piola work over accepted compatible states"
            ),
        }


@dataclass
class LiveFiniteStrainCellFields:
    """Derived cell fields refreshed from active Explicit state at output time."""

    fields: tuple[object, ...]
    _evaluators: tuple[object, ...]

    def update(self) -> tuple[object, ...]:
        for field, evaluator in zip(self.fields, self._evaluators, strict=True):
            field.interpolate(evaluator)
            field.x.scatter_forward()
        return self.fields

    def __iter__(self):
        return iter(self.fields)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "live_finite_strain_cell_fields",
            "variables": tuple(field.name for field in self.fields),
            "update": "active_state_at_saved_explicit_frame",
            "location": "DG0_cell_sample",
        }


def finite_strain_cell_fields(
    displacement,
    properties,
    *,
    variables=("F", "E", "GREEN", "P", "S", "MISES", "J", "SENER", "EVOL"),
    pressure=None,
    velocity=None,
    density=None,
) -> tuple[object, ...]:
    """Create requested standard P0 finite-strain cell fields.

    ``E`` follows finite-strain output convention and resolves to logarithmic
    strain ``LE``. Green--Lagrange strain remains available explicitly as
    ``GREEN``. Fields are centroid samples except ``EVOL``, which is assembled
    as the reference-element integral of ``J``.
    """

    function = getattr(displacement, "value", displacement)
    domain = function.function_space.mesh
    F = hyperelasticity.deformation_gradient(function)
    E = hyperelasticity.green_lagrange_strain(function)
    J = ufl.det(F)
    if isinstance(properties, hyperelasticity.MixedNeoHookeanProperties):
        if pressure is None:
            raise ValueError(
                "Mixed finite-strain output requires the independent pressure field."
            )
        pressure_function = getattr(pressure, "value", pressure)
        P = hyperelasticity.mixed_first_piola(
            function, pressure_function, properties
        )
        sigma = hyperelasticity.mixed_cauchy_stress(
            function, pressure_function, properties
        )
        psi = hyperelasticity.mixed_strain_energy_density(
            function, pressure_function, properties
        )
    else:
        pressure_function = None
        P = hyperelasticity.first_piola(function, properties)
        sigma = hyperelasticity.cauchy_stress(function, properties)
        psi = hyperelasticity.strain_energy_density(function, properties)
    identity = ufl.Identity(F.ufl_shape[0])
    deviator = sigma - ufl.tr(sigma) / 3.0 * identity
    von_mises = ufl.sqrt(1.5 * ufl.inner(deviator, deviator))
    requested = resolve_field_variables(variables, finite_strain=True)
    kinetic_density = None
    if any(variable.key == "KED" for variable in requested):
        if velocity is None:
            raise ValueError("KED output requires the current velocity field.")
        velocity_function = getattr(velocity, "value", velocity)
        selected_density = (
            getattr(properties, "density", None) if density is None else density
        )
        if selected_density is None:
            raise ValueError("KED output requires a material or explicit density.")
        kinetic_density = 0.5 * selected_density * ufl.inner(
            velocity_function,
            velocity_function,
        )
    sampled_F = None
    fields = []
    for variable in requested:
        if variable.key in {"U", "V", "A"}:
            continue
        if variable.key == "F":
            field = _cell_sample(F, domain, variable.key)
            sampled_F = field
        elif variable.key == "LE":
            if sampled_F is None:
                sampled_F = _cell_sample(F, domain, "F")
            field = _logarithmic_strain_field(sampled_F, variable.key)
        elif variable.key == "GREEN":
            field = _cell_sample(E, domain, variable.key)
        elif variable.key == "P":
            field = _cell_sample(P, domain, variable.key)
        elif variable.key == "PRESSURE":
            if pressure_function is None:
                raise ValueError(
                    "PRESSURE is available only for a mixed displacement-pressure material."
                )
            field = _cell_sample(pressure_function, domain, variable.key)
        elif variable.key == "S":
            field = _cell_sample(sigma, domain, variable.key)
        elif variable.key == "MISES":
            field = _cell_sample(von_mises, domain, variable.key)
        elif variable.key == "J":
            field = _cell_sample(J, domain, variable.key)
        elif variable.key == "SENER":
            field = _cell_sample(psi, domain, variable.key)
        elif variable.key == "KED":
            field = _cell_sample(kinetic_density, domain, variable.key)
        elif variable.key == "EVOL":
            field = _current_element_volume(J, domain, name=variable.key)
        else:
            raise NotImplementedError(
                f"Finite-strain output does not provide {variable.key!r}."
            )
        fields.append(field)
    return tuple(fields)


def finite_strain_dynamic_cell_fields(
    displacement,
    velocity,
    properties,
    *,
    variables=("SENER", "KED", "J"),
    pressure=None,
    density=None,
) -> LiveFiniteStrainCellFields:
    """Create reusable SED/KED/stress fields for Explicit saved frames.

    The returned object owns compiled DOLFINx expressions that reference the
    active displacement and velocity Functions. ``ExplicitDynamicsStep``
    recognizes it inside ``fields=`` and calls ``update()`` immediately before
    every time-series write.
    """

    function = getattr(displacement, "value", displacement)
    velocity_function = getattr(velocity, "value", velocity)
    if velocity_function.function_space.mesh is not function.function_space.mesh:
        raise ValueError("Dynamic displacement and velocity must share one mesh.")
    domain = function.function_space.mesh
    F = hyperelasticity.deformation_gradient(function)
    green = hyperelasticity.green_lagrange_strain(function)
    J = ufl.det(F)
    if isinstance(properties, hyperelasticity.MixedNeoHookeanProperties):
        if pressure is None:
            raise ValueError("Mixed dynamic fields require the independent pressure.")
        pressure_function = getattr(pressure, "value", pressure)
        P = hyperelasticity.mixed_first_piola(function, pressure_function, properties)
        sigma = hyperelasticity.mixed_cauchy_stress(function, pressure_function, properties)
        psi = hyperelasticity.mixed_strain_energy_density(
            function, pressure_function, properties
        )
    else:
        P = hyperelasticity.first_piola(function, properties)
        sigma = hyperelasticity.cauchy_stress(function, properties)
        psi = hyperelasticity.strain_energy_density(function, properties)
    selected_density = getattr(properties, "density", None) if density is None else density
    kinetic = (
        None
        if selected_density is None
        else 0.5 * selected_density * ufl.inner(velocity_function, velocity_function)
    )
    identity = ufl.Identity(F.ufl_shape[0])
    deviator = sigma - ufl.tr(sigma) / 3.0 * identity
    expressions = {
        "F": F,
        "GREEN": green,
        "P": P,
        "S": sigma,
        "MISES": ufl.sqrt(1.5 * ufl.inner(deviator, deviator)),
        "J": J,
        "SENER": psi,
        "KED": kinetic,
    }
    requested = resolve_field_variables(variables, finite_strain=True)
    outputs = []
    evaluators = []
    for variable in requested:
        if variable.location != "cells":
            raise ValueError(
                f"Live finite-strain derived fields require cell variables, got {variable.key}."
            )
        if variable.key not in expressions or expressions[variable.key] is None:
            if variable.key == "KED":
                raise ValueError("KED output requires a material or explicit density.")
            raise NotImplementedError(
                f"Live finite-strain output does not provide {variable.key!r}."
            )
        expression = expressions[variable.key]
        shape = tuple(expression.ufl_shape)
        element = ("DG", 0) if not shape else ("DG", 0, shape)
        space = fem.functionspace(domain, element)
        output = fem.Function(space, name=variable.key)
        evaluator = fem.Expression(expression, space.element.interpolation_points)
        outputs.append(output)
        evaluators.append(evaluator)
    live = LiveFiniteStrainCellFields(tuple(outputs), tuple(evaluators))
    live.update()
    return live


def homogenize_periodic_cell(
    displacement,
    properties,
    *,
    pressure=None,
    macro_deformation_gradient,
    cell_reference_volume: float,
    load_factor: float,
) -> HomogenizedFrame:
    """Return volume-normalized macroscopic finite-strain response.

    Voids carry zero stress. Integrals are therefore over the solid mesh but
    divided by the complete periodic-cell volume, matching an RVE effective
    stress rather than a matrix-phase average.
    """

    function = getattr(displacement, "value", displacement)
    domain = function.function_space.mesh
    dx = ufl.dx(domain=domain)
    Fbar = np.asarray(macro_deformation_gradient, dtype=float)
    if Fbar.shape != (3, 3) or np.linalg.det(Fbar) <= 0.0:
        raise ValueError("macro_deformation_gradient must be a positive-J 3x3 matrix.")
    if not np.isfinite(cell_reference_volume) or cell_reference_volume <= 0.0:
        raise ValueError("cell_reference_volume must be finite and positive.")

    F = hyperelasticity.deformation_gradient(function)
    J = ufl.det(F)
    if isinstance(properties, hyperelasticity.MixedNeoHookeanProperties):
        if pressure is None:
            raise ValueError(
                "Mixed periodic homogenization requires the independent pressure field."
            )
        pressure_function = getattr(pressure, "value", pressure)
        P = hyperelasticity.mixed_first_piola(
            function, pressure_function, properties
        )
        sigma = hyperelasticity.mixed_cauchy_stress(
            function, pressure_function, properties
        )
        psi = hyperelasticity.mixed_strain_energy_density(
            function, pressure_function, properties
        )
    else:
        P = hyperelasticity.first_piola(function, properties)
        sigma = hyperelasticity.cauchy_stress(function, properties)
        psi = hyperelasticity.strain_energy_density(function, properties)
    reference_solid_volume = float(integral(ufl.as_ufl(1.0), measure=dx))
    current_solid_volume = float(integral(J, measure=dx))
    Pbar = np.asarray(integral(P, measure=dx), dtype=float) / cell_reference_volume
    Jbar = float(np.linalg.det(Fbar))
    sigma_bar = (
        np.asarray(integral(J * sigma, measure=dx), dtype=float)
        / (Jbar * cell_reference_volume)
    )
    P_from_sigma = Jbar * sigma_bar @ np.linalg.inv(Fbar).T
    green = 0.5 * (Fbar.T @ Fbar - np.eye(3))
    logarithmic = _logarithmic_strain(Fbar)
    return HomogenizedFrame(
        load_factor=float(load_factor),
        deformation_gradient=Fbar,
        green_lagrange_strain=green,
        logarithmic_strain=logarithmic,
        first_piola_stress=Pbar,
        cauchy_stress=sigma_bar,
        deformation_jacobian=Jbar,
        strain_energy_density=float(integral(psi, measure=dx))
        / cell_reference_volume,
        solid_reference_fraction=reference_solid_volume / cell_reference_volume,
        solid_current_fraction=current_solid_volume / (Jbar * cell_reference_volume),
        stress_consistency_error=float(np.max(np.abs(Pbar - P_from_sigma))),
    )


def homogenize_periodic_path(
    snapshots,
    properties,
    *,
    constraint,
) -> tuple[HomogenizedFrame, ...]:
    """Homogenize every saved state of an affine periodic-cell analysis."""

    selected = tuple(snapshots)
    if not selected:
        raise ValueError("homogenize_periodic_path requires saved snapshots.")
    return tuple(
        _homogenize_snapshot(snapshot, properties, constraint)
        for snapshot in selected
    )


def cauchy_stress_invariants(
    stress,
    *,
    relative_tolerance: float = 1.0e-12,
) -> StressStateInvariants:
    """Return triaxiality and normalized Lode state from a 3D Cauchy tensor.

    The normalized Lode convention is ``+1`` in axisymmetric tension, ``0``
    in pure shear, and ``-1`` in axisymmetric compression.  Hydrostatic and
    zero states have no defined Lode angle or triaxiality.
    """

    selected = np.asarray(stress, dtype=float)
    if selected.shape != (3, 3) or not np.all(np.isfinite(selected)):
        raise ValueError("stress must be one finite 3x3 Cauchy tensor.")
    tolerance = float(relative_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive.")
    symmetry_error = float(np.max(np.abs(selected - selected.T)))
    scale = max(float(np.linalg.norm(selected)), np.finfo(float).tiny)
    if symmetry_error > tolerance * scale:
        raise ValueError(
            "Cauchy stress is not symmetric within relative_tolerance; "
            f"relative skew={symmetry_error / scale:.3e}."
        )
    symmetric = 0.5 * (selected + selected.T)
    mean = float(np.trace(symmetric) / 3.0)
    deviator = symmetric - mean * np.eye(3)
    q = float(np.sqrt(max(0.0, 1.5 * np.sum(deviator * deviator))))
    j3 = float(np.linalg.det(deviator))
    stress_scale = max(float(np.linalg.norm(symmetric)), np.finfo(float).tiny)
    if q <= tolerance * stress_scale:
        return StressStateInvariants(
            mean,
            q,
            j3,
            None,
            None,
            False,
            symmetry_error,
        )
    argument = float(np.clip(13.5 * j3 / q**3, -1.0, 1.0))
    lode = float(1.0 - 2.0 / np.pi * np.arccos(argument))
    return StressStateInvariants(
        mean,
        q,
        j3,
        mean / q,
        lode,
        True,
        symmetry_error,
    )


def hill_mandel_increment(
    start_snapshot,
    snapshot,
    properties,
    *,
    constraint,
    start_frame: HomogenizedFrame | None = None,
    frame: HomogenizedFrame | None = None,
) -> HillMandelIncrement:
    """Compare microscopic and macroscopic first-Piola work increments.

    This finite-strain evidence uses the same trapezoidal stress rule at both
    scales.  It is the discrete compatible-state form of the Hill--Mandel
    condition for a quasistatic periodic/affine cell without body-force or
    inertia power terms.
    """

    if float(snapshot.load_factor) <= float(start_snapshot.load_factor):
        raise ValueError("Hill-Mandel increments require increasing load factors.")
    first = (
        _homogenize_snapshot(start_snapshot, properties, constraint)
        if start_frame is None
        else start_frame
    )
    second = (
        _homogenize_snapshot(snapshot, properties, constraint)
        if frame is None
        else frame
    )
    start_u = getattr(start_snapshot.solution, "value", start_snapshot.solution)
    end_u = getattr(snapshot.solution, "value", snapshot.solution)
    domain = start_u.function_space.mesh
    if end_u.function_space.mesh is not domain:
        raise ValueError("Hill-Mandel snapshots must share one mesh.")
    start_pressure = getattr(start_snapshot, "fields", {}).get("PRESSURE")
    end_pressure = getattr(snapshot, "fields", {}).get("PRESSURE")
    P0 = _first_piola_expression(start_u, properties, pressure=start_pressure)
    P1 = _first_piola_expression(end_u, properties, pressure=end_pressure)
    F0 = hyperelasticity.deformation_gradient(start_u)
    F1 = hyperelasticity.deformation_gradient(end_u)
    microscopic = float(
        integral(
            0.5 * ufl.inner(P0 + P1, F1 - F0),
            measure=ufl.dx(domain=domain),
        )
    ) / float(constraint.reference_cell_volume)
    macroscopic = float(
        0.5
        * np.sum(
            (first.first_piola_stress + second.first_piola_stress)
            * (second.deformation_gradient - first.deformation_gradient)
        )
    )
    residual = microscopic - macroscopic
    scale = max(abs(microscopic), abs(macroscopic), np.finfo(float).tiny)
    return HillMandelIncrement(
        float(start_snapshot.load_factor),
        float(snapshot.load_factor),
        microscopic,
        macroscopic,
        residual,
        abs(residual) / scale,
    )


def hill_mandel_periodic_path(
    snapshots,
    properties,
    *,
    constraint,
    frames=None,
) -> tuple[HillMandelIncrement, ...]:
    """Evaluate Hill--Mandel evidence between consecutive saved states."""

    selected = tuple(snapshots)
    selected_frames = (
        homogenize_periodic_path(selected, properties, constraint=constraint)
        if frames is None
        else tuple(frames)
    )
    if len(selected) != len(selected_frames):
        raise ValueError("snapshots and frames must have the same length.")
    return tuple(
        hill_mandel_increment(
            selected[index - 1],
            selected[index],
            properties,
            constraint=constraint,
            start_frame=selected_frames[index - 1],
            frame=selected_frames[index],
        )
        for index in range(1, len(selected))
    )


def _homogenize_snapshot(snapshot, properties, constraint) -> HomogenizedFrame:
    return homogenize_periodic_cell(
        snapshot.solution,
        properties,
        pressure=getattr(snapshot, "fields", {}).get("PRESSURE"),
        macro_deformation_gradient=(
            constraint.measured_deformation_gradient(snapshot.solution)
            if hasattr(constraint, "measured_deformation_gradient")
            else constraint.deformation_gradient_at(snapshot.load_factor)
        ),
        cell_reference_volume=constraint.reference_cell_volume,
        load_factor=snapshot.load_factor,
    )


def _first_piola_expression(displacement, properties, *, pressure=None):
    function = getattr(displacement, "value", displacement)
    if isinstance(properties, hyperelasticity.MixedNeoHookeanProperties):
        if pressure is None:
            raise ValueError("Mixed Hill-Mandel evidence requires pressure fields.")
        pressure_function = getattr(pressure, "value", pressure)
        return hyperelasticity.mixed_first_piola(
            function,
            pressure_function,
            properties,
        )
    return hyperelasticity.first_piola(function, properties)


def finite_strain_diagnostics(
    displacement,
    *,
    constraint=None,
    quadrature_degree: int = 4,
) -> dict[str, object]:
    """Evaluate reusable physical checks for a finite-deformation solution."""

    function = getattr(displacement, "value", displacement)
    domain = function.function_space.mesh
    measures = hyperelasticity.kinematics(function)
    dx = ufl.dx(domain=domain)
    minimum_J, maximum_J = _quadrature_extrema(
        measures.jacobian,
        domain,
        degree=quadrature_degree,
    )
    dimension = int(function.ufl_shape[0])
    coefficients = np.asarray(function.x.array).reshape(-1, dimension)
    local_maximum = (
        0.0
        if coefficients.size == 0
        else float(np.max(np.linalg.norm(coefficients, axis=1)))
    )
    maximum_displacement = float(
        domain.comm.allreduce(local_maximum, op=_mpi_max())
    )
    reference_volume = float(integral(ufl.as_ufl(1.0), measure=dx))
    if reference_volume <= 0.0:
        raise ValueError("Finite-strain diagnostics require positive volume.")
    values: dict[str, object] = {
        "average_deformation_gradient": integral(
            measures.deformation_gradient,
            measure=dx,
        )
        / reference_volume,
        "average_J": integral(measures.jacobian, measure=dx)
        / reference_volume,
        "minimum_quadrature_J": minimum_J,
        "maximum_quadrature_J": maximum_J,
        "maximum_displacement": maximum_displacement,
    }
    if constraint is not None:
        measured = (
            constraint.measured_deformation_gradient(function)
            if hasattr(constraint, "measured_deformation_gradient")
            else None
        )
        values.update(
            {
                "nominal_deformation_gradient": constraint.deformation_gradient,
                "measured_macro_deformation_gradient": measured,
                "macro_control_prescribed_mask": np.asarray(
                    [
                        [0.0 if value is None else 1.0 for value in row]
                        for row in constraint.control_displacements
                    ],
                    dtype=float,
                ),
                "periodic_equation_max_error": constraint.mismatch(),
            }
        )
        if not getattr(constraint, "has_free_macro_dofs", False):
            values["target_deformation_gradient"] = constraint.deformation_gradient
    return values


def write_homogenized_history(
    path: str | Path,
    frames,
    *,
    hill_mandel=(),
    increment_info=(),
) -> Path:
    """Write an exact, compact NumPy history for plotting and ML reuse."""

    selected = tuple(frames)
    if not selected:
        raise ValueError("write_homogenized_history requires at least one frame.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stress_states = tuple(
        cauchy_stress_invariants(frame.cauchy_stress) for frame in selected
    )
    hill = _aligned_hill_mandel(selected, hill_mandel)
    convergence = _aligned_increment_evidence(selected, increment_info)
    np.savez_compressed(
        output,
        load_factor=np.asarray([frame.load_factor for frame in selected]),
        deformation_gradient=np.asarray(
            [frame.deformation_gradient for frame in selected]
        ),
        green_lagrange_strain=np.asarray(
            [frame.green_lagrange_strain for frame in selected]
        ),
        logarithmic_strain=np.asarray(
            [frame.logarithmic_strain for frame in selected]
        ),
        first_piola_stress=np.asarray(
            [frame.first_piola_stress for frame in selected]
        ),
        cauchy_stress=np.asarray([frame.cauchy_stress for frame in selected]),
        deformation_jacobian=np.asarray(
            [frame.deformation_jacobian for frame in selected]
        ),
        strain_energy_density=np.asarray(
            [frame.strain_energy_density for frame in selected]
        ),
        solid_reference_fraction=np.asarray(
            [frame.solid_reference_fraction for frame in selected]
        ),
        solid_current_fraction=np.asarray(
            [frame.solid_current_fraction for frame in selected]
        ),
        stress_consistency_error=np.asarray(
            [frame.stress_consistency_error for frame in selected]
        ),
        mean_cauchy_stress=np.asarray(
            [state.mean_stress for state in stress_states]
        ),
        von_mises_cauchy_stress=np.asarray(
            [state.von_mises_stress for state in stress_states]
        ),
        stress_triaxiality=np.asarray(
            [np.nan if state.triaxiality is None else state.triaxiality for state in stress_states]
        ),
        normalized_lode_parameter=np.asarray(
            [
                np.nan
                if state.normalized_lode_parameter is None
                else state.normalized_lode_parameter
                for state in stress_states
            ]
        ),
        stress_state_defined=np.asarray(
            [state.deviatoric_state_defined for state in stress_states],
            dtype=bool,
        ),
        hill_mandel_microscopic_work_density=np.asarray(
            [item[0] for item in hill]
        ),
        hill_mandel_macroscopic_work_density=np.asarray(
            [item[1] for item in hill]
        ),
        hill_mandel_residual=np.asarray([item[2] for item in hill]),
        hill_mandel_relative_error=np.asarray([item[3] for item in hill]),
        accepted_increment_defined=np.asarray(
            [item[0] for item in convergence], dtype=bool
        ),
        accepted_increment_size=np.asarray([item[1] for item in convergence]),
        accepted_newton_iterations=np.asarray([item[2] for item in convergence]),
        accepted_residual_norm=np.asarray([item[3] for item in convergence]),
        accepted_periodic_equation_mismatch=np.asarray(
            [item[4] for item in convergence]
        ),
        accepted_attempt=np.asarray([item[5] for item in convergence]),
    )
    return output


def write_homogenized_csv(
    path: str | Path,
    frames,
    *,
    hill_mandel=(),
    increment_info=(),
) -> Path:
    """Write flattened macro tensors in a human-readable table."""

    selected = tuple(frames)
    if not selected:
        raise ValueError("write_homogenized_csv requires at least one frame.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stress_states = tuple(
        cauchy_stress_invariants(frame.cauchy_stress) for frame in selected
    )
    hill = _aligned_hill_mandel(selected, hill_mandel)
    convergence = _aligned_increment_evidence(selected, increment_info)
    tensor_names = (
        "deformation_gradient",
        "green_lagrange_strain",
        "logarithmic_strain",
        "first_piola_stress",
        "cauchy_stress",
    )
    component_names = tuple(f"{i + 1}{j + 1}" for i in range(3) for j in range(3))
    header = ["load_factor"]
    for tensor in tensor_names:
        header.extend(f"{tensor}_{component}" for component in component_names)
    header.extend(
        (
            "deformation_jacobian",
            "strain_energy_density",
            "solid_reference_fraction",
            "solid_current_fraction",
            "stress_consistency_error",
            "mean_cauchy_stress",
            "von_mises_cauchy_stress",
            "stress_triaxiality",
            "normalized_lode_parameter",
            "stress_state_defined",
            "hill_mandel_microscopic_work_density",
            "hill_mandel_macroscopic_work_density",
            "hill_mandel_residual",
            "hill_mandel_relative_error",
            "accepted_increment_defined",
            "accepted_increment_size",
            "accepted_newton_iterations",
            "accepted_residual_norm",
            "accepted_periodic_equation_mismatch",
            "accepted_attempt",
        )
    )
    rows = []
    for index, frame in enumerate(selected):
        state = stress_states[index]
        row = [frame.load_factor]
        for tensor in tensor_names:
            row.extend(np.asarray(getattr(frame, tensor)).reshape(-1))
        row.extend(
            (
                frame.deformation_jacobian,
                frame.strain_energy_density,
                frame.solid_reference_fraction,
                frame.solid_current_fraction,
                frame.stress_consistency_error,
                state.mean_stress,
                state.von_mises_stress,
                np.nan if state.triaxiality is None else state.triaxiality,
                (
                    np.nan
                    if state.normalized_lode_parameter is None
                    else state.normalized_lode_parameter
                ),
                float(state.deviatoric_state_defined),
                *hill[index],
                *convergence[index],
            )
        )
        rows.append(row)
    np.savetxt(
        output,
        np.asarray(rows, dtype=float),
        delimiter=",",
        header=",".join(header),
        comments="",
    )
    return output


def _aligned_hill_mandel(frames, increments):
    selected = tuple(increments)
    if selected and len(selected) != len(frames) - 1:
        raise ValueError("Hill-Mandel increments must connect consecutive frames.")
    if not selected:
        return tuple((np.nan, np.nan, np.nan, np.nan) for _ in frames)
    return (
        (0.0, 0.0, 0.0, 0.0),
        *tuple(
            (
                item.microscopic_work_density,
                item.macroscopic_work_density,
                item.residual,
                item.relative_error,
            )
            for item in selected
        ),
    )


def _aligned_increment_evidence(frames, increment_info):
    """Align accepted-solve evidence with macro frames.

    The initial state has no nonlinear increment.  Missing evidence remains
    explicit through the leading boolean and NaN numerical values in files;
    result histories use a zero placeholder plus the same validity channel.
    """

    selected = tuple(increment_info)
    if selected and len(selected) == len(frames) - 1:
        selected = (None, *selected)
    if selected and len(selected) != len(frames):
        raise ValueError(
            "Accepted-increment evidence must align with frames or connect "
            "consecutive frames."
        )
    if not selected:
        selected = (None,) * len(frames)

    aligned = []
    for item in selected:
        if item is None:
            aligned.append((False, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        start = _evidence_value(item, "start_load_factor")
        end = _evidence_value(item, "load_factor")
        aligned.append(
            (
                True,
                float(end) - float(start),
                float(_evidence_value(item, "iterations")),
                float(_evidence_value(item, "residual_norm")),
                float(_evidence_value(item, "equation_mismatch")),
                float(_evidence_value(item, "attempt", default=1)),
            )
        )
    return tuple(aligned)


def _evidence_value(item, name: str, *, default=None):
    if isinstance(item, dict):
        value = item.get(name, default)
    else:
        value = getattr(item, name, default)
    if value is None:
        raise ValueError(f"Accepted-increment evidence is missing {name!r}.")
    return value


def _cell_sample(expression, domain, name: str):
    shape = tuple(expression.ufl_shape)
    element = ("DG", 0) if not shape else ("DG", 0, shape)
    space = fem.functionspace(domain, element)
    output = fem.Function(space, name=name)
    evaluator = fem.Expression(expression, space.element.interpolation_points)
    output.interpolate(evaluator)
    return output


def _current_element_volume(J, domain, *, name="EVOL"):
    space = fem.functionspace(domain, ("DG", 0))
    output = fem.Function(space, name=name)
    test = ufl.TestFunction(space)
    vector = fem_petsc.assemble_vector(
        fem.form(test * J * ufl.dx(domain=domain))
    )
    owned_size = int(
        space.dofmap.index_map.size_local
        * space.dofmap.index_map_bs
    )
    if vector.array_r.size != owned_size:
        raise RuntimeError(
            "Element-volume vector does not match the owned DG0 dof count."
        )
    output.x.array[:owned_size] = vector.array_r
    output.x.scatter_forward()
    vector.destroy()
    return output


def _logarithmic_strain_field(deformation_gradient, name: str):
    """Return DG0 spatial Hencky strain ``log(V)`` from a DG0 ``F`` field."""

    space = deformation_gradient.function_space
    output = fem.Function(space, name=name)
    dimension = deformation_gradient.ufl_shape[0]
    values = np.asarray(deformation_gradient.x.array).reshape(
        -1,
        dimension,
        dimension,
    )
    logarithmic = np.empty_like(values)
    for index, F in enumerate(values):
        eigenvalues, eigenvectors = np.linalg.eigh(F @ F.T)
        if np.any(eigenvalues <= 0.0):
            raise ValueError("Logarithmic strain requires positive principal stretches.")
        logarithmic[index] = (
            eigenvectors
            @ np.diag(0.5 * np.log(eigenvalues))
            @ eigenvectors.T
        )
    output.x.array[:] = logarithmic.reshape(-1)
    output.x.scatter_forward()
    return output


def _logarithmic_strain(F: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(F.T @ F)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("Logarithmic strain requires a positive-definite F.T F.")
    return eigenvectors @ np.diag(0.5 * np.log(eigenvalues)) @ eigenvectors.T


def _quadrature_extrema(expression, domain, *, degree: int) -> tuple[float, float]:
    from .quantities import quadrature_extrema

    return quadrature_extrema(expression, domain, degree=degree)


def _mpi_max():
    from mpi4py import MPI

    return MPI.MAX
