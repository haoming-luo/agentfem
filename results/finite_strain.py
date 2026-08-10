"""Finite-strain field output and periodic-cell homogenization."""

from __future__ import annotations

from dataclasses import dataclass
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
        homogenize_periodic_cell(
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
        for snapshot in selected
    )


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
) -> Path:
    """Write an exact, compact NumPy history for plotting and ML reuse."""

    selected = tuple(frames)
    if not selected:
        raise ValueError("write_homogenized_history requires at least one frame.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
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
    )
    return output


def write_homogenized_csv(path: str | Path, frames) -> Path:
    """Write flattened macro tensors in a human-readable table."""

    selected = tuple(frames)
    if not selected:
        raise ValueError("write_homogenized_csv requires at least one frame.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
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
        )
    )
    rows = []
    for frame in selected:
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
