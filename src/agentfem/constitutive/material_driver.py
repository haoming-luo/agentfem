"""Solver-neutral quadrature driver for material-point providers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

from .quadrature import MaterialQuadratureState
from .quadrature import QuadratureField
from .quadrature import QuadratureMaterialMap
from .user_material import MaterialPointInput, UserMaterial, validated_material_update


@dataclass(frozen=True)
class MaterialPointBatchResult:
    """Responses from one atomic integration-point constitutive update."""

    cauchy_stress: np.ndarray
    consistent_tangent: np.ndarray
    state_new: np.ndarray
    strain_energy_density: np.ndarray
    suggested_time_scale: np.ndarray
    committed: bool
    stored_energy_density_components: Mapping[str, np.ndarray] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        stress = np.asarray(self.cauchy_stress, dtype=float)
        tangent = np.asarray(self.consistent_tangent, dtype=float)
        state = np.asarray(self.state_new, dtype=float)
        energy = np.asarray(self.strain_energy_density, dtype=float).reshape(-1)
        scale = np.asarray(self.suggested_time_scale, dtype=float).reshape(-1)
        count = len(stress)
        if stress.shape != (count, 3, 3):
            raise ValueError("Batch Cauchy stress must have shape (points, 3, 3).")
        if tangent.shape != (count, 9, 9):
            raise ValueError("Batch tangent must have shape (points, 9, 9).")
        if state.ndim != 2 or len(state) != count:
            raise ValueError("Batch state must have shape (points, state_size).")
        if len(energy) != count or len(scale) != count:
            raise ValueError("Batch energy and time scale must have one value per point.")
        for label, value in (
            ("stress", stress),
            ("tangent", tangent),
            ("state", state),
            ("energy", energy),
            ("time scale", scale),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Batch material {label} must be finite.")
        if np.any(scale <= 0.0):
            raise ValueError("Batch material time scales must be positive.")
        components = {}
        for name, value in self.stored_energy_density_components.items():
            key = str(name).strip().upper()
            selected = np.asarray(value, dtype=float).reshape(-1)
            if not key or len(selected) != count or not np.all(np.isfinite(selected)):
                raise ValueError(
                    "Batch stored-energy components require a nonempty name and "
                    "one finite value per point."
                )
            if key in components:
                raise ValueError(f"Duplicate batch stored-energy component {key!r}.")
            components[key] = selected.copy()
        if components and not np.allclose(
            np.sum(tuple(components.values()), axis=0),
            energy,
            rtol=2.0e-12,
            atol=2.0e-14 * max(1.0, float(np.max(np.abs(energy), initial=0.0))),
        ):
            raise ValueError(
                "Batch stored-energy components must sum to strain energy."
            )
        object.__setattr__(self, "cauchy_stress", stress.copy())
        object.__setattr__(self, "consistent_tangent", tangent.copy())
        object.__setattr__(self, "state_new", state.copy())
        object.__setattr__(self, "strain_energy_density", energy.copy())
        object.__setattr__(self, "suggested_time_scale", scale.copy())
        object.__setattr__(
            self,
            "stored_energy_density_components",
            {name: values.copy() for name, values in components.items()},
        )

    @property
    def point_count(self) -> int:
        return len(self.cauchy_stress)

    @property
    def minimum_suggested_time_scale(self) -> float:
        return float(np.min(self.suggested_time_scale))

    def summary(self) -> dict[str, object]:
        return {
            "kind": "material_point_batch_result",
            "point_count": self.point_count,
            "committed": self.committed,
            "minimum_suggested_time_scale": self.minimum_suggested_time_scale,
            "state_size": self.state_new.shape[1],
            "stress_measure": "cauchy",
            "tangent_measure": "first_piola_deformation_gradient",
            "stored_energy_density_components": tuple(
                self.stored_energy_density_components
            ),
        }


@dataclass
class MaterialQuadratureResponse:
    """Quadrature stress/tangent fields sharing one typed state transaction."""

    state: MaterialQuadratureState
    first_piola_stress: QuadratureField
    cauchy_stress: QuadratureField
    tangent: QuadratureField
    strain_energy_density: QuadratureField
    stored_energy_density_components: dict[str, QuadratureField] = field(
        default_factory=dict
    )

    @classmethod
    def create(
        cls,
        domain,
        state_schema,
        *,
        degree: int = 2,
        scheme: str = "default",
        stored_energy_component_names=(),
    ):
        state = MaterialQuadratureState.create(
            domain,
            state_schema,
            degree=degree,
            scheme=scheme,
        )
        common = {"degree": int(degree), "scheme": str(scheme)}
        component_names = tuple(
            dict.fromkeys(str(name).strip().upper() for name in stored_energy_component_names)
        )
        if any(not name for name in component_names):
            raise ValueError("Stored-energy component names must be nonempty.")
        return cls(
            state=state,
            first_piola_stress=QuadratureField.create(
                domain, name="P", value_shape=(3, 3), **common
            ),
            cauchy_stress=QuadratureField.create(
                domain, name="S", value_shape=(3, 3), **common
            ),
            tangent=QuadratureField.create(
                domain, name="DPDF", value_shape=(3, 3, 3, 3), **common
            ),
            strain_energy_density=QuadratureField.create(
                domain, name="SENER", **common
            ),
            stored_energy_density_components={
                name: QuadratureField.create(domain, name=name, **common)
                for name in component_names
            },
        )

    @property
    def domain(self):
        return self.state.domain

    @property
    def measure(self):
        return self.state.measure

    def update(
        self,
        material: UserMaterial | QuadratureMaterialMap,
        *,
        deformation_gradient_old,
        deformation_gradient_new,
        time: float,
        time_increment: float,
        properties=(),
        commit: bool = False,
    ) -> MaterialPointBatchResult:
        comm = self.domain.comm
        contract_problem = None
        try:
            declared_components = (
                material.require_common_stored_energy_component_names()
                if isinstance(material, QuadratureMaterialMap)
                else tuple(
                    str(name).strip().upper()
                    for name in getattr(
                        material,
                        "stored_energy_component_names",
                        (),
                    )
                )
            )
            if set(declared_components) != set(
                self.stored_energy_density_components
            ):
                raise ValueError(
                    "Material stored-energy declaration differs from the "
                    "quadrature response contract."
                )
        except Exception as exc:
            contract_problem = f"{type(exc).__name__}: {exc}"
        _raise_collective_material_problem(
            comm,
            contract_problem,
            context="quadrature response contract",
        )
        result = update_material_points(
            material,
            self.state,
            deformation_gradient_old=deformation_gradient_old,
            deformation_gradient_new=deformation_gradient_new,
            time=time,
            time_increment=time_increment,
            properties=properties,
            # Commit only after all derived response fields have been built.
            # Otherwise a postprocessing failure could commit the internal
            # variables while leaving P/S/DPDF at the previous boundary.
            commit=False,
        )
        first_piola = None
        postprocessing_problem = None
        try:
            new_gradients = _point_array(
                deformation_gradient_new,
                point_count=result.point_count,
                label="deformation_gradient_new",
            )
            first_piola = np.asarray(
                [
                    np.linalg.det(gradient)
                    * stress
                    @ np.linalg.inv(gradient).T
                    for gradient, stress in zip(
                        new_gradients, result.cauchy_stress, strict=True
                    )
                ]
            )
            if not np.all(np.isfinite(first_piola)):
                raise ValueError("First Piola stress must be finite.")
            if set(result.stored_energy_density_components) != set(
                self.stored_energy_density_components
            ):
                raise ValueError(
                    "Material response stored-energy components differ from "
                    "the quadrature response contract."
                )
        except Exception as exc:
            postprocessing_problem = f"{type(exc).__name__}: {exc}"
        problems = comm.allgather(postprocessing_problem)
        if any(problem is not None for problem in problems):
            self.state.rollback()
            rank = next(
                index for index, problem in enumerate(problems) if problem is not None
            )
            raise RuntimeError(
                f"Rank {rank}: quadrature response postprocessing failed: "
                f"{problems[rank]}"
            )
        assignments = [
            (self.first_piola_stress, first_piola),
            (self.cauchy_stress, result.cauchy_stress),
            (
                self.tangent,
                result.consistent_tangent.reshape((-1, 3, 3, 3, 3)),
            ),
            (self.strain_energy_density, result.strain_energy_density),
            *(
                (field, result.stored_energy_density_components[name])
                for name, field in self.stored_energy_density_components.items()
            ),
        ]
        assignment_problem = None
        try:
            for field, values in assignments:
                selected = np.asarray(values)
                expected = int(field.function.x.array.size)
                if selected.size != expected or not np.all(np.isfinite(selected)):
                    raise ValueError(
                        f"{field.function.name} requires {expected} finite "
                        f"coefficient values, got {selected.size}."
                    )
        except Exception as exc:
            assignment_problem = f"{type(exc).__name__}: {exc}"
        _raise_collective_material_problem(
            comm,
            assignment_problem,
            context="quadrature response assignment contract",
        )
        assignment_problem = None
        try:
            for field, values in assignments:
                field.assign(values)
        except Exception as exc:
            assignment_problem = f"{type(exc).__name__}: {exc}"
        problems = comm.allgather(assignment_problem)
        if any(problem is not None for problem in problems):
            self.state.rollback()
            rank = next(
                index for index, problem in enumerate(problems) if problem is not None
            )
            raise RuntimeError(
                f"Rank {rank}: quadrature response assignment failed: "
                f"{problems[rank]}"
            )
        if commit:
            self.state.commit()
            result = replace(result, committed=True)
        return result

    def commit(self) -> None:
        self.state.commit()

    def rollback(self) -> None:
        self.state.rollback()

    def snapshot(self) -> dict[str, np.ndarray]:
        return self.state.snapshot()

    def restore(self, snapshot) -> None:
        self.state.restore(snapshot)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "material_quadrature_response",
            "state": self.state.summary(),
            "fields": {
                "first_piola_stress": "P",
                "cauchy_stress": "S",
                "tangent": "DPDF",
                "strain_energy_density": "SENER",
                "stored_energy_density_components": {
                    name: name for name in self.stored_energy_density_components
                },
            },
        }


def _raise_collective_material_problem(comm, local_problem, *, context: str) -> None:
    """Raise the first rank-local setup error before constitutive collectives."""

    problems = comm.allgather(local_problem)
    if not any(problem is not None for problem in problems):
        return
    rank = next(index for index, problem in enumerate(problems) if problem is not None)
    raise RuntimeError(f"Rank {rank}: {context} failed: {problems[rank]}")


def _point_array(value, *, point_count: int, label: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.shape == (3, 3):
        selected = np.broadcast_to(selected, (point_count, 3, 3)).copy()
    if selected.shape != (point_count, 3, 3) or not np.all(np.isfinite(selected)):
        raise ValueError(f"{label} must have shape (3, 3) or (points, 3, 3).")
    determinants = np.linalg.det(selected)
    if np.any(determinants <= 0.0):
        raise ValueError(f"Every {label} must have positive determinant.")
    return selected


def _optional_point_scalars(value, *, point_count: int, label: str):
    if value is None:
        return None
    selected = np.asarray(value, dtype=float)
    if selected.ndim == 0:
        selected = np.full(point_count, float(selected))
    selected = selected.reshape(-1)
    if len(selected) != point_count or not np.all(np.isfinite(selected)):
        raise ValueError(f"{label} must be scalar or have one value per point.")
    return selected


def update_material_points(
    material: UserMaterial | QuadratureMaterialMap,
    state: MaterialQuadratureState,
    *,
    deformation_gradient_old,
    deformation_gradient_new,
    time: float,
    time_increment: float,
    properties=(),
    temperature=None,
    temperature_increment=None,
    field_variables=None,
    commit: bool = False,
) -> MaterialPointBatchResult:
    """Update every local quadrature point as one rollback-safe transaction.

    The driver always reads the committed state and writes the trial state.
    ``commit=False`` is the correct choice inside a global Newton iteration;
    the caller commits only after the structural increment converges.  Any
    local exception restores trial storage to the committed state before the
    exception is propagated.
    """

    input_problem = None
    try:
        regional = isinstance(material, QuadratureMaterialMap)
        if regional:
            if material.domain is not state.domain:
                raise ValueError(
                    "Quadrature material map and state must use the same mesh."
                )
            material_schema = material.require_common_state_schema()
            material.require_common_tangent_convention()
        else:
            material_schema = material.state_schema
        if state.state_schema.identity != material_schema.identity:
            raise ValueError("Quadrature and material state schemas do not match.")
        if state.state_schema.summary() != material_schema.summary():
            raise ValueError(
                "Quadrature and material state schema definitions do not match."
            )
        committed_state = state.committed_state_vectors()
        point_count = len(committed_state)
        points_per_cell = len(state.reference_field.points)
        if regional and point_count != len(material.cell_regions) * points_per_cell:
            raise ValueError(
                "Quadrature material regions and point storage do not align."
            )
        old_gradients = _point_array(
            deformation_gradient_old,
            point_count=point_count,
            label="deformation_gradient_old",
        )
        new_gradients = _point_array(
            deformation_gradient_new,
            point_count=point_count,
            label="deformation_gradient_new",
        )
        temperatures = _optional_point_scalars(
            temperature,
            point_count=point_count,
            label="temperature",
        )
        temperature_increments = _optional_point_scalars(
            temperature_increment,
            point_count=point_count,
            label="temperature_increment",
        )
        selected_properties = np.asarray(properties, dtype=float).reshape(-1)
        if not np.all(np.isfinite(selected_properties)):
            raise ValueError("properties must contain finite values.")
        if regional and selected_properties.size:
            raise ValueError(
                "Regional material providers own their parameters; pass no "
                "shared properties array to a QuadratureMaterialMap."
            )
        fields = (
            None
            if field_variables is None
            else np.asarray(field_variables, dtype=float)
        )
        if fields is not None:
            if fields.ndim == 1:
                fields = np.broadcast_to(
                    fields,
                    (point_count, len(fields)),
                ).copy()
            if (
                fields.ndim != 2
                or len(fields) != point_count
                or not np.all(np.isfinite(fields))
            ):
                raise ValueError(
                    "field_variables must be one vector or one vector per point."
                )
    except Exception as exc:
        input_problem = f"{type(exc).__name__}: {exc}"
    input_problems = state.domain.comm.allgather(input_problem)
    if any(problem is not None for problem in input_problems):
        rank = next(
            index
            for index, problem in enumerate(input_problems)
            if problem is not None
        )
        raise RuntimeError(
            f"Rank {rank}: invalid material-point batch input: "
            f"{input_problems[rank]}"
        )

    stress = np.empty((point_count, 3, 3), dtype=float)
    tangent = np.empty((point_count, 9, 9), dtype=float)
    state_new = np.empty_like(committed_state)
    energy = np.empty(point_count, dtype=float)
    energy_components: dict[str, np.ndarray] | None = None
    scales = np.empty(point_count, dtype=float)
    state.begin()
    local_problem = None
    try:
        for index in range(point_count):
            try:
                selected_material = (
                    material.material_for_point(
                        index,
                        points_per_cell=points_per_cell,
                    )
                    if regional
                    else material
                )
                response = validated_material_update(
                    selected_material,
                    MaterialPointInput(
                        deformation_gradient_old=old_gradients[index],
                        deformation_gradient_new=new_gradients[index],
                        time=float(time),
                        time_increment=float(time_increment),
                        properties=selected_properties,
                        state_old=committed_state[index],
                        state_schema=state.state_schema,
                        temperature=(
                            None
                            if temperatures is None
                            else float(temperatures[index])
                        ),
                        temperature_increment=(
                            None
                            if temperature_increments is None
                            else float(temperature_increments[index])
                        ),
                        field_variables=None if fields is None else fields[index],
                    ),
                )
            except Exception as exc:
                local_problem = (
                    f"material update failed at local quadrature point {index}: "
                    f"{type(exc).__name__}: {exc}"
                )
                break
            stress[index] = response.cauchy_stress
            tangent[index] = response.consistent_tangent
            state_new[index] = response.state_new
            energy[index] = (
                0.0
                if response.strain_energy_density is None
                else response.strain_energy_density
            )
            point_components = dict(response.stored_energy_density_components)
            declared_component_names = getattr(
                selected_material,
                "stored_energy_component_names",
                None,
            )
            declared_components = (
                None
                if declared_component_names is None
                else tuple(
                    str(name).strip().upper()
                    for name in declared_component_names
                )
            )
            if (
                declared_components is not None
                and set(point_components) != set(declared_components)
            ):
                local_problem = (
                    "material update returned stored-energy components that "
                    "differ from its declared contract at local quadrature "
                    f"point {index}"
                )
                break
            if energy_components is None:
                energy_components = {
                    name: np.empty(point_count, dtype=float)
                    for name in point_components
                }
            if set(point_components) != set(energy_components):
                local_problem = (
                    "material update changed the stored-energy component contract "
                    f"at local quadrature point {index}"
                )
                break
            for name, values in energy_components.items():
                values[index] = point_components[name]
            scales[index] = response.suggested_time_scale
        problems = state.domain.comm.allgather(local_problem)
        if any(problem is not None for problem in problems):
            rank = next(
                index for index, problem in enumerate(problems) if problem is not None
            )
            raise RuntimeError(f"Rank {rank}: {problems[rank]}")
        batch = None
        batch_problem = None
        try:
            batch = MaterialPointBatchResult(
                cauchy_stress=stress,
                consistent_tangent=tangent,
                state_new=state_new,
                strain_energy_density=energy,
                suggested_time_scale=scales,
                committed=bool(commit),
                stored_energy_density_components=(
                    {} if energy_components is None else energy_components
                ),
            )
        except Exception as exc:
            batch_problem = f"{type(exc).__name__}: {exc}"
        problems = state.domain.comm.allgather(batch_problem)
        if any(problem is not None for problem in problems):
            rank = next(
                index for index, problem in enumerate(problems) if problem is not None
            )
            raise RuntimeError(
                f"Rank {rank}: material-point batch validation failed: "
                f"{problems[rank]}"
            )
        state.assign_trial_state_vectors(state_new)
        if commit:
            state.commit()
    except Exception:
        state.rollback()
        raise
    return batch
