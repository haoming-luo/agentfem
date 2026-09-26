# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Solver-neutral quadrature driver for material-point providers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

from .quadrature import MaterialQuadratureState
from .quadrature import QuadratureField
from .quadrature import QuadratureMaterialMap
from .user_material import (
    BatchedUserMaterial,
    MaterialPointBatchInput,
    MaterialPointInput,
    UserMaterial,
    validated_material_batch_update,
)
from .small_strain_user_material import (
    SmallStrainMaterialPointBatchInput,
    SmallStrainMaterialPointBatchOutput,
    SmallStrainUserMaterial,
    small_strain_matrix_to_tensor,
    validated_small_strain_batch_update,
)


@dataclass(frozen=True)
class MaterialPointBatchResult:
    """Responses from one atomic integration-point constitutive update."""

    cauchy_stress: np.ndarray
    consistent_tangent: np.ndarray
    state_new: np.ndarray
    strain_energy_density: np.ndarray
    suggested_time_scale: np.ndarray
    committed: bool
    material_group_count: int = 1
    provider_batch_calls: int = 0
    scalar_fallback_points: int = 0
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
            raise ValueError(
                "Batch energy and time scale must have one value per point."
            )
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
        group_count = int(self.material_group_count)
        batch_calls = int(self.provider_batch_calls)
        scalar_points = int(self.scalar_fallback_points)
        if group_count < 1:
            raise ValueError("material_group_count must be positive.")
        if batch_calls < 0 or scalar_points < 0:
            raise ValueError("Batch evaluation counters must be nonnegative.")
        if scalar_points > count:
            raise ValueError("scalar_fallback_points cannot exceed point_count.")
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
        object.__setattr__(self, "material_group_count", group_count)
        object.__setattr__(self, "provider_batch_calls", batch_calls)
        object.__setattr__(self, "scalar_fallback_points", scalar_points)
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
            "evaluation": {
                "material_groups": self.material_group_count,
                "provider_batch_calls": self.provider_batch_calls,
                "scalar_fallback_points": self.scalar_fallback_points,
            },
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
            dict.fromkeys(
                str(name).strip().upper() for name in stored_energy_component_names
            )
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
            if set(declared_components) != set(self.stored_energy_density_components):
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
                    np.linalg.det(gradient) * stress @ np.linalg.inv(gradient).T
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
                f"Rank {rank}: quadrature response assignment failed: {problems[rank]}"
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


@dataclass
class SmallStrainMaterialQuadratureResponse:
    """Rollback-safe local state and fields for a generic small-strain material.

    The response stores the provider's native state schema and its unambiguous
    Cauchy/small-strain tangent.  A global Newton procedure may consume these
    fields without treating the model as finite strain or knowing which
    framework executed the local update.
    """

    state: MaterialQuadratureState
    cauchy_stress: QuadratureField
    tangent: QuadratureField
    stored_energy_density: QuadratureField | None = None
    dissipation_density_increment: QuadratureField | None = None
    stored_energy_density_components: dict[str, QuadratureField] = field(
        default_factory=dict
    )
    last_diagnostics: tuple[Mapping[str, object], ...] = field(
        default_factory=tuple,
        init=False,
    )
    last_applicability: tuple[str, ...] = field(default_factory=tuple, init=False)

    @classmethod
    def create(
        cls,
        domain,
        state_schema,
        *,
        degree: int = 2,
        scheme: str = "default",
        stored_energy: bool = False,
        dissipation: bool = False,
        stored_energy_component_names=(),
    ):
        state = MaterialQuadratureState.create(
            domain, state_schema, degree=degree, scheme=scheme
        )
        common = {"degree": int(degree), "scheme": str(scheme)}
        component_names = tuple(
            dict.fromkeys(
                str(name).strip().upper() for name in stored_energy_component_names
            )
        )
        if any(not name for name in component_names):
            raise ValueError("Stored-energy component names must be nonempty.")
        if component_names and not stored_energy:
            raise ValueError("Stored-energy components require stored_energy=True.")
        return cls(
            state=state,
            cauchy_stress=QuadratureField.create(
                domain, name="S", value_shape=(3, 3), **common
            ),
            tangent=QuadratureField.create(
                domain, name="DDSDDE", value_shape=(3, 3, 3, 3), **common
            ),
            stored_energy_density=(
                QuadratureField.create(domain, name="SENER", **common)
                if stored_energy
                else None
            ),
            dissipation_density_increment=(
                QuadratureField.create(domain, name="DENER_INC", **common)
                if dissipation
                else None
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
        material: SmallStrainUserMaterial,
        *,
        strain_old,
        strain_new,
        time: float,
        time_increment: float,
        parameters=None,
        temperature=None,
        temperature_increment=None,
        field_variables=None,
        commit: bool = False,
    ) -> SmallStrainMaterialPointBatchOutput:
        """Update all local integration points as one collective transaction."""

        comm = self.domain.comm
        result = None
        local_problem = None
        self.state.begin()
        try:
            committed = self.state.committed_state_vectors()
            count = len(committed)
            selected_parameters = (
                getattr(material, "parameters", {})
                if parameters is None
                else parameters
            )
            request = SmallStrainMaterialPointBatchInput(
                strain_old=_small_strain_point_array(
                    strain_old, point_count=count, label="strain_old"
                ),
                strain_new=_small_strain_point_array(
                    strain_new, point_count=count, label="strain_new"
                ),
                time=time,
                time_increment=time_increment,
                parameters=selected_parameters,
                state_old=committed,
                state_schema=self.state.state_schema,
                parameter_schema=material.parameter_schema,
                temperature=_optional_point_scalars(
                    temperature, point_count=count, label="temperature"
                ),
                temperature_increment=_optional_point_scalars(
                    temperature_increment,
                    point_count=count,
                    label="temperature_increment",
                ),
                field_variables=(
                    {}
                    if field_variables is None
                    else {
                        str(name): _optional_point_scalars(
                            value,
                            point_count=count,
                            label=f"field variable {name!r}",
                        )
                        for name, value in field_variables.items()
                    }
                ),
            )
            result = validated_small_strain_batch_update(material, request)
            expected_components = set(self.stored_energy_density_components)
            if set(result.stored_energy_density_components) != expected_components:
                raise ValueError(
                    "Material response stored-energy components differ from the "
                    "quadrature response contract."
                )
            if (result.stored_energy_density is not None) != (
                self.stored_energy_density is not None
            ):
                raise ValueError(
                    "Material stored-energy response differs from the quadrature "
                    "response contract."
                )
            if (result.dissipation_density_increment is not None) != (
                self.dissipation_density_increment is not None
            ):
                raise ValueError(
                    "Material dissipation response differs from the quadrature "
                    "response contract."
                )
        except Exception as exc:
            local_problem = f"{type(exc).__name__}: {exc}"
        problems = comm.allgather(local_problem)
        if any(problem is not None for problem in problems):
            self.state.rollback()
            rank = next(index for index, problem in enumerate(problems) if problem)
            raise RuntimeError(
                f"Rank {rank}: small-strain material batch update failed: "
                f"{problems[rank]}"
            )

        assignments = [
            (self.cauchy_stress, result.cauchy_stress),
            (
                self.tangent,
                np.asarray(
                    [
                        small_strain_matrix_to_tensor(
                            tangent, result.tangent_convention
                        )
                        for tangent in result.consistent_tangent
                    ]
                ),
            ),
            *(
                ((self.stored_energy_density, result.stored_energy_density),)
                if self.stored_energy_density is not None
                else ()
            ),
            *(
                (
                    (
                        self.dissipation_density_increment,
                        result.dissipation_density_increment,
                    ),
                )
                if self.dissipation_density_increment is not None
                else ()
            ),
            *(
                (
                    field,
                    result.stored_energy_density_components[name],
                )
                for name, field in self.stored_energy_density_components.items()
            ),
        ]
        local_problem = None
        field_backups = {
            id(selected_field): selected_field.function.x.array.copy()
            for selected_field, _ in assignments
        }
        try:
            self.state.assign_trial_state_vectors(result.state_new)
            for selected_field, values in assignments:
                expected = int(selected_field.function.x.array.size)
                supplied = np.asarray(values, dtype=float)
                if supplied.size != expected or not np.all(np.isfinite(supplied)):
                    raise ValueError(
                        f"{selected_field.function.name} requires {expected} finite "
                        f"values, got {supplied.size}."
                    )
            for selected_field, values in assignments:
                selected_field.assign(values)
        except Exception as exc:
            local_problem = f"{type(exc).__name__}: {exc}"
        problems = comm.allgather(local_problem)
        if any(problem is not None for problem in problems):
            self.state.rollback()
            for selected_field, _ in assignments:
                selected_field.function.x.array[:] = field_backups[id(selected_field)]
                selected_field.function.x.scatter_forward()
            rank = next(index for index, problem in enumerate(problems) if problem)
            raise RuntimeError(
                f"Rank {rank}: small-strain quadrature assignment failed: "
                f"{problems[rank]}"
            )
        self.last_diagnostics = result.diagnostics
        self.last_applicability = result.applicability
        if commit:
            self.state.commit()
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
            "kind": "small_strain_material_quadrature_response",
            "state": self.state.summary(),
            "fields": {
                "cauchy_stress": "S",
                "consistent_tangent": "DDSDDE",
                "stored_energy_density": (
                    None if self.stored_energy_density is None else "SENER"
                ),
                "dissipation_density_increment": (
                    None if self.dissipation_density_increment is None else "DENER_INC"
                ),
                "stored_energy_density_components": {
                    name: name for name in self.stored_energy_density_components
                },
            },
            "applicability_counts": {
                status: self.last_applicability.count(status)
                for status in sorted(set(self.last_applicability))
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


def _small_strain_point_array(value, *, point_count: int, label: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.shape == (3, 3):
        selected = np.broadcast_to(selected, (point_count, 3, 3)).copy()
    if selected.shape != (point_count, 3, 3) or not np.all(np.isfinite(selected)):
        raise ValueError(f"{label} must have shape (3, 3) or (points, 3, 3).")
    skew = np.max(np.abs(selected - np.swapaxes(selected, 1, 2)), axis=(1, 2))
    scale = np.maximum(np.linalg.norm(selected, axis=(1, 2)), np.finfo(float).tiny)
    if np.any(skew > 1.0e-10 * scale):
        raise ValueError(f"Every {label} tensor must be symmetric.")
    return 0.5 * (selected + np.swapaxes(selected, 1, 2))


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
            index for index, problem in enumerate(input_problems) if problem is not None
        )
        raise RuntimeError(
            f"Rank {rank}: invalid material-point batch input: {input_problems[rank]}"
        )

    stress = np.empty((point_count, 3, 3), dtype=float)
    tangent = np.empty((point_count, 9, 9), dtype=float)
    state_new = np.empty_like(committed_state)
    energy = np.empty(point_count, dtype=float)
    energy_components: dict[str, np.ndarray] | None = None
    scales = np.empty(point_count, dtype=float)
    material_group_count = 0
    provider_batch_calls = 0
    scalar_fallback_points = 0
    state.begin()
    local_problem = None
    try:
        point_materials = []
        point_inputs = []
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
                point_materials.append(selected_material)
                point_inputs.append(
                    MaterialPointInput(
                        deformation_gradient_old=old_gradients[index],
                        deformation_gradient_new=new_gradients[index],
                        time=float(time),
                        time_increment=float(time_increment),
                        properties=selected_properties,
                        state_old=committed_state[index],
                        state_schema=state.state_schema,
                        temperature=(
                            None if temperatures is None else float(temperatures[index])
                        ),
                        temperature_increment=(
                            None
                            if temperature_increments is None
                            else float(temperature_increments[index])
                        ),
                        field_variables=None if fields is None else fields[index],
                    )
                )
            except Exception as exc:
                local_problem = (
                    f"material input failed at local quadrature point {index}: "
                    f"{type(exc).__name__}: {exc}"
                )
                break
        responses = [None] * point_count
        if local_problem is None:
            groups: dict[int, tuple[UserMaterial, list[int]]] = {}
            for index, selected_material in enumerate(point_materials):
                identity = id(selected_material)
                if identity not in groups:
                    groups[identity] = (selected_material, [])
                groups[identity][1].append(index)
            for selected_material, indices in groups.values():
                material_group_count += 1
                if isinstance(selected_material, BatchedUserMaterial):
                    provider_batch_calls += 1
                else:
                    scalar_fallback_points += len(indices)
                try:
                    batch_response = validated_material_batch_update(
                        selected_material,
                        MaterialPointBatchInput(
                            tuple(point_inputs[index] for index in indices)
                        ),
                    )
                    for index, response in zip(
                        indices,
                        batch_response.responses,
                        strict=True,
                    ):
                        responses[index] = response
                except Exception as exc:
                    local_problem = (
                        "material batch update failed for local quadrature "
                        f"points {indices[0]}..{indices[-1]}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    break
        if local_problem is None:
            for index, response in enumerate(responses):
                selected_material = point_materials[index]
                if response is None:
                    local_problem = (
                        "material batch provider omitted local quadrature "
                        f"point {index}"
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
                if declared_components is not None and set(point_components) != set(
                    declared_components
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
                        "material update changed the stored-energy component "
                        "contract at local quadrature point "
                        f"{index}"
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
                material_group_count=material_group_count,
                provider_batch_calls=provider_batch_calls,
                scalar_fallback_points=scalar_fallback_points,
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
                f"Rank {rank}: material-point batch validation failed: {problems[rank]}"
            )
        state.assign_trial_state_vectors(state_new)
        if commit:
            state.commit()
    except Exception:
        state.rollback()
        raise
    return batch
