"""Solver-neutral quadrature driver for material-point providers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quadrature import MaterialQuadratureState
from .quadrature import QuadratureField
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
        object.__setattr__(self, "cauchy_stress", stress.copy())
        object.__setattr__(self, "consistent_tangent", tangent.copy())
        object.__setattr__(self, "state_new", state.copy())
        object.__setattr__(self, "strain_energy_density", energy.copy())
        object.__setattr__(self, "suggested_time_scale", scale.copy())

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
        }


@dataclass
class MaterialQuadratureResponse:
    """Quadrature stress/tangent fields sharing one typed state transaction."""

    state: MaterialQuadratureState
    first_piola_stress: QuadratureField
    cauchy_stress: QuadratureField
    tangent: QuadratureField
    strain_energy_density: QuadratureField

    @classmethod
    def create(cls, domain, state_schema, *, degree: int = 2, scheme: str = "default"):
        state = MaterialQuadratureState.create(
            domain,
            state_schema,
            degree=degree,
            scheme=scheme,
        )
        common = {"degree": int(degree), "scheme": str(scheme)}
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
        )

    @property
    def domain(self):
        return self.state.domain

    @property
    def measure(self):
        return self.state.measure

    def update(
        self,
        material: UserMaterial,
        *,
        deformation_gradient_old,
        deformation_gradient_new,
        time: float,
        time_increment: float,
        properties=(),
        commit: bool = False,
    ) -> MaterialPointBatchResult:
        result = update_material_points(
            material,
            self.state,
            deformation_gradient_old=deformation_gradient_old,
            deformation_gradient_new=deformation_gradient_new,
            time=time,
            time_increment=time_increment,
            properties=properties,
            commit=commit,
        )
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
        self.first_piola_stress.assign(first_piola)
        self.cauchy_stress.assign(result.cauchy_stress)
        self.tangent.assign(result.consistent_tangent.reshape((-1, 3, 3, 3, 3)))
        self.strain_energy_density.assign(result.strain_energy_density)
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
            },
        }


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
    material: UserMaterial,
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

    if state.state_schema.identity != material.state_schema.identity:
        raise ValueError("Quadrature and material state schemas do not match.")
    committed_state = state.committed_state_vectors()
    point_count = len(committed_state)
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
    fields = None if field_variables is None else np.asarray(field_variables, dtype=float)
    if fields is not None:
        if fields.ndim == 1:
            fields = np.broadcast_to(fields, (point_count, len(fields))).copy()
        if fields.ndim != 2 or len(fields) != point_count or not np.all(np.isfinite(fields)):
            raise ValueError(
                "field_variables must be one vector or one vector per point."
            )

    stress = np.empty((point_count, 3, 3), dtype=float)
    tangent = np.empty((point_count, 9, 9), dtype=float)
    state_new = np.empty_like(committed_state)
    energy = np.empty(point_count, dtype=float)
    scales = np.empty(point_count, dtype=float)
    state.begin()
    local_problem = None
    try:
        for index in range(point_count):
            try:
                response = validated_material_update(
                    material,
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
            scales[index] = response.suggested_time_scale
        problems = state.domain.comm.allgather(local_problem)
        if any(problem is not None for problem in problems):
            rank = next(
                index for index, problem in enumerate(problems) if problem is not None
            )
            raise RuntimeError(f"Rank {rank}: {problems[rank]}")
        state.assign_trial_state_vectors(state_new)
        if commit:
            state.commit()
    except Exception:
        state.rollback()
        raise
    return MaterialPointBatchResult(
        cauchy_stress=stress,
        consistent_tangent=tangent,
        state_new=state_new,
        strain_energy_density=energy,
        suggested_time_scale=scales,
        committed=bool(commit),
    )
