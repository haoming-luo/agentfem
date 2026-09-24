# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Solver-neutral small-strain material-point contracts.

The finite-element core owns tensor conventions, state transactions and
validation.  A provider may implement the constitutive update with NumPy,
compiled code, a neural network, or another runtime; none of those execution
choices enter this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from .user_material import (
    MaterialStateSchema,
    MaterialTangentConvention,
)


_VOIGT_ORDER = ("xx", "yy", "zz", "xy", "yz", "xz")
_DOMAIN_STATUSES = {"in_domain", "warning", "out_of_domain", "invalid_state"}
_CHANNEL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _finite_vector(value, *, size: int, label: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float).reshape(-1)
    if selected.shape != (size,) or not np.all(np.isfinite(selected)):
        raise ValueError(f"{label} must contain {size} finite values.")
    return selected.copy()


def _finite_parameters(values: Mapping[str, float]) -> Mapping[str, float]:
    selected: dict[str, float] = {}
    for name, value in dict(values).items():
        key = str(name).strip()
        number = float(value)
        if not key or not np.isfinite(number):
            raise ValueError("Material parameter names and values must be finite.")
        if key in selected:
            raise ValueError(f"Duplicate material parameter {key!r}.")
        selected[key] = number
    return MappingProxyType(selected)


def _finite_metadata(
    values: Mapping[str, object], *, label: str
) -> Mapping[str, object]:
    selected: dict[str, object] = {}
    for name, value in dict(values).items():
        key = str(name).strip()
        if not _CHANNEL_NAME.fullmatch(key):
            raise ValueError(
                f"{label} names must be stable identifiers; received {key!r}."
            )
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{label} {key!r} must be finite.")
        if not isinstance(value, (str, bool, int, float, type(None))):
            raise TypeError(
                f"{label} {key!r} must be a JSON scalar, got {type(value).__name__}."
            )
        selected[key] = value
    return MappingProxyType(selected)


@dataclass(frozen=True)
class SmallStrainMaterialPointInput:
    """One small-strain constitutive increment with named parameters."""

    strain_old: np.ndarray
    strain_new: np.ndarray
    time: float
    time_increment: float
    parameters: Mapping[str, float]
    state_old: np.ndarray
    state_schema: MaterialStateSchema
    temperature: float | None = None
    temperature_increment: float | None = None
    field_variables: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "strain_old",
            _finite_vector(self.strain_old, size=6, label="strain_old"),
        )
        object.__setattr__(
            self,
            "strain_new",
            _finite_vector(self.strain_new, size=6, label="strain_new"),
        )
        if not np.isfinite(self.time) or not np.isfinite(self.time_increment):
            raise ValueError("Material-point time values must be finite.")
        if self.time_increment <= 0.0:
            raise ValueError("Material-point time_increment must be positive.")
        if not isinstance(self.state_schema, MaterialStateSchema):
            raise TypeError("state_schema must be a MaterialStateSchema.")
        object.__setattr__(
            self,
            "state_old",
            self.state_schema.validate(self.state_old, label="state_old"),
        )
        object.__setattr__(self, "parameters", _finite_parameters(self.parameters))
        for name in ("temperature", "temperature_increment"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when provided.")
        if self.field_variables is not None:
            fields = np.asarray(self.field_variables, dtype=float).reshape(-1)
            if not np.all(np.isfinite(fields)):
                raise ValueError("field_variables must contain finite values.")
            object.__setattr__(self, "field_variables", fields.copy())


@dataclass(frozen=True)
class SmallStrainMaterialPointOutput:
    """Stress, discrete tangent, state and trust diagnostics for one point."""

    cauchy_stress: np.ndarray
    consistent_tangent: np.ndarray
    state_new: np.ndarray
    tangent_convention: MaterialTangentConvention
    state_schema: MaterialStateSchema
    stored_energy_density: float | None = None
    dissipated_energy_density: float | None = None
    energy_density_components: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    suggested_time_scale: float = 1.0
    applicability_status: str = "in_domain"

    def __post_init__(self) -> None:
        stress = _finite_vector(self.cauchy_stress, size=6, label="cauchy_stress")
        if not isinstance(self.tangent_convention, MaterialTangentConvention):
            raise TypeError("tangent_convention must be a MaterialTangentConvention.")
        convention = self.tangent_convention
        if (
            convention.stress_measure != "cauchy"
            or convention.kinematic_measure != "small_strain"
            or convention.storage != "matrix_6x6"
        ):
            raise ValueError(
                "Small-strain output requires a Cauchy/small-strain 6x6 tangent."
            )
        tangent = convention.validate(self.consistent_tangent)
        if not isinstance(self.state_schema, MaterialStateSchema):
            raise TypeError("state_schema must be a MaterialStateSchema.")
        state = self.state_schema.validate(self.state_new, label="state_new")
        for name in ("stored_energy_density", "dissipated_energy_density"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when provided.")
        components: dict[str, float] = {}
        for name, value in dict(self.energy_density_components).items():
            key = str(name).strip()
            number = float(value)
            if not _CHANNEL_NAME.fullmatch(key) or not np.isfinite(number):
                raise ValueError(
                    "Energy component names must be stable identifiers and values "
                    "must be finite."
                )
            components[key] = number
        if not np.isfinite(self.suggested_time_scale) or self.suggested_time_scale <= 0:
            raise ValueError("suggested_time_scale must be finite and positive.")
        status = str(self.applicability_status).strip().lower()
        if status not in _DOMAIN_STATUSES:
            raise ValueError(
                f"applicability_status must be one of {sorted(_DOMAIN_STATUSES)!r}."
            )
        object.__setattr__(self, "cauchy_stress", stress)
        object.__setattr__(self, "consistent_tangent", tangent)
        object.__setattr__(self, "state_new", state)
        object.__setattr__(
            self, "energy_density_components", MappingProxyType(components)
        )
        object.__setattr__(
            self,
            "diagnostics",
            _finite_metadata(self.diagnostics, label="diagnostic"),
        )
        object.__setattr__(self, "applicability_status", status)

    def stress_tensor(self) -> np.ndarray:
        """Return the symmetric 3x3 stress tensor in physical components."""

        xx, yy, zz, xy, yz, xz = self.cauchy_stress
        return np.asarray(((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)))

    def summary(self) -> dict[str, object]:
        return {
            "kind": "small_strain_material_point_output",
            "state_schema": self.state_schema.summary(),
            "tangent_convention": self.tangent_convention.summary(),
            "applicability_status": self.applicability_status,
            "suggested_time_scale": self.suggested_time_scale,
            "diagnostics": dict(self.diagnostics),
            "energy_density_components": dict(self.energy_density_components),
        }


@dataclass(frozen=True)
class SmallStrainMaterialBatchInput:
    """Vectorized input for all local integration points on one MPI rank."""

    strain_old: np.ndarray
    strain_new: np.ndarray
    time: float
    time_increment: float
    parameters: Mapping[str, float]
    state_old: np.ndarray
    state_schema: MaterialStateSchema
    temperature: np.ndarray | None = None
    temperature_increment: np.ndarray | None = None
    field_variables: np.ndarray | None = None

    def __post_init__(self) -> None:
        old = np.asarray(self.strain_old, dtype=float)
        new = np.asarray(self.strain_new, dtype=float)
        state = np.asarray(self.state_old, dtype=float)
        if old.ndim != 2 or old.shape[1:] != (6,) or not np.all(np.isfinite(old)):
            raise ValueError("strain_old must have shape (points, 6) and be finite.")
        if new.shape != old.shape or not np.all(np.isfinite(new)):
            raise ValueError("strain_new must match strain_old and be finite.")
        if not isinstance(self.state_schema, MaterialStateSchema):
            raise TypeError("state_schema must be a MaterialStateSchema.")
        if state.shape != (len(old), self.state_schema.size):
            raise ValueError("state_old must have shape (points, state_schema.size).")
        if not np.all(np.isfinite(state)):
            raise ValueError("state_old must be finite.")
        if not np.isfinite(self.time) or not np.isfinite(self.time_increment):
            raise ValueError("Batch time values must be finite.")
        if self.time_increment <= 0:
            raise ValueError("Batch time_increment must be positive.")
        object.__setattr__(self, "strain_old", old.copy())
        object.__setattr__(self, "strain_new", new.copy())
        object.__setattr__(self, "state_old", state.copy())
        object.__setattr__(self, "parameters", _finite_parameters(self.parameters))
        for name in ("temperature", "temperature_increment"):
            value = getattr(self, name)
            if value is None:
                continue
            selected = np.asarray(value, dtype=float).reshape(-1)
            if len(selected) != len(old) or not np.all(np.isfinite(selected)):
                raise ValueError(f"{name} must contain one finite value per point.")
            object.__setattr__(self, name, selected.copy())
        if self.field_variables is not None:
            fields = np.asarray(self.field_variables, dtype=float)
            if (
                fields.ndim != 2
                or len(fields) != len(old)
                or not np.all(np.isfinite(fields))
            ):
                raise ValueError(
                    "field_variables must have shape (points, variables) and be finite."
                )
            object.__setattr__(self, "field_variables", fields.copy())

    @property
    def point_count(self) -> int:
        return len(self.strain_new)

    def point(self, index: int) -> SmallStrainMaterialPointInput:
        return SmallStrainMaterialPointInput(
            strain_old=self.strain_old[index],
            strain_new=self.strain_new[index],
            time=self.time,
            time_increment=self.time_increment,
            parameters=self.parameters,
            state_old=self.state_old[index],
            state_schema=self.state_schema,
            temperature=None
            if self.temperature is None
            else float(self.temperature[index]),
            temperature_increment=(
                None
                if self.temperature_increment is None
                else float(self.temperature_increment[index])
            ),
            field_variables=(
                None if self.field_variables is None else self.field_variables[index]
            ),
        )


@dataclass(frozen=True)
class SmallStrainMaterialBatchOutput:
    """Validated vectorized constitutive response with no partial acceptance."""

    cauchy_stress: np.ndarray
    consistent_tangent: np.ndarray
    state_new: np.ndarray
    tangent_convention: MaterialTangentConvention
    state_schema: MaterialStateSchema
    stored_energy_density: np.ndarray
    dissipated_energy_density: np.ndarray
    suggested_time_scale: np.ndarray
    applicability_status: tuple[str, ...]
    energy_density_components: Mapping[str, np.ndarray] = field(default_factory=dict)
    diagnostics: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stress = np.asarray(self.cauchy_stress, dtype=float)
        tangent = np.asarray(self.consistent_tangent, dtype=float)
        state = np.asarray(self.state_new, dtype=float)
        count = len(stress)
        if stress.shape != (count, 6) or not np.all(np.isfinite(stress)):
            raise ValueError("Batch Cauchy stress must have shape (points, 6).")
        if tangent.shape != (count, 6, 6) or not np.all(np.isfinite(tangent)):
            raise ValueError("Batch tangent must have shape (points, 6, 6).")
        if not isinstance(self.tangent_convention, MaterialTangentConvention):
            raise TypeError("tangent_convention must be a MaterialTangentConvention.")
        if (
            self.tangent_convention.stress_measure != "cauchy"
            or self.tangent_convention.kinematic_measure != "small_strain"
            or self.tangent_convention.storage != "matrix_6x6"
        ):
            raise ValueError("Batch output requires a Cauchy/small-strain 6x6 tangent.")
        if not isinstance(self.state_schema, MaterialStateSchema):
            raise TypeError("state_schema must be a MaterialStateSchema.")
        if state.shape != (count, self.state_schema.size) or not np.all(
            np.isfinite(state)
        ):
            raise ValueError("Batch state must have shape (points, state_schema.size).")
        stored = np.asarray(self.stored_energy_density, dtype=float).reshape(-1)
        dissipated = np.asarray(self.dissipated_energy_density, dtype=float).reshape(-1)
        scales = np.asarray(self.suggested_time_scale, dtype=float).reshape(-1)
        if any(len(values) != count for values in (stored, dissipated, scales)):
            raise ValueError(
                "Batch energy and time-scale arrays require one value per point."
            )
        if not all(
            np.all(np.isfinite(values)) for values in (stored, dissipated, scales)
        ):
            raise ValueError("Batch energy and time-scale arrays must be finite.")
        if np.any(scales <= 0):
            raise ValueError("Batch suggested_time_scale values must be positive.")
        statuses = tuple(
            str(value).strip().lower() for value in self.applicability_status
        )
        if len(statuses) != count or any(
            value not in _DOMAIN_STATUSES for value in statuses
        ):
            raise ValueError("Batch applicability status is incomplete or invalid.")
        components: dict[str, np.ndarray] = {}
        for name, values in dict(self.energy_density_components).items():
            key = str(name).strip()
            selected = np.asarray(values, dtype=float).reshape(-1)
            if (
                not _CHANNEL_NAME.fullmatch(key)
                or len(selected) != count
                or not np.all(np.isfinite(selected))
            ):
                raise ValueError(
                    "Batch energy component names must be stable identifiers and "
                    "provide one finite value per point."
                )
            components[key] = selected.copy()
        diagnostics: dict[str, np.ndarray] = {}
        for name, values in dict(self.diagnostics).items():
            key = str(name).strip()
            selected = np.asarray(values)
            if not _CHANNEL_NAME.fullmatch(key) or selected.shape != (count,):
                raise ValueError(
                    "Batch diagnostic names must be stable identifiers and provide "
                    "one scalar value per point."
                )
            if np.issubdtype(selected.dtype, np.number) and not np.all(
                np.isfinite(selected)
            ):
                raise ValueError(f"Batch diagnostic {key!r} must be finite.")
            diagnostics[key] = selected.copy()
        object.__setattr__(self, "cauchy_stress", stress.copy())
        object.__setattr__(self, "consistent_tangent", tangent.copy())
        object.__setattr__(self, "state_new", state.copy())
        object.__setattr__(self, "stored_energy_density", stored.copy())
        object.__setattr__(self, "dissipated_energy_density", dissipated.copy())
        object.__setattr__(self, "suggested_time_scale", scales.copy())
        object.__setattr__(self, "applicability_status", statuses)
        object.__setattr__(
            self, "energy_density_components", MappingProxyType(components)
        )
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))

    @classmethod
    def from_points(
        cls,
        outputs: tuple[SmallStrainMaterialPointOutput, ...],
    ) -> "SmallStrainMaterialBatchOutput":
        if not outputs:
            raise ValueError("A material batch must contain at least one point.")
        convention = outputs[0].tangent_convention
        schema = outputs[0].state_schema
        if any(item.tangent_convention != convention for item in outputs):
            raise ValueError("All batch points must use one tangent convention.")
        if any(item.state_schema.summary() != schema.summary() for item in outputs):
            raise ValueError("All batch points must use one state schema.")
        names = sorted(set().union(*(item.diagnostics for item in outputs)))
        diagnostics = {
            name: np.asarray([item.diagnostics.get(name, np.nan) for item in outputs])
            for name in names
        }
        if any(
            np.any(~np.isfinite(value.astype(float))) for value in diagnostics.values()
        ):
            diagnostics = {}
        component_names = sorted(
            set().union(*(item.energy_density_components for item in outputs))
        )
        energy_components = {
            name: np.asarray(
                [item.energy_density_components.get(name, 0.0) for item in outputs]
            )
            for name in component_names
        }
        return cls(
            cauchy_stress=np.stack([item.cauchy_stress for item in outputs]),
            consistent_tangent=np.stack([item.consistent_tangent for item in outputs]),
            state_new=np.stack([item.state_new for item in outputs]),
            tangent_convention=convention,
            state_schema=schema,
            stored_energy_density=np.asarray(
                [
                    0.0
                    if item.stored_energy_density is None
                    else item.stored_energy_density
                    for item in outputs
                ]
            ),
            dissipated_energy_density=np.asarray(
                [
                    0.0
                    if item.dissipated_energy_density is None
                    else item.dissipated_energy_density
                    for item in outputs
                ]
            ),
            suggested_time_scale=np.asarray(
                [item.suggested_time_scale for item in outputs]
            ),
            applicability_status=tuple(item.applicability_status for item in outputs),
            energy_density_components=energy_components,
            diagnostics=diagnostics,
        )

    @property
    def point_count(self) -> int:
        return len(self.cauchy_stress)


@runtime_checkable
class SmallStrainUserMaterial(Protocol):
    """Protocol for a pure small-strain discrete material update."""

    name: str
    state_schema: MaterialStateSchema
    tangent_convention: MaterialTangentConvention

    def update(
        self,
        point: SmallStrainMaterialPointInput,
    ) -> SmallStrainMaterialPointOutput: ...


@runtime_checkable
class BatchedSmallStrainUserMaterial(SmallStrainUserMaterial, Protocol):
    """Optional vectorized material interface used by production FE paths."""

    def update_batch(
        self,
        request: SmallStrainMaterialBatchInput,
    ) -> SmallStrainMaterialBatchOutput: ...


def validated_small_strain_update(
    material: SmallStrainUserMaterial,
    point: SmallStrainMaterialPointInput,
) -> SmallStrainMaterialPointOutput:
    if not isinstance(material, SmallStrainUserMaterial):
        raise TypeError(
            "Small-strain material must declare name, state_schema, "
            "tangent_convention and update()."
        )
    if point.state_schema.summary() != material.state_schema.summary():
        raise ValueError("Material-point state schema does not match the material.")
    response = material.update(point)
    if not isinstance(response, SmallStrainMaterialPointOutput):
        raise TypeError("Small-strain update() returned the wrong response type.")
    if response.state_schema.summary() != material.state_schema.summary():
        raise ValueError("Material response changed the state schema.")
    if response.tangent_convention != material.tangent_convention:
        raise ValueError("Material response changed the tangent convention.")
    return response


def update_small_strain_material_batch(
    material: SmallStrainUserMaterial,
    request: SmallStrainMaterialBatchInput,
) -> SmallStrainMaterialBatchOutput:
    """Execute one atomic rank-local batch, with a scalar compatibility fallback."""

    if request.state_schema.summary() != material.state_schema.summary():
        raise ValueError("Batch state schema does not match the material.")
    update_batch = getattr(material, "update_batch", None)
    if callable(update_batch):
        response = update_batch(request)
        if not isinstance(response, SmallStrainMaterialBatchOutput):
            raise TypeError("update_batch() returned the wrong response type.")
        if response.point_count != request.point_count:
            raise ValueError("Batch material response changed the point count.")
        if response.state_schema.summary() != material.state_schema.summary():
            raise ValueError("Batch material response changed the state schema.")
        if response.tangent_convention != material.tangent_convention:
            raise ValueError("Batch material response changed the tangent convention.")
        return response
    return SmallStrainMaterialBatchOutput.from_points(
        tuple(
            validated_small_strain_update(material, request.point(i))
            for i in range(request.point_count)
        )
    )


@dataclass(frozen=True)
class SmallStrainMaterialTangentCheck:
    relative_error: float
    maximum_absolute_error: float
    relative_step: float
    tolerance: float
    accepted: bool
    convention: MaterialTangentConvention

    def summary(self) -> dict[str, object]:
        return {
            "kind": "small_strain_material_tangent_check",
            "method": "central_difference_fixed_old_state",
            "relative_error": self.relative_error,
            "maximum_absolute_error": self.maximum_absolute_error,
            "relative_step": self.relative_step,
            "tolerance": self.tolerance,
            "accepted": self.accepted,
            "convention": self.convention.summary(),
        }


def check_small_strain_material_tangent(
    material: SmallStrainUserMaterial,
    point: SmallStrainMaterialPointInput,
    *,
    relative_step: float = 1.0e-7,
    tolerance: float = 1.0e-5,
) -> SmallStrainMaterialTangentCheck:
    """Check ``d sigma / d epsilon`` while holding old state fixed."""

    step = float(relative_step)
    selected_tolerance = float(tolerance)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("relative_step must be finite and positive.")
    if not np.isfinite(selected_tolerance) or selected_tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    baseline = validated_small_strain_update(material, point)
    numerical = np.empty((6, 6), dtype=float)
    for column in range(6):
        increment = step * max(1.0, abs(float(point.strain_new[column])))
        plus = point.strain_new.copy()
        minus = point.strain_new.copy()
        plus[column] += increment
        minus[column] -= increment
        plus_response = validated_small_strain_update(
            material, replace(point, strain_new=plus)
        )
        minus_response = validated_small_strain_update(
            material, replace(point, strain_new=minus)
        )
        numerical[:, column] = (
            plus_response.cauchy_stress - minus_response.cauchy_stress
        ) / (2.0 * increment)
    difference = baseline.consistent_tangent - numerical
    denominator = max(float(np.linalg.norm(numerical)), np.finfo(float).tiny)
    error = float(np.linalg.norm(difference) / denominator)
    return SmallStrainMaterialTangentCheck(
        relative_error=error,
        maximum_absolute_error=float(np.max(np.abs(difference), initial=0.0)),
        relative_step=step,
        tolerance=selected_tolerance,
        accepted=error <= selected_tolerance,
        convention=baseline.tangent_convention,
    )


def voigt_stress_to_tensor(values) -> np.ndarray:
    xx, yy, zz, xy, yz, xz = _finite_vector(values, size=6, label="stress")
    return np.asarray(((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)))


__all__ = [
    "BatchedSmallStrainUserMaterial",
    "SmallStrainMaterialBatchInput",
    "SmallStrainMaterialBatchOutput",
    "SmallStrainMaterialPointInput",
    "SmallStrainMaterialPointOutput",
    "SmallStrainMaterialTangentCheck",
    "SmallStrainUserMaterial",
    "check_small_strain_material_tangent",
    "update_small_strain_material_batch",
    "validated_small_strain_update",
    "voigt_stress_to_tensor",
]
