# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Framework-neutral small-strain material-point contracts.

The finite-element core owns the mechanics-facing boundary: named parameters,
state identity, stress/tangent conventions, atomic batch semantics, energy and
applicability diagnostics.  A native, learned, or externally adapted material
owns the local update itself.  No machine-learning runtime is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from .user_material import (
    MaterialStateSchema,
    MaterialTangentConvention,
)


_COMPONENT_ORDER = ("xx", "yy", "zz", "xy", "yz", "xz")
_APPLICABILITY = {"in_domain", "warning", "out_of_domain", "invalid_state"}


def small_strain_tangent_convention(
    *, shear_convention: str = "tensor"
) -> MaterialTangentConvention:
    """Return the canonical 3D Cauchy/small-strain matrix convention."""

    return MaterialTangentConvention(
        stress_measure="cauchy",
        kinematic_measure="small_strain",
        configuration="reference",
        storage="matrix_6x6",
        component_order=_COMPONENT_ORDER,
        shear_convention=shear_convention,
        symmetric=True,
    )


@dataclass(frozen=True)
class MaterialParameter:
    """One named, unit-aware constitutive parameter."""

    name: str
    unit: str | None = None
    lower: float | None = None
    upper: float | None = None
    default: float | None = None
    description: str = "Material parameter."

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name or not name.replace("_", "a").isalnum() or not name[0].isalpha():
            raise ValueError(
                "Material parameter names must start with a letter and contain "
                "only letters, digits, or underscores."
            )
        lower = None if self.lower is None else float(self.lower)
        upper = None if self.upper is None else float(self.upper)
        default = None if self.default is None else float(self.default)
        if any(
            value is not None and not np.isfinite(value)
            for value in (lower, upper, default)
        ):
            raise ValueError("Material parameter bounds and defaults must be finite.")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(
                "Material parameter lower bound cannot exceed upper bound."
            )
        if default is not None:
            self.validate(default)
        if not str(self.description).strip():
            raise ValueError("Material parameter description must not be empty.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "unit", None if self.unit is None else str(self.unit))
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "default", default)

    @property
    def required(self) -> bool:
        return self.default is None

    def validate(self, value) -> float:
        selected = float(value)
        if not np.isfinite(selected):
            raise ValueError(f"Material parameter {self.name!r} must be finite.")
        if self.lower is not None and selected < self.lower:
            raise ValueError(
                f"Material parameter {self.name!r} must be >= {self.lower}."
            )
        if self.upper is not None and selected > self.upper:
            raise ValueError(
                f"Material parameter {self.name!r} must be <= {self.upper}."
            )
        return selected

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "lower": self.lower,
            "upper": self.upper,
            "default": self.default,
            "required": self.required,
            "description": self.description,
        }


@dataclass(frozen=True)
class MaterialParameterSchema:
    """Stable named parameter layout for native and external materials."""

    name: str
    parameters: tuple[MaterialParameter, ...] = ()
    version: str = "0.1.0"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        version = str(self.version).strip()
        parameters = tuple(self.parameters)
        if not name or not version:
            raise ValueError("Material parameter schema name and version are required.")
        if any(not isinstance(item, MaterialParameter) for item in parameters):
            raise TypeError("Parameter schemas must contain MaterialParameter records.")
        names = tuple(item.name for item in parameters)
        if len(set(names)) != len(names):
            raise ValueError("Material parameter names must be unique.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "parameters", parameters)

    @property
    def identity(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.summary(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return f"sha256:{sha256(encoded).hexdigest()}"

    def validate(self, values: Mapping[str, object]) -> Mapping[str, float]:
        if not isinstance(values, Mapping):
            raise TypeError("Material parameters must be supplied as a named mapping.")
        supplied = {str(name): value for name, value in values.items()}
        expected = {item.name for item in self.parameters}
        unknown = set(supplied) - expected
        if unknown:
            raise ValueError(f"Unknown material parameters: {sorted(unknown)!r}.")
        normalized: dict[str, float] = {}
        for item in self.parameters:
            if item.name in supplied:
                normalized[item.name] = item.validate(supplied[item.name])
            elif item.default is not None:
                normalized[item.name] = item.default
            else:
                raise ValueError(
                    f"Required material parameter {item.name!r} is missing."
                )
        return MappingProxyType(normalized)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "material_parameter_schema",
            "name": self.name,
            "version": self.version,
            "identity": self.identity,
            "parameters": tuple(item.summary() for item in self.parameters),
        }


@dataclass(frozen=True)
class SmallStrainMaterialPointInput:
    """One three-dimensional small-strain constitutive update request."""

    strain_old: np.ndarray
    strain_new: np.ndarray
    time: float
    time_increment: float
    parameters: Mapping[str, float]
    state_old: np.ndarray
    state_schema: MaterialStateSchema
    parameter_schema: MaterialParameterSchema
    temperature: float | None = None
    temperature_increment: float | None = None
    field_variables: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("strain_old", "strain_new"):
            value = _symmetric_tensor(getattr(self, name), label=name)
            object.__setattr__(self, name, value)
        if not np.isfinite(self.time) or not np.isfinite(self.time_increment):
            raise ValueError("Material-point time values must be finite.")
        if self.time_increment <= 0.0:
            raise ValueError("Material-point time_increment must be positive.")
        if not isinstance(self.state_schema, MaterialStateSchema):
            raise TypeError("state_schema must be a MaterialStateSchema.")
        if not isinstance(self.parameter_schema, MaterialParameterSchema):
            raise TypeError("parameter_schema must be a MaterialParameterSchema.")
        object.__setattr__(
            self,
            "state_old",
            self.state_schema.validate(self.state_old, label="state_old"),
        )
        object.__setattr__(
            self, "parameters", self.parameter_schema.validate(self.parameters)
        )
        for name in ("temperature", "temperature_increment"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when provided.")
        object.__setattr__(
            self,
            "field_variables",
            _finite_scalar_mapping(self.field_variables, label="field_variables"),
        )


@dataclass(frozen=True)
class SmallStrainMaterialPointOutput:
    """Stress, consistent tangent, state, energy and diagnostics for one point."""

    cauchy_stress: np.ndarray
    consistent_tangent: np.ndarray
    state_new: np.ndarray
    state_schema: MaterialStateSchema
    tangent_convention: MaterialTangentConvention = field(
        default_factory=small_strain_tangent_convention
    )
    stored_energy_density: float | None = None
    stored_energy_density_components: Mapping[str, float] = field(default_factory=dict)
    dissipation_density_increment: float | None = None
    energy_balance_residual: float | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    applicability: str = "in_domain"
    suggested_time_scale: float = 1.0

    def __post_init__(self) -> None:
        stress = _symmetric_tensor(self.cauchy_stress, label="cauchy_stress")
        if not isinstance(self.state_schema, MaterialStateSchema):
            raise TypeError("state_schema must be a MaterialStateSchema.")
        state = self.state_schema.validate(self.state_new, label="state_new")
        convention = self.tangent_convention
        if not isinstance(convention, MaterialTangentConvention):
            raise TypeError("tangent_convention must be a MaterialTangentConvention.")
        _require_small_strain_convention(convention)
        tangent = convention.validate(self.consistent_tangent)
        applicability = str(self.applicability).strip().lower()
        if applicability not in _APPLICABILITY:
            raise ValueError(
                f"Unknown applicability {applicability!r}; expected "
                f"{sorted(_APPLICABILITY)!r}."
            )
        scale = float(self.suggested_time_scale)
        if not np.isfinite(scale) or not 0.0 < scale <= 1.0:
            raise ValueError("suggested_time_scale must be in the interval (0, 1].")
        stored = _optional_finite(self.stored_energy_density, "stored_energy_density")
        dissipation = _optional_finite(
            self.dissipation_density_increment,
            "dissipation_density_increment",
        )
        balance = _optional_finite(
            self.energy_balance_residual, "energy_balance_residual"
        )
        components = _finite_scalar_mapping(
            self.stored_energy_density_components,
            label="stored_energy_density_components",
            normalize_names=True,
        )
        if components and stored is None:
            raise ValueError("Stored-energy components require stored_energy_density.")
        if components and not np.isclose(
            sum(components.values()),
            stored,
            rtol=2.0e-12,
            atol=2.0e-14 * max(1.0, abs(stored)),
        ):
            raise ValueError(
                "Stored-energy components must sum to stored_energy_density."
            )
        diagnostics = _json_mapping(self.diagnostics, label="diagnostics")
        object.__setattr__(self, "cauchy_stress", stress)
        object.__setattr__(self, "consistent_tangent", tangent)
        object.__setattr__(self, "state_new", state)
        object.__setattr__(self, "applicability", applicability)
        object.__setattr__(self, "suggested_time_scale", scale)
        object.__setattr__(self, "stored_energy_density", stored)
        object.__setattr__(self, "dissipation_density_increment", dissipation)
        object.__setattr__(self, "energy_balance_residual", balance)
        object.__setattr__(self, "stored_energy_density_components", components)
        object.__setattr__(self, "diagnostics", diagnostics)

    def require_usable(self) -> "SmallStrainMaterialPointOutput":
        if self.applicability in {"out_of_domain", "invalid_state"}:
            raise MaterialApplicabilityError(
                self.applicability,
                self.diagnostics.get("message", "Material response is not usable."),
            )
        return self

    def summary(self) -> dict[str, object]:
        return {
            "kind": "small_strain_material_point_output",
            "applicability": self.applicability,
            "suggested_time_scale": self.suggested_time_scale,
            "state_schema": self.state_schema.summary(),
            "tangent_convention": self.tangent_convention.summary(),
            "stored_energy_density_defined": self.stored_energy_density is not None,
            "stored_energy_density_components": tuple(
                self.stored_energy_density_components
            ),
            "dissipation_defined": self.dissipation_density_increment is not None,
            "energy_balance_defined": self.energy_balance_residual is not None,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class SmallStrainMaterialPointBatchInput:
    """Vectorized request for all local points owned by one provider call."""

    strain_old: np.ndarray
    strain_new: np.ndarray
    time: float
    time_increment: float
    parameters: Mapping[str, float]
    state_old: np.ndarray
    state_schema: MaterialStateSchema
    parameter_schema: MaterialParameterSchema
    temperature: np.ndarray | None = None
    temperature_increment: np.ndarray | None = None
    field_variables: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        old = _symmetric_tensor_batch(self.strain_old, label="strain_old")
        new = _symmetric_tensor_batch(self.strain_new, label="strain_new")
        if old.shape != new.shape:
            raise ValueError("strain_old and strain_new batch shapes must match.")
        count = len(old)
        if not np.isfinite(self.time) or not np.isfinite(self.time_increment):
            raise ValueError("Material-point time values must be finite.")
        if self.time_increment <= 0.0:
            raise ValueError("Material-point time_increment must be positive.")
        if not isinstance(self.state_schema, MaterialStateSchema):
            raise TypeError("state_schema must be a MaterialStateSchema.")
        if not isinstance(self.parameter_schema, MaterialParameterSchema):
            raise TypeError("parameter_schema must be a MaterialParameterSchema.")
        states = np.asarray(self.state_old, dtype=float)
        if states.shape != (count, self.state_schema.size) or not np.all(
            np.isfinite(states)
        ):
            raise ValueError(
                "state_old must have shape (points, state_schema.size) and be finite."
            )
        temperatures = _optional_point_values(
            self.temperature, count=count, label="temperature"
        )
        temperature_increments = _optional_point_values(
            self.temperature_increment,
            count=count,
            label="temperature_increment",
        )
        fields = {
            str(name): _point_values(value, count=count, label=f"field {name!r}")
            for name, value in self.field_variables.items()
        }
        object.__setattr__(self, "strain_old", old)
        object.__setattr__(self, "strain_new", new)
        object.__setattr__(self, "state_old", states.copy())
        object.__setattr__(
            self, "parameters", self.parameter_schema.validate(self.parameters)
        )
        object.__setattr__(self, "temperature", temperatures)
        object.__setattr__(self, "temperature_increment", temperature_increments)
        object.__setattr__(self, "field_variables", MappingProxyType(fields))

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
            parameter_schema=self.parameter_schema,
            temperature=None if self.temperature is None else self.temperature[index],
            temperature_increment=(
                None
                if self.temperature_increment is None
                else self.temperature_increment[index]
            ),
            field_variables={
                name: values[index] for name, values in self.field_variables.items()
            },
        )


@dataclass(frozen=True)
class SmallStrainMaterialPointBatchOutput:
    """Validated vectorized constitutive response."""

    cauchy_stress: np.ndarray
    consistent_tangent: np.ndarray
    state_new: np.ndarray
    state_schema: MaterialStateSchema
    tangent_convention: MaterialTangentConvention = field(
        default_factory=small_strain_tangent_convention
    )
    stored_energy_density: np.ndarray | None = None
    stored_energy_density_components: Mapping[str, np.ndarray] = field(
        default_factory=dict
    )
    dissipation_density_increment: np.ndarray | None = None
    energy_balance_residual: np.ndarray | None = None
    diagnostics: tuple[Mapping[str, object], ...] = ()
    applicability: tuple[str, ...] = ()
    suggested_time_scale: np.ndarray | float = 1.0

    def __post_init__(self) -> None:
        stress = _symmetric_tensor_batch(self.cauchy_stress, label="cauchy_stress")
        count = len(stress)
        if not isinstance(self.state_schema, MaterialStateSchema):
            raise TypeError("state_schema must be a MaterialStateSchema.")
        state = np.asarray(self.state_new, dtype=float)
        if state.shape != (count, self.state_schema.size) or not np.all(
            np.isfinite(state)
        ):
            raise ValueError("state_new batch shape or values are invalid.")
        convention = self.tangent_convention
        if not isinstance(convention, MaterialTangentConvention):
            raise TypeError("tangent_convention must be a MaterialTangentConvention.")
        _require_small_strain_convention(convention)
        tangent = np.asarray(self.consistent_tangent, dtype=float)
        if tangent.shape != (count, *convention.array_shape) or not np.all(
            np.isfinite(tangent)
        ):
            raise ValueError("consistent_tangent must have shape (points, 6, 6).")
        scale = _point_values(
            self.suggested_time_scale,
            count=count,
            label="suggested_time_scale",
        )
        if np.any(scale <= 0.0) or np.any(scale > 1.0):
            raise ValueError("suggested_time_scale values must lie in (0, 1].")
        stored = _optional_point_values(
            self.stored_energy_density,
            count=count,
            label="stored_energy_density",
        )
        dissipation = _optional_point_values(
            self.dissipation_density_increment,
            count=count,
            label="dissipation_density_increment",
        )
        balance = _optional_point_values(
            self.energy_balance_residual,
            count=count,
            label="energy_balance_residual",
        )
        components: dict[str, np.ndarray] = {}
        for name, values in self.stored_energy_density_components.items():
            key = str(name).strip().upper()
            if not key or key in components:
                raise ValueError(
                    "Stored-energy component names must be nonempty and unique "
                    "after normalization."
                )
            components[key] = _point_values(
                values,
                count=count,
                label=f"stored-energy component {name!r}",
            )
        if components and stored is None:
            raise ValueError("Stored-energy components require stored_energy_density.")
        if components and not np.allclose(
            np.sum(tuple(components.values()), axis=0),
            stored,
            rtol=2.0e-12,
            atol=2.0e-14,
        ):
            raise ValueError(
                "Stored-energy components must sum to stored_energy_density."
            )
        applicability = (
            ("in_domain",) * count
            if not self.applicability
            else tuple(str(item).lower() for item in self.applicability)
        )
        if len(applicability) != count or any(
            item not in _APPLICABILITY for item in applicability
        ):
            raise ValueError("applicability must provide one valid status per point.")
        # Keep the common no-diagnostics case allocation-free.  Creating one
        # empty dictionary per integration point is material at production
        # quadrature counts and conveys no additional information.
        diagnostics = (
            ()
            if not self.diagnostics
            else tuple(
                _json_mapping(item, label=f"diagnostics[{index}]")
                for index, item in enumerate(self.diagnostics)
            )
        )
        if diagnostics and len(diagnostics) != count:
            raise ValueError("diagnostics must provide one mapping per point.")
        object.__setattr__(self, "cauchy_stress", stress)
        object.__setattr__(self, "consistent_tangent", tangent.copy())
        object.__setattr__(self, "state_new", state.copy())
        object.__setattr__(self, "stored_energy_density", stored)
        object.__setattr__(self, "dissipation_density_increment", dissipation)
        object.__setattr__(self, "energy_balance_residual", balance)
        object.__setattr__(self, "stored_energy_density_components", components)
        object.__setattr__(self, "applicability", applicability)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "suggested_time_scale", scale)

    @property
    def point_count(self) -> int:
        return len(self.cauchy_stress)

    @property
    def minimum_suggested_time_scale(self) -> float:
        return float(np.min(self.suggested_time_scale))

    def require_usable(self) -> "SmallStrainMaterialPointBatchOutput":
        rejected = [
            index
            for index, status in enumerate(self.applicability)
            if status in {"out_of_domain", "invalid_state"}
        ]
        if rejected:
            first = rejected[0]
            diagnostic = self.diagnostics[first] if self.diagnostics else {}
            raise MaterialApplicabilityError(
                self.applicability[first],
                f"Material response rejected local point {first}: "
                f"{diagnostic.get('message', 'no diagnostic message')}.",
            )
        return self

    def summary(self) -> dict[str, object]:
        counts = {
            status: self.applicability.count(status)
            for status in sorted(_APPLICABILITY)
        }
        return {
            "kind": "small_strain_material_point_batch_output",
            "point_count": self.point_count,
            "state_size": self.state_new.shape[1],
            "applicability_counts": counts,
            "minimum_suggested_time_scale": self.minimum_suggested_time_scale,
            "state_schema": self.state_schema.summary(),
            "tangent_convention": self.tangent_convention.summary(),
            "stored_energy_density_defined": self.stored_energy_density is not None,
            "stored_energy_density_components": tuple(
                self.stored_energy_density_components
            ),
            "dissipation_defined": self.dissipation_density_increment is not None,
            "energy_balance_defined": self.energy_balance_residual is not None,
        }


class MaterialApplicabilityError(RuntimeError):
    """A material refused to extrapolate or accepted state was invalid."""

    def __init__(self, status: str, message: str) -> None:
        self.code = f"AFM-MATERIAL-{str(status).upper().replace('_', '-')}"
        self.status = str(status)
        super().__init__(f"{self.code}: {message}")


@runtime_checkable
class SmallStrainUserMaterial(Protocol):
    """Scalar update protocol implemented by native or external providers."""

    name: str
    state_schema: MaterialStateSchema
    parameter_schema: MaterialParameterSchema
    tangent_convention: MaterialTangentConvention

    def update(
        self, point: SmallStrainMaterialPointInput
    ) -> SmallStrainMaterialPointOutput:
        """Advance one point from committed old state to trial new state."""


def validated_small_strain_update(
    material: SmallStrainUserMaterial,
    point: SmallStrainMaterialPointInput,
    *,
    require_usable: bool = True,
) -> SmallStrainMaterialPointOutput:
    """Run one update and fail closed on schema or convention drift."""

    if not isinstance(material, SmallStrainUserMaterial):
        raise TypeError(
            "A small-strain material must declare name, state_schema, "
            "parameter_schema, tangent_convention, and update()."
        )
    _require_contract_match(material, point)
    response = material.update(point)
    if not isinstance(response, SmallStrainMaterialPointOutput):
        raise TypeError(
            "Small-strain material update() must return SmallStrainMaterialPointOutput."
        )
    _require_response_match(material, response)
    return response.require_usable() if require_usable else response


def validated_small_strain_batch_update(
    material: SmallStrainUserMaterial,
    request: SmallStrainMaterialPointBatchInput,
    *,
    require_usable: bool = True,
) -> SmallStrainMaterialPointBatchOutput:
    """Use a provider batch kernel when available, otherwise a scalar fallback."""

    if not isinstance(material, SmallStrainUserMaterial):
        raise TypeError("material does not implement SmallStrainUserMaterial.")
    _require_contract_match(material, request)
    update_batch = getattr(material, "update_batch", None)
    if callable(update_batch):
        response = update_batch(request)
        if not isinstance(response, SmallStrainMaterialPointBatchOutput):
            raise TypeError(
                "Small-strain update_batch() must return "
                "SmallStrainMaterialPointBatchOutput."
            )
    else:
        scalar = [
            validated_small_strain_update(
                material,
                request.point(index),
                require_usable=False,
            )
            for index in range(request.point_count)
        ]
        response = _stack_scalar_responses(scalar)
    if response.state_schema.summary() != material.state_schema.summary():
        raise ValueError("Material batch response changed the state schema.")
    if response.tangent_convention != material.tangent_convention:
        raise ValueError("Material batch response changed the tangent convention.")
    if response.point_count != request.point_count:
        raise ValueError("Material batch response changed the point count.")
    return response.require_usable() if require_usable else response


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
    """Check the discrete ``d sigma / d epsilon`` at fixed old state."""

    step = float(relative_step)
    selected_tolerance = float(tolerance)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("relative_step must be finite and positive.")
    if not np.isfinite(selected_tolerance) or selected_tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    baseline = validated_small_strain_update(material, point)
    numerical = np.empty((6, 6), dtype=float)
    base = point.strain_new
    for column, (row, component) in enumerate(
        ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
    ):
        increment = step * max(1.0, abs(float(base[row, component])))
        plus = base.copy()
        minus = base.copy()
        plus[row, component] += increment
        minus[row, component] -= increment
        denominator = 2.0 * increment
        if row != component:
            plus[component, row] += increment
            minus[component, row] -= increment
            if baseline.tangent_convention.shear_convention == "engineering":
                denominator *= 2.0
        plus_response = validated_small_strain_update(
            material, replace(point, strain_new=plus)
        )
        minus_response = validated_small_strain_update(
            material, replace(point, strain_new=minus)
        )
        numerical[:, column] = (
            _symmetric_voigt(plus_response.cauchy_stress)
            - _symmetric_voigt(minus_response.cauchy_stress)
        ) / denominator
    declared = baseline.consistent_tangent
    difference = numerical - declared
    denominator = max(float(np.linalg.norm(numerical)), np.finfo(float).tiny)
    relative_error = float(np.linalg.norm(difference) / denominator)
    maximum = float(np.max(np.abs(difference), initial=0.0))
    return SmallStrainMaterialTangentCheck(
        relative_error=relative_error,
        maximum_absolute_error=maximum,
        relative_step=step,
        tolerance=selected_tolerance,
        accepted=bool(relative_error <= selected_tolerance),
        convention=baseline.tangent_convention,
    )


def small_strain_matrix_to_tensor(
    matrix, convention: MaterialTangentConvention
) -> np.ndarray:
    """Expand a declared 6x6 small-strain tangent to a minor-symmetric tensor."""

    _require_small_strain_convention(convention)
    values = convention.validate(matrix)
    pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
    tensor = np.zeros((3, 3, 3, 3), dtype=float)
    for row, (i, j) in enumerate(pairs):
        for column, (k, l) in enumerate(pairs):
            value = values[row, column]
            if k != l and convention.shear_convention == "tensor":
                value *= 0.5
            tensor[i, j, k, l] = value
            tensor[j, i, k, l] = value
            tensor[i, j, l, k] = value
            tensor[j, i, l, k] = value
    return tensor


def _stack_scalar_responses(
    responses: list[SmallStrainMaterialPointOutput],
) -> SmallStrainMaterialPointBatchOutput:
    if not responses:
        raise ValueError("A material batch must contain at least one point.")
    stored_defined = [item.stored_energy_density is not None for item in responses]
    dissipation_defined = [
        item.dissipation_density_increment is not None for item in responses
    ]
    balance_defined = [item.energy_balance_residual is not None for item in responses]
    if (
        len(set(stored_defined)) > 1
        or len(set(dissipation_defined)) > 1
        or len(set(balance_defined)) > 1
    ):
        raise ValueError("Scalar material responses changed their energy contract.")
    component_names = tuple(responses[0].stored_energy_density_components)
    if any(
        tuple(item.stored_energy_density_components) != component_names
        for item in responses
    ):
        raise ValueError("Scalar material responses changed stored-energy components.")
    return SmallStrainMaterialPointBatchOutput(
        cauchy_stress=np.asarray([item.cauchy_stress for item in responses]),
        consistent_tangent=np.asarray([item.consistent_tangent for item in responses]),
        state_new=np.asarray([item.state_new for item in responses]),
        state_schema=responses[0].state_schema,
        tangent_convention=responses[0].tangent_convention,
        stored_energy_density=(
            np.asarray([item.stored_energy_density for item in responses])
            if stored_defined[0]
            else None
        ),
        stored_energy_density_components={
            name: np.asarray(
                [item.stored_energy_density_components[name] for item in responses]
            )
            for name in component_names
        },
        dissipation_density_increment=(
            np.asarray([item.dissipation_density_increment for item in responses])
            if dissipation_defined[0]
            else None
        ),
        energy_balance_residual=(
            np.asarray([item.energy_balance_residual for item in responses])
            if balance_defined[0]
            else None
        ),
        diagnostics=tuple(item.diagnostics for item in responses),
        applicability=tuple(item.applicability for item in responses),
        suggested_time_scale=np.asarray(
            [item.suggested_time_scale for item in responses]
        ),
    )


def _require_contract_match(material, request) -> None:
    if request.state_schema.summary() != material.state_schema.summary():
        raise ValueError("Material request state schema does not match the material.")
    if request.parameter_schema.summary() != material.parameter_schema.summary():
        raise ValueError(
            "Material request parameter schema does not match the material."
        )
    _require_small_strain_convention(material.tangent_convention)


def _require_response_match(material, response) -> None:
    if response.state_schema.summary() != material.state_schema.summary():
        raise ValueError("Material response changed the state schema.")
    if response.tangent_convention != material.tangent_convention:
        raise ValueError("Material response changed the tangent convention.")


def _require_small_strain_convention(convention) -> None:
    if (
        convention.stress_measure != "cauchy"
        or convention.kinematic_measure != "small_strain"
        or convention.configuration != "reference"
        or convention.storage != "matrix_6x6"
        or convention.component_order != _COMPONENT_ORDER
        or convention.shear_convention not in {"tensor", "engineering"}
    ):
        raise ValueError(
            "Small-strain materials require Cauchy stress, small strain, "
            "reference configuration, matrix_6x6 storage, component order "
            "xx,yy,zz,xy,yz,xz, and an explicit tensor/engineering shear convention."
        )


def _symmetric_tensor(value, *, label: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.shape != (3, 3) or not np.all(np.isfinite(selected)):
        raise ValueError(f"{label} must be a finite 3x3 tensor.")
    scale = max(float(np.linalg.norm(selected)), np.finfo(float).tiny)
    if np.max(np.abs(selected - selected.T)) > 1.0e-10 * scale:
        raise ValueError(f"{label} must be symmetric.")
    return 0.5 * (selected + selected.T)


def _symmetric_tensor_batch(value, *, label: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if (
        selected.ndim != 3
        or selected.shape[1:] != (3, 3)
        or not np.all(np.isfinite(selected))
    ):
        raise ValueError(f"{label} must have shape (points, 3, 3) and be finite.")
    if len(selected) == 0:
        raise ValueError(f"{label} must contain at least one point.")
    skew = np.max(np.abs(selected - np.swapaxes(selected, 1, 2)), axis=(1, 2))
    scale = np.maximum(np.linalg.norm(selected, axis=(1, 2)), np.finfo(float).tiny)
    if np.any(skew > 1.0e-10 * scale):
        raise ValueError(f"Every {label} tensor must be symmetric.")
    return 0.5 * (selected + np.swapaxes(selected, 1, 2))


def _symmetric_voigt(value) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    return selected[(0, 1, 2, 0, 1, 0), (0, 1, 2, 1, 2, 2)]


def _optional_finite(value, label: str) -> float | None:
    if value is None:
        return None
    selected = float(value)
    if not np.isfinite(selected):
        raise ValueError(f"{label} must be finite when provided.")
    return selected


def _finite_scalar_mapping(
    values: Mapping[str, object], *, label: str, normalize_names: bool = False
) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    normalized: dict[str, float] = {}
    for name, value in values.items():
        key = str(name).strip()
        key = key.upper() if normalize_names else key
        if not key or key in normalized:
            raise ValueError(f"{label} names must be nonempty and unique.")
        selected = float(value)
        if not np.isfinite(selected):
            raise ValueError(f"{label}[{key!r}] must be finite.")
        normalized[key] = selected
    return MappingProxyType(normalized)


def _json_mapping(values: Mapping[str, object], *, label: str) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    try:
        normalized = json.loads(
            json.dumps(dict(values), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must contain JSON-shaped finite values.") from exc
    return MappingProxyType(normalized)


def _point_values(value, *, count: int, label: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.ndim == 0:
        selected = np.full(count, float(selected))
    selected = selected.reshape(-1)
    if len(selected) != count or not np.all(np.isfinite(selected)):
        raise ValueError(f"{label} must provide one finite value per point.")
    return selected.copy()


def _optional_point_values(value, *, count: int, label: str):
    return None if value is None else _point_values(value, count=count, label=label)


__all__ = [
    "MaterialApplicabilityError",
    "MaterialParameter",
    "MaterialParameterSchema",
    "SmallStrainMaterialPointBatchInput",
    "SmallStrainMaterialPointBatchOutput",
    "SmallStrainMaterialPointInput",
    "SmallStrainMaterialPointOutput",
    "SmallStrainMaterialTangentCheck",
    "SmallStrainUserMaterial",
    "check_small_strain_material_tangent",
    "small_strain_matrix_to_tensor",
    "small_strain_tangent_convention",
    "validated_small_strain_batch_update",
    "validated_small_strain_update",
]
