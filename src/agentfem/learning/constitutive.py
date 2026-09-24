# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Framework-neutral contracts for learned constitutive materials.

AgentFEM records scientific identity and solver compatibility here.  Model
loading, tensors, devices, automatic differentiation and framework runtimes
belong to separately installed providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import re
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from ..constitutive.small_strain_material import SmallStrainUserMaterial
from ..constitutive.user_material import (
    MaterialStateSchema,
    MaterialStateVariable,
    MaterialTangentConvention,
)


_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _required_text(value, label: str) -> str:
    selected = str(value).strip()
    if not selected:
        raise ValueError(f"{label} must be non-empty.")
    return selected


def _optional_text(value, label: str) -> str | None:
    return None if value is None else _required_text(value, label)


def _json_value(value, *, label: str):
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, label=f"{label}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item, label=label) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"{label} must not contain non-finite values.")
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    raise TypeError(f"{label} must be JSON-compatible, got {type(value).__name__}.")


@dataclass(frozen=True)
class MaterialParameterSpec:
    """Named, unit-bearing parameter accepted by one material artifact."""

    name: str
    unit: str
    default: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    required: bool = True
    description: str = "Material parameter."

    def __post_init__(self) -> None:
        name = _required_text(self.name, "MaterialParameterSpec.name")
        if not _NAME.fullmatch(name):
            raise ValueError("Material parameter names use identifier syntax.")
        unit = _required_text(self.unit, "MaterialParameterSpec.unit")
        description = _required_text(
            self.description, "MaterialParameterSpec.description"
        )
        for label in ("default", "minimum", "maximum"):
            value = getattr(self, label)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"MaterialParameterSpec.{label} must be finite.")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("Material parameter minimum exceeds maximum.")
        if self.default is not None:
            self.validate(self.default)
        if not self.required and self.default is None:
            raise ValueError("An optional material parameter requires a default.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "description", description)

    def validate(self, value) -> float:
        selected = float(value)
        if not np.isfinite(selected):
            raise ValueError(f"Material parameter {self.name!r} must be finite.")
        if self.minimum is not None and selected < self.minimum:
            raise ValueError(f"Material parameter {self.name!r} is below its minimum.")
        if self.maximum is not None and selected > self.maximum:
            raise ValueError(f"Material parameter {self.name!r} exceeds its maximum.")
        return selected

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "required": self.required,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> "MaterialParameterSpec":
        return cls(**dict(record))


@dataclass(frozen=True)
class LearnedConstitutiveSpec:
    """Immutable scientific identity for one learned material artifact."""

    provider: str
    architecture_id: str
    artifact: str
    tangent_convention: MaterialTangentConvention
    parameter_schema: tuple[MaterialParameterSpec, ...]
    state_schema: MaterialStateSchema
    parameters: Mapping[str, float]
    model_name: str = "learned_constitutive"
    model_version: str = "0.1.0"
    revision: str | None = None
    artifact_sha256: str | None = None
    required_inputs: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ("stress", "state")
    dtype_policy: str = "float64"
    batch_capable: bool = True
    applicability_domain: Mapping[str, object] = field(default_factory=dict)
    dataset_id: str | None = None
    dataset_revision: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider = _required_text(self.provider, "LearnedConstitutiveSpec.provider")
        architecture = _required_text(
            self.architecture_id, "LearnedConstitutiveSpec.architecture_id"
        )
        artifact = _required_text(self.artifact, "LearnedConstitutiveSpec.artifact")
        if not _NAME.fullmatch(architecture):
            raise ValueError("architecture_id must use stable identifier syntax.")
        if not isinstance(self.tangent_convention, MaterialTangentConvention):
            raise TypeError("tangent_convention must be a MaterialTangentConvention.")
        if not isinstance(self.state_schema, MaterialStateSchema):
            raise TypeError("state_schema must be a MaterialStateSchema.")
        schema = tuple(self.parameter_schema)
        if any(not isinstance(item, MaterialParameterSpec) for item in schema):
            raise TypeError("parameter_schema entries must be MaterialParameterSpec.")
        names = tuple(item.name for item in schema)
        if len(set(names)) != len(names):
            raise ValueError("Material parameter schema names must be unique.")
        supplied = dict(self.parameters)
        unknown = set(supplied) - set(names)
        if unknown:
            raise ValueError(f"Unknown learned-material parameters: {sorted(unknown)!r}.")
        normalized: dict[str, float] = {}
        for item in schema:
            if item.name in supplied:
                value = supplied[item.name]
            elif item.default is not None:
                value = item.default
            elif item.required:
                raise ValueError(f"Missing required material parameter {item.name!r}.")
            else:
                continue
            normalized[item.name] = item.validate(value)
        checksum = self.artifact_sha256
        if checksum is not None:
            checksum = str(checksum).strip().lower().removeprefix("sha256:")
            if not _SHA256.fullmatch(checksum):
                raise ValueError("artifact_sha256 must contain 64 hexadecimal digits.")
        required_inputs = tuple(
            _required_text(item, "required_inputs") for item in self.required_inputs
        )
        capabilities = tuple(
            _required_text(item, "capabilities") for item in self.capabilities
        )
        if len(set(required_inputs)) != len(required_inputs):
            raise ValueError("required_inputs must be unique.")
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("capabilities must be unique.")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "architecture_id", architecture)
        object.__setattr__(self, "artifact", artifact)
        object.__setattr__(self, "parameter_schema", schema)
        object.__setattr__(self, "parameters", MappingProxyType(normalized))
        object.__setattr__(self, "model_name", _required_text(self.model_name, "model_name"))
        object.__setattr__(
            self, "model_version", _required_text(self.model_version, "model_version")
        )
        object.__setattr__(self, "revision", _optional_text(self.revision, "revision"))
        object.__setattr__(self, "artifact_sha256", checksum)
        object.__setattr__(self, "required_inputs", required_inputs)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "dtype_policy", _required_text(self.dtype_policy, "dtype_policy"))
        object.__setattr__(
            self,
            "applicability_domain",
            MappingProxyType(_json_value(self.applicability_domain, label="applicability_domain")),
        )
        object.__setattr__(self, "dataset_id", _optional_text(self.dataset_id, "dataset_id"))
        object.__setattr__(
            self,
            "dataset_revision",
            _optional_text(self.dataset_revision, "dataset_revision"),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_json_value(self.metadata, label="metadata")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "agentfem.learned_constitutive",
            "schema_version": "0.1.0",
            "provider": self.provider,
            "architecture_id": self.architecture_id,
            "artifact": self.artifact,
            "revision": self.revision,
            "artifact_sha256": self.artifact_sha256,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "tangent_convention": self.tangent_convention.summary(),
            "parameter_schema": tuple(item.summary() for item in self.parameter_schema),
            "state_schema": self.state_schema.summary(),
            "parameters": dict(self.parameters),
            "required_inputs": self.required_inputs,
            "capabilities": self.capabilities,
            "dtype_policy": self.dtype_policy,
            "batch_capable": self.batch_capable,
            "applicability_domain": dict(self.applicability_domain),
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "metadata": dict(self.metadata),
        }

    summary = to_dict

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return f"sha256:{sha256(payload).hexdigest()}"

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> "LearnedConstitutiveSpec":
        selected = dict(record)
        if selected.get("schema") not in {None, "agentfem.learned_constitutive"}:
            raise ValueError("Unsupported learned constitutive schema.")
        tangent_record = dict(selected.pop("tangent_convention"))
        tangent_record.pop("kind", None)
        tangent_record.pop("array_shape", None)
        state_record = dict(selected.pop("state_schema"))
        state_record.pop("kind", None)
        identity = state_record.pop("identity", None)
        variables = []
        for variable in state_record.pop("variables"):
            item = dict(variable)
            item.pop("size", None)
            variables.append(MaterialStateVariable(**item))
        if identity is None or "@" not in identity:
            raise ValueError("State schema identity is missing from serialized spec.")
        state_name, state_version = identity.rsplit("@", 1)
        state_schema = MaterialStateSchema(
            state_name,
            tuple(variables),
            version=state_version,
        )
        selected.pop("schema", None)
        selected.pop("schema_version", None)
        parameter_schema = tuple(
            MaterialParameterSpec.from_dict(item)
            for item in selected.pop("parameter_schema")
        )
        return cls(
            **selected,
            tangent_convention=MaterialTangentConvention(**tangent_record),
            parameter_schema=parameter_schema,
            state_schema=state_schema,
        )


class LearnedConstitutiveProviderError(RuntimeError):
    """A learned material provider is missing or incompatible."""


@dataclass(frozen=True)
class LearnedConstitutiveMaterialBinding:
    """Framework-neutral binding of immutable science to one implementation."""

    # The binding owns committed integration-point history even when the
    # provider happens to expose an elastic reference model. Declaring this
    # at the material boundary lets the ordinary Step resolver select the
    # stateful nonlinear procedure before provider lowering.
    stateful_constitutive = True

    specification: LearnedConstitutiveSpec
    implementation: SmallStrainUserMaterial
    runtime_evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.specification, LearnedConstitutiveSpec):
            raise TypeError("specification must be a LearnedConstitutiveSpec.")
        if not isinstance(self.implementation, SmallStrainUserMaterial):
            raise TypeError("implementation must satisfy SmallStrainUserMaterial.")
        if not isinstance(self.runtime_evidence, Mapping):
            raise TypeError("runtime_evidence must be a mapping.")
        object.__setattr__(
            self,
            "runtime_evidence",
            MappingProxyType(_json_value(self.runtime_evidence, label="runtime_evidence")),
        )

    @property
    def name(self) -> str:
        return self.specification.model_name

    @property
    def state_schema(self) -> MaterialStateSchema:
        return self.implementation.state_schema

    @property
    def tangent_convention(self) -> MaterialTangentConvention:
        return self.implementation.tangent_convention

    @property
    def parameters(self) -> Mapping[str, float]:
        return self.specification.parameters

    def _check_parameters(self, values: Mapping[str, float]) -> None:
        selected = {str(name): float(value) for name, value in values.items()}
        if selected != dict(self.parameters):
            raise ValueError(
                "A learned material update must use the parameters frozen in its "
                "LearnedConstitutiveSpec. Create another spec for another material."
            )

    def update(self, point):
        self._check_parameters(point.parameters)
        return self.implementation.update(point)

    def update_batch(self, request):
        self._check_parameters(request.parameters)
        batch = getattr(self.implementation, "update_batch", None)
        if callable(batch):
            return batch(request)
        from ..constitutive.small_strain_material import (
            SmallStrainMaterialBatchOutput,
            validated_small_strain_update,
        )

        return SmallStrainMaterialBatchOutput.from_points(
            tuple(
                validated_small_strain_update(self.implementation, request.point(index))
                for index in range(request.point_count)
            )
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "learned_constitutive_material",
            "specification_fingerprint": self.specification.fingerprint,
            "specification": self.specification.to_dict(),
            "runtime": dict(self.runtime_evidence),
        }

    as_dict = summary


@runtime_checkable
class LearnedConstitutiveProvider(Protocol):
    """Runtime provider that lowers one immutable spec to a material."""

    name: str

    def create(self, specification: LearnedConstitutiveSpec) -> SmallStrainUserMaterial:
        ...

    def evidence(self, specification: LearnedConstitutiveSpec) -> Mapping[str, object]:
        ...


_PROVIDERS: dict[str, LearnedConstitutiveProvider] = {}


def register_learned_constitutive_provider(
    provider: LearnedConstitutiveProvider,
    *,
    replace: bool = False,
) -> LearnedConstitutiveProvider:
    if not isinstance(provider, LearnedConstitutiveProvider):
        raise TypeError("Learned constitutive provider must expose name and create().")
    name = _required_text(provider.name, "provider.name")
    if name in _PROVIDERS and not replace:
        raise ValueError(f"Learned constitutive provider {name!r} is already registered.")
    _PROVIDERS[name] = provider
    return provider


def learned_constitutive_providers() -> tuple[str, ...]:
    return tuple(sorted(_PROVIDERS))


def resolve_learned_constitutive_provider(name: str) -> LearnedConstitutiveProvider:
    selected = _required_text(name, "provider")
    try:
        return _PROVIDERS[selected]
    except KeyError as exc:
        raise LearnedConstitutiveProviderError(
            f"Learned constitutive provider {selected!r} is not active. "
            f"Available providers={learned_constitutive_providers()!r}. "
            "Install and explicitly load the extension that owns this runtime."
        ) from exc


def load_learned_constitutive(
    specification: LearnedConstitutiveSpec,
) -> LearnedConstitutiveMaterialBinding:
    if not isinstance(specification, LearnedConstitutiveSpec):
        raise TypeError("specification must be a LearnedConstitutiveSpec.")
    provider = resolve_learned_constitutive_provider(specification.provider)
    material = provider.create(specification)
    if not isinstance(material, SmallStrainUserMaterial):
        raise TypeError(
            f"Provider {provider.name!r} did not return a SmallStrainUserMaterial."
        )
    if material.state_schema.summary() != specification.state_schema.summary():
        raise ValueError("Provider material state schema differs from its specification.")
    if material.tangent_convention != specification.tangent_convention:
        raise ValueError("Provider material tangent differs from its specification.")
    evidence = provider.evidence(specification)
    return LearnedConstitutiveMaterialBinding(specification, material, evidence)


def learned_constitutive_evidence(
    specification: LearnedConstitutiveSpec,
) -> dict[str, object]:
    """Return portable scientific identity plus provider runtime evidence."""

    if not isinstance(specification, LearnedConstitutiveSpec):
        raise TypeError("specification must be a LearnedConstitutiveSpec.")
    provider = resolve_learned_constitutive_provider(specification.provider)
    runtime = provider.evidence(specification)
    if not isinstance(runtime, Mapping):
        raise TypeError("Learned constitutive provider evidence must be a mapping.")
    return {
        "kind": "learned_constitutive_execution_evidence",
        "specification_fingerprint": specification.fingerprint,
        "specification": specification.to_dict(),
        "runtime": _json_value(runtime, label="provider_evidence"),
    }


def learned_constitutive(**kwargs) -> LearnedConstitutiveSpec:
    """Create a framework-neutral learned-material declaration."""

    return LearnedConstitutiveSpec(**kwargs)


__all__ = [
    "LearnedConstitutiveProvider",
    "LearnedConstitutiveProviderError",
    "LearnedConstitutiveMaterialBinding",
    "LearnedConstitutiveSpec",
    "MaterialParameterSpec",
    "learned_constitutive",
    "learned_constitutive_evidence",
    "learned_constitutive_providers",
    "load_learned_constitutive",
    "register_learned_constitutive_provider",
    "resolve_learned_constitutive_provider",
]
