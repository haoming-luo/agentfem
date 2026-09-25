# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral learned-constitutive descriptions and discovery.

This module stores scientific identity and compatibility only.  Executable
models, tensor frameworks, devices, automatic differentiation and weight
loading belong to explicitly activated extension providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping
from uuid import uuid4

from ..constitutive.small_strain_user_material import (
    MaterialParameter,
    MaterialParameterSchema,
    SmallStrainMaterialPointBatchInput,
    SmallStrainMaterialPointBatchOutput,
    SmallStrainMaterialPointInput,
    SmallStrainMaterialPointOutput,
    SmallStrainUserMaterial,
)
from ..constitutive.user_material import (
    MaterialStateSchema,
    MaterialStateVariable,
    MaterialTangentConvention,
)
from ..provenance import content_fingerprint


@dataclass(frozen=True)
class LearnedConstitutiveSpec:
    """Immutable scientific identity for an externally executed local model."""

    provider: str
    architecture: str
    artifact: str
    revision: str
    artifact_sha256: str
    parameter_schema: MaterialParameterSchema
    parameters: Mapping[str, float]
    state_schema: MaterialStateSchema
    tangent_convention: MaterialTangentConvention
    required_inputs: tuple[str, ...] = ("strain", "state")
    capabilities: tuple[str, ...] = ("stress", "consistent_tangent", "batch")
    dtype_policy: str = "float64"
    batch_update: bool = True
    applicability_domain: Mapping[str, object] = field(default_factory=dict)
    dataset: Mapping[str, object] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = "0.1.0"

    def __post_init__(self) -> None:
        for name in ("provider", "architecture", "artifact", "revision"):
            selected = str(getattr(self, name)).strip()
            if not selected:
                raise ValueError(f"LearnedConstitutiveSpec.{name} must be non-empty.")
            object.__setattr__(self, name, selected)
        digest = str(self.artifact_sha256).strip().lower()
        if len(digest) != 64 or any(item not in "0123456789abcdef" for item in digest):
            raise ValueError("artifact_sha256 must be a 64-character SHA-256 digest.")
        if not isinstance(self.parameter_schema, MaterialParameterSchema):
            raise TypeError("parameter_schema must be a MaterialParameterSchema.")
        if not isinstance(self.state_schema, MaterialStateSchema):
            raise TypeError("state_schema must be a MaterialStateSchema.")
        if not isinstance(self.tangent_convention, MaterialTangentConvention):
            raise TypeError("tangent_convention must be a MaterialTangentConvention.")
        if self.tangent_convention.kinematic_measure != "small_strain":
            raise ValueError(
                "The first learned-constitutive contract requires an explicit "
                "small-strain tangent convention."
            )
        parameters = self.parameter_schema.validate(self.parameters)
        required = _unique_names(self.required_inputs, "required_inputs")
        capabilities = _unique_names(self.capabilities, "capabilities")
        if not {"strain", "state"}.issubset(required):
            raise ValueError("required_inputs must include 'strain' and 'state'.")
        required_capabilities = {"stress", "consistent_tangent"}
        missing_required = required_capabilities - set(capabilities)
        if missing_required:
            raise ValueError(
                "capabilities must include the mechanics outputs "
                f"{sorted(required_capabilities)!r}; missing "
                f"{sorted(missing_required)!r}."
            )
        if self.batch_update and "batch" not in capabilities:
            raise ValueError("batch_update=True requires the 'batch' capability.")
        dtype = str(self.dtype_policy).strip().lower()
        if dtype not in {"float32", "float64", "provider_declared"}:
            raise ValueError(
                "dtype_policy must be float32, float64, or provider_declared."
            )
        object.__setattr__(self, "artifact_sha256", digest)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "required_inputs", required)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "dtype_policy", dtype)
        object.__setattr__(
            self,
            "applicability_domain",
            _json_mapping(self.applicability_domain, "applicability_domain"),
        )
        object.__setattr__(self, "dataset", _json_mapping(self.dataset, "dataset"))
        object.__setattr__(
            self, "provenance", _json_mapping(self.provenance, "provenance")
        )
        schema_version = str(self.schema_version).strip()
        if not schema_version:
            raise ValueError("schema_version must be non-empty.")
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(self.summary())

    def summary(self) -> dict[str, object]:
        return {
            "kind": "learned_constitutive_spec",
            "schema_version": self.schema_version,
            "provider": self.provider,
            "architecture": self.architecture,
            "artifact": self.artifact,
            "revision": self.revision,
            "artifact_sha256": self.artifact_sha256,
            "kinematics": "small_strain",
            "parameter_schema": self.parameter_schema.summary(),
            "parameters": dict(self.parameters),
            "state_schema": self.state_schema.summary(),
            "tangent_convention": self.tangent_convention.summary(),
            "required_inputs": self.required_inputs,
            "capabilities": self.capabilities,
            "dtype_policy": self.dtype_policy,
            "batch_update": self.batch_update,
            "applicability_domain": dict(self.applicability_domain),
            "dataset": dict(self.dataset),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_summary(cls, record: Mapping[str, object]) -> "LearnedConstitutiveSpec":
        """Restore one specification from its dependency-free JSON shape."""

        if not isinstance(record, Mapping):
            raise TypeError("Learned constitutive specification must be a mapping.")
        if record.get("kind") != "learned_constitutive_spec":
            raise ValueError("Record is not a learned_constitutive_spec.")
        state_schema = _state_schema_from_summary(record.get("state_schema"))
        parameter_schema = _parameter_schema_from_summary(
            record.get("parameter_schema")
        )
        tangent = _tangent_convention_from_summary(record.get("tangent_convention"))
        return cls(
            provider=record["provider"],
            architecture=record["architecture"],
            artifact=record["artifact"],
            revision=record["revision"],
            artifact_sha256=record["artifact_sha256"],
            parameter_schema=parameter_schema,
            parameters=record.get("parameters", {}),
            state_schema=state_schema,
            tangent_convention=tangent,
            required_inputs=tuple(record.get("required_inputs", ("strain", "state"))),
            capabilities=tuple(
                record.get(
                    "capabilities",
                    ("stress", "consistent_tangent", "batch"),
                )
            ),
            dtype_policy=record.get("dtype_policy", "float64"),
            batch_update=bool(record.get("batch_update", True)),
            applicability_domain=record.get("applicability_domain", {}),
            dataset=record.get("dataset", {}),
            provenance=record.get("provenance", {}),
            schema_version=record.get("schema_version", "0.1.0"),
        )

    def write(self, path: str | Path) -> Path:
        """Write a portable, fingerprinted specification manifest."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "agentfem.learned_constitutive_spec",
            "schema_version": "0.1.0",
            "fingerprint": self.fingerprint,
            "specification": self.summary(),
        }
        temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return output

    @classmethod
    def read(cls, path: str | Path) -> "LearnedConstitutiveSpec":
        """Read a specification and reject malformed or altered identity."""

        selected = Path(path)
        try:
            payload = json.loads(selected.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LearnedConstitutiveProviderError(
                "AFM-LEARNED-MATERIAL-SPEC-READ",
                f"Cannot read learned-material specification {selected}: {exc}",
            ) from exc
        if not isinstance(payload, Mapping):
            raise LearnedConstitutiveProviderError(
                "AFM-LEARNED-MATERIAL-SPEC-SCHEMA",
                "Learned-material specification manifest must be a mapping.",
            )
        if payload.get("schema") != "agentfem.learned_constitutive_spec":
            raise LearnedConstitutiveProviderError(
                "AFM-LEARNED-MATERIAL-SPEC-SCHEMA",
                "Learned-material specification manifest has an unknown schema.",
            )
        try:
            specification = cls.from_summary(payload.get("specification"))
        except (KeyError, TypeError, ValueError) as exc:
            raise LearnedConstitutiveProviderError(
                "AFM-LEARNED-MATERIAL-SPEC-SCHEMA",
                f"Learned-material specification content is invalid: {exc}",
            ) from exc
        if payload.get("fingerprint") != specification.fingerprint:
            raise LearnedConstitutiveProviderError(
                "AFM-LEARNED-MATERIAL-SPEC-FINGERPRINT",
                "Learned-material specification fingerprint does not match its content.",
            )
        return specification

    def verify_artifact(self, path: str | Path) -> Path:
        """Verify a prepared local artifact without downloading or executing it."""

        selected = Path(path).expanduser().resolve()
        if not selected.is_file():
            raise LearnedConstitutiveProviderError(
                "AFM-LEARNED-MATERIAL-ARTIFACT-MISSING",
                f"Prepared model artifact does not exist: {selected}",
            )
        hasher = sha256()
        with selected.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
        if digest != self.artifact_sha256:
            raise LearnedConstitutiveProviderError(
                "AFM-LEARNED-MATERIAL-CHECKSUM",
                f"Artifact SHA-256 {digest} does not match {self.artifact_sha256}.",
            )
        return selected


@dataclass(frozen=True)
class LearnedConstitutiveProvider:
    """Registered factory owned by an activated extension package."""

    name: str
    version: str
    factory: Callable[[LearnedConstitutiveSpec], SmallStrainUserMaterial]
    architectures: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        version = str(self.version).strip()
        if not name or not version or not callable(self.factory):
            raise TypeError(
                "A learned-material provider needs name, version, and factory."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(
            self, "architectures", _unique_names(self.architectures, "architectures")
        )
        object.__setattr__(
            self, "capabilities", _unique_names(self.capabilities, "capabilities")
        )

    def supports(self, specification: LearnedConstitutiveSpec) -> bool:
        return (
            not self.architectures or specification.architecture in self.architectures
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "architectures": self.architectures,
            "capabilities": self.capabilities,
        }


class LearnedConstitutiveProviderError(RuntimeError):
    """Addressable provider, artifact, or compatibility failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


_PROVIDERS: dict[str, LearnedConstitutiveProvider] = {}


def register_learned_constitutive_provider(
    provider: LearnedConstitutiveProvider, *, replace: bool = False
) -> None:
    """Register one already-activated framework provider."""

    if not isinstance(provider, LearnedConstitutiveProvider):
        raise TypeError("provider must be a LearnedConstitutiveProvider.")
    if provider.name in _PROVIDERS and not replace:
        raise LearnedConstitutiveProviderError(
            "AFM-LEARNED-MATERIAL-PROVIDER-CONFLICT",
            f"Provider {provider.name!r} is already registered.",
        )
    _PROVIDERS[provider.name] = provider


def learned_constitutive_providers() -> tuple[LearnedConstitutiveProvider, ...]:
    return tuple(_PROVIDERS[name] for name in sorted(_PROVIDERS))


def resolve_learned_constitutive_provider(
    specification: LearnedConstitutiveSpec,
) -> LearnedConstitutiveProvider:
    provider = _PROVIDERS.get(specification.provider)
    if provider is None:
        raise LearnedConstitutiveProviderError(
            "AFM-LEARNED-MATERIAL-PROVIDER-MISSING",
            f"Provider {specification.provider!r} is not active. Explicitly load "
            "the extension that owns it before constructing the material.",
        )
    if not provider.supports(specification):
        raise LearnedConstitutiveProviderError(
            "AFM-LEARNED-MATERIAL-ARCHITECTURE",
            f"Provider {provider.name!r} does not support architecture "
            f"{specification.architecture!r}.",
        )
    missing = set(specification.capabilities) - set(provider.capabilities)
    if missing:
        raise LearnedConstitutiveProviderError(
            "AFM-LEARNED-MATERIAL-CAPABILITY",
            f"Provider {provider.name!r} lacks capabilities {sorted(missing)!r}.",
        )
    return provider


class LearnedConstitutiveMaterial:
    """Ordinary small-strain material delegating execution to one provider."""

    def __init__(self, specification: LearnedConstitutiveSpec) -> None:
        if not isinstance(specification, LearnedConstitutiveSpec):
            raise TypeError("specification must be a LearnedConstitutiveSpec.")
        provider = resolve_learned_constitutive_provider(specification)
        implementation = provider.factory(specification)
        if not isinstance(implementation, SmallStrainUserMaterial):
            raise LearnedConstitutiveProviderError(
                "AFM-LEARNED-MATERIAL-CONTRACT",
                "Provider factory did not return a SmallStrainUserMaterial.",
            )
        if (
            implementation.state_schema.summary()
            != specification.state_schema.summary()
        ):
            raise LearnedConstitutiveProviderError(
                "AFM-LEARNED-MATERIAL-STATE-SCHEMA",
                "Provider material state schema differs from the specification.",
            )
        if (
            implementation.parameter_schema.summary()
            != specification.parameter_schema.summary()
        ):
            raise LearnedConstitutiveProviderError(
                "AFM-LEARNED-MATERIAL-PARAMETER-SCHEMA",
                "Provider material parameter schema differs from the specification.",
            )
        if implementation.tangent_convention != specification.tangent_convention:
            raise LearnedConstitutiveProviderError(
                "AFM-LEARNED-MATERIAL-TANGENT",
                "Provider tangent convention differs from the specification.",
            )
        if specification.batch_update and not callable(
            getattr(implementation, "update_batch", None)
        ):
            raise LearnedConstitutiveProviderError(
                "AFM-LEARNED-MATERIAL-BATCH",
                "Specification requires batch update but provider returned only a "
                "scalar update implementation.",
            )
        self.specification = specification
        self.provider = provider
        self.implementation = implementation
        self.name = getattr(implementation, "name", specification.architecture)
        self.state_schema = specification.state_schema
        self.parameter_schema = specification.parameter_schema
        self.parameters = specification.parameters
        self.tangent_convention = specification.tangent_convention
        if not specification.batch_update:
            # Shadow the optional class method so the core batch driver uses
            # its validated scalar fallback rather than inventing provider
            # batching that the specification did not declare.
            self.update_batch = None

    def update(
        self, point: SmallStrainMaterialPointInput
    ) -> SmallStrainMaterialPointOutput:
        return self.implementation.update(point)

    def update_batch(
        self, request: SmallStrainMaterialPointBatchInput
    ) -> SmallStrainMaterialPointBatchOutput:
        selected = getattr(self.implementation, "update_batch", None)
        if not callable(selected):
            raise LearnedConstitutiveProviderError(
                "AFM-LEARNED-MATERIAL-BATCH",
                "The selected provider material does not expose update_batch().",
            )
        return selected(request)

    def evidence(
        self,
        *,
        runtime: Mapping[str, object] | None = None,
        diagnostics: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "contract": "agentfem.learning.learned_constitutive",
            "specification_fingerprint": self.specification.fingerprint,
            "provider": self.provider.summary(),
            "specification": self.specification.summary(),
            "runtime": dict(_json_mapping(runtime or {}, "runtime")),
            "diagnostics": dict(_json_mapping(diagnostics or {}, "diagnostics")),
        }

    def summary(self) -> dict[str, object]:
        return {
            "kind": "learned_constitutive_material",
            "name": self.name,
            "provider": self.provider.summary(),
            "specification": self.specification.summary(),
            "specification_fingerprint": self.specification.fingerprint,
        }

    def as_dict(self) -> dict[str, object]:
        return self.summary()


def learned_constitutive(
    *,
    provider: str,
    architecture: str,
    artifact: str,
    revision: str,
    artifact_sha256: str,
    parameter_schema: MaterialParameterSchema,
    parameters: Mapping[str, float],
    state_schema: MaterialStateSchema,
    tangent_convention: MaterialTangentConvention,
    **options,
) -> LearnedConstitutiveSpec:
    """Construct a descriptive specification without loading executable code."""

    return LearnedConstitutiveSpec(
        provider=provider,
        architecture=architecture,
        artifact=artifact,
        revision=revision,
        artifact_sha256=artifact_sha256,
        parameter_schema=parameter_schema,
        parameters=parameters,
        state_schema=state_schema,
        tangent_convention=tangent_convention,
        **options,
    )


def material(specification: LearnedConstitutiveSpec) -> LearnedConstitutiveMaterial:
    """Bind one immutable specification to an explicitly active provider."""

    return LearnedConstitutiveMaterial(specification)


def record_learned_constitutive_evidence(
    result,
    selected_material: LearnedConstitutiveMaterial,
    *,
    runtime: Mapping[str, object],
    diagnostics: Mapping[str, object],
):
    """Attach reserved, portable provider evidence to a SimulationResult."""

    from ..results import SimulationResult

    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult.")
    if not isinstance(selected_material, LearnedConstitutiveMaterial):
        raise TypeError("selected_material must be a LearnedConstitutiveMaterial.")
    evidence = selected_material.evidence(runtime=runtime, diagnostics=diagnostics)
    existing = result.metadata.get("learned_constitutive")
    if existing is not None and existing != evidence:
        raise ValueError(
            "SimulationResult.metadata['learned_constitutive'] is reserved for "
            "the AgentFEM learned-material boundary."
        )
    result.metadata["learned_constitutive"] = evidence
    return result


def _unique_names(values, label: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if any(not value for value in normalized) or len(set(normalized)) != len(
        normalized
    ):
        raise ValueError(f"{label} must contain unique nonempty names.")
    return normalized


def _json_mapping(values: Mapping[str, object], label: str) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    try:
        normalized = json.loads(
            json.dumps(dict(values), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must contain JSON-shaped finite values.") from exc
    return MappingProxyType(normalized)


def _parameter_schema_from_summary(record) -> MaterialParameterSchema:
    if (
        not isinstance(record, Mapping)
        or record.get("kind") != "material_parameter_schema"
    ):
        raise ValueError("parameter_schema is not a material-parameter schema.")
    name, version = _schema_identity(record)
    return MaterialParameterSchema(
        name=name,
        version=version,
        parameters=tuple(
            MaterialParameter(
                name=item["name"],
                unit=item.get("unit"),
                lower=item.get("lower"),
                upper=item.get("upper"),
                default=item.get("default"),
                description=item.get("description", "Material parameter."),
            )
            for item in record.get("parameters", ())
        ),
    )


def _state_schema_from_summary(record) -> MaterialStateSchema:
    if not isinstance(record, Mapping) or record.get("kind") != "material_state_schema":
        raise ValueError("state_schema is not a material-state schema.")
    name, version = _schema_identity(record)
    return MaterialStateSchema(
        name=name,
        version=version,
        variables=tuple(
            MaterialStateVariable(
                name=item["name"],
                shape=tuple(item.get("shape", ())),
                initial_value=item.get("initial_value", 0.0),
                unit=item.get("unit"),
                description=item.get("description", "Material internal variable."),
                output_name=item.get("output_name"),
            )
            for item in record.get("variables", ())
        ),
    )


def _tangent_convention_from_summary(record) -> MaterialTangentConvention:
    if (
        not isinstance(record, Mapping)
        or record.get("kind") != "material_tangent_convention"
    ):
        raise ValueError("tangent_convention is not a material tangent convention.")
    return MaterialTangentConvention(
        stress_measure=record["stress_measure"],
        kinematic_measure=record["kinematic_measure"],
        configuration=record["configuration"],
        storage=record["storage"],
        component_order=tuple(record["component_order"]),
        shear_convention=record.get("shear_convention", "tensor"),
        objective_rate=record.get("objective_rate", "not_applicable"),
        symmetric=bool(record.get("symmetric", True)),
    )


def _schema_identity(record: Mapping[str, object]) -> tuple[str, str]:
    name = str(record.get("name", "")).strip()
    version = str(record.get("version", "")).strip()
    if name and version:
        return name, version
    identity = str(record.get("identity", ""))
    if "@" not in identity:
        raise ValueError("Schema summary must provide name/version or identity.")
    parsed_name, parsed_version = identity.rsplit("@", 1)
    if not parsed_name or not parsed_version:
        raise ValueError("Schema identity must have the form name@version.")
    return parsed_name, parsed_version


__all__ = [
    "LearnedConstitutiveMaterial",
    "LearnedConstitutiveProvider",
    "LearnedConstitutiveProviderError",
    "LearnedConstitutiveSpec",
    "learned_constitutive",
    "learned_constitutive_providers",
    "material",
    "record_learned_constitutive_evidence",
    "register_learned_constitutive_provider",
    "resolve_learned_constitutive_provider",
]
