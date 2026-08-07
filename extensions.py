"""Explicit, inspectable extension packages for the AgentFEM open core.

Extensions are discovered through the standard ``agentfem.extensions`` Python
entry-point group.  Discovery never imports extension code; activation is an
explicit project or user decision.  This keeps optional and proprietary
packages outside the Apache-2.0 distribution while giving them a stable way to
register scientific capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from typing import Callable, Mapping


ENTRY_POINT_GROUP = "agentfem.extensions"
EXTENSION_API_VERSION = 1


class ExtensionError(RuntimeError):
    """An installed extension could not be discovered or activated safely."""


@dataclass(frozen=True)
class ExtensionSpec:
    """Identity and compatibility contract published by one extension."""

    name: str
    version: str
    api_version: int = EXTENSION_API_VERSION
    description: str = ""
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ExtensionSpec.name must be non-empty.")
        if not self.version.strip():
            raise ValueError("ExtensionSpec.version must be non-empty.")
        if int(self.api_version) <= 0:
            raise ValueError("ExtensionSpec.api_version must be positive.")
        object.__setattr__(
            self,
            "capabilities",
            tuple(str(item) for item in self.capabilities),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "api_version": self.api_version,
            "description": self.description,
            "capabilities": self.capabilities,
        }


@dataclass(frozen=True)
class Extension:
    """One loadable extension and its side-effect-free registration callback."""

    spec: ExtensionSpec
    register: Callable[["ExtensionContext"], None]


@dataclass(frozen=True)
class ExtensionDescriptor:
    """Package metadata visible without importing extension code."""

    name: str
    value: str
    distribution: str | None = None
    distribution_version: str | None = None
    loaded: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "entry_point": self.value,
            "distribution": self.distribution,
            "distribution_version": self.distribution_version,
            "loaded": self.loaded,
        }


@dataclass
class ExtensionContext:
    """Staging area exposed to an extension during activation.

    Registration is committed only after the extension callback returns and
    the complete plan has passed conflict and schema checks.  An extension can
    still expose ordinary Python objects from its own package; this context is
    only for capabilities that must participate in AgentFEM discovery.
    """

    extension: ExtensionSpec
    _step_providers: list[tuple[object, bool]] = field(default_factory=list)
    _backends: list[tuple[str, object, bool]] = field(default_factory=list)
    _materials: list[tuple[str, dict[str, object], bool]] = field(default_factory=list)

    def add_step_provider(self, provider, *, replace: bool = False) -> None:
        """Stage a ``StepProvider`` for model validation and lowering."""

        self._step_providers.append((provider, bool(replace)))

    def add_backend(self, name: str, factory, *, replace: bool = False) -> None:
        """Stage a lazy numerical backend factory."""

        self._backends.append((str(name), factory, bool(replace)))

    def add_material(
        self,
        name: str,
        record: Mapping[str, object],
        *,
        replace: bool = False,
    ) -> None:
        """Stage a private or public material record without writing it to core."""

        self._materials.append((str(name), dict(record), bool(replace)))

    def summary(self) -> dict[str, object]:
        return {
            "step_providers": tuple(
                getattr(provider, "name", type(provider).__name__)
                for provider, _ in self._step_providers
            ),
            "backends": tuple(name for name, _, _ in self._backends),
            "materials": tuple(name for name, _, _ in self._materials),
        }

    def commit(self) -> None:
        """Validate the complete registration plan, then publish it."""

        from . import backends, materials
        from .step_providers import StepProvider, register_step_provider, step_providers

        provider_names = {item.name for item in step_providers()}
        backend_names = set(backends.available_backends())
        material_names = set(materials.list_materials())
        staged_provider_names: set[str] = set()
        staged_backend_names: set[str] = set()
        staged_material_names: set[str] = set()

        for provider, replace in self._step_providers:
            if not isinstance(provider, StepProvider):
                raise TypeError(
                    "Extension step providers must be StepProvider instances; "
                    f"received {type(provider).__name__}."
                )
            _check_conflict(
                "step provider",
                provider.name,
                provider_names,
                staged_provider_names,
                replace=replace,
            )
            staged_provider_names.add(provider.name)

        for name, factory, replace in self._backends:
            normalized = name.strip().lower().replace("-", "_")
            if not normalized or not callable(factory):
                raise TypeError("Extension backends require a name and callable factory.")
            _check_conflict(
                "backend",
                normalized,
                backend_names,
                staged_backend_names,
                replace=replace,
            )
            staged_backend_names.add(normalized)

        for name, record, replace in self._materials:
            materials.validate_material_record(name, record)
            _check_conflict(
                "material",
                name,
                material_names,
                staged_material_names,
                replace=replace,
            )
            staged_material_names.add(name)

        for provider, replace in self._step_providers:
            register_step_provider(provider, replace=replace)
        for name, factory, replace in self._backends:
            backends.register_backend(name, factory, overwrite=replace)
        for name, record, replace in self._materials:
            materials.register_material(name, record, overwrite=replace)


@dataclass(frozen=True)
class LoadedExtension:
    """Activated identity and the capabilities registered into this process."""

    spec: ExtensionSpec
    distribution: str | None
    distribution_version: str | None
    registrations: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            **self.spec.as_dict(),
            "distribution": self.distribution,
            "distribution_version": self.distribution_version,
            "registrations": dict(self.registrations),
        }


_LOADED: dict[str, LoadedExtension] = {}


def discover_extensions() -> tuple[ExtensionDescriptor, ...]:
    """Return installed extension metadata without importing extension code."""

    return tuple(_descriptor(item) for item in _entry_points())


def extension_status() -> dict[str, object]:
    """Return the machine-facing installed and activated extension inventory."""

    return {
        "schema": "agentfem.extensions",
        "schema_version": "0.1.0",
        "extension_api_version": EXTENSION_API_VERSION,
        "installed": tuple(item.as_dict() for item in discover_extensions()),
        "loaded": tuple(item.as_dict() for item in loaded_extensions()),
    }


def loaded_extensions() -> tuple[LoadedExtension, ...]:
    """Return activated extensions in stable name order."""

    return tuple(_LOADED[name] for name in sorted(_LOADED))


def missing_extensions(names) -> tuple[str, ...]:
    """Return required names that are not advertised by installed packages."""

    installed = {item.name for item in discover_extensions()}
    return tuple(sorted({str(name) for name in names} - installed))


def load_extension(name: str) -> LoadedExtension:
    """Explicitly import, validate, and activate one installed extension."""

    selected_name = str(name).strip()
    if selected_name in _LOADED:
        return _LOADED[selected_name]
    matches = [item for item in _entry_points() if item.name == selected_name]
    if not matches:
        raise ExtensionError(
            f"AgentFEM extension {selected_name!r} is not installed. "
            f"Available extensions={tuple(item.name for item in _entry_points())!r}."
        )
    if len(matches) != 1:
        sources = tuple(_distribution_name(item) for item in matches)
        raise ExtensionError(
            f"Extension name {selected_name!r} is ambiguous across {sources!r}."
        )
    entry_point = matches[0]
    try:
        candidate = entry_point.load()
        extension = _coerce_extension(candidate)
        if extension.spec.name != selected_name:
            raise ExtensionError(
                f"Entry point {selected_name!r} published spec name "
                f"{extension.spec.name!r}."
            )
        if extension.spec.api_version != EXTENSION_API_VERSION:
            raise ExtensionError(
                f"Extension {selected_name!r} requires API "
                f"{extension.spec.api_version}; this AgentFEM runtime supports "
                f"{EXTENSION_API_VERSION}."
            )
        context = ExtensionContext(extension=extension.spec)
        extension.register(context)
        context.commit()
    except ExtensionError:
        raise
    except Exception as exc:
        raise ExtensionError(
            f"Could not activate AgentFEM extension {selected_name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    loaded = LoadedExtension(
        spec=extension.spec,
        distribution=_distribution_name(entry_point),
        distribution_version=_distribution_version(entry_point),
        registrations=context.summary(),
    )
    _LOADED[selected_name] = loaded
    return loaded


def load_extensions(names) -> tuple[LoadedExtension, ...]:
    """Activate required extensions in declaration order."""

    return tuple(load_extension(name) for name in names)


def _entry_points():
    selected = metadata.entry_points()
    points = (
        selected.select(group=ENTRY_POINT_GROUP)
        if hasattr(selected, "select")
        else selected.get(ENTRY_POINT_GROUP, ())
    )
    return tuple(sorted(points, key=lambda item: (item.name, item.value)))


def _descriptor(entry_point) -> ExtensionDescriptor:
    return ExtensionDescriptor(
        name=entry_point.name,
        value=entry_point.value,
        distribution=_distribution_name(entry_point),
        distribution_version=_distribution_version(entry_point),
        loaded=entry_point.name in _LOADED,
    )


def _distribution_name(entry_point) -> str | None:
    distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        return None
    metadata_record = getattr(distribution, "metadata", {})
    return metadata_record.get("Name") or getattr(distribution, "name", None)


def _distribution_version(entry_point) -> str | None:
    distribution = getattr(entry_point, "dist", None)
    return None if distribution is None else getattr(distribution, "version", None)


def _coerce_extension(candidate) -> Extension:
    if isinstance(candidate, Extension):
        return candidate
    if callable(candidate) and not (
        hasattr(candidate, "spec") and hasattr(candidate, "register")
    ):
        candidate = candidate()
    spec = getattr(candidate, "spec", None)
    register = getattr(candidate, "register", None)
    if not isinstance(spec, ExtensionSpec) or not callable(register):
        raise TypeError(
            "An AgentFEM extension entry point must expose Extension(spec, register) "
            "or a zero-argument factory returning that object."
        )
    return Extension(spec=spec, register=register)


def _check_conflict(
    kind: str,
    name: str,
    existing: set[str],
    staged: set[str],
    *,
    replace: bool,
) -> None:
    if name in staged:
        raise ExtensionError(f"Extension stages duplicate {kind} {name!r}.")
    if name in existing and not replace:
        raise ExtensionError(
            f"Extension {kind} {name!r} conflicts with an existing registration."
        )


__all__ = [
    "ENTRY_POINT_GROUP",
    "EXTENSION_API_VERSION",
    "Extension",
    "ExtensionContext",
    "ExtensionDescriptor",
    "ExtensionError",
    "ExtensionSpec",
    "LoadedExtension",
    "discover_extensions",
    "extension_status",
    "load_extension",
    "load_extensions",
    "loaded_extensions",
    "missing_extensions",
]
