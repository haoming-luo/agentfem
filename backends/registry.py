"""Explicit backend registration and default selection."""

from __future__ import annotations

from collections.abc import Callable

from .base import BackendAdapter, BackendDescriptor


BackendFactory = Callable[[], BackendAdapter]
_FACTORIES: dict[str, BackendFactory] = {}
_INSTANCES: dict[str, BackendAdapter] = {}
_DEFAULT_BACKEND = "fenicsx"


def register_backend(
    name: str,
    factory: BackendFactory,
    *,
    overwrite: bool = False,
) -> None:
    """Register a lazy backend factory.

    Registration is intentionally explicit so importing AgentFEM does not
    trigger every optional backend dependency.
    """

    normalized = _normalize(name)
    if normalized in _FACTORIES and not overwrite:
        raise ValueError(f"Backend {normalized!r} is already registered.")
    _FACTORIES[normalized] = factory
    _INSTANCES.pop(normalized, None)


def get_backend(name: str | None = None) -> BackendAdapter:
    """Return a lazily constructed backend adapter."""

    normalized = _normalize(name or _DEFAULT_BACKEND)
    if normalized not in _FACTORIES:
        raise KeyError(
            f"Unknown backend {normalized!r}. Registered backends: "
            f"{sorted(_FACTORIES)}."
        )
    if normalized not in _INSTANCES:
        backend = _FACTORIES[normalized]()
        if not isinstance(backend, BackendAdapter):
            raise TypeError(
                f"Backend factory {normalized!r} returned "
                f"{type(backend).__name__}, expected BackendAdapter."
            )
        _INSTANCES[normalized] = backend
    return _INSTANCES[normalized]


def set_default_backend(name: str) -> None:
    """Select the process-local default backend by registered name."""

    global _DEFAULT_BACKEND
    normalized = _normalize(name)
    if normalized not in _FACTORIES:
        raise KeyError(
            f"Cannot select unknown backend {normalized!r}. "
            f"Registered backends: {sorted(_FACTORIES)}."
        )
    _DEFAULT_BACKEND = normalized


def default_backend_name() -> str:
    return _DEFAULT_BACKEND


def available_backends() -> tuple[str, ...]:
    """Return registered backend names without importing their dependencies."""

    return tuple(sorted(_FACTORIES))


def backend_descriptors() -> tuple[BackendDescriptor, ...]:
    """Return descriptors for all registered backends."""

    return tuple(get_backend(name).descriptor for name in available_backends())


def _normalize(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("Backend name must not be empty.")
    return normalized


__all__ = [
    "available_backends",
    "backend_descriptors",
    "default_backend_name",
    "get_backend",
    "register_backend",
    "set_default_backend",
]
