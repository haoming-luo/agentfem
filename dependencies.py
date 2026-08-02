"""Optional-dependency boundaries with actionable, agent-readable errors.

AgentFEM deliberately keeps format converters, visualisation, and learning
libraries outside the numerical core. Optional imports should therefore fail
at the capability boundary, not while importing the public package.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, metadata


class OptionalDependencyError(ImportError):
    """Raised when a requested optional capability is not installed."""

    def __init__(self, *, package: str, extra: str, capability: str):
        self.package = package
        self.extra = extra
        self.capability = capability
        super().__init__(
            f"{capability} requires optional package {package!r}. "
            f"Install it with `python -m pip install 'agentfem[{extra}]'` "
            "inside the active FEniCSx environment."
        )


@dataclass(frozen=True)
class DependencyStatus:
    """Inspectable availability record for one optional integration."""

    package: str
    extra: str
    capability: str
    available: bool
    version: str | None = None

    def summary(self) -> dict[str, object]:
        return {
            "package": self.package,
            "extra": self.extra,
            "capability": self.capability,
            "available": self.available,
            "version": self.version,
        }


def require(package: str, *, extra: str, capability: str):
    """Import an optional package or raise an installation-specific error."""

    try:
        return import_module(package)
    except (ImportError, OSError) as exc:
        # OSError covers installed binary wheels whose shared libraries cannot
        # be loaded. For users this is still an unavailable capability.
        raise OptionalDependencyError(
            package=package,
            extra=extra,
            capability=capability,
        ) from exc


def status(package: str, *, extra: str, capability: str) -> DependencyStatus:
    """Return package availability without importing compiled extensions."""

    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        return DependencyStatus(package, extra, capability, False)
    return DependencyStatus(package, extra, capability, True, version)


__all__ = [
    "DependencyStatus",
    "OptionalDependencyError",
    "require",
    "status",
]
