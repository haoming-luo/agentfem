"""Backend contracts for progressive AgentFEM lowering.

The contract is intentionally small.  It creates a real compilation seam
without pretending that all AgentFEM semantics are backend independent today.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


BACKEND_API_VERSION = "0.1"


@dataclass(frozen=True)
class BackendDescriptor:
    """Inspectable backend identity and capability statement."""

    name: str
    version: str
    capabilities: tuple[str, ...]
    api_version: str = BACKEND_API_VERSION
    status: str = "available"
    notes: str | None = None

    def supports(self, capability: str) -> bool:
        """Return whether the backend explicitly advertises a capability."""

        return capability in self.capabilities

    def as_dict(self) -> dict[str, object]:
        """Return a stable backend record for AF-IR and provenance."""

        result: dict[str, object] = {
            "name": self.name,
            "version": self.version,
            "api_version": self.api_version,
            "status": self.status,
            "capabilities": self.capabilities,
        }
        if self.notes is not None:
            result["notes"] = self.notes
        return result

    def summary(self) -> dict[str, object]:
        return self.as_dict()


class BackendAdapter(ABC):
    """Minimal interface used by operator compilation and assembly."""

    @property
    @abstractmethod
    def descriptor(self) -> BackendDescriptor:
        """Return backend identity and supported lowering operations."""

    @abstractmethod
    def compile_form(self, expression):
        """Compile a backend expression into an executable form."""

    @abstractmethod
    def assemble_matrix(self, expression, *, bcs=None):
        """Compile and assemble a matrix-valued expression."""

    @abstractmethod
    def assemble_vector(self, expression):
        """Compile and assemble a vector-valued expression."""


__all__ = ["BACKEND_API_VERSION", "BackendAdapter", "BackendDescriptor"]
