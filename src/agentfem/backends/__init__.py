"""Backend capability and lowering interfaces.

AgentFEM remains FEniCSx-first.  This module exposes a narrow, honest seam for
compilation and assembly so future backends can be evaluated against explicit
scientific semantics instead of being anticipated through generic wrappers.
"""

from .base import BACKEND_API_VERSION, BackendAdapter, BackendDescriptor
from .fenicsx import FEniCSxBackend
from .registry import (
    available_backends,
    backend_descriptors,
    default_backend_name,
    get_backend,
    register_backend,
    set_default_backend,
)
from .runtime import (
    RuntimeCapabilityError,
    RuntimeProfile,
    RuntimeSelectionError,
    current_runtime,
    require_capabilities,
)


register_backend("fenicsx", FEniCSxBackend)


__all__ = [
    "BACKEND_API_VERSION",
    "BackendAdapter",
    "BackendDescriptor",
    "FEniCSxBackend",
    "available_backends",
    "backend_descriptors",
    "default_backend_name",
    "get_backend",
    "register_backend",
    "set_default_backend",
    "RuntimeCapabilityError",
    "RuntimeProfile",
    "RuntimeSelectionError",
    "current_runtime",
    "require_capabilities",
]
