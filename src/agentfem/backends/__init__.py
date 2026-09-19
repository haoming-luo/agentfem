# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Backend capability and lowering interfaces.

AgentFEM remains FEniCSx-first.  This module exposes a narrow, honest seam for
compilation and assembly so future backends can be evaluated against explicit
scientific semantics instead of being anticipated through generic wrappers.
"""

from .base import BACKEND_API_VERSION, BackendAdapter, BackendDescriptor
from ._additive import AdditiveTangentMatrix, create_additive_tangent_matrix
from ._fenicsx_nonlinear import FEniCSxTangentAction, fenicsx_tangent_action
from .fenicsx import FEniCSxBackend
from .registry import (
    available_backends,
    backend_descriptors,
    default_backend_name,
    get_backend,
    register_backend,
    set_default_backend,
)


register_backend("fenicsx", FEniCSxBackend)


__all__ = [
    "BACKEND_API_VERSION",
    "AdditiveTangentMatrix",
    "BackendAdapter",
    "BackendDescriptor",
    "FEniCSxBackend",
    "FEniCSxTangentAction",
    "available_backends",
    "backend_descriptors",
    "default_backend_name",
    "get_backend",
    "fenicsx_tangent_action",
    "register_backend",
    "set_default_backend",
    "create_additive_tangent_matrix",
]
