"""Experimental, versioned scientific records for AgentFEM.

AF-IR 0.1 records the supported public model semantics while the executable
implementation remains intentionally FEniCSx-first.  The schema will evolve
with migrations as semantic coverage grows.
"""

from .export import WORKFLOW_ORDER, describe, describe_many, model_document
from .schema import (
    AFIR_SCHEMA,
    AFIR_SCHEMA_VERSION,
    AFIR_STATUS,
    IRDocument,
    IRSerializationError,
    to_json_safe,
    write_document,
)
from .values import describe_value

__all__ = [
    "AFIR_SCHEMA",
    "AFIR_SCHEMA_VERSION",
    "AFIR_STATUS",
    "IRDocument",
    "IRSerializationError",
    "WORKFLOW_ORDER",
    "describe",
    "describe_many",
    "describe_value",
    "model_document",
    "to_json_safe",
    "write_document",
]
