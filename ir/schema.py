"""Versioned AF-IR document primitives.

The first AF-IR release is deliberately a faithful record of the public
AgentFEM model rather than a claim of backend-neutral executable coverage.
Documents are JSON-safe, versioned, and explicit about their maturity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from math import isfinite
from pathlib import Path
from typing import Mapping


AFIR_SCHEMA = "agentfem.af-ir"
AFIR_SCHEMA_VERSION = "0.1.0"
AFIR_STATUS = "experimental"


class IRSerializationError(ValueError):
    """Raised when a value cannot be represented without hiding its meaning."""


@dataclass(frozen=True)
class IRDocument:
    """Canonical envelope for an AF-IR artifact."""

    document_type: str
    root: Mapping[str, object]
    generator: Mapping[str, object]
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema: str = AFIR_SCHEMA
    schema_version: str = AFIR_SCHEMA_VERSION
    status: str = AFIR_STATUS

    def __post_init__(self) -> None:
        if not self.document_type:
            raise ValueError("IRDocument.document_type must not be empty.")
        if self.schema != AFIR_SCHEMA:
            raise ValueError(
                f"IRDocument schema must be {AFIR_SCHEMA!r}, got {self.schema!r}."
            )

    def as_dict(self) -> dict[str, object]:
        """Return the canonical JSON-safe document mapping."""

        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "status": self.status,
            "document_type": self.document_type,
            "generator": to_json_safe(self.generator, path="generator"),
            "metadata": to_json_safe(self.metadata, path="metadata"),
            "root": to_json_safe(self.root, path="root"),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize with deterministic key ordering."""

        return json.dumps(
            self.as_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )


def write_document(
    document: IRDocument | Mapping[str, object],
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """Write one deterministic AF-IR JSON document and return its path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(document, IRDocument):
        text = document.to_json(indent=indent)
    else:
        text = json.dumps(
            to_json_safe(document),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    output.write_text(text + "\n", encoding="utf-8")
    return output


def to_json_safe(value, *, path: str = "$"):
    """Convert scientific summaries to deterministic JSON-safe values.

    Backend runtime objects are represented by a typed opaque marker rather
    than by ``repr()``, whose memory addresses are unstable and scientifically
    meaningless.  Public objects should expose ``to_ir()``, ``as_dict()``, or
    ``summary()`` to retain richer semantics.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise IRSerializationError(f"{path} contains non-finite float {value!r}.")
        return value
    if isinstance(value, Enum):
        return to_json_safe(value.value, path=path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result = {}
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, (str, int, float, bool)):
                raise IRSerializationError(
                    f"{path} has unsupported mapping key type {type(key).__name__}."
                )
            result[str(key)] = to_json_safe(value[key], path=f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list, set, frozenset)):
        return [
            to_json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return to_json_safe(item_method(), path=path)
        except (TypeError, ValueError):
            pass
    list_method = getattr(value, "tolist", None)
    if callable(list_method):
        try:
            return to_json_safe(list_method(), path=path)
        except (TypeError, ValueError):
            pass

    for method_name in ("to_ir", "as_dict", "summary"):
        method = getattr(value, method_name, None)
        if callable(method):
            converted = method()
            if converted is value:
                break
            return to_json_safe(converted, path=path)

    value_type = type(value)
    return {
        "kind": "opaque_runtime_object",
        "python_type": f"{value_type.__module__}.{value_type.__qualname__}",
        "serializable": False,
    }


__all__ = [
    "AFIR_SCHEMA",
    "AFIR_SCHEMA_VERSION",
    "AFIR_STATUS",
    "IRDocument",
    "IRSerializationError",
    "to_json_safe",
    "write_document",
]
