"""Dataset schema primitives with units, shapes, and provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Quantity:
    """One scalar, curve, vector, or sampled-field output contract."""

    name: str
    shape: tuple[int, ...] = ()
    unit: str | None = None
    kind: str = "quantity_of_interest"
    description: str = ""
    field_encoding: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("Quantity.name must not be empty.")
        shape = tuple(int(value) for value in self.shape)
        if any(value <= 0 for value in shape):
            raise ValueError("Quantity.shape dimensions must be positive.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "kind", str(self.kind).strip() or "quantity_of_interest")
        object.__setattr__(
            self,
            "field_encoding",
            None if self.field_encoding is None else dict(self.field_encoding),
        )
        if self.kind in {"field", "sampled_field"} and self.field_encoding is None:
            raise ValueError(
                f"Field quantity {self.name!r} requires field_encoding metadata."
            )

    @property
    def size(self) -> int:
        return prod(self.shape) if self.shape else 1

    def validate(self, value) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.shape != self.shape:
            raise ValueError(
                f"Output {self.name!r} requires shape {self.shape}, got {array.shape}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Output {self.name!r} contains non-finite values.")
        return array

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": self.shape,
            "size": self.size,
            "unit": self.unit,
            "kind": self.kind,
            "description": self.description,
            "field_encoding": self.field_encoding,
        }


@dataclass(frozen=True)
class Sample:
    """One successful simulation sample and its scientific lineage."""

    case_id: str
    inputs: Mapping[str, object]
    outputs: Mapping[str, object]
    provenance: Mapping[str, object] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        case_id = str(self.case_id).strip()
        if not case_id:
            raise ValueError("Sample.case_id must not be empty.")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "inputs", dict(self.inputs))
        object.__setattr__(self, "outputs", dict(self.outputs))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "artifacts", dict(self.artifacts))


def decode_quantities(
    quantities: tuple[Quantity, ...],
    row,
) -> dict[str, object]:
    """Restore one flattened numeric row to declared named quantities."""

    selected_quantities = tuple(quantities)
    values = np.asarray(row, dtype=float).reshape(-1)
    expected = sum(quantity.size for quantity in selected_quantities)
    if values.size != expected:
        raise ValueError(f"Expected {expected} output values, got {values.size}.")
    result: dict[str, object] = {}
    offset = 0
    for quantity in selected_quantities:
        selected = values[offset : offset + quantity.size].reshape(quantity.shape)
        result[quantity.name] = float(selected) if quantity.shape == () else selected
        offset += quantity.size
    return result


__all__ = ["Quantity", "Sample", "decode_quantities"]
