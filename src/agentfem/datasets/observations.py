"""Portable rectilinear observations for FEM, experiments, and publications."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RectilinearObservation:
    """One scalar field on explicit physical ``x``/``y`` axes.

    ``values`` follows image/scientific-array layout ``(len(y), len(x))``.
    This is intentionally different from the internal neural-operator layout,
    whose axes follow the declared :class:`ObservationGrid` order.  The
    conversion is explicit so a transpose can never be hidden in a figure.
    """

    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    quantity: str
    unit: str | None = None
    coordinate_names: tuple[str, str] = ("x", "y")
    coordinate_system: str = "cartesian"
    coordinate_unit: str | None = None
    configuration: str = "reference"
    mask: np.ndarray | None = None
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float).reshape(-1)
        y = np.asarray(self.y, dtype=float).reshape(-1)
        values = np.asarray(self.values, dtype=float)
        if x.size < 2 or y.size < 2:
            raise ValueError("RectilinearObservation axes need at least two points.")
        if (
            np.any(~np.isfinite(x))
            or np.any(~np.isfinite(y))
            or np.any(np.diff(x) <= 0.0)
            or np.any(np.diff(y) <= 0.0)
        ):
            raise ValueError("RectilinearObservation axes must be finite and increasing.")
        expected = (y.size, x.size)
        if values.shape != expected or np.any(~np.isfinite(values)):
            raise ValueError(
                f"RectilinearObservation.values must be finite with shape {expected}."
            )
        names = tuple(str(name).strip() for name in self.coordinate_names)
        if len(names) != 2 or any(not name for name in names) or names[0] == names[1]:
            raise ValueError("RectilinearObservation needs two distinct coordinate names.")
        quantity = str(self.quantity).strip()
        if not quantity:
            raise ValueError("RectilinearObservation.quantity must not be empty.")
        coordinate_system = str(self.coordinate_system).strip()
        if not coordinate_system:
            raise ValueError("RectilinearObservation.coordinate_system must not be empty.")
        coordinate_unit = (
            None if self.coordinate_unit is None else str(self.coordinate_unit).strip()
        )
        if coordinate_unit == "":
            raise ValueError("RectilinearObservation.coordinate_unit must not be empty.")
        mask = None if self.mask is None else np.asarray(self.mask, dtype=bool)
        if mask is not None and mask.shape != expected:
            raise ValueError(f"RectilinearObservation.mask must have shape {expected}.")
        configuration = str(self.configuration).strip().lower().replace("-", "_")
        if configuration not in {"reference", "current"}:
            raise ValueError(
                "RectilinearObservation.configuration must be 'reference' or 'current'."
            )
        object.__setattr__(self, "x", x.copy())
        object.__setattr__(self, "y", y.copy())
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "coordinate_names", names)
        object.__setattr__(self, "coordinate_system", coordinate_system)
        object.__setattr__(self, "coordinate_unit", coordinate_unit)
        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(self, "mask", None if mask is None else mask.copy())
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_field_sample(
        cls,
        sample,
        *,
        component: int | None = None,
        quantity: str | None = None,
    ) -> "RectilinearObservation":
        """Convert a 2D structured :class:`FEMFieldSample` without ambiguity."""

        encoding = dict(sample.encoding)
        if encoding.get("representation") != "structured_grid":
            raise ValueError("Rectilinear conversion requires a structured-grid sample.")
        details = dict(encoding.get("metadata", {}))
        grid = dict(details.get("observation_grid", {}))
        names = tuple(grid.get("axis_names", ()))
        axes = dict(grid.get("axes", {}))
        if len(names) != 2 or any(name not in axes for name in names):
            raise ValueError("Rectilinear conversion requires exactly two recorded axes.")
        values = np.asarray(sample.values, dtype=float)
        grid_shape = tuple(int(value) for value in grid.get("shape", ()))
        if values.shape[:2] != grid_shape:
            raise ValueError("Field-sample values do not match the recorded observation grid.")
        trailing = values.shape[2:]
        if trailing:
            flat = values.reshape((*grid_shape, -1))
            if component is None:
                if flat.shape[-1] != 1:
                    raise ValueError(
                        "A vector/tensor field requires an explicit flattened component."
                    )
                selected = flat[..., 0]
            else:
                index = int(component)
                if not 0 <= index < flat.shape[-1]:
                    raise ValueError("Requested field component is out of range.")
                selected = flat[..., index]
        else:
            if component not in {None, 0}:
                raise ValueError("A scalar field only has component 0.")
            selected = values
        mask = None
        if sample.mask is not None:
            mask = np.asarray(sample.mask, dtype=bool).T
        return cls(
            x=np.asarray(axes[names[0]], dtype=float),
            y=np.asarray(axes[names[1]], dtype=float),
            values=np.asarray(selected, dtype=float).T,
            quantity=str(quantity or encoding.get("name") or "field"),
            unit=encoding.get("unit"),
            coordinate_names=(names[0], names[1]),
            coordinate_system=str(grid.get("coordinate_system", "cartesian")),
            coordinate_unit=grid.get("coordinate_unit"),
            configuration=str(details.get("configuration", "reference")),
            mask=mask,
            metadata={
                "source": "FEMFieldSample",
                "field_sample_metadata": dict(sample.metadata),
                "coordinate_map": details.get("coordinate_map"),
                "component": component,
            },
        )

    def write(self, path: str | Path) -> Path:
        location = Path(path)
        if location.suffix.lower() != ".npz":
            location = location.with_suffix(".npz")
        location.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, object] = {
            "schema": np.asarray("agentfem.rectilinear-observation.v1"),
            "x": self.x,
            "y": self.y,
            "values": self.values,
            "record_json": np.asarray(
                json.dumps(
                    {
                        "quantity": self.quantity,
                        "unit": self.unit,
                        "coordinate_names": self.coordinate_names,
                        "coordinate_system": self.coordinate_system,
                        "coordinate_unit": self.coordinate_unit,
                        "configuration": self.configuration,
                        "metadata": dict(self.metadata or {}),
                    },
                    sort_keys=True,
                )
            ),
        }
        if self.mask is not None:
            arrays["mask"] = self.mask
        np.savez_compressed(location, **arrays)
        return location

    @classmethod
    def read(cls, path: str | Path) -> "RectilinearObservation":
        with np.load(Path(path), allow_pickle=False) as archive:
            if str(archive["schema"]) != "agentfem.rectilinear-observation.v1":
                raise ValueError("Unsupported rectilinear-observation schema.")
            record = json.loads(str(archive["record_json"]))
            return cls(
                x=archive["x"],
                y=archive["y"],
                values=archive["values"],
                quantity=record["quantity"],
                unit=record.get("unit"),
                coordinate_names=tuple(record["coordinate_names"]),
                coordinate_system=record["coordinate_system"],
                coordinate_unit=record.get("coordinate_unit"),
                configuration=record["configuration"],
                mask=archive["mask"] if "mask" in archive.files else None,
                metadata=record.get("metadata", {}),
            )

    def summary(self) -> dict[str, object]:
        valid = (
            self.values.size
            if self.mask is None
            else int(np.count_nonzero(self.mask))
        )
        return {
            "kind": "rectilinear_observation",
            "quantity": self.quantity,
            "unit": self.unit,
            "shape": self.values.shape,
            "valid_samples": valid,
            "coordinate_names": self.coordinate_names,
            "coordinate_system": self.coordinate_system,
            "coordinate_unit": self.coordinate_unit,
            "configuration": self.configuration,
            "bounds": [
                [float(self.x[0]), float(self.x[-1])],
                [float(self.y[0]), float(self.y[-1])],
            ],
            "metadata": dict(self.metadata or {}),
        }


__all__ = ["RectilinearObservation"]
