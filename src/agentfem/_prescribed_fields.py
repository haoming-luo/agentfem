# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Frozen reference-coordinate inputs in the existing quadrature point order."""
from dataclasses import dataclass
from hashlib import sha256
import numpy as np


@dataclass(frozen=True)
class PrescribedField:
    """Sample a scalar/FE field or source(X, time) on declared physical knots.

    X has shape (points, geometric_dimension); returned samples have shape
    (points, *value_shape). Values are frozen when bound to a quadrature field,
    then interpolated linearly in physical time. This is a prescribed input,
    not a transport equation or a live/current-coordinate coupling.
    """
    source: object
    name: str
    unit: str
    times: tuple[float, ...]
    value_shape: tuple[int, ...] = ()
    coordinate: str = "reference"
    time_unit: str | None = None

    def __post_init__(self):
        times = np.asarray(self.times, dtype=float)
        if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("AFM-PRESET-TIME: at least two finite increasing physical times required.")
        if not isinstance(self.name, str) or not self.name.strip() or not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("AFM-PRESET-IDENTITY: declare field name and unit.")
        if self.time_unit is not None and (not isinstance(self.time_unit, str) or not self.time_unit.strip()):
            raise ValueError("AFM-PRESET-TIME: time_unit must be a nonempty unit label.")
        if self.coordinate != "reference":
            raise NotImplementedError("AFM-PRESET-COORDINATE: only frozen reference coordinates are admitted.")
        if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in self.value_shape):
            raise ValueError("AFM-PRESET-SHAPE: value_shape requires positive integer dimensions.")
        object.__setattr__(self, "times", tuple(float(t) for t in times))
        object.__setattr__(self, "value_shape", tuple(self.value_shape))

    def summary(self):
        return {"kind": "prescribed_field_declaration", "name": self.name, "unit": self.unit,
            "times": list(self.times), "time_unit": self.time_unit,
            "coordinate": self.coordinate, "value_shape": list(self.value_shape),
            "interpolation": "linear", "sampling": "frozen_when_bound_to_quadrature"}

    def bind(self, quadrature):
        """Collectively freeze samples, retaining cell/point layout and identity."""
        from dolfinx import fem
        import ufl
        from .fields import unwrap
        domain = quadrature.function.function_space.mesh
        cells = domain.topology.index_map(domain.topology.dim)
        indices = np.arange(cells.size_local + cells.num_ghosts, dtype=np.int32)
        X = np.asarray(fem.Expression(ufl.SpatialCoordinate(domain), quadrature.points).eval(domain, indices))
        X = X.reshape(-1, domain.geometry.dim)
        source = unwrap(self.source)
        contract = (self.name, self.unit, self.times, self.value_shape, self.time_unit,
            "fe" if hasattr(source, "ufl_shape") else "callable" if callable(source) else "constant")
        contracts = domain.comm.allgather(contract)
        if any(item != contracts[0] for item in contracts):
            raise ValueError("AFM-PRESET-MAPPING: field contracts/source kinds must agree on every rank.")
        source_error = None
        if hasattr(source, "ufl_shape") and (
            source.ufl_shape != self.value_shape or ufl.domain.extract_unique_domain(source) != domain.ufl_domain()
        ):
            source_error = "FE input must have the declared value shape and the same reference mesh."
        source_errors = domain.comm.allgather(source_error)
        if any(source_errors):
            raise ValueError("AFM-PRESET-MAPPING: " + "; ".join(e for e in source_errors if e))
        fe_values = None
        if hasattr(source, "ufl_shape"):
            if hasattr(source, "x"):
                source.x.scatter_forward()
            fe_values = np.asarray(fem.Expression(source, quadrature.points).eval(domain, indices))
            fe_values = fe_values.reshape((len(X), *self.value_shape))
        error, values = None, None
        try:
            samples = []
            for time in self.times:
                if fe_values is not None:
                    sample = fe_values
                elif callable(source):
                    sample = source(X.copy(), time)
                else:
                    sample = np.broadcast_to(source, (len(X), *self.value_shape))
                if np.iscomplexobj(sample):
                    raise ValueError("Prescribed samples must be real.")
                sample = np.asarray(sample, dtype=float)
                if sample.shape != (len(X), *self.value_shape) or not np.isfinite(sample).all():
                    raise ValueError(f"Expected finite samples with shape {(len(X), *self.value_shape)}.")
                samples.append(sample.copy())
            values = np.asarray(samples)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = domain.comm.allgather(error)
        if any(errors):
            raise ValueError("AFM-PRESET-MAPPING: " + "; ".join(e for e in errors if e))
        return FrozenPrescribedField(self, values, X, quadrature.owned_cell_keys,
            len(quadrature.points), int(cells.size_local))


class FrozenPrescribedField:
    """Detached knot values; source mutation cannot alter a bound input."""
    def __init__(self, declaration, values, coordinates, cell_keys, points_per_cell, owned_cells):
        self.declaration = declaration
        self.times = declaration.times
        self.values = np.frombuffer(values.tobytes(), dtype=float).reshape(values.shape)
        self.coordinates = np.frombuffer(coordinates.tobytes(), dtype=float).reshape(coordinates.shape)
        self.cell_keys = tuple(int(k) for k in cell_keys)
        self.points_per_cell, self.owned_cells = points_per_cell, owned_cells

    def at(self, time):
        time = float(time)
        if not np.isfinite(time) or not self.times[0] <= time <= self.times[-1]:
            raise ValueError("AFM-PRESET-TIME: time outside frozen interval; extrapolation is not admitted.")
        right = min(int(np.searchsorted(self.times, time, side="right")), len(self.times)-1)
        left = max(0, right-1)
        weight = (time-self.times[left]) / (self.times[right]-self.times[left])
        return (1-weight)*self.values[left] + weight*self.values[right]

    def summary(self):
        source = self.declaration
        digest = sha256(self.values.tobytes() + self.coordinates.tobytes()).hexdigest()
        return {"kind": "prescribed_quadrature_field", "name": source.name,
            "unit": source.unit, "coordinate": source.coordinate, "time_unit": source.time_unit,
            "times": list(self.times), "interpolation": "linear", "values_sha256": digest,
            "value_shape": list(source.value_shape), "cell_keys": list(self.cell_keys),
            "points_per_cell": self.points_per_cell, "owned_cells": self.owned_cells,
            "layout": "partition_cell_point_component", "source": "frozen_at_construction"}


def prescribed(source, *, name, unit, times, value_shape=(), coordinate="reference", time_unit=None):
    """Declare a reference-coordinate input for quadrature mapping, without solving."""
    return PrescribedField(source, name, unit, tuple(times), tuple(value_shape), coordinate, time_unit)
