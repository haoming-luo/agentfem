"""Distributed diagnostics for finite-element fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from dolfinx import fem
from mpi4py import MPI

from . import fields as field_api
from .kernel import dofs


def comm_of(obj=None, default=MPI.COMM_WORLD):
    """Return the MPI communicator associated with an object when possible."""

    if obj is None:
        return default
    if hasattr(obj, "rank") and hasattr(obj, "size"):
        return obj
    if hasattr(obj, "comm"):
        return obj.comm
    if hasattr(obj, "mesh"):
        return obj.mesh.comm
    if hasattr(obj, "domain") and hasattr(obj.domain, "comm"):
        return obj.domain.comm
    if hasattr(obj, "function_space"):
        return obj.function_space.mesh.comm
    value = getattr(obj, "value", None)
    if value is not None and hasattr(value, "function_space"):
        return value.function_space.mesh.comm
    function = getattr(obj, "function", None)
    if function is not None and hasattr(function, "function_space"):
        return function.function_space.mesh.comm
    return default


def is_root(obj=None, *, root: int = 0) -> bool:
    """Return whether the current MPI rank is the selected reporting rank."""

    return comm_of(obj).rank == root


def print_on_root(obj, *args, root: int = 0, flush: bool = True, **kwargs) -> None:
    """Print a message only on the selected MPI root rank.

    ``flush`` defaults to ``True`` because long-running finite-element solves
    should show progress messages immediately.
    """

    if is_root(obj, root=root):
        print(*args, flush=flush, **kwargs)


def kinetic_energy(mass_lumped: np.ndarray, velocity: fem.Function) -> float:
    """Global kinetic energy from a lumped mass vector and velocity field."""

    velocity = field_api.unwrap(velocity)
    local = 0.5 * float(np.sum(mass_lumped * dofs.owned_array(velocity) ** 2))
    return velocity.function_space.mesh.comm.allreduce(local, op=MPI.SUM)


def max_abs(function: fem.Function) -> float:
    """Global max absolute value of a finite-element field."""

    function = field_api.unwrap(function)
    local = float(np.max(np.abs(function.x.array)))
    return function.function_space.mesh.comm.allreduce(local, op=MPI.MAX)


def max_magnitude(function) -> float:
    """Global maximum magnitude of a scalar or vector finite-element field."""

    return magnitude_stats(function).max


@dataclass(frozen=True)
class FieldStats:
    """Distributed scalar statistics for a finite-element field."""

    name: str
    maximum: float
    mean: float
    minimum: float = 0.0
    count: int = 0

    @property
    def max(self) -> float:
        """Compatibility alias for the maximum value."""

        return self.maximum

    def summary(self) -> dict[str, object]:
        """Return a compact agent-readable summary."""

        return {
            "name": self.name,
            "kind": "field_stats",
            "max": self.maximum,
            "mean": self.mean,
            "min": self.minimum,
            "count": self.count,
        }


@dataclass(frozen=True)
class ScalarDiagnostic:
    """Named scalar diagnostic evaluated on demand."""

    name: str
    value: Callable[[], float]

    def evaluate(self) -> float:
        return float(self.value())


@dataclass(frozen=True)
class DiagnosticSet:
    """Ordered collection of scalar diagnostics."""

    diagnostics: tuple[ScalarDiagnostic, ...]

    @classmethod
    def create(cls, *diagnostics: ScalarDiagnostic):
        return cls(tuple(diagnostics))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(diagnostic.name for diagnostic in self.diagnostics)

    def evaluate(self) -> dict[str, float]:
        return {
            diagnostic.name: diagnostic.evaluate()
            for diagnostic in self.diagnostics
        }


def magnitude_stats(function, *, on=None, name: str | None = None) -> FieldStats:
    """Return distributed magnitude statistics for a scalar or vector field.

    ``on`` may be a geometric marker callable or a named mesh region with a
    marker, such as ``mesh.face(...)``. Cell regions are not supported yet
    because nodal field statistics need a dof-selection marker.
    """

    function = field_api.unwrap(function)
    marker = getattr(on, "marker", on)
    if marker is None:
        values = _field_magnitudes(function)
    else:
        values = _field_magnitudes_on_marker(function, marker)
    comm = function.function_space.mesh.comm
    local_count = int(len(values))
    if local_count:
        local_max = float(np.max(values))
        local_min = float(np.min(values))
        local_sum = float(np.sum(values))
    else:
        local_max = 0.0
        local_min = np.inf
        local_sum = 0.0
    global_count = comm.allreduce(local_count, op=MPI.SUM)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    global_min = comm.allreduce(local_min, op=MPI.MIN)
    if global_count == 0:
        global_min = 0.0
    return FieldStats(
        name=name or f"{function.name}_magnitude",
        maximum=global_max,
        mean=0.0 if global_count == 0 else global_sum / global_count,
        minimum=float(global_min),
        count=int(global_count),
    )


def field_stats(function, *, on=None, name: str | None = None) -> FieldStats:
    """Alias for ``magnitude_stats`` for application-level diagnostics."""

    return magnitude_stats(function, on=on, name=name)


def _field_magnitudes(function) -> np.ndarray:
    values = dofs.owned_array(function)
    shape = getattr(function, "ufl_shape", ())
    if len(shape) == 1:
        dim = int(shape[0])
        if dim > 0 and len(values) % dim == 0:
            return np.linalg.norm(values.reshape((-1, dim)), axis=1)
    return np.abs(values)


def _field_magnitudes_on_marker(function, marker) -> np.ndarray:
    V = function.function_space
    values = function.x.array
    shape = getattr(function, "ufl_shape", ())
    if len(shape) != 1:
        dofs_selected = fem.locate_dofs_geometrical(V, marker)
        return np.abs(values[np.asarray(dofs_selected, dtype=np.int32)])

    components = []
    for component in range(V.num_sub_spaces):
        Vc, _ = V.sub(component).collapse()
        parent, _ = fem.locate_dofs_geometrical((V.sub(component), Vc), marker)
        components.append(values[np.asarray(parent, dtype=np.int32)])
    if not components or any(len(component) == 0 for component in components):
        return np.zeros(0, dtype=float)
    return np.sqrt(sum(component**2 for component in components))
