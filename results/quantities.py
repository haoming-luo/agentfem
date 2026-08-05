"""MPI-safe quantities of interest assembled from finite-element expressions."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import sqrt

import numpy as np
import basix
import ufl
from dolfinx import fem
from dolfinx import geometry as geometry_api
from mpi4py import MPI

from .. import fields as field_api
from ..kernel import dofs


@dataclass(frozen=True)
class PathSample:
    """Values sampled along one straight physical-space path."""

    coordinates: np.ndarray
    distance: np.ndarray
    values: np.ndarray
    field_name: str

    def add_to(
        self,
        result,
        *,
        name: str | None = None,
        unit: str | None = None,
        distance_unit: str | None = None,
        description: str = "",
    ):
        """Attach the path as a standard result history and return it."""

        selected_name = name or f"{self.field_name}_path"
        selected_description = description or (
            f"{self.field_name} sampled along a straight physical-space path."
        )
        return result.add_history(
            selected_name,
            self.distance,
            self.values,
            unit=unit,
            abscissa_name="distance",
            abscissa_unit=distance_unit,
            description=selected_description,
        )


def integral(expression, *, measure=ufl.dx, comm=None):
    """Return the global integral of a scalar, vector, or tensor expression."""

    shape = tuple(getattr(expression, "ufl_shape", ()))
    selected_comm = comm or _comm_from_measure(measure)
    if not shape:
        return _assemble_component(expression, measure, selected_comm)
    values = np.empty(shape, dtype=float)
    for index in np.ndindex(shape):
        values[index] = _assemble_component(
            expression[index],
            measure,
            selected_comm,
        )
    return values


def average(expression, *, measure=ufl.dx, comm=None):
    """Return the measure-weighted global average of an expression."""

    selected_comm = comm or _comm_from_measure(measure)
    volume = _assemble_component(ufl.as_ufl(1.0), measure, selected_comm)
    if volume <= 0.0:
        raise ValueError("average requires a measure with positive total weight.")
    return integral(expression, measure=measure, comm=selected_comm) / volume


def l2_norm(expression, *, measure=ufl.dx, comm=None) -> float:
    """Return ``sqrt(integral(inner(value, value)))`` globally."""

    selected_comm = comm or _comm_from_measure(measure)
    squared = _assemble_component(
        ufl.inner(expression, expression),
        measure,
        selected_comm,
    )
    return sqrt(max(0.0, squared))


def quadrature_extrema(
    expression,
    domain,
    *,
    degree: int = 4,
) -> tuple[float, float]:
    """Return global min/max sampled at Basix quadrature points.

    This is useful for bounded nonlinear diagnostics such as ``det(F)``.  It is
    a quadrature-point diagnostic, not a mathematical proof of an element-wise
    bound.
    """

    if tuple(getattr(expression, "ufl_shape", ())):
        raise ValueError("quadrature_extrema currently requires a scalar expression.")
    points, _ = basix.make_quadrature(domain.basix_cell(), int(degree))
    evaluator = fem.Expression(expression, points)
    cells = np.arange(
        domain.topology.index_map(domain.topology.dim).size_local,
        dtype=np.int32,
    )
    values = np.asarray(evaluator.eval(domain, cells), dtype=float)
    local_min = float(np.min(values)) if values.size else np.inf
    local_max = float(np.max(values)) if values.size else -np.inf
    return (
        float(domain.comm.allreduce(local_min, op=MPI.MIN)),
        float(domain.comm.allreduce(local_max, op=MPI.MAX)),
    )


def region_integral(expression, *, on):
    """Integrate a scalar, vector, or tensor over a named mesh region."""

    return integral(expression, measure=_region_measure(on))


def region_average(expression, *, on):
    """Return a measure-weighted average over a named mesh region."""

    return average(expression, measure=_region_measure(on))


def boundary_resultant(traction, *, on):
    """Integrate a traction/flux expression over a named boundary."""

    return integral(traction, measure=_region_measure(on))


def field_extrema(field, *, magnitude: bool = False) -> dict[str, object]:
    """Return MPI-global extrema of owned field dofs or nodal magnitudes."""

    function = field_api.unwrap(field)
    values = np.asarray(dofs.owned_array(function), dtype=float)
    shape = tuple(getattr(function, "ufl_shape", ()))
    if magnitude:
        if len(shape) != 1:
            raise ValueError("field_extrema(magnitude=True) requires a vector field.")
        components = int(shape[0])
        if values.size % components:
            raise ValueError("Vector dof storage is incompatible with its value shape.")
        values = np.linalg.norm(values.reshape(-1, components), axis=1)
    local_min = float(np.min(values)) if values.size else np.inf
    local_max = float(np.max(values)) if values.size else -np.inf
    comm = function.function_space.mesh.comm
    return {
        "minimum": float(comm.allreduce(local_min, op=MPI.MIN)),
        "maximum": float(comm.allreduce(local_max, op=MPI.MAX)),
        "magnitude": bool(magnitude),
    }


def probe(field, *, at, padding: float = 1.0e-10):
    """Return one scalar, vector, or tensor field value at a physical point."""

    function = field_api.unwrap(field)
    point = _single_point(at, function.function_space.mesh)
    value = sample_points(function, point, padding=padding)[0]
    return value.item() if np.asarray(value).ndim == 0 else np.asarray(value).copy()


def sample_points(
    field,
    points,
    *,
    padding: float = 1.0e-10,
    missing: str = "raise",
) -> np.ndarray:
    """Evaluate a finite-element field at common physical points under MPI.

    Every rank must call this function with identical point coordinates.  One
    deterministic rank evaluates each point in an owned cell, then the values
    are shared collectively.  For discontinuous fields, a point exactly on an
    interelement boundary uses the lowest-rank, lowest-local-cell candidate;
    sample inside the intended cell when a one-sided value is required.
    """

    function = field_api.unwrap(field)
    domain = function.function_space.mesh
    comm = domain.comm
    selected_padding = float(padding)
    if selected_padding < 0.0:
        raise ValueError("sample_points padding must be non-negative.")
    selected_missing = str(missing).lower()
    if selected_missing not in {"raise", "nan"}:
        raise ValueError("sample_points missing must be 'raise' or 'nan'.")
    coordinates = _point_array(points, domain)
    _require_collective_points(coordinates, comm)
    point_count = int(coordinates.shape[0])
    if point_count == 0:
        shape = tuple(getattr(function, "ufl_shape", ()))
        return np.empty((0, *shape), dtype=function.x.array.dtype)

    topology = domain.topology
    owned_count = int(topology.index_map(topology.dim).size_local)
    local_cells = np.full(point_count, -1, dtype=np.int32)
    if owned_count:
        owned_cells = np.arange(owned_count, dtype=np.int32)
        tree = geometry_api.bb_tree(
            domain,
            topology.dim,
            padding=selected_padding,
            entities=owned_cells,
        )
        candidates = geometry_api.compute_collisions_points(tree, coordinates)
        collisions = geometry_api.compute_colliding_cells(
            domain,
            candidates,
            coordinates,
        )
        for index in range(point_count):
            cells = np.asarray(collisions.links(index), dtype=np.int32)
            cells = cells[cells < owned_count]
            if cells.size:
                local_cells[index] = int(np.min(cells))

    local_owner = np.where(local_cells >= 0, comm.rank, comm.size).astype(np.int32)
    owner = np.empty_like(local_owner)
    comm.Allreduce(local_owner, owner, op=MPI.MIN)
    missing_indices = np.flatnonzero(owner == comm.size)
    if missing_indices.size and selected_missing == "raise":
        listed = ", ".join(str(int(index)) for index in missing_indices[:8])
        raise ValueError(
            "sample_points could not locate point indices "
            f"[{listed}] in the mesh."
        )

    value_shape = tuple(getattr(function, "ufl_shape", ()))
    value_size = int(np.prod(value_shape, dtype=int)) if value_shape else 1
    local_values = np.zeros(
        (point_count, value_size),
        dtype=function.x.array.dtype,
    )
    selected = np.flatnonzero(owner == comm.rank)
    if selected.size:
        evaluated = np.asarray(
            function.eval(coordinates[selected], local_cells[selected])
        ).reshape(selected.size, value_size)
        local_values[selected] = evaluated
    values = np.empty_like(local_values)
    comm.Allreduce(local_values, values, op=MPI.SUM)
    if missing_indices.size:
        values[missing_indices] = np.nan
    if value_shape:
        return values.reshape((point_count, *value_shape))
    return values[:, 0]


def sample_path(
    field,
    *,
    start,
    end,
    count: int = 101,
    padding: float = 1.0e-10,
    missing: str = "raise",
) -> PathSample:
    """Sample a field along the straight segment from ``start`` to ``end``."""

    function = field_api.unwrap(field)
    domain = function.function_space.mesh
    selected_count = int(count)
    if selected_count < 2:
        raise ValueError("sample_path count must be at least two.")
    start_point = _single_point(start, domain)[0, : domain.geometry.dim]
    end_point = _single_point(end, domain)[0, : domain.geometry.dim]
    coordinates = np.linspace(start_point, end_point, selected_count)
    values = sample_points(
        function,
        coordinates,
        padding=padding,
        missing=missing,
    )
    distance = np.linalg.norm(coordinates - coordinates[0], axis=1)
    return PathSample(
        coordinates=coordinates,
        distance=distance,
        values=values,
        field_name=str(getattr(function, "name", "field")),
    )


def _assemble_component(expression, measure, comm) -> float:
    local = fem.assemble_scalar(fem.form(expression * measure))
    return float(comm.allreduce(local, op=MPI.SUM))


def _single_point(point, domain) -> np.ndarray:
    selected = np.asarray(point, dtype=domain.geometry.x.dtype)
    if selected.ndim != 1:
        raise ValueError("A probe point must be one coordinate vector.")
    return _point_array(selected.reshape(1, -1), domain)


def _point_array(points, domain) -> np.ndarray:
    selected = np.asarray(points, dtype=domain.geometry.x.dtype)
    if selected.ndim == 1 and domain.geometry.dim == 1:
        selected = selected.reshape(-1, 1)
    if selected.ndim != 2:
        raise ValueError("Point coordinates must have shape (number, dimension).")
    geometric_dimension = int(domain.geometry.dim)
    storage_dimension = int(domain.geometry.x.shape[1])
    if selected.shape[1] == storage_dimension:
        return np.ascontiguousarray(selected)
    if selected.shape[1] != geometric_dimension:
        raise ValueError(
            "Point coordinates must match the mesh geometric dimension "
            f"{geometric_dimension}."
        )
    coordinates = np.zeros(
        (selected.shape[0], storage_dimension),
        dtype=domain.geometry.x.dtype,
    )
    coordinates[:, :geometric_dimension] = selected
    return coordinates


def _require_collective_points(points: np.ndarray, comm) -> None:
    digest = sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
    identity = (tuple(int(value) for value in points.shape), digest)
    identities = comm.allgather(identity)
    if any(item != identities[0] for item in identities[1:]):
        raise ValueError(
            "sample_points requires identical coordinates on every MPI rank."
        )


def _comm_from_measure(measure):
    domain = measure.ufl_domain()
    cargo = None if domain is None else domain.ufl_cargo()
    comm = getattr(cargo, "comm", None)
    if comm is None:
        raise ValueError(
            "Could not infer an MPI communicator from the integration measure; "
            "pass comm=... explicitly."
        )
    return comm


def _region_measure(region):
    measure = getattr(region, "measure", None)
    if measure is None:
        raise ValueError("A named cell or boundary region with a measure is required.")
    return measure
