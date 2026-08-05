"""MPI-safe quantities of interest assembled from finite-element expressions."""

from __future__ import annotations

from math import sqrt

import numpy as np
import basix
import ufl
from dolfinx import fem
from mpi4py import MPI

from .. import fields as field_api
from ..kernel import dofs


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


def _assemble_component(expression, measure, comm) -> float:
    local = fem.assemble_scalar(fem.form(expression * measure))
    return float(comm.allreduce(local, op=MPI.SUM))


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
