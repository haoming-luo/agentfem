"""Optional exact multi-point-constraint construction backends.

This module owns reusable geometric MPC construction.  Scientific adapters
may consume it, but benchmark-specific code must not carry a private periodic
implementation.  Solver lowering remains explicit because DOLFINx and
``dolfinx_mpc`` linear/nonlinear problems have different assembly contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI


@dataclass(frozen=True)
class RectangularPeriodicMPC:
    """Exact rectangular periodic relation and construction diagnostics."""

    backend: object
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    axes: tuple[int, ...]
    tolerance: float
    name: str = "rectangular_periodic_mpc"

    @property
    def strict(self) -> bool:
        return True

    @property
    def supports_parallel(self) -> bool:
        return True

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "periodic_constraint",
            "method": "dolfinx_mpc",
            "enforcement": "exact_multi_point_constraint",
            "lower": self.lower,
            "upper": self.upper,
            "axes": self.axes,
            "tolerance": self.tolerance,
            "strict": True,
            "supports_parallel": True,
        }


def rectangular_periodic_mpc(
    target,
    *,
    axes=None,
    bcs=(),
    tolerance: float | None = None,
    name: str = "rectangular_periodic_mpc",
) -> RectangularPeriodicMPC:
    """Constrain maximum faces of a rectangular mesh to minimum faces.

    Corners are mapped directly to the opposite corner, avoiding chained
    slave relations.  ``target`` may be a FunctionSpace, Function, or an
    AgentFEM field exposing ``space``/``value``.  The optional dependency is
    imported only when this backend is requested.
    """

    try:
        import dolfinx_mpc
    except ImportError as exc:
        raise ImportError(
            "Exact rectangular periodic constraints require optional "
            "`dolfinx_mpc`. Install a matching build or use the serial "
            "central-difference projection backend."
        ) from exc

    space = _space(target)
    domain = space.mesh
    dimension = int(domain.geometry.dim)
    selected_axes = (
        tuple(range(dimension))
        if axes is None
        else tuple(sorted({int(axis) for axis in axes}))
    )
    if not selected_axes or any(axis < 0 or axis >= dimension for axis in selected_axes):
        raise ValueError(
            f"axes must select dimensions in [0, {dimension}); got {selected_axes!r}."
        )
    coordinates = np.asarray(domain.geometry.x[:, :dimension], dtype=float)
    lower = np.asarray(
        [
            domain.comm.allreduce(float(coordinates[:, axis].min()), op=MPI.MIN)
            for axis in range(dimension)
        ]
    )
    upper = np.asarray(
        [
            domain.comm.allreduce(float(coordinates[:, axis].max()), op=MPI.MAX)
            for axis in range(dimension)
        ]
    )
    span = upper - lower
    if np.any(span <= 0.0):
        raise ValueError("Rectangular periodic bounds must have positive span.")
    selected_tolerance = (
        100.0 * np.finfo(float).eps * max(1.0, float(np.max(span)))
        if tolerance is None
        else float(tolerance)
    )
    if not np.isfinite(selected_tolerance) or selected_tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite.")

    def slave_boundary(x):
        selected = np.zeros(x.shape[1], dtype=bool)
        for axis in selected_axes:
            selected |= np.isclose(
                x[axis], upper[axis], atol=selected_tolerance, rtol=0.0
            )
        return selected

    def matching_point(x):
        mapped = x.copy()
        for axis in selected_axes:
            on_maximum = np.isclose(
                x[axis], upper[axis], atol=selected_tolerance, rtol=0.0
            )
            mapped[axis, on_maximum] -= span[axis]
        return mapped

    backend = dolfinx_mpc.MultiPointConstraint(space)
    backend.create_periodic_constraint_geometrical(
        space,
        slave_boundary,
        matching_point,
        bcs=[getattr(item, "bc", item) for item in bcs],
        tol=selected_tolerance,
    )
    backend.finalize()
    return RectangularPeriodicMPC(
        backend=backend,
        lower=tuple(float(value) for value in lower),
        upper=tuple(float(value) for value in upper),
        axes=selected_axes,
        tolerance=selected_tolerance,
        name=name,
    )


def _space(target):
    if hasattr(target, "space"):
        return target.space
    if hasattr(target, "function_space"):
        return target.function_space
    if hasattr(target, "value") and hasattr(target.value, "function_space"):
        return target.value.function_space
    return target


__all__ = ["RectangularPeriodicMPC", "rectangular_periodic_mpc"]
