"""Optional exact multi-point-constraint construction backends.

This module owns reusable geometric MPC construction.  Scientific adapters
may consume it, but benchmark-specific code must not carry a private periodic
implementation.  Solver lowering remains explicit because DOLFINx and
``dolfinx_mpc`` linear/nonlinear problems have different assembly contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

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
    construction_diagnostics: Mapping[str, object] | None = None

    @property
    def strict(self) -> bool:
        return True

    @property
    def supports_parallel(self) -> bool:
        return True

    def summary(self) -> dict[str, object]:
        summary = {
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
        if self.construction_diagnostics is not None:
            summary["diagnostics"] = dict(self.construction_diagnostics)
        return summary

    def diagnostics(self) -> dict[str, object]:
        """Return immutable construction evidence for the exact MPC graph.

        These counts verify that every declared slave has exactly one unit-
        coefficient master relation.  They do not claim access to the
        eliminated constraint multiplier, a boundary reaction distribution,
        or macroscopic work.
        """

        if self.construction_diagnostics is None:
            return {
                "status": "unavailable",
                "reason": "constructed outside rectangular_periodic_mpc",
            }
        return dict(self.construction_diagnostics)


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
    diagnostics = _construction_diagnostics(
        backend,
        domain=domain,
        axes=selected_axes,
        span=span,
        tolerance=selected_tolerance,
    )
    if diagnostics["status"] != "valid":
        raise RuntimeError(
            "The exact rectangular periodic MPC graph is incomplete or "
            f"ambiguous: {dict(diagnostics)!r}."
        )
    return RectangularPeriodicMPC(
        backend=backend,
        lower=tuple(float(value) for value in lower),
        upper=tuple(float(value) for value in upper),
        axes=selected_axes,
        tolerance=selected_tolerance,
        name=name,
        construction_diagnostics=diagnostics,
    )


def _construction_diagnostics(
    backend,
    *,
    domain,
    axes: tuple[int, ...],
    span: np.ndarray,
    tolerance: float,
) -> Mapping[str, object]:
    """Describe the finalized one-master periodic elimination graph."""

    slaves = np.asarray(backend.slaves, dtype=np.int64).reshape(-1)
    # dolfinx_mpc appends ghost slaves after the locally owned prefix.  Only
    # that prefix may participate in global cardinalities; summing the full
    # array double-counts relations shared by MPI partitions.
    owned_count = int(backend.num_local_slaves)
    owned_slaves = slaves[:owned_count]
    local_relations = 0
    local_unmatched = 0
    local_ambiguous = 0
    for slave in owned_slaves:
        masters = np.asarray(backend.masters.links(int(slave))).reshape(-1)
        count = int(masters.size)
        local_relations += count
        local_unmatched += int(count == 0)
        local_ambiguous += int(count > 1)

    coefficients, _ = backend.coefficients()
    coefficients = np.asarray(coefficients)
    local_has_nonunit = int(
        np.any(~np.isclose(coefficients, 1.0, atol=tolerance, rtol=0.0))
    )
    comm = domain.comm

    def total(value: int) -> int:
        return int(comm.allreduce(int(value), op=MPI.SUM))

    global_slaves = total(owned_count)
    ghost_slave_copies = total(slaves.size - owned_count)
    global_relations = total(local_relations)
    global_unmatched = total(local_unmatched)
    global_ambiguous = total(local_ambiguous)
    global_has_nonunit = bool(
        comm.allreduce(local_has_nonunit, op=MPI.MAX)
    )
    ranks_with_slaves = total(int(owned_count > 0))
    valid = bool(
        global_slaves > 0
        and global_relations == global_slaves
        and global_unmatched == 0
        and global_ambiguous == 0
        and not global_has_nonunit
    )
    return MappingProxyType(
        {
            "status": "valid" if valid else "invalid",
            "global_slave_dofs": global_slaves,
            "global_master_relations": global_relations,
            "ghost_slave_copies": ghost_slave_copies,
            "unmatched_slave_dofs": global_unmatched,
            "multiply_matched_slave_dofs": global_ambiguous,
            "nonunit_coefficients_detected": global_has_nonunit,
            "ranks_with_slaves": ranks_with_slaves,
            "comm_size": int(comm.size),
            "axes": tuple(int(axis) for axis in axes),
            "periods": tuple(float(span[axis]) for axis in axes),
            "coordinate_tolerance": float(tolerance),
            "reaction_distribution": "unavailable_without_provider_dual",
            "macroscopic_work": "unavailable_without_work_coordinate",
        }
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
