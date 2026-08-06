"""Mesh-quality evidence for supported simplex solver domains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dolfinx.cpp.mesh import entities_to_geometry
from mpi4py import MPI


@dataclass(frozen=True)
class MeshQualityReport:
    """MPI-global mean-ratio quality summary for owned cells."""

    cell_type: str
    metric: str
    minimum: float
    mean: float
    maximum: float
    threshold: float
    poor_cells: int
    invalid_cells: int
    global_cells: int

    @property
    def valid(self) -> bool:
        return self.invalid_cells == 0

    @property
    def acceptable(self) -> bool:
        return self.valid and self.poor_cells == 0

    def summary(self) -> dict[str, object]:
        return {
            "kind": "mesh_quality_report",
            "cell_type": self.cell_type,
            "metric": self.metric,
            "range": [self.minimum, self.maximum],
            "mean": self.mean,
            "threshold": self.threshold,
            "poor_cells": self.poor_cells,
            "invalid_cells": self.invalid_cells,
            "global_cells": self.global_cells,
            "valid": self.valid,
            "acceptable": self.acceptable,
            "interpretation": "1 is equilateral; 0 is degenerate",
        }


def cell_quality(domain) -> np.ndarray:
    """Return owned-cell simplex mean-ratio values in ``[0, 1]``."""

    tdim = int(domain.topology.dim)
    cell_type = str(domain.topology.cell_type).lower()
    if "triangle" in cell_type:
        corner_count = 3
        evaluator = _triangle_quality
    elif "tetra" in cell_type:
        corner_count = 4
        evaluator = _tetrahedron_quality
    else:
        raise NotImplementedError(
            "Mesh-quality mean ratio currently supports triangle and "
            f"tetrahedron domains, not {domain.topology.cell_type}."
        )

    domain.topology.create_connectivity(tdim, 0)
    domain.topology.create_connectivity(0, tdim)
    connectivity = domain.topology.connectivity(tdim, 0)
    owned_cells = int(domain.topology.index_map(tdim).size_local)
    values = np.empty(owned_cells, dtype=float)
    for cell in range(owned_cells):
        vertices = np.asarray(connectivity.links(cell), dtype=np.int32)
        if vertices.size != corner_count:
            raise RuntimeError(
                f"{domain.topology.cell_type} cell {cell} has "
                f"{vertices.size} topological vertices, expected {corner_count}."
            )
        geometry_nodes = np.asarray(
            entities_to_geometry(domain._cpp_object, 0, vertices, False),
            dtype=np.int32,
        ).reshape(-1)
        points = np.asarray(domain.geometry.x[geometry_nodes], dtype=float)
        values[cell] = evaluator(points)
    return values


def audit(domain, *, threshold: float = 0.1, strict: bool = False) -> MeshQualityReport:
    """Create a collective preflight report and optionally reject poor cells."""

    selected_threshold = float(threshold)
    if not 0.0 <= selected_threshold <= 1.0 or not np.isfinite(selected_threshold):
        raise ValueError("mesh-quality threshold must lie in [0, 1].")
    values = cell_quality(domain)
    comm = domain.comm
    local_count = int(values.size)
    global_count = int(comm.allreduce(local_count, op=MPI.SUM))
    if global_count == 0:
        raise ValueError("mesh-quality audit requires at least one cell.")
    local_minimum = float(np.min(values)) if values.size else np.inf
    local_maximum = float(np.max(values)) if values.size else -np.inf
    minimum = float(comm.allreduce(local_minimum, op=MPI.MIN))
    maximum = float(comm.allreduce(local_maximum, op=MPI.MAX))
    total = float(comm.allreduce(float(np.sum(values)), op=MPI.SUM))
    invalid = int(
        comm.allreduce(int(np.count_nonzero(~np.isfinite(values) | (values <= 0.0))), op=MPI.SUM)
    )
    poor = int(
        comm.allreduce(int(np.count_nonzero(values < selected_threshold)), op=MPI.SUM)
    )
    report = MeshQualityReport(
        cell_type=str(domain.topology.cell_type),
        metric="simplex_mean_ratio",
        minimum=minimum,
        mean=total / global_count,
        maximum=maximum,
        threshold=selected_threshold,
        poor_cells=poor,
        invalid_cells=invalid,
        global_cells=global_count,
    )
    if strict and not report.acceptable:
        raise ValueError(
            "Mesh quality failed: "
            f"minimum={minimum:.6g}, poor_cells={poor}, invalid_cells={invalid}."
        )
    return report


def _triangle_quality(points: np.ndarray) -> float:
    edges = (points[1] - points[0], points[2] - points[1], points[0] - points[2])
    squared = sum(float(np.dot(edge, edge)) for edge in edges)
    area = 0.5 * float(np.linalg.norm(np.cross(edges[0], points[2] - points[0])))
    if squared <= 0.0 or area <= np.finfo(float).eps:
        return 0.0
    return min(1.0, 4.0 * np.sqrt(3.0) * area / squared)


def _tetrahedron_quality(points: np.ndarray) -> float:
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    squared = sum(
        float(np.dot(points[right] - points[left], points[right] - points[left]))
        for left, right in pairs
    )
    volume = abs(
        float(
            np.linalg.det(
                np.column_stack(
                    (points[1] - points[0], points[2] - points[0], points[3] - points[0])
                )
            )
        )
    ) / 6.0
    if squared <= 0.0 or volume <= np.finfo(float).eps:
        return 0.0
    return min(1.0, 12.0 * (3.0 * volume) ** (2.0 / 3.0) / squared)


__all__ = ["MeshQualityReport", "audit", "cell_quality"]
