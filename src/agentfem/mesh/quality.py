# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Collective mesh-quality evidence for supported solver domains.

Simplex cells use a normalized mean-ratio metric. Tensor-product and mixed-
facet cells use a sampled scaled Jacobian computed from the actual coordinate
element, so curved high-order geometry is inspected rather than reduced to its
corner topology.
"""

from __future__ import annotations

from dataclasses import dataclass

import basix
import numpy as np
from dolfinx.cpp.mesh import entities_to_geometry
from mpi4py import MPI


@dataclass(frozen=True)
class GeometryIdentity:
    """Inspectable identity of the coordinate finite element on one domain."""

    cell_type: str
    topological_dimension: int
    geometric_dimension: int
    basis_family: str
    degree: int
    variant: str | None
    mapping: str
    nodes_per_cell: int
    high_order: bool
    embedded: bool

    def summary(self) -> dict[str, object]:
        return {
            "kind": "geometry_identity",
            "cell_type": self.cell_type,
            "topological_dimension": self.topological_dimension,
            "geometric_dimension": self.geometric_dimension,
            "basis_family": self.basis_family,
            "degree": self.degree,
            "variant": self.variant,
            "mapping": self.mapping,
            "nodes_per_cell": self.nodes_per_cell,
            "high_order": self.high_order,
            "embedded": self.embedded,
        }


@dataclass(frozen=True)
class MeshQualityReport:
    """MPI-global quality summary for owned cells."""

    cell_type: str
    metric: str
    minimum: float
    mean: float
    maximum: float
    threshold: float
    poor_cells: int
    invalid_cells: int
    global_cells: int
    geometry_degree: int
    samples_per_cell: int
    interpretation: str
    geometry: GeometryIdentity | None = None

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
            "geometry_degree": self.geometry_degree,
            "samples_per_cell": self.samples_per_cell,
            "interpretation": self.interpretation,
            "geometry": (
                None if self.geometry is None else self.geometry.summary()
            ),
        }


def describe_geometry(domain) -> GeometryIdentity:
    """Describe the real Basix coordinate basis used by a DOLFINx mesh."""

    element = _basix_coordinate_element(domain)
    family = _enum_name(element.family)
    variant = None
    if family == "P":
        variant = _enum_name(element.lagrange_variant)
    elif family == "serendipity":
        variant = "serendipity"
    topological_dimension = int(domain.topology.dim)
    geometric_dimension = int(domain.geometry.dim)
    degree = int(element.degree)
    return GeometryIdentity(
        cell_type=_enum_name(element.cell_type),
        topological_dimension=topological_dimension,
        geometric_dimension=geometric_dimension,
        basis_family=family,
        degree=degree,
        variant=variant,
        mapping=_enum_name(element.map_type),
        nodes_per_cell=int(element.dim),
        high_order=degree > 1,
        embedded=geometric_dimension != topological_dimension,
    )


def cell_quality(domain) -> np.ndarray:
    """Return one normalized quality value per owned cell in ``[0, 1]``.

    Triangles and tetrahedra use the simplex mean ratio. Quadrilaterals,
    hexahedra, prisms, and pyramids use the minimum sampled scaled Jacobian of
    the coordinate map. The latter samples the real geometry element and
    therefore includes high-order curvature when present.
    """

    tdim = int(domain.topology.dim)
    cell_type = str(domain.topology.cell_type).lower()
    if "triangle" in cell_type:
        corner_count = 3
        evaluator = _triangle_quality
    elif "tetra" in cell_type:
        corner_count = 4
        evaluator = _tetrahedron_quality
    elif any(
        value in cell_type
        for value in ("quadrilateral", "hexahedron", "prism", "pyramid")
    ):
        return _sampled_scaled_jacobian_quality(domain)[0]
    else:
        raise NotImplementedError(
            "Mesh-quality preflight supports triangle, tetrahedron, "
            "quadrilateral, hexahedron, prism, and pyramid domains, not "
            f"{domain.topology.cell_type}."
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
    if int(_coordinate_element(domain).degree) > 1:
        sampled, _degree, _sample_count = _sampled_scaled_jacobian_quality(domain)
        # Corner shape alone can look excellent while a curved coordinate map
        # is close to singular between vertices.  Retain the simplex shape
        # metric, but let the sampled high-order map lower it continuously
        # instead of noticing only an already-folded map.
        values = np.minimum(values, sampled)
    return values


def audit(domain, *, threshold: float = 0.1, strict: bool = False) -> MeshQualityReport:
    """Create a collective preflight report and optionally reject poor cells."""

    selected_threshold = float(threshold)
    if not 0.0 <= selected_threshold <= 1.0 or not np.isfinite(selected_threshold):
        raise ValueError("mesh-quality threshold must lie in [0, 1].")
    cell_type = str(domain.topology.cell_type).lower()
    if "triangle" in cell_type or "tetra" in cell_type:
        values = cell_quality(domain)
        geometry_degree = int(_coordinate_element(domain).degree)
        if geometry_degree > 1:
            _sampled, _degree, samples_per_cell = (
                _sampled_scaled_jacobian_quality(domain)
            )
            metric = "simplex_mean_ratio_with_sampled_scaled_jacobian"
            interpretation = (
                "minimum of corner mean ratio and sampled coordinate-map "
                "scaled Jacobian; 0 is singular or folded"
            )
        else:
            metric = "simplex_mean_ratio"
            samples_per_cell = 1
            interpretation = "1 is equilateral; 0 is degenerate"
    elif any(
        value in cell_type
        for value in ("quadrilateral", "hexahedron", "prism", "pyramid")
    ):
        values, geometry_degree, samples_per_cell = (
            _sampled_scaled_jacobian_quality(domain)
        )
        metric = "sampled_scaled_jacobian"
        interpretation = "1 is locally orthogonal; 0 is singular or folded"
    else:
        # Keep the public supported-cell diagnostic in one place.
        values = cell_quality(domain)
        raise AssertionError("unreachable after cell_quality validation")
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
        metric=metric,
        minimum=minimum,
        mean=total / global_count,
        maximum=maximum,
        threshold=selected_threshold,
        poor_cells=poor,
        invalid_cells=invalid,
        global_cells=global_count,
        geometry_degree=geometry_degree,
        samples_per_cell=samples_per_cell,
        interpretation=interpretation,
        geometry=describe_geometry(domain),
    )
    if strict and not report.acceptable:
        raise ValueError(
            "Mesh quality failed: "
            f"minimum={minimum:.6g}, poor_cells={poor}, invalid_cells={invalid}."
        )
    return report


def _coordinate_element(domain):
    cmaps = getattr(domain.geometry, "cmaps", ())
    if len(cmaps) != 1:
        raise NotImplementedError(
            "Mesh-quality preflight requires one coordinate element per domain; "
            "mixed-topology geometry must be split into explicit solver domains."
        )
    return cmaps[0]


def _basix_coordinate_element(domain):
    """Reconstruct the Basix geometry basis owned by one DOLFINx mesh."""

    cmap = _coordinate_element(domain)
    cell = domain.basix_cell()
    degree = int(cmap.degree)
    candidates = (
        (
            basix.ElementFamily.P,
            {"lagrange_variant": basix.LagrangeVariant(int(cmap.variant))},
        ),
        (basix.ElementFamily.serendipity, {}),
    )
    errors = []
    for family, options in candidates:
        try:
            element = basix.create_element(family, cell, degree, **options)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        if int(element.dim) == int(cmap.dim):
            return element
    raise NotImplementedError(
        "The active coordinate element cannot yet be reconstructed for "
        f"quality sampling: cell={cell.name}, degree={degree}, dofs={cmap.dim}. "
        f"Basix attempts: {tuple(errors)}"
    )


def _sampled_scaled_jacobian_quality(domain) -> tuple[np.ndarray, int, int]:
    """Sample coordinate-map scaled Jacobians on every owned cell."""

    element = _basix_coordinate_element(domain)
    cell = element.cell_type
    tdim = int(domain.topology.dim)
    gdim = int(domain.geometry.dim)
    degree = int(_coordinate_element(domain).degree)
    quadrature_points, _ = basix.make_quadrature(cell, max(2, 2 * degree))
    reference_vertices = np.asarray(basix.cell.geometry(cell), dtype=float)
    if cell == basix.CellType.pyramid:
        # The collapsed-coordinate pyramid basis has no unique in-plane
        # derivative at its apex. Interior quadrature plus all base vertices
        # inspect the map without labelling the valid reference pyramid
        # singular solely because of that coordinate representation.
        apex = int(np.argmax(reference_vertices[:, -1]))
        reference_vertices = np.delete(reference_vertices, apex, axis=0)
    points = np.unique(
        np.vstack((quadrature_points, reference_vertices)), axis=0
    )
    table = element.tabulate(1, points)
    derivatives = np.asarray(table[1 : 1 + tdim, :, :, 0], dtype=float)
    dofmap = domain.geometry.dofmaps[0]
    owned_cells = int(domain.topology.index_map(tdim).size_local)
    values = np.empty(owned_cells, dtype=float)
    tolerance = 64.0 * np.finfo(float).eps

    for cell_index in range(owned_cells):
        geometry_nodes = np.asarray(dofmap[cell_index], dtype=np.int32)
        coordinates = np.asarray(
            domain.geometry.x[geometry_nodes, :gdim], dtype=float
        )
        minimum = 1.0
        signs: set[int] = set()
        for point_index in range(points.shape[0]):
            gradient = derivatives[:, point_index, :].T
            jacobian = coordinates.T @ gradient
            column_norms = np.linalg.norm(jacobian, axis=0)
            scale = float(np.prod(column_norms))
            gram = jacobian.T @ jacobian
            determinant = float(np.linalg.det(gram))
            if scale <= tolerance or determinant <= tolerance * scale * scale:
                minimum = 0.0
                continue
            measure = float(np.sqrt(max(0.0, determinant)))
            minimum = min(minimum, measure / scale)
            if gdim == tdim:
                signed = float(np.linalg.det(jacobian))
                if abs(signed) > tolerance * scale:
                    signs.add(1 if signed > 0.0 else -1)
        if len(signs) > 1:
            minimum = 0.0
        values[cell_index] = min(1.0, max(0.0, minimum))
    return values, degree, int(points.shape[0])


def _enum_name(value) -> str:
    """Return a stable enum label without exposing implementation reprs."""

    name = getattr(value, "name", None)
    return str(name if name is not None else value).split(".")[-1]


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


__all__ = [
    "GeometryIdentity",
    "MeshQualityReport",
    "audit",
    "cell_quality",
    "describe_geometry",
]
