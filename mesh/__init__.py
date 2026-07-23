"""Mesh import, marking, and measure helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ufl
from dolfinx import io
from dolfinx import mesh
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

from . import formats


@dataclass(frozen=True)
class FEMMesh:
    """DOLFINx mesh plus optional cell and facet tags.

    The object can be unpacked as ``domain, cell_tags, facet_tags`` for simple
    application code.
    """

    domain: mesh.Mesh
    cell_tags: mesh.MeshTags | None = None
    facet_tags: mesh.MeshTags | None = None

    def __iter__(self):
        yield self.domain
        yield self.cell_tags
        yield self.facet_tags

    def summary(self) -> "MeshSummary":
        """Return a compact summary of the mesh and available tags."""

        return summarize_mesh(self.domain, self.cell_tags, self.facet_tags)


@dataclass(frozen=True)
class TagSummary:
    """Summary of integer mesh tags on one topological entity dimension."""

    entity_dim: int
    tags: tuple[int, ...]
    counts: dict[int, int]


@dataclass(frozen=True)
class MeshSummary:
    """Human- and agent-readable mesh summary."""

    geometric_dim: int
    topological_dim: int
    local_cells: int
    global_cells: int
    local_vertices: int
    global_vertices: int
    cell_tags: TagSummary | None = None
    facet_tags: TagSummary | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a plain dictionary for logging, docs, or agent inspection."""

        return {
            "geometric_dim": self.geometric_dim,
            "topological_dim": self.topological_dim,
            "local_cells": self.local_cells,
            "global_cells": self.global_cells,
            "local_vertices": self.local_vertices,
            "global_vertices": self.global_vertices,
            "cell_tags": None if self.cell_tags is None else self.cell_tags.counts,
            "facet_tags": None if self.facet_tags is None else self.facet_tags.counts,
        }


@dataclass(frozen=True)
class BoundaryRegion:
    """Named exterior boundary region on a mesh.

    A boundary region stores both the geometric marker used for dof location and
    the tagged boundary measure used for weak boundary terms.
    """

    name: str
    domain: object
    marker: object
    tag: int
    facet_tags: object
    ds: object

    @property
    def measure(self):
        """Boundary integration measure restricted to this region."""

        return self.ds(self.tag)

    def summary(self) -> dict[str, object]:
        """Return a compact region summary."""

        return {
            "name": self.name,
            "kind": "boundary_region",
            "tag": self.tag,
        }


def import_gmsh_model(
    model,
    comm: MPI.Comm = MPI.COMM_WORLD,
    *,
    model_rank: int = 0,
    gdim: int = 3,
) -> FEMMesh:
    """Convert an in-memory Gmsh model to a DOLFINx mesh.

    The Gmsh model must already contain physical groups if cell and facet tags
    are required downstream.
    """

    mesh_data = gmshio.model_to_mesh(model, comm, model_rank, gdim=gdim)
    return FEMMesh(mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags)


def rectangle(
    lower,
    upper,
    cells,
    comm: MPI.Comm = MPI.COMM_WORLD,
    *,
    cell_type: str | mesh.CellType = "quadrilateral",
):
    """Create a structured 2D rectangular mesh.

    ``cell_type`` controls the geometric cell shape, such as ``"quadrilateral"``
    or ``"triangle"``. The finite-element interpolation degree is selected later
    when creating fields or spaces.
    """

    return mesh.create_rectangle(
        comm,
        [np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)],
        list(cells),
        cell_type=_cell_type(cell_type),
    )


def _cell_type(cell_type: str | mesh.CellType):
    if isinstance(cell_type, mesh.CellType):
        return cell_type
    try:
        return getattr(mesh.CellType, cell_type)
    except AttributeError as exc:
        raise ValueError(f"Unknown cell type: {cell_type!r}.") from exc


def read_gmsh_mesh(
    path: str | Path,
    comm: MPI.Comm = MPI.COMM_WORLD,
    *,
    model_rank: int = 0,
    gdim: int = 3,
) -> FEMMesh:
    """Read a ``.msh`` file with Gmsh and convert it to a DOLFINx mesh."""

    import gmsh

    path = Path(path)
    was_initialized = gmsh.isInitialized()
    if not was_initialized:
        gmsh.initialize()
    try:
        if comm.rank == model_rank:
            gmsh.open(str(path))
        return import_gmsh_model(gmsh.model, comm, model_rank=model_rank, gdim=gdim)
    finally:
        if not was_initialized:
            gmsh.finalize()


def read_xdmf_mesh(
    path: str | Path,
    comm: MPI.Comm = MPI.COMM_WORLD,
    *,
    mesh_name: str = "mesh",
    cell_tags_name: str | None = None,
    facet_tags_name: str | None = None,
) -> FEMMesh:
    """Read a DOLFINx XDMF mesh and optional cell/facet meshtags."""

    path = Path(path)
    with io.XDMFFile(comm, str(path), "r") as xdmf:
        domain = xdmf.read_mesh(name=mesh_name)
        cell_tags = None
        facet_tags = None
        tdim = domain.topology.dim
        if cell_tags_name is not None:
            domain.topology.create_connectivity(tdim, tdim)
            cell_tags = xdmf.read_meshtags(domain, name=cell_tags_name)
        if facet_tags_name is not None:
            domain.topology.create_connectivity(tdim - 1, tdim)
            facet_tags = xdmf.read_meshtags(domain, name=facet_tags_name)
    return FEMMesh(domain, cell_tags, facet_tags)


def write_xdmf_mesh(
    path: str | Path,
    domain,
    comm: MPI.Comm | None = None,
    *,
    mode: str = "w",
) -> None:
    """Write a DOLFINx mesh to XDMF."""

    path = Path(path)
    comm = comm or domain.comm
    with io.XDMFFile(comm, str(path), mode) as xdmf:
        xdmf.write_mesh(domain)


def convert_external_mesh_to_xdmf(*args, **kwargs):
    """Convert Abaqus/NASTRAN/COMSOL-like external meshes to XDMF.

    This is a convenience entry point that delegates to
    ``agentfem.mesh.formats.convert_to_xdmf``. It requires the optional
    dependency ``meshio`` only when called.
    """

    from .formats import convert_to_xdmf

    return convert_to_xdmf(*args, **kwargs)


def summarize_tags(tags) -> TagSummary | None:
    """Summarize a DOLFINx meshtags object."""

    if tags is None:
        return None
    values, counts = np.unique(np.asarray(tags.values, dtype=np.int32), return_counts=True)
    counts_by_tag = {int(tag): int(count) for tag, count in zip(values, counts)}
    return TagSummary(
        entity_dim=int(tags.dim),
        tags=tuple(int(tag) for tag in values),
        counts=counts_by_tag,
    )


def summarize_mesh(domain, cell_tags=None, facet_tags=None) -> MeshSummary:
    """Return local/global mesh size and tag summaries."""

    tdim = domain.topology.dim
    domain.topology.create_entities(0)
    domain.topology.create_entities(tdim)
    cell_map = domain.topology.index_map(tdim)
    vertex_map = domain.topology.index_map(0)
    local_cells = cell_map.size_local
    local_vertices = vertex_map.size_local
    global_cells = domain.comm.allreduce(local_cells, op=MPI.SUM)
    global_vertices = domain.comm.allreduce(local_vertices, op=MPI.SUM)
    return MeshSummary(
        geometric_dim=domain.geometry.dim,
        topological_dim=tdim,
        local_cells=local_cells,
        global_cells=global_cells,
        local_vertices=local_vertices,
        global_vertices=global_vertices,
        cell_tags=summarize_tags(cell_tags),
        facet_tags=summarize_tags(facet_tags),
    )


def require_tags(tags, required: int | tuple[int, ...] | list[int], *, name: str = "tags") -> None:
    """Raise a modeling error if required integer tags are absent."""

    required_tags = (required,) if isinstance(required, int) else tuple(required)
    available = set() if tags is None else set(int(tag) for tag in np.unique(tags.values))
    missing = [tag for tag in required_tags if tag not in available]
    if missing:
        raise ValueError(
            f"Missing required {name}: {missing}. "
            f"Available {name}: {sorted(available)}."
        )


def require_cell_tags(cell_tags, required: int | tuple[int, ...] | list[int]) -> None:
    """Require cell/material region tags."""

    require_tags(cell_tags, required, name="cell tags")


def require_facet_tags(facet_tags, required: int | tuple[int, ...] | list[int]) -> None:
    """Require boundary/facet tags."""

    require_tags(facet_tags, required, name="facet tags")


def boundary(domain, marker, *, name: str = "boundary", tag: int = 1) -> BoundaryRegion:
    """Create a named exterior boundary region from a geometric marker."""

    ds, facet_tags = tagged_boundary_measure(domain, marker, tag=tag)
    return BoundaryRegion(
        name=name,
        domain=domain,
        marker=marker,
        tag=tag,
        facet_tags=facet_tags,
        ds=ds,
    )


def boundary_region(domain, marker, *, name: str = "boundary", tag: int = 1) -> BoundaryRegion:
    """Alias for ``boundary`` when a more explicit name reads better."""

    return boundary(domain, marker, name=name, tag=tag)


def region_measure(location):
    """Return a region's restricted measure or pass through a measure."""

    return location.measure if hasattr(location, "measure") else location


def region_marker(location):
    """Return a region's marker or pass through a marker callable."""

    return location.marker if hasattr(location, "marker") else location


def locate_boundary_facets(domain, marker):
    """Locate exterior facets using a geometrical marker."""

    facet_dim = domain.topology.dim - 1
    return mesh.locate_entities_boundary(domain, facet_dim, marker)


def mark_facets(domain, facets, tag: int):
    """Create a meshtags object for a set of facets."""

    facet_dim = domain.topology.dim - 1
    facets = np.asarray(facets, dtype=np.int32)
    values = np.full(len(facets), tag, dtype=np.int32)
    return mesh.meshtags(domain, facet_dim, facets, values)


def mark_boundary_facets(domain, marker, tag: int):
    """Locate and tag exterior facets in one step."""

    return mark_facets(domain, locate_boundary_facets(domain, marker), tag)


def boundary_measure(domain, facet_tags=None):
    """Create a boundary integration measure."""

    return ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)


def tagged_boundary_measure(domain, marker, tag: int):
    """Locate/tag exterior facets and return ``(ds, facet_tags)``."""

    facet_tags = mark_boundary_facets(domain, marker, tag)
    return boundary_measure(domain, facet_tags), facet_tags
