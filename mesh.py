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
    ``agentfem.mesh_formats.convert_to_xdmf``. It requires the optional
    dependency ``meshio`` only when called.
    """

    from .mesh_formats import convert_to_xdmf

    return convert_to_xdmf(*args, **kwargs)


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
