"""Mesh import, marking, and measure helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import io
from dolfinx import mesh
from mpi4py import MPI

from agentfem import dependencies

from . import formats
from . import quality
from . import abaqus
from . import selectors as select
from .specs import SUPPORTED_GEOMETRIES, from_geometry_spec
from .regions import RegionSet
from .selectors import Selector, ball, box, disk, layer, plane, where

audit_quality = quality.audit
cell_quality = quality.cell_quality

# A short public alias for configuration, GUI, and agent-authored models.
from_spec = from_geometry_spec


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

    ``selection`` makes the source of truth explicit. Geometric regions use a
    marker; tagged regions use imported facet ids; hybrid legacy regions use
    the facet tag for both weak terms and strong dof location and can be
    audited against their marker.
    """

    name: str
    domain: object
    marker: object | None
    tag: int
    facet_tags: object
    ds: object
    selection: str = "auto"

    def __post_init__(self) -> None:
        selected = str(self.selection).lower().replace("-", "_")
        if selected == "auto":
            if self.marker is None:
                selected = "tagged"
            elif self.facet_tags is None:
                selected = "geometric"
            else:
                selected = "hybrid"
        if selected not in {"geometric", "tagged", "hybrid"}:
            raise ValueError(
                "BoundaryRegion.selection must be geometric, tagged, or hybrid."
            )
        if selected == "geometric" and self.marker is None:
            raise ValueError("A geometric BoundaryRegion requires marker=....")
        if selected in {"tagged", "hybrid"} and self.facet_tags is None:
            raise ValueError("A tagged BoundaryRegion requires facet_tags=....")
        object.__setattr__(self, "selection", selected)

    @property
    def measure(self):
        """Boundary integration measure restricted to this region."""

        return self.ds(self.tag)

    @property
    def facets(self) -> np.ndarray:
        """Return local facet entities selected by this region's tag."""

        if self.facet_tags is None:
            return np.empty(0, dtype=np.int32)
        return np.asarray(self.facet_tags.find(self.tag), dtype=np.int32)

    def audit(self, *, strict: bool = False) -> dict[str, object]:
        """Return geometric/tag evidence and optionally require consistency."""

        return audit_boundary_region(self, strict=strict)

    def summary(self) -> dict[str, object]:
        """Return a compact region summary."""

        return {
            "name": self.name,
            "kind": "boundary_region",
            "tag": self.tag,
            "selection": self.selection,
            "local_facets": int(len(self.facets)),
        }


@dataclass(frozen=True)
class CellRegion:
    """Named cell/material region on a mesh.

    A cell region stores the tagged cell measure used for material-dependent
    domain integrals such as stiffness, mass, heat capacity, or body sources.
    """

    name: str
    domain: object
    tag: int
    cell_tags: object
    dx: object

    @property
    def measure(self):
        """Domain integration measure restricted to this cell region."""

        return self.dx(self.tag)

    def summary(self) -> dict[str, object]:
        """Return a compact region summary."""

        return {
            "name": self.name,
            "kind": "cell_region",
            "tag": self.tag,
        }


@dataclass(frozen=True)
class NodeRegion:
    """Named source-node region, including high-order geometry nodes."""

    name: str
    domain: object
    coordinates: np.ndarray
    source_labels: tuple[int, ...] = ()
    tolerance: float = 1.0e-9
    selection: str = "source_nodes"

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != self.domain.geometry.dim:
            raise ValueError("NodeRegion coordinates have incompatible shape.")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("NodeRegion coordinates must be finite.")
        if float(self.tolerance) <= 0.0:
            raise ValueError("NodeRegion tolerance must be positive.")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(
            self,
            "source_labels",
            tuple(int(value) for value in self.source_labels),
        )

    def marker(self, x) -> np.ndarray:
        """Select finite-element dof coordinates matching source nodes."""

        selected = np.zeros(x.shape[1], dtype=bool)
        for coordinate in self.coordinates:
            selected |= np.all(
                np.isclose(
                    x[: self.domain.geometry.dim],
                    coordinate[:, None],
                    rtol=0.0,
                    atol=float(self.tolerance),
                ),
                axis=0,
            )
        return selected

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "node_region",
            "selection": self.selection,
            "global_nodes": len(self.source_labels),
            "source_labels": self.source_labels,
            "tolerance": float(self.tolerance),
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

    # Keep the adapter import at the capability boundary. The AgentFEM mesh
    # namespace and all structured/XDMF paths remain independent of Gmsh.
    from dolfinx.io import gmsh as gmshio

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


def cuboid(
    lower,
    upper,
    cells,
    comm: MPI.Comm = MPI.COMM_WORLD,
    *,
    cell_type: str | mesh.CellType = "hexahedron",
):
    """Create a structured 3D cuboid mesh.

    The name remains distinct from :func:`box`, which is the long-standing
    axis-aligned region selector. Use ``cell_type="tetrahedron"`` when a
    simplicial subdivision is preferred.
    """

    lower_point = np.asarray(lower, dtype=float)
    upper_point = np.asarray(upper, dtype=float)
    selected_cells = tuple(int(value) for value in cells)
    if lower_point.shape != (3,) or upper_point.shape != (3,):
        raise ValueError("cuboid lower and upper points must have three coordinates.")
    if len(selected_cells) != 3 or any(value <= 0 for value in selected_cells):
        raise ValueError("cuboid cells must contain three positive integers.")
    if np.any(upper_point <= lower_point):
        raise ValueError("cuboid upper coordinates must exceed lower coordinates.")
    return mesh.create_box(
        comm,
        [lower_point, upper_point],
        list(selected_cells),
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

    gmsh = require_gmsh()

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


def require_gmsh():
    """Return the optional Gmsh Python API used only for direct Gmsh import."""

    return dependencies.require(
        "gmsh",
        extra="gmsh",
        capability="Reading a Gmsh model or .msh file through the Gmsh API",
    )


def optional_mesh_capabilities() -> tuple[dependencies.DependencyStatus, ...]:
    """Return availability of optional mesh-format integrations."""

    return (
        dependencies.status(
            "meshio",
            extra="mesh-formats",
            capability="External CAE mesh conversion",
        ),
        dependencies.status(
            "gmsh",
            extra="gmsh",
            capability="Gmsh model and .msh import",
        ),
    )


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


def convert_external_mesh_bundle(*args, **kwargs):
    """Convert selected source topologies into explicit solver-domain files."""

    return formats.convert_topology_bundle(*args, **kwargs)


def inspect_external_mesh(path):
    """Inventory external element blocks and named sets before conversion."""

    return formats.inspect_external_mesh(path)


def split_gmsh_physical_interface(*args, **kwargs):
    """Lower named Gmsh physical cell/surface groups to a split interface."""

    return formats.split_gmsh_physical_interface(*args, **kwargs)


def read_abaqus_mesh(
    path: str | Path,
    converted_path: str | Path,
    comm: MPI.Comm = MPI.COMM_WORLD,
    *,
    cell_type: str | None = None,
    reuse_conversion: bool = True,
) -> abaqus.AbaqusMeshImport:
    """Convert and read an Abaqus mesh while retaining source node labels.

    ``path`` may use a nonstandard extension such as ``.dat``; the reader
    explicitly selects Abaqus keyword syntax rather than guessing by suffix.
    """

    source = Path(path)
    converted = Path(converted_path)
    if comm.rank == 0:
        converted.parent.mkdir(parents=True, exist_ok=True)
    comm.barrier()
    nodes = abaqus.read_node_table(source)
    conversion = None
    conversion_error = None
    if comm.rank == 0:
        try:
            conversion = (
                formats.reusable_conversion(source, converted, cell_type=cell_type)
                if reuse_conversion
                else None
            )
            if conversion is None:
                conversion = formats.convert_abaqus_inp_to_xdmf(
                    source,
                    converted,
                    cell_type=cell_type,
                )
        except Exception as exc:  # broadcast avoids stranding non-root ranks
            conversion_error = (type(exc).__name__, str(exc))
    conversion, conversion_error = comm.bcast(
        (conversion, conversion_error),
        root=0,
    )
    if conversion_error is not None:
        error_type, message = conversion_error
        raise RuntimeError(
            f"Abaqus mesh conversion failed on MPI rank zero "
            f"({error_type}): {message}"
        )
    comm.barrier()
    imported = read_converted_xdmf(conversion, comm=comm)
    if comm.size == 1 and nodes.labels.size != imported.domain.geometry.x.shape[0]:
        raise ValueError(
            "Abaqus node count does not match the converted DOLFINx geometry: "
            f"{nodes.labels.size} source nodes versus "
            f"{imported.domain.geometry.x.shape[0]} geometry nodes."
        )
    return abaqus.AbaqusMeshImport(imported, nodes, conversion)


def external_mesh_formats() -> dict[str, str]:
    """Return common external formats supported through optional ``meshio``."""

    return formats.describe_supported_external_formats()


def read_converted_xdmf(
    conversion,
    comm: MPI.Comm = MPI.COMM_WORLD,
    *,
    mesh_name: str = "Grid",
    tag_grid_name: str = "Grid",
) -> FEMMesh:
    """Read a :class:`mesh.formats.MeshConversionResult` into DOLFINx.

    meshio writes named set values as XDMF attributes on a grid. DOLFINx
    distinguishes the grid ``name`` from ``attribute_name``; this helper owns
    that interoperability detail for both cell and separate facet files.
    """

    with io.XDMFFile(comm, str(conversion.mesh_path), "r") as xdmf:
        domain = xdmf.read_mesh(name=mesh_name)
        cell_tags = None
        if conversion.region_tags:
            tdim = domain.topology.dim
            domain.topology.create_connectivity(tdim, tdim)
            cell_tags = xdmf.read_meshtags(
                domain,
                name=tag_grid_name,
                attribute_name="agentfem_region",
            )

    facet_tags = None
    if conversion.facet_path is not None and conversion.boundary_tags:
        with io.XDMFFile(comm, str(conversion.facet_path), "r") as xdmf:
            tdim = domain.topology.dim
            domain.topology.create_connectivity(tdim - 1, tdim)
            facet_tags = xdmf.read_meshtags(
                domain,
                name=tag_grid_name,
                attribute_name="agentfem_boundary",
            )
    return FEMMesh(domain, cell_tags, facet_tags)


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


def require_tags(
    tags,
    required: int | tuple[int, ...] | list[int],
    *,
    name: str = "tags",
    comm=None,
) -> None:
    """Raise if required tags are absent globally.

    A valid distributed physical group may have no local entities on one MPI
    rank, so callers that own a mesh should pass its communicator.
    """

    required_tags = (required,) if isinstance(required, int) else tuple(required)
    local_available = (
        set() if tags is None else set(int(tag) for tag in np.unique(tags.values))
    )
    if comm is None:
        available = local_available
    else:
        available = set().union(*comm.allgather(tuple(sorted(local_available))))
    missing = [tag for tag in required_tags if tag not in available]
    if missing:
        raise ValueError(
            f"Missing required {name}: {missing}. "
            f"Available {name}: {sorted(available)}."
        )


def require_cell_tags(
    cell_tags,
    required: int | tuple[int, ...] | list[int],
    *,
    comm=None,
) -> None:
    """Require cell/material region tags."""

    require_tags(cell_tags, required, name="cell tags", comm=comm)


def require_facet_tags(
    facet_tags,
    required: int | tuple[int, ...] | list[int],
    *,
    comm=None,
) -> None:
    """Require boundary/facet tags."""

    require_tags(facet_tags, required, name="facet tags", comm=comm)


def boundary(domain, marker, *, name: str = "boundary", tag: int = 1) -> BoundaryRegion:
    """Create a named exterior boundary region from a geometric marker."""

    marker = where(marker)
    ds, facet_tags = tagged_boundary_measure(domain, marker, tag=tag)
    return BoundaryRegion(
        name=name,
        domain=domain,
        marker=marker,
        tag=tag,
        facet_tags=facet_tags,
        ds=ds,
        selection="geometric",
    )


def tagged_boundary_region(
    domain,
    facet_tags,
    *,
    tag: int,
    name: str = "tagged_boundary",
    marker=None,
) -> BoundaryRegion:
    """Create a boundary whose canonical selection is an imported facet tag.

    ``marker`` is optional and, when supplied, is retained only as independent
    audit evidence. Strong constraints and weak loads both consume ``tag``.
    """

    expected_dim = domain.topology.dim - 1
    if int(facet_tags.dim) != expected_dim:
        raise ValueError(
            "tagged_boundary_region requires facet tags on topological "
            f"dimension {expected_dim}, received dimension {facet_tags.dim}."
        )
    require_facet_tags(facet_tags, int(tag), comm=domain.comm)
    return BoundaryRegion(
        name=str(name),
        domain=domain,
        marker=marker,
        tag=int(tag),
        facet_tags=facet_tags,
        ds=boundary_measure(domain, facet_tags),
        selection="tagged" if marker is None else "hybrid",
    )


def audit_boundary_region(
    region: BoundaryRegion,
    *,
    strict: bool = False,
) -> dict[str, object]:
    """Inspect a boundary's identity, size, orientation, and tag/marker agreement.

    Imported physical tags are the canonical identity for tagged and hybrid
    regions. A marker attached to a hybrid region is independent audit evidence,
    never a second silently competing definition.
    """

    domain = region.domain
    comm = domain.comm
    fdim = domain.topology.dim - 1
    domain.topology.create_entities(fdim)
    facet_map = domain.topology.index_map(fdim)
    owned_limit = facet_map.size_local

    tagged = np.unique(region.facets)
    tagged_owned = tagged[tagged < owned_limit]
    geometric_owned = None
    if region.marker is not None:
        geometric = np.unique(
            mesh.locate_entities_boundary(domain, fdim, region.marker)
        )
        geometric_owned = geometric[geometric < owned_limit]

    local_tagged = int(tagged_owned.size)
    global_tagged = int(comm.allreduce(local_tagged, op=MPI.SUM))
    evidence: dict[str, object] = {
        "name": region.name,
        "tag": int(region.tag),
        "selection": region.selection,
        "local_tagged_facets": local_tagged,
        "global_tagged_facets": global_tagged,
    }

    if geometric_owned is not None:
        marker_only = np.setdiff1d(geometric_owned, tagged_owned, assume_unique=True)
        tag_only = np.setdiff1d(tagged_owned, geometric_owned, assume_unique=True)
        global_geometric = int(comm.allreduce(geometric_owned.size, op=MPI.SUM))
        global_marker_only = int(comm.allreduce(marker_only.size, op=MPI.SUM))
        global_tag_only = int(comm.allreduce(tag_only.size, op=MPI.SUM))
        evidence.update(
            {
                "global_geometric_facets": global_geometric,
                "marker_only_facets": global_marker_only,
                "tag_only_facets": global_tag_only,
                "consistent": global_marker_only == 0 and global_tag_only == 0,
            }
        )
    else:
        evidence["consistent"] = None

    local_measure = fem.assemble_scalar(fem.form(1.0 * region.measure))
    evidence["measure"] = float(comm.allreduce(local_measure, op=MPI.SUM))

    normal = ufl.FacetNormal(domain)
    normal_result = []
    for component in range(domain.geometry.dim):
        local = fem.assemble_scalar(fem.form(normal[component] * region.measure))
        normal_result.append(float(comm.allreduce(local, op=MPI.SUM)))
    evidence["integrated_normal"] = tuple(normal_result)

    if tagged_owned.size:
        midpoints = mesh.compute_midpoints(domain, fdim, tagged_owned)
        local_min = np.min(midpoints[:, : domain.geometry.dim], axis=0)
        local_max = np.max(midpoints[:, : domain.geometry.dim], axis=0)
    else:
        local_min = np.full(domain.geometry.dim, np.inf)
        local_max = np.full(domain.geometry.dim, -np.inf)
    global_min = np.empty_like(local_min)
    global_max = np.empty_like(local_max)
    comm.Allreduce(local_min, global_min, op=MPI.MIN)
    comm.Allreduce(local_max, global_max, op=MPI.MAX)
    evidence["midpoint_bounds"] = (
        None
        if not np.all(np.isfinite(global_min))
        else (tuple(global_min.tolist()), tuple(global_max.tolist()))
    )

    problems = []
    if global_tagged == 0:
        problems.append("selects no facets")
    if evidence.get("consistent") is False:
        problems.append(
            "tag and marker disagree "
            f"(marker-only={evidence['marker_only_facets']}, "
            f"tag-only={evidence['tag_only_facets']})"
        )
    evidence["valid"] = not problems
    evidence["issues"] = tuple(problems)
    if strict and problems:
        raise ValueError(f"Boundary {region.name!r} failed audit: {'; '.join(problems)}.")
    return evidence


def face(
    domain,
    *,
    axis: str | int,
    value: float,
    name: str | None = None,
    tag: int = 1,
    tolerance: float | None = None,
) -> BoundaryRegion:
    """Create a planar exterior boundary region such as ``x = 0``.

    This is the common application-level shortcut for rectangular/box-like
    domains. More complex boundaries can still use ``boundary(domain, marker)``.
    """

    axis_id = _axis_id(axis, domain.geometry.dim)
    atol = _coordinate_tolerance(domain) if tolerance is None else float(tolerance)

    def marker(x):
        return np.isclose(x[axis_id], value, rtol=0.0, atol=atol)

    label = name or f"{_axis_name(axis_id)}_{value:g}"
    return boundary(domain, where(marker, name=label), name=label, tag=tag)


def boundary_region(domain, marker, *, name: str = "boundary", tag: int = 1) -> BoundaryRegion:
    """Alias for ``boundary`` when a more explicit name reads better."""

    return boundary(domain, marker, name=name, tag=tag)


def cell_region(domain, cell_tags=None, *, tag: int, name: str = "cell_region", marker=None) -> CellRegion:
    """Create a named cell/material region.

    Pass existing ``cell_tags`` from an imported mesh, or pass a geometric
    ``marker`` to locate and tag cells directly.
    """

    if cell_tags is None:
        if marker is None:
            raise ValueError("cell_region requires cell_tags or marker.")
        cell_tags = mark_cells(domain, locate_cells(domain, marker), tag)
    else:
        require_cell_tags(cell_tags, tag, comm=domain.comm)
    return CellRegion(
        name=name,
        domain=domain,
        tag=tag,
        cell_tags=cell_tags,
        dx=cell_measure(domain, cell_tags),
    )


def region_measure(location):
    """Return a region's restricted measure or pass through a measure."""

    return location.measure if hasattr(location, "measure") else location


def region_marker(location):
    """Return a region's marker or pass through a marker callable."""

    return location.marker if hasattr(location, "marker") else location


def cells(domain, *, name: str, where, tag: int = 1) -> CellRegion:
    """Create a named cell region from a selector."""

    return cell_region(domain, tag=tag, name=name, marker=where)


def partition_cells(domain, **regions) -> RegionSet:
    """Partition mesh cells into named cell regions.

    The input values are selectors or vectorized coordinate predicates. Every
    local cell must belong to exactly one region. Internally this creates
    cell tags, but user code receives named ``CellRegion`` objects.
    """

    if not regions:
        raise ValueError("partition_cells requires at least one named region.")
    tag_to_name = {tag: name for tag, name in enumerate(regions, start=1)}
    tag_to_selector = {
        tag: where(regions[name], name=name)
        for tag, name in tag_to_name.items()
    }
    cell_tags = mark_cell_regions(domain, tag_to_selector)
    region_objects = {
        name: cell_region(domain, cell_tags, tag=tag, name=name)
        for tag, name in tag_to_name.items()
    }
    return RegionSet(domain=domain, regions=region_objects, tags=cell_tags, kind="cell")


def partition_boundaries(domain, **regions) -> RegionSet:
    """Create named exterior boundary regions from selectors."""

    if not regions:
        raise ValueError("partition_boundaries requires at least one named boundary.")
    region_objects = {}
    for tag, (name, selector) in enumerate(regions.items(), start=1):
        region_objects[name] = boundary(
            domain,
            where(selector, name=name),
            name=name,
            tag=tag,
        )
    return RegionSet(domain=domain, regions=region_objects, tags=None, kind="boundary")


def locate_cells(domain, marker):
    """Locate cells using a geometrical marker."""

    return mesh.locate_entities(domain, domain.topology.dim, where(marker))


def mark_cells(domain, cells, tag: int):
    """Create cell meshtags for a set of cells."""

    tdim = domain.topology.dim
    cells = np.asarray(cells, dtype=np.int32)
    cells = np.sort(cells)
    values = np.full(len(cells), tag, dtype=np.int32)
    return mesh.meshtags(domain, tdim, cells, values)


def mark_cell_regions(domain, tag_to_marker: dict[int, object]):
    """Create cell meshtags from several geometric cell markers.

    Each marker should select one material/domain region. Regions must not
    overlap on the same local mesh partition, and every local cell must be
    selected by exactly one marker. Markers are evaluated at cell midpoints so
    complementary material definitions do not leave untagged interface cells.
    """

    tdim = domain.topology.dim
    cell_map = domain.topology.index_map(tdim)
    cells = np.arange(cell_map.size_local, dtype=np.int32)
    midpoints = mesh.compute_midpoints(domain, tdim, cells).T
    values = np.full(len(cells), -1, dtype=np.int32)
    for tag, marker in tag_to_marker.items():
        selector = where(marker)
        selected = np.asarray(selector(midpoints), dtype=bool)
        if selected.shape != values.shape:
            raise ValueError(
                "Cell region marker must return one boolean per cell midpoint. "
                f"Expected shape {values.shape}, got {selected.shape}."
            )
        overlap = selected & (values >= 0)
        if np.any(overlap):
            sample = cells[overlap][:5].tolist()
            raise ValueError(f"Cell region markers overlap on cells: {sample}.")
        values[selected] = int(tag)
    if len(tag_to_marker) == 0:
        raise ValueError("mark_cell_regions requires at least one marker.")
    missing = values < 0
    if np.any(missing):
        sample = cells[missing][:5].tolist()
        raise ValueError(f"Cell region markers leave untagged cells: {sample}.")
    order = np.argsort(cells)
    return mesh.meshtags(domain, tdim, cells[order], values[order])


def tag_field(domain, tags, *, name: str = "Tag"):
    """Create a DG0 visualization field from cell tags.

    The current implementation supports cell tags. It is useful for writing
    material ids, partitions, or element groups to XDMF for ParaView inspection.
    """

    if tags is None:
        raise ValueError("tag_field requires a MeshTags object.")
    if int(tags.dim) != int(domain.topology.dim):
        raise ValueError(
            "tag_field currently supports cell tags only. "
            f"Got tag dimension {tags.dim}, mesh cell dimension {domain.topology.dim}."
        )
    Q = fem.functionspace(domain, ("DG", 0))
    result = fem.Function(Q, name=name)
    values = result.x.array
    dofmap = Q.dofmap
    for cell, tag in zip(tags.indices, tags.values):
        values[dofmap.cell_dofs(int(cell))] = float(tag)
    result.x.scatter_forward()
    return result


def locate_boundary_facets(domain, marker):
    """Locate exterior facets using a geometrical marker."""

    facet_dim = domain.topology.dim - 1
    return mesh.locate_entities_boundary(domain, facet_dim, where(marker))


def mark_facets(domain, facets, tag: int):
    """Create a meshtags object for a set of facets."""

    facet_dim = domain.topology.dim - 1
    facets = np.asarray(facets, dtype=np.int32)
    facets = np.sort(facets)
    values = np.full(len(facets), tag, dtype=np.int32)
    return mesh.meshtags(domain, facet_dim, facets, values)


def mark_boundary_facets(domain, marker, tag: int):
    """Locate and tag exterior facets in one step."""

    return mark_facets(domain, locate_boundary_facets(domain, marker), tag)


def boundary_measure(domain, facet_tags=None):
    """Create a boundary integration measure."""

    return ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)


def cell_measure(domain, cell_tags=None):
    """Create a domain integration measure."""

    return ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)


def facet_normal(domain):
    """Return the outward facet normal for boundary models."""

    return ufl.FacetNormal(domain)


def tagged_boundary_measure(domain, marker, tag: int):
    """Locate/tag exterior facets and return ``(ds, facet_tags)``."""

    facet_tags = mark_boundary_facets(domain, marker, tag)
    return boundary_measure(domain, facet_tags), facet_tags


def _axis_id(axis: str | int, gdim: int) -> int:
    if isinstance(axis, str):
        names = {"x": 0, "y": 1, "z": 2}
        key = axis.lower()
        if key not in names:
            raise ValueError("axis must be 'x', 'y', 'z', or an integer.")
        axis_id = names[key]
    else:
        axis_id = int(axis)
    if axis_id < 0 or axis_id >= int(gdim):
        raise ValueError(f"axis {axis!r} is outside geometric dimension {gdim}.")
    return axis_id


def _axis_name(axis_id: int) -> str:
    return ("x", "y", "z")[axis_id] if axis_id < 3 else f"axis{axis_id}"


def _coordinate_tolerance(domain) -> float:
    coords = domain.geometry.x
    if coords.size == 0:
        return 1.0e-12
    span = float(np.max(coords) - np.min(coords))
    return max(1.0, span) * 1.0e-12
