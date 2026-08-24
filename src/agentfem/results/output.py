"""Declarative field output and compact ParaView-ready time series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import warnings
import xml.etree.ElementTree as ET

import numpy as np
from dolfinx import fem, plot
from dolfinx.io import XDMFFile
import h5py

from .field_catalog import resolve_field_variables
from .finite_strain import finite_strain_cell_fields


@dataclass(frozen=True)
class FieldOutputArtifacts:
    """Files and final live fields produced by one output plan."""

    reference_xdmf: Path | None
    unified_xdmf: Path | None
    deformed_pvd: Path | None
    deformed_frames: tuple[Path, ...]
    final_fields: tuple[object, ...]


@dataclass(frozen=True)
class ResultFieldArtifacts:
    """One completed-result field dataset and its explicit layout contract."""

    xdmf: Path
    hdf5: Path | None
    paraview: Path | None
    backend: str
    layout: str
    geometry: str
    warp_field: str | None
    field_names: tuple[str, ...]
    omitted_fields: tuple[str, ...] = ()
    warp_field_semantic: str | None = None
    physical_components: int | None = None
    stored_components: int | None = None
    geometry_dimension: int = 3
    physical_model_dimension: int = 3
    warp_compatible: bool = False

    def summary(self) -> dict[str, object]:
        visualization = self.paraview if self.paraview is not None else self.xdmf
        summary = {
            "status": "completed",
            "backend": self.backend,
            "layout": self.layout,
            "geometry": self.geometry,
            "scientific_artifact": str(self.xdmf),
            "scientific_xdmf_layout": (
                "collective_grid_per_field"
                if self.paraview is not None
                else "single_uniform_grid"
            ),
            "recommended_visualization_artifact": str(visualization),
            "visualization_geometry_datasets_per_time": 1,
            "visualization_requires_extract_block": False,
            "warp_field": self.warp_field,
            "warp_field_semantic": self.warp_field_semantic,
            "physical_components": self.physical_components,
            "stored_components": self.stored_components,
            "geometry_dimension": self.geometry_dimension,
            "physical_model_dimension": self.physical_model_dimension,
            "warp_compatible": self.warp_compatible,
            "field_aliases": (
                {}
                if self.warp_field is None or self.warp_field_semantic is None
                else {self.warp_field_semantic: self.warp_field}
            ),
        }
        if self.paraview is not None:
            summary["paraview"] = str(self.paraview)
        return summary


@dataclass(frozen=True)
class FieldOutput:
    """What fields to save, how often, and in which configuration."""

    variables: tuple[str, ...] = ("U", "S", "E", "MISES")
    every: int | str | None = None
    intervals: int | None = None
    configuration: str = "both"
    deformation_scale: float = 1.0
    backend: str = "xdmf"

    def __post_init__(self) -> None:
        every = self.every
        if isinstance(every, str):
            normalized_every = every.lower().replace("-", "_").strip()
            if normalized_every not in {"increment", "every_increment"}:
                raise ValueError(
                    "FieldOutput.every must be a positive integer or 'increment'."
                )
            every = 1
        if every is None and self.intervals is None:
            every = 1
        if every is not None and int(every) <= 0:
            raise ValueError("FieldOutput.every must be positive.")
        if self.intervals is not None and int(self.intervals) <= 0:
            raise ValueError("FieldOutput.intervals must be positive.")
        if every is not None and self.intervals is not None:
            raise ValueError(
                "Choose output every n increments or evenly spaced intervals, not both."
            )
        configuration = self.configuration.lower().replace("-", "_")
        if configuration not in {"reference", "deformed", "both"}:
            raise ValueError(
                "FieldOutput.configuration must be reference, deformed, or both."
            )
        if not np.isfinite(self.deformation_scale):
            raise ValueError("FieldOutput.deformation_scale must be finite.")
        backend = self.backend.lower().replace("-", "_")
        if backend not in {"xdmf", "pvd", "both"}:
            raise ValueError("FieldOutput.backend must be xdmf, pvd, or both.")
        resolve_field_variables(self.variables, finite_strain=True)
        object.__setattr__(self, "every", None if every is None else int(every))
        object.__setattr__(
            self,
            "intervals",
            None if self.intervals is None else int(self.intervals),
        )
        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(
            self,
            "variables",
            tuple(str(value).upper() for value in self.variables),
        )

    def required_factors(self) -> tuple[float, ...]:
        """Return exact normalized output marks for interval-based output."""

        if self.intervals is None:
            return ()
        return tuple(
            index / self.intervals
            for index in range(1, self.intervals + 1)
        )

    def write_finite_strain(
        self,
        directory,
        *,
        domain,
        snapshots,
        material,
        basename: str = "results",
    ) -> FieldOutputArtifacts:
        """Write a unified XDMF/HDF5 series and/or a legacy PVD series."""

        selected = tuple(snapshots)
        if not selected:
            raise ValueError("FieldOutput requires at least one saved snapshot.")
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        variables = resolve_field_variables(self.variables, finite_strain=True)
        cell_variables = tuple(
            variable.key for variable in variables if variable.key != "U"
        )
        per_frame_fields = []
        for snapshot in selected:
            fields = finite_strain_cell_fields(
                snapshot.solution,
                material,
                variables=cell_variables,
                pressure=getattr(snapshot, "fields", {}).get("PRESSURE"),
            )
            per_frame_fields.append(fields)

        reference_xdmf = None
        unified_xdmf = None
        if self.backend in {"xdmf", "both"}:
            if domain.comm.size > 1:
                reference_xdmf = write_parallel_xdmf_series(
                    output / f"{basename}_parallel.xdmf",
                    selected,
                    per_frame_fields,
                )
            else:
                unified_xdmf = output / f"{basename}.xdmf"
                geometry_scale = (
                    0.0
                    if self.configuration == "reference"
                    else self.deformation_scale
                )
                unified_xdmf = write_unified_xdmf_series(
                    unified_xdmf,
                    selected,
                    per_frame_fields,
                    deformation_scale=geometry_scale,
                    store_reference_geometry=True,
                )

        pvd_path = None
        frame_paths = ()
        if domain.comm.size > 1 and self.backend in {"xdmf", "both"}:
            pvd_path = write_parallel_vtk_series(
                output / f"{basename}.pvd",
                selected,
                per_frame_fields,
            )
        if self.backend in {"pvd", "both"}:
            if domain.comm.size > 1:
                if pvd_path is None:
                    pvd_path = write_parallel_vtk_series(
                        output / f"{basename}.pvd",
                        selected,
                        per_frame_fields,
                    )
            else:
                pvd_path, frame_paths = write_deformed_vtk_series(
                    output / f"{basename}_deformed.pvd",
                    selected,
                    per_frame_fields,
                    deformation_scale=self.deformation_scale,
                )
        return FieldOutputArtifacts(
            reference_xdmf,
            unified_xdmf,
            pvd_path,
            frame_paths,
            tuple(per_frame_fields[-1]),
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "field_output",
            "variables": self.variables,
            "every": self.every,
            "intervals": self.intervals,
            "configuration": self.configuration,
            "deformation_scale": self.deformation_scale,
            "backend": self.backend,
            "finite_strain_aliases": {"E": "LE"},
        }


def write_parallel_vtk_series(path, snapshots, fields_by_frame) -> Path:
    """Write collective single-dataset ParaView frames under MPI.

    The mesh remains in the reference configuration and ``U`` is a point-data
    vector on the same dataset as every DG0 cell field. ParaView can therefore
    apply one Warp By Vector filter without retaining unwarped duplicate
    blocks. The PVD/PVTU/VTU family is intentionally a presentation backend,
    not compact checkpoint storage.
    """

    from .. import io as io_api

    selected_snapshots = tuple(snapshots)
    selected_fields = tuple(tuple(frame) for frame in fields_by_frame)
    if not selected_snapshots or len(selected_snapshots) != len(selected_fields):
        raise ValueError(
            "Parallel VTK output requires one field collection per snapshot."
        )
    output = Path(path)
    domain = selected_snapshots[0].solution.function_space.mesh
    with io_api.ParaViewTimeSeries(output, domain) as writer:
        for snapshot, frame_fields in zip(selected_snapshots, selected_fields):
            writer.write_fields(
                float(snapshot.load_factor),
                snapshot.solution,
                *frame_fields,
            )
    return output

def field_output(
    *variables,
    every: int | str | None = None,
    intervals: int | None = None,
    configuration: str = "deformed",
    deformation_scale: float = 1.0,
    backend: str = "xdmf",
) -> FieldOutput:
    """Create a concise, inspectable field-output request."""

    return FieldOutput(
        variables=tuple(variables) if variables else FieldOutput().variables,
        every=every,
        intervals=intervals,
        configuration=configuration,
        deformation_scale=deformation_scale,
        backend=backend,
    )


def write_parallel_xdmf_series(
    xdmf_path,
    snapshots,
    cell_fields,
) -> Path:
    """Collectively write reference-configuration MPI field histories.

    DOLFINx owns the distributed topology, HDF5 layout, and collective I/O.
    Directly deformed geometry remains a serial presentation product; the
    displacement field in this scientific file can be warped in ParaView.
    """

    selected = tuple(snapshots)
    fields_by_frame = tuple(cell_fields)
    if not selected or len(selected) != len(fields_by_frame):
        raise ValueError(
            "Parallel XDMF requires equal non-empty snapshot and field frames."
        )
    domain = selected[0].solution.function_space.mesh
    xdmf = Path(xdmf_path)
    if domain.comm.rank == 0:
        xdmf.parent.mkdir(parents=True, exist_ok=True)
    domain.comm.barrier()
    with XDMFFile(domain.comm, xdmf, "w") as writer:
        writer.write_mesh(domain)
        for snapshot, fields in zip(selected, fields_by_frame):
            time = float(snapshot.load_factor)
            writer.write_function(snapshot.solution, time)
            for field in fields:
                writer.write_function(field, time)
    return xdmf


class UnifiedXDMFTimeSeries:
    """Incremental single-grid XDMF/HDF5 writer for serial result histories.

    Every accepted frame is one XDMF ``Uniform`` grid carrying the primary
    point field and all compatible point/cell attributes.  Unlike repeated
    DOLFINx ``write_function`` calls, this layout opens in ParaView as one
    geometry per time value and therefore needs no ``Extract Block`` step.
    """

    def __init__(
        self,
        path,
        *,
        deformation_scale: float = 0.0,
        store_reference_geometry: bool = True,
        compression: int = 4,
    ) -> None:
        self.path = Path(path)
        self.h5_path = self.path.with_suffix(".h5")
        self.deformation_scale = float(deformation_scale)
        self.store_reference_geometry = bool(store_reference_geometry)
        self.compression = int(compression)
        self._h5 = None
        self._root = None
        self._temporal = None
        self._frame_count = 0
        self._field_contract = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
        if exc_type is None:
            if self._frame_count == 0:
                raise ValueError("UnifiedXDMFTimeSeries wrote no result frames.")
            tree = ET.ElementTree(self._root)
            ET.indent(tree, space="  ")
            tree.write(self.path, encoding="utf-8", xml_declaration=True)
        return False

    def _initialize(self, primary) -> None:
        domain = primary.function_space.mesh
        if domain.comm.size != 1:
            raise NotImplementedError(
                "Unified compressed XDMF is a serial writer. Under MPI use "
                "ParaViewTimeSeries, which keeps one dataset per saved time."
            )
        topology, cell_types, coordinates = plot.vtk_mesh(primary.function_space)
        nodes_per_cell = int(topology[0])
        connectivity = np.asarray(topology).reshape(
            -1, nodes_per_cell + 1
        )[:, 1:]
        unique_cell_types = np.unique(cell_types)
        if unique_cell_types.size != 1:
            raise ValueError("Unified XDMF currently requires one VTK cell type.")
        primary_shape = tuple(getattr(primary, "ufl_shape", ()))
        if len(primary_shape) > 1:
            raise NotImplementedError(
                "Unified XDMF primary fields support scalar or vector fields."
            )
        vector_primary = len(primary_shape) == 1
        primary_name = (
            "U" if vector_primary else str(getattr(primary, "name", "Primary"))
        )
        topology_type = _xdmf_topology_type(
            domain.topology.cell_type.name,
            nodes_per_cell,
        )
        effective_scale = self.deformation_scale if vector_primary else 0.0

        self._connectivity = connectivity
        self._reference_coordinates = coordinates
        self._topology_type = topology_type
        self._nodes_per_cell = nodes_per_cell
        self._cell_count = len(cell_types)
        self._point_count = coordinates.shape[0]
        self._geometry_dimension = int(coordinates.shape[1])
        self._physical_model_dimension = int(domain.geometry.dim)
        self._primary_shape = primary_shape
        self._primary_name = primary_name
        self._vector_primary = vector_primary
        self._effective_scale = effective_scale
        self._h5_options = {
            "compression": "gzip",
            "compression_opts": self.compression,
            "shuffle": True,
        }

        self._root = ET.Element("Xdmf", Version="3.0")
        xml_domain = ET.SubElement(self._root, "Domain")
        self._temporal = ET.SubElement(
            xml_domain,
            "Grid",
            Name="Results",
            GridType="Collection",
            CollectionType="Temporal",
        )
        self._h5 = h5py.File(self.h5_path, "w")
        self._h5.attrs["agentfem_schema"] = "agentfem.unified-xdmf"
        self._h5.attrs["schema_version"] = "0.1.0"
        self._h5.attrs["deformation_scale"] = effective_scale
        self._h5.attrs["vtk_cell_type"] = int(unique_cell_types[0])
        self._h5.attrs["xdmf_topology_type"] = topology_type
        self._h5.attrs["nodes_per_cell"] = nodes_per_cell
        self._h5.attrs["point_count"] = self._point_count
        self._h5.attrs["cell_count"] = self._cell_count
        self._h5.attrs["primary_field"] = primary_name
        self._h5.attrs["primary_semantic_name"] = (
            "Displacement" if vector_primary else primary_name
        )
        self._h5.attrs["geometry_dimension"] = self._geometry_dimension
        self._h5.attrs["physical_model_dimension"] = self._physical_model_dimension
        self._h5.attrs["primary_physical_components"] = (
            int(primary_shape[0]) if vector_primary else 1
        )
        self._h5.attrs["primary_storage_components"] = (
            self._geometry_dimension if vector_primary else 1
        )
        self._h5.attrs["warp_compatible"] = bool(vector_primary)
        self._h5.attrs["geometry_mode"] = (
            "deformed" if effective_scale != 0.0 else "reference"
        )
        mesh_group = self._h5.create_group("Mesh")
        mesh_group.create_dataset(
            "Topology", data=connectivity, **self._h5_options
        )
        if self.store_reference_geometry:
            mesh_group.create_dataset(
                "ReferenceGeometry", data=coordinates, **self._h5_options
            )

    def write_fields(self, time: float, *fields) -> None:
        """Append one time value with one primary and any auxiliary fields."""

        if not fields:
            raise ValueError("UnifiedXDMFTimeSeries requires at least one field.")
        primary, *auxiliary = fields
        if self._h5 is None:
            self._initialize(primary)
        if tuple(getattr(primary, "ufl_shape", ())) != self._primary_shape:
            raise ValueError("Unified XDMF primary-field shape changed between frames.")

        value_size = (
            int(np.prod(self._primary_shape)) if self._primary_shape else 1
        )
        primary_values = np.asarray(primary.x.array).reshape(-1, value_size)
        if primary_values.shape[0] != self._point_count:
            raise ValueError("Unified XDMF primary-field dofs must match mesh points.")
        displacement = np.zeros_like(self._reference_coordinates)
        if self._vector_primary:
            displacement[:, : int(self._primary_shape[0])] = primary_values
        geometry = (
            self._reference_coordinates + self._effective_scale * displacement
        )

        frame_name = f"{self._frame_count:04d}"
        frame_group = self._h5.create_group(f"Frames/{frame_name}")
        frame_group.attrs["load_factor"] = float(time)
        frame_group.attrs["coordinate"] = float(time)
        frame_group.create_dataset(
            "Geometry", data=geometry, **self._h5_options
        )
        point_group = frame_group.create_group("Point")
        cell_group = frame_group.create_group("Cell")
        stored_primary = displacement if self._vector_primary else primary_values[:, 0]
        primary_dataset = point_group.create_dataset(
            self._primary_name, data=stored_primary, **self._h5_options
        )
        primary_dataset.attrs["physical_components"] = (
            int(self._primary_shape[0]) if self._vector_primary else 1
        )
        primary_dataset.attrs["stored_components"] = (
            self._geometry_dimension if self._vector_primary else 1
        )
        if self._vector_primary:
            primary_dataset.attrs["semantic_name"] = "Displacement"
        attributes = [(self._primary_name, "Node", stored_primary)]
        if self._vector_primary:
            magnitude = np.linalg.norm(primary_values, axis=1)
            point_group.create_dataset("UMAG", data=magnitude, **self._h5_options)
            attributes.append(("UMAG", "Node", magnitude))

        for field in auxiliary:
            name = str(getattr(field, "name", "Field"))
            if name in {self._primary_name, "UMAG"}:
                continue
            center, shaped = _unified_field_values(
                field,
                solution_space=primary.function_space,
                point_count=self._point_count,
                cell_count=self._cell_count,
            )
            shaped = _visualization_vector_values(
                shaped,
                field,
                components=self._geometry_dimension,
            )
            group = point_group if center == "Node" else cell_group
            dataset = group.create_dataset(name, data=shaped, **self._h5_options)
            field_shape = tuple(getattr(field, "ufl_shape", ()))
            if len(field_shape) == 1:
                dataset.attrs["physical_components"] = int(field_shape[0])
                dataset.attrs["stored_components"] = int(np.asarray(shaped).shape[-1])
            attributes.append((name, center, shaped))

        contract = tuple(
            (name, center, tuple(np.asarray(values).shape[1:]))
            for name, center, values in attributes
        )
        if self._field_contract is None:
            self._field_contract = contract
        elif contract != self._field_contract:
            raise ValueError(
                "Unified XDMF field names, locations, or shapes changed "
                "between frames."
            )

        grid = ET.SubElement(
            self._temporal,
            "Grid",
            Name=f"Frame_{frame_name}",
            GridType="Uniform",
        )
        ET.SubElement(grid, "Time", Value=f"{float(time):.16g}")
        topology_xml = ET.SubElement(
            grid,
            "Topology",
            TopologyType=self._topology_type,
            NumberOfElements=str(self._cell_count),
            NodesPerElement=str(self._nodes_per_cell),
        )
        _data_item(
            topology_xml,
            dimensions=self._connectivity.shape,
            number_type="Int",
            value=f"{self.h5_path.name}:/Mesh/Topology",
        )
        geometry_xml = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
        _data_item(
            geometry_xml,
            dimensions=geometry.shape,
            value=f"{self.h5_path.name}:/Frames/{frame_name}/Geometry",
        )
        for name, center, values in attributes:
            group_name = "Point" if center == "Node" else "Cell"
            _attribute(
                grid,
                name,
                center,
                values,
                f"{self.h5_path.name}:/Frames/{frame_name}/{group_name}/{name}",
            )
        self._frame_count += 1


def write_unified_xdmf_series(
    xdmf_path,
    snapshots,
    cell_fields,
    *,
    deformation_scale: float = 1.0,
    store_reference_geometry: bool = True,
    compression: int = 4,
) -> Path:
    """Write one temporal XDMF and one compressed HDF5 heavy-data file.

    Each temporal grid owns its geometry and all point/cell attributes, while
    every frame references one shared topology dataset. A vector primary field
    is interpreted as displacement and may deform the geometry. A scalar
    primary field, such as temperature, is written on the reference geometry.
    Reference coordinates remain in HDF5 for provenance.
    """

    selected = tuple(snapshots)
    fields_by_frame = tuple(cell_fields)
    if not selected or len(selected) != len(fields_by_frame):
        raise ValueError(
            "Unified XDMF requires equal non-empty snapshot and field frames."
        )
    xdmf = Path(xdmf_path)
    with UnifiedXDMFTimeSeries(
        xdmf,
        deformation_scale=deformation_scale,
        store_reference_geometry=store_reference_geometry,
        compression=compression,
    ) as writer:
        for snapshot, fields in zip(selected, fields_by_frame):
            writer.write_fields(
                float(snapshot.load_factor),
                snapshot.solution,
                *tuple(fields),
            )
    return xdmf


def write_result_fields(
    result,
    path,
    *,
    time: float = 0.0,
    names=(),
    deformation_scale: float = 0.0,
) -> ResultFieldArtifacts:
    """Write the live, visualization-ready fields of one SimulationResult.

    Integration-point fields remain first-class entries in ``SimulationResult``
    but are not silently presented as ordinary nodal/cell visualization data.
    Their explicitly recovered ``*_CELL`` counterparts are written instead.
    """

    requested = tuple(str(item) for item in names)
    records = tuple(result.fields.values())
    if requested:
        missing = tuple(name for name in requested if name not in result.fields)
        if missing:
            raise KeyError(f"Unknown result fields requested for output: {missing!r}.")
        records = tuple(result.fields[name] for name in requested)
    live = tuple(item for item in records if item.field is not None)
    if not live:
        raise ValueError("Result field output requires at least one live field.")
    forbidden = tuple(
        item.name for item in live if item.location == "quadrature_points"
    )
    if requested and forbidden:
        raise ValueError(
            "Quadrature fields cannot be written as ordinary XDMF attributes: "
            f"{forbidden!r}. Request their recovered *_CELL fields or use the "
            "quadrature-state export contract."
        )
    writable = tuple(
        item for item in live if item.location != "quadrature_points"
    )
    if not writable:
        raise ValueError(
            "No visualization-ready live fields remain after excluding "
            "quadrature-point state."
        )
    primary = next(
        (
            item
            for item in writable
            if item.processing.get("method") == "primary_finite_element_solution"
        ),
        writable[0],
    )
    solution = getattr(primary.field, "value", primary.field)
    auxiliary = tuple(
        getattr(item.field, "value", item.field)
        for item in writable
        if item is not primary
    )
    domain = solution.function_space.mesh
    physical_shape = tuple(getattr(solution, "ufl_shape", ()))
    vector_solution = len(physical_shape) == 1
    physical_components = int(physical_shape[0]) if vector_solution else None
    geometry_dimension = int(domain.geometry.x.shape[1])
    stored_components = geometry_dimension if vector_solution else None
    selected_path = Path(path)
    if domain.comm.size == 1:
        write_unified_xdmf_series(
            selected_path,
            (SimpleNamespace(solution=solution, load_factor=float(time)),),
            (auxiliary,),
            deformation_scale=float(deformation_scale),
        )
        layout = "single_uniform_grid"
        backend = "agentfem_unified_xdmf"
        paraview = None
    else:
        from .. import io

        output_fields = []
        for function in (solution, *auxiliary):
            if function is solution:
                coordinate_maps = getattr(domain.geometry, "cmaps", ())
                degree = int(
                    getattr(coordinate_maps[0], "degree", 1)
                    if coordinate_maps
                    else 1
                )
                function = io.interpolate_for_xdmf(
                    function,
                    degree=degree,
                    name=getattr(function, "name", primary.name),
                )
            output_fields.append(function)
        with io.XDMFTimeSeries(selected_path, domain) as writer:
            writer.write_fields(float(time), *output_fields)
        paraview = selected_path.with_suffix(".pvd")
        with io.ParaViewTimeSeries(paraview, domain) as writer:
            writer.write_fields(float(time), *output_fields)
        layout = "scientific_xdmf_plus_single_paraview_dataset"
        backend = "dolfinx_collective_xdmf_and_vtk"
    hdf5 = selected_path.with_suffix(".h5")
    return ResultFieldArtifacts(
        xdmf=selected_path,
        hdf5=hdf5 if hdf5.exists() else None,
        paraview=paraview,
        backend=backend,
        layout=layout,
        geometry=("deformed" if float(deformation_scale) != 0.0 else "reference"),
        warp_field=("U" if vector_solution else None),
        warp_field_semantic=("Displacement" if vector_solution else None),
        physical_components=physical_components,
        stored_components=stored_components,
        geometry_dimension=geometry_dimension,
        physical_model_dimension=int(domain.geometry.dim),
        warp_compatible=bool(vector_solution and stored_components == geometry_dimension),
        field_names=tuple(item.name for item in writable),
        omitted_fields=tuple(
            item.name for item in live if item.location == "quadrature_points"
        ),
    )


def attach_result_field_output(
    result,
    path,
    *,
    time: float = 0.0,
    names=(),
    deformation_scale: float = 0.0,
    strict: bool = False,
):
    """Write and register one result dataset without hiding solve success."""

    selected_path = Path(path)
    try:
        artifacts = write_result_fields(
            result,
            selected_path,
            time=time,
            names=names,
            deformation_scale=deformation_scale,
        )
        result.metadata["field_output"] = artifacts.summary()
        result.metadata["field_output_fields"] = {
            "included": artifacts.field_names,
            "omitted": artifacts.omitted_fields,
        }
        result.add_artifact("fields_xdmf", artifacts.xdmf)
        if artifacts.hdf5 is not None:
            result.add_artifact("fields_hdf5", artifacts.hdf5)
        if artifacts.paraview is not None:
            result.add_artifact("fields_paraview", artifacts.paraview)
    except Exception as exc:
        result.status = "completed_with_output_errors"
        result.metadata["field_output"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "requested_path": str(selected_path),
        }
        if strict:
            raise
        warnings.warn(
            f"Simulation completed, but field output failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    return result


def _unified_field_values(
    field,
    *,
    solution_space,
    point_count: int,
    cell_count: int,
) -> tuple[str, np.ndarray]:
    """Return XDMF center and values for one supported finite-element field."""

    function = field.field if hasattr(field, "field") else field
    element = function.function_space.element
    basix_element = getattr(element, "basix_element", None)
    discontinuous = getattr(basix_element, "discontinuous", None)
    if discontinuous is None:
        family = str(function.function_space.ufl_element().family()).lower()
        discontinuous = "discontinuous" in family or family in {"dg", "dp"}
    values = np.asarray(function.x.array)
    value_shape = tuple(getattr(function, "ufl_shape", ()))
    value_size = int(np.prod(value_shape)) if value_shape else 1

    if discontinuous:
        degree = _element_degree(function.function_space)
        if degree != 0 or values.size != cell_count * value_size:
            raise ValueError(
                f"Discontinuous field {function.name!r} is degree {degree}; "
                "single-grid XDMF currently accepts cellwise DG0 fields. "
                "Project to DG0 for cell output or use a dedicated high-order "
                "visualization backend."
            )
        shaped = values.reshape(cell_count, value_size)
        return "Cell", shaped[:, 0] if value_size == 1 else shaped

    if values.size != point_count * value_size:
        degree = _element_degree(solution_space)
        target_element = (
            ("Lagrange", degree)
            if not value_shape
            else ("Lagrange", degree, value_shape)
        )
        target_space = fem.functionspace(solution_space.mesh, target_element)
        interpolated = fem.Function(target_space, name=function.name)
        interpolated.interpolate(function)
        values = np.asarray(interpolated.x.array)
    if values.size != point_count * value_size:
        raise ValueError(
            f"Point field {function.name!r} does not align with the output mesh "
            f"({values.size} values for {point_count} points)."
        )
    shaped = values.reshape(point_count, value_size)
    return "Node", shaped[:, 0] if value_size == 1 else shaped


def _visualization_vector_values(values, field, *, components: int = 3) -> np.ndarray:
    """Pad physical vectors to the coordinate dimension for VTK/ParaView.

    A two-dimensional finite-element field remains a two-component unknown in
    memory.  XDMF/VTK presentation geometry is nevertheless stored as XYZ, so
    a zero out-of-plane component is added only to the visualization array.
    Scalars and tensors are returned unchanged.
    """

    array = np.asarray(values)
    shape = tuple(getattr(field, "ufl_shape", ()))
    if len(shape) != 1 or array.ndim != 2:
        return array
    physical_components = int(shape[0])
    selected_components = int(components)
    if physical_components >= selected_components:
        return array
    padded = np.zeros((array.shape[0], selected_components), dtype=array.dtype)
    padded[:, :physical_components] = array
    return padded


def _element_degree(space) -> int:
    degree = getattr(space.element, "degree", None)
    if degree is None:
        degree = space.ufl_element().degree
        if callable(degree):
            degree = degree()
    if isinstance(degree, tuple):
        degree = max(degree)
    return int(degree)


def read_unified_xdmf_series(xdmf_path) -> tuple[object, ...]:
    """Read AgentFEM's compact XDMF/HDF5 frames as PyVista grids.

    This reader is intentionally small and schema-specific. General XDMF
    readers such as ParaView remain the primary interoperability path.
    """

    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "Reading unified XDMF frames requires optional `pyvista`."
        ) from exc

    xdmf = Path(xdmf_path)
    h5_path = xdmf.with_suffix(".h5")
    grids = []
    with h5py.File(h5_path, "r") as h5:
        if h5.attrs.get("agentfem_schema") != "agentfem.unified-xdmf":
            raise ValueError(f"{h5_path} is not an AgentFEM unified XDMF store.")
        connectivity = np.asarray(h5["Mesh/Topology"])
        nodes_per_cell = int(h5.attrs["nodes_per_cell"])
        vtk_cell_type = int(h5.attrs["vtk_cell_type"])
        vtk_topology = np.column_stack(
            (
                np.full(connectivity.shape[0], nodes_per_cell, dtype=np.int64),
                connectivity,
            )
        ).reshape(-1)
        cell_types = np.full(
            connectivity.shape[0],
            vtk_cell_type,
            dtype=np.uint8,
        )
        for frame_name in sorted(h5["Frames"]):
            frame = h5[f"Frames/{frame_name}"]
            grid = pv.UnstructuredGrid(
                vtk_topology,
                cell_types,
                np.asarray(frame["Geometry"]),
            )
            for name, dataset in frame["Point"].items():
                grid.point_data[name] = np.asarray(dataset)
            for name, dataset in frame["Cell"].items():
                grid.cell_data[name] = np.asarray(dataset)
            grid.field_data["load_factor"] = np.asarray(
                [float(frame.attrs["load_factor"])]
            )
            grids.append(grid)
    return tuple(grids)


def write_deformed_vtk_series(
    pvd_path,
    snapshots,
    cell_fields,
    *,
    deformation_scale: float = 1.0,
) -> tuple[Path, tuple[Path, ...]]:
    """Write one deformed VTU grid per frame and a ParaView PVD collection.

    Every VTU contains the deformed coordinates, nodal displacement, and all
    requested cell fields on the same grid. This is a presentation product;
    the reference-configuration XDMF remains the scientific FE record.
    """

    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "Deformed PVD/VTU output requires the optional dependency `pyvista`."
        ) from exc

    selected = tuple(snapshots)
    fields_by_frame = tuple(cell_fields)
    if len(selected) != len(fields_by_frame):
        raise ValueError("Snapshots and cell-field frames must have equal length.")
    pvd = Path(pvd_path)
    pvd.parent.mkdir(parents=True, exist_ok=True)
    frame_directory = pvd.parent / f"{pvd.stem}_frames"
    frame_directory.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for frame_index, (snapshot, fields) in enumerate(
        zip(selected, fields_by_frame)
    ):
        solution = snapshot.solution
        topology, cell_types, coordinates = plot.vtk_mesh(
            solution.function_space
        )
        grid = pv.UnstructuredGrid(topology, cell_types, coordinates)
        value_dimension = solution.ufl_shape[0]
        displacement_values = np.asarray(solution.x.array).reshape(
            -1,
            value_dimension,
        )
        if displacement_values.shape[0] != coordinates.shape[0]:
            raise ValueError(
                "Deformed output requires displacement dofs to match VTK points."
            )
        displacement = np.zeros_like(coordinates)
        displacement[:, :value_dimension] = displacement_values
        grid.points = coordinates + float(deformation_scale) * displacement
        grid.point_data["U"] = _visualization_vector_values(
            displacement_values,
            solution,
            components=coordinates.shape[1],
        )
        grid.point_data["UMAG"] = np.linalg.norm(displacement_values, axis=1)
        cell_count = len(cell_types)
        for field in fields:
            values = np.asarray(field.x.array)
            if values.size % cell_count:
                raise ValueError(
                    f"Cell field {field.name!r} does not align with VTK cells."
                )
            shaped = values.reshape(cell_count, -1)
            grid.cell_data[field.name] = (
                shaped[:, 0] if shaped.shape[1] == 1 else shaped
            )
        frame_path = frame_directory / f"{pvd.stem}_{frame_index:04d}.vtu"
        grid.save(frame_path, binary=True)
        frame_paths.append(frame_path)

    vtk_file = ET.Element("VTKFile", type="Collection", version="0.1")
    collection = ET.SubElement(vtk_file, "Collection")
    for snapshot, frame_path in zip(selected, frame_paths):
        ET.SubElement(
            collection,
            "DataSet",
            timestep=f"{snapshot.load_factor:.16g}",
            group="",
            part="0",
            file=str(frame_path.relative_to(pvd.parent)),
        )
    tree = ET.ElementTree(vtk_file)
    ET.indent(tree, space="  ")
    tree.write(pvd, encoding="utf-8", xml_declaration=True)
    return pvd, tuple(frame_paths)


def _copy_named(function, name: str):
    output = fem.Function(function.function_space, name=name)
    output.x.array[:] = function.x.array
    output.x.scatter_forward()
    return output


def _xdmf_topology_type(cell_name: str, nodes_per_cell: int) -> str:
    normalized = str(cell_name).lower()
    topology_types = {
        ("interval", 2): "Polyline",
        ("interval", 3): "Edge_3",
        ("triangle", 3): "Triangle",
        ("triangle", 6): "Triangle_6",
        ("quadrilateral", 4): "Quadrilateral",
        ("quadrilateral", 8): "Quadrilateral_8",
        ("quadrilateral", 9): "Quadrilateral_9",
        ("tetrahedron", 4): "Tetrahedron",
        ("tetrahedron", 10): "Tetrahedron_10",
        ("hexahedron", 8): "Hexahedron",
        ("hexahedron", 20): "Hexahedron_20",
        ("hexahedron", 27): "Hexahedron_27",
        ("prism", 6): "Wedge",
        ("pyramid", 5): "Pyramid",
    }
    try:
        return topology_types[(normalized, int(nodes_per_cell))]
    except KeyError as exc:
        raise NotImplementedError(
            "No XDMF topology mapping for "
            f"{cell_name!r} with {nodes_per_cell} nodes per cell."
        ) from exc


def _data_item(
    parent,
    *,
    dimensions,
    value: str,
    number_type: str = "Float",
    precision: int = 8,
):
    attributes = {
        "Dimensions": " ".join(str(int(item)) for item in dimensions),
        "NumberType": number_type,
        "Format": "HDF",
    }
    if number_type != "Int":
        attributes["Precision"] = str(int(precision))
    item = ET.SubElement(parent, "DataItem", **attributes)
    item.text = value
    return item


def _attribute(grid, name: str, center: str, values, hdf_reference: str):
    array = np.asarray(values)
    component_count = 1 if array.ndim == 1 else int(array.shape[-1])
    if component_count == 1:
        attribute_type = "Scalar"
    elif component_count in {2, 3}:
        attribute_type = "Vector"
    elif component_count == 6:
        attribute_type = "Tensor6"
    elif component_count in {4, 9}:
        attribute_type = "Tensor"
    else:
        attribute_type = "Matrix"
    attribute = ET.SubElement(
        grid,
        "Attribute",
        Name=name,
        AttributeType=attribute_type,
        Center=center,
    )
    _data_item(
        attribute,
        dimensions=array.shape,
        value=hdf_reference,
    )
    return attribute
