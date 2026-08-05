"""Declarative field output and compact ParaView-ready time series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
        if self.backend in {"pvd", "both"}:
            if domain.comm.size > 1:
                raise NotImplementedError(
                    "PVD presentation output is serial. Use backend='xdmf' "
                    "for collective MPI output, then render after the solve."
                )
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
    every frame references one shared topology dataset. With
    ``deformation_scale=1`` ParaView displays the physical deformed shape
    directly. Reference coordinates remain in HDF5 for provenance.
    """

    selected = tuple(snapshots)
    # ``cell_fields`` is retained as the positional API name for compatibility;
    # frames may now contain both point and cell fields.
    fields_by_frame = tuple(cell_fields)
    if not selected or len(selected) != len(fields_by_frame):
        raise ValueError(
            "Unified XDMF requires equal non-empty snapshot and field frames."
        )
    solution0 = selected[0].solution
    domain = solution0.function_space.mesh
    if domain.comm.size != 1:
        raise NotImplementedError(
            "Unified compressed XDMF currently supports serial output. "
            "Use the DOLFINx writer or add collective parallel HDF5 support."
        )
    xdmf = Path(xdmf_path)
    xdmf.parent.mkdir(parents=True, exist_ok=True)
    h5_path = xdmf.with_suffix(".h5")
    topology, cell_types, reference_coordinates = plot.vtk_mesh(
        solution0.function_space
    )
    nodes_per_cell = int(topology[0])
    connectivity = np.asarray(topology).reshape(-1, nodes_per_cell + 1)[:, 1:]
    topology_type = _xdmf_topology_type(
        domain.topology.cell_type.name,
        nodes_per_cell,
    )
    cell_count = len(cell_types)
    point_count = reference_coordinates.shape[0]
    h5_options = {
        "compression": "gzip",
        "compression_opts": int(compression),
        "shuffle": True,
    }

    root = ET.Element("Xdmf", Version="3.0")
    xml_domain = ET.SubElement(root, "Domain")
    temporal = ET.SubElement(
        xml_domain,
        "Grid",
        Name="Results",
        GridType="Collection",
        CollectionType="Temporal",
    )
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["agentfem_schema"] = "agentfem.unified-xdmf"
        h5.attrs["schema_version"] = "0.1.0"
        h5.attrs["deformation_scale"] = float(deformation_scale)
        unique_cell_types = np.unique(cell_types)
        if unique_cell_types.size != 1:
            raise ValueError("Unified XDMF currently requires one VTK cell type.")
        h5.attrs["vtk_cell_type"] = int(unique_cell_types[0])
        h5.attrs["xdmf_topology_type"] = topology_type
        h5.attrs["nodes_per_cell"] = nodes_per_cell
        h5.attrs["point_count"] = point_count
        h5.attrs["cell_count"] = cell_count
        mesh_group = h5.create_group("Mesh")
        mesh_group.create_dataset(
            "Topology",
            data=connectivity,
            **h5_options,
        )
        if store_reference_geometry:
            mesh_group.create_dataset(
                "ReferenceGeometry",
                data=reference_coordinates,
                **h5_options,
            )
        for index, (snapshot, fields) in enumerate(
            zip(selected, fields_by_frame)
        ):
            frame_name = f"{index:04d}"
            frame_group = h5.create_group(f"Frames/{frame_name}")
            frame_group.attrs["load_factor"] = float(snapshot.load_factor)
            solution = snapshot.solution
            value_dimension = solution.ufl_shape[0]
            displacement_values = np.asarray(solution.x.array).reshape(
                -1,
                value_dimension,
            )
            if displacement_values.shape[0] != point_count:
                raise ValueError(
                    "Unified XDMF displacement dofs must match mesh points."
                )
            displacement = np.zeros_like(reference_coordinates)
            displacement[:, :value_dimension] = displacement_values
            geometry = (
                reference_coordinates
                + float(deformation_scale) * displacement
            )
            frame_group.create_dataset("Geometry", data=geometry, **h5_options)
            point_group = frame_group.create_group("Point")
            point_group.create_dataset(
                "U",
                data=displacement_values,
                **h5_options,
            )
            point_group.create_dataset(
                "UMAG",
                data=np.linalg.norm(displacement_values, axis=1),
                **h5_options,
            )
            cell_group = frame_group.create_group("Cell")
            shaped_point_fields = []
            shaped_cell_fields = []
            for field in fields:
                name = str(getattr(field, "name", "Field"))
                if name in {"U", "UMAG"}:
                    continue
                center, shaped = _unified_field_values(
                    field,
                    solution_space=solution.function_space,
                    point_count=point_count,
                    cell_count=cell_count,
                )
                group = point_group if center == "Node" else cell_group
                group.create_dataset(name, data=shaped, **h5_options)
                selected = shaped_point_fields if center == "Node" else shaped_cell_fields
                selected.append((name, shaped))

            grid = ET.SubElement(
                temporal,
                "Grid",
                Name=f"Frame_{frame_name}",
                GridType="Uniform",
            )
            ET.SubElement(grid, "Time", Value=f"{snapshot.load_factor:.16g}")
            topology_xml = ET.SubElement(
                grid,
                "Topology",
                TopologyType=topology_type,
                NumberOfElements=str(cell_count),
                NodesPerElement=str(nodes_per_cell),
            )
            _data_item(
                topology_xml,
                dimensions=connectivity.shape,
                number_type="Int",
                value=f"{h5_path.name}:/Mesh/Topology",
            )
            geometry_xml = ET.SubElement(
                grid,
                "Geometry",
                GeometryType="XYZ",
            )
            _data_item(
                geometry_xml,
                dimensions=geometry.shape,
                value=f"{h5_path.name}:/Frames/{frame_name}/Geometry",
            )
            _attribute(
                grid,
                "U",
                "Node",
                displacement_values,
                f"{h5_path.name}:/Frames/{frame_name}/Point/U",
            )
            magnitude = np.linalg.norm(displacement_values, axis=1)
            _attribute(
                grid,
                "UMAG",
                "Node",
                magnitude,
                f"{h5_path.name}:/Frames/{frame_name}/Point/UMAG",
            )
            for field_name, shaped in shaped_point_fields:
                _attribute(
                    grid,
                    field_name,
                    "Node",
                    shaped,
                    f"{h5_path.name}:/Frames/{frame_name}/Point/{field_name}",
                )
            for field_name, shaped in shaped_cell_fields:
                _attribute(
                    grid,
                    field_name,
                    "Cell",
                    shaped,
                    f"{h5_path.name}:/Frames/{frame_name}/Cell/{field_name}",
                )
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(xdmf, encoding="utf-8", xml_declaration=True)
    return xdmf


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
        grid.point_data["U"] = displacement_values
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
