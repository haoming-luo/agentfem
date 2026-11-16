# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Composite-specific field recovery and exact piecewise-linear XDMF output."""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from ..provenance import collective_call


class CompositeTetCellFields:
    def __init__(self, operator, variables):
        from dolfinx import fem

        self.operator = operator
        domain = operator.displacement.function_space.mesh

        def select():
            names = (variables,) if isinstance(variables, str) else tuple(variables)
            names = tuple(dict.fromkeys(str(n).upper() for n in names))
            if set(names) - {"U", "V", "A", "S", "E", "MISES", "SENER"}:
                raise ValueError("Composite fields: U, V, A, S, E, MISES, SENER only.")
            return names

        names = collective_call(
            select, comm=domain.comm, label="Composite output selection"
        )
        if any(other != names for other in domain.comm.allgather(names)):
            raise ValueError("Composite output selection differs between ranks.")
        self.fields = tuple(
            fem.Function(
                fem.functionspace(
                    domain, ("DG", 0, (3, 3)) if name in {"S", "E"} else ("DG", 0)
                ),
                name=name,
            )
            for name in names
            if name not in {"U", "V", "A"}
        )
        for f in self.fields:
            f._agentfem_processing = {
                "formulation": "composite_tet10",
                "representation": "cell_volume_average",
                "source": "projected_strain_and_declared_material",
                "interelement_smoothing": False,
                "shear_convention": "tensor",
                "stabilization_stress_included": False,
            }

    def update(self):
        op = self.operator
        op.displacement.x.scatter_forward()

        def local():
            u = op.displacement.x.array.reshape(-1, 3)[op.cell_nodes].reshape(-1, 30)
            for cell, (e, C, uc) in enumerate(zip(op.elements, op.materials, u)):
                uc = (uc.reshape(10, 3) - uc.reshape(10, 3).mean(axis=0)).ravel()
                eps = e.strain_matrices(mean_dilatation=op.policy.mean_dilatation) @ uc
                stress = eps @ C.T
                tensor = stress[:, [[0, 3, 5], [3, 1, 4], [5, 4, 2]]]
                strain_tensor = eps.copy()
                strain_tensor[:, 3:] *= 0.5
                strain_tensor = strain_tensor[:, [[0, 3, 5], [3, 1, 4], [5, 4, 2]]]
                deviator = (
                    tensor
                    - np.trace(tensor, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
                )
                values = {
                    "S": tensor.mean(0).ravel(),
                    "E": strain_tensor.mean(0).ravel(),
                    "MISES": np.sqrt(1.5 * np.sum(deviator**2, axis=(1, 2))).mean(),
                    "SENER": 0.5 * uc @ op.stiffness[cell] @ uc / e.volume,
                }
                for field in self.fields:
                    dof = field.function_space.dofmap.cell_dofs(cell)[0]
                    width = int(np.prod(field.ufl_shape)) if field.ufl_shape else 1
                    field.x.array.reshape(-1, width)[dof] = values[field.name]

        collective_call(local, comm=op.comm, label="Composite field recovery")
        for f in self.fields:
            f.x.scatter_forward()
        return self.fields


class CompositeTetTimeSeries:
    """Collective gather-to-root scientific output, no parallel HDF5 requirement.

    Native ten-node coefficients and connectivity are retained. The added centre
    is the exact tied interpolation value; twelve linear tetrahedra represent
    the displacement field exactly. DG0 outputs are labelled cell averages.
    This bounded writer gathers the dataset on rank zero, not a scalable HPC
    distributed writer. It never substitutes quadratic interpolation.
    """

    def __init__(self, path, primary):
        self.path = Path(path).with_suffix(".xdmf")
        self.h5_path = self.path.with_suffix(".h5")
        self.comm = primary.function_space.mesh.comm
        self.h5 = None
        self.frames = 0
        self.root = ET.Element("Xdmf", Version="3.0")
        domain = ET.SubElement(self.root, "Domain")
        ET.SubElement(
            domain,
            "Information",
            Name="solver_element",
            Value="composite_tet10_not_C3D10M",
        )
        ET.SubElement(
            domain,
            "Information",
            Name="interpolation",
            Value="exact_12_subtet_piecewise_linear",
        )
        self.collection = ET.SubElement(
            domain,
            "Grid",
            Name="Results",
            GridType="Collection",
            CollectionType="Temporal",
        )
        self.last_time = None
        self.signature = None
        self.failed = False

    def __enter__(self):
        def prepare():
            if self.comm.rank == 0:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.path.exists() or self.h5_path.exists():
                    raise FileExistsError(
                        "Composite output already exists; use a fresh path."
                    )
                if self.path.with_suffix(".partial.xdmf").exists():
                    raise FileExistsError(
                        "Composite partial output exists; use a fresh path."
                    )

        collective_call(prepare, comm=self.comm, label="Composite output directory")
        return self

    def __exit__(self, exc_type, exc, tb):
        def close():
            if self.comm.rank == 0:
                if self.h5 is not None:
                    self.h5.close()
                if exc_type is None and not self.failed:
                    if not self.frames:
                        raise ValueError("Composite output contains no frames.")
                    self._publish(self.path)
                    self.path.with_suffix(".partial.xdmf").unlink(missing_ok=True)

        collective_call(close, comm=self.comm, label="Composite output close")
        return False

    def _publish(self, path):
        from ..checkpointing import atomic_write_text

        ET.indent(self.root, space="  ")
        atomic_write_text(
            path, ET.tostring(self.root, encoding="unicode", xml_declaration=True)
        )

    def write_fields(self, time, *fields):
        from ..elements._composite_tet_dolfinx import local_order, is_composite_space

        V = fields[0].function_space
        domain = V.mesh

        def pack():
            if not np.isfinite(time) or (
                self.last_time is not None and time <= self.last_time
            ):
                raise ValueError("Composite output times must increase.")
            order = local_order(V)
            count = domain.topology.index_map(3).size_local
            imap = V.dofmap.index_map
            n = imap.size_local
            ids = imap.local_to_global(np.arange(n, dtype=np.int32))
            cells = np.array(
                [V.dofmap.cell_dofs(c)[order] for c in range(count)], dtype=np.int32
            ).reshape(-1, 10)
            cell_ids = imap.local_to_global(cells.ravel()).reshape(-1, 10)
            attributes = []
            for i, f in enumerate(fields):
                if f.function_space.mesh is not domain:
                    raise ValueError("Composite fields must share one mesh.")
                width = int(np.prod(f.ufl_shape)) if f.ufl_shape else 1
                if is_composite_space(f.function_space):
                    if not np.array_equal(f.function_space.dofmap.list, V.dofmap.list):
                        raise ValueError("Composite nodal layouts differ.")
                    values = f.x.array.reshape(-1, width)[:n].copy()
                    center = "Node"
                elif (
                    f.function_space.element.basix_element.discontinuous
                    and f.function_space.element.basix_element.degree == 0
                ):
                    ds = np.array(
                        [f.function_space.dofmap.cell_dofs(c)[0] for c in range(count)],
                        dtype=int,
                    )
                    values = f.x.array.reshape(-1, width)[ds].copy()
                    center = "Cell"
                else:
                    raise NotImplementedError(
                        "Composite output supports native nodal fields and DG0 only."
                    )
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite composite output.")
                name = {"Displacement": "U", "Velocity": "V", "Acceleration": "A"}.get(
                    f.name, f.name
                )
                if i == 0:
                    name = "U"
                attributes.append(
                    (name, center, values, getattr(f, "_agentfem_processing", {}))
                )
            if len({a[0] for a in attributes}) != len(attributes):
                raise ValueError("Output field names must be unique.")
            return (
                ids,
                V.tabulate_dof_coordinates()[:n].copy(),
                cell_ids,
                attributes,
                domain.topology.original_cell_index[:count].copy(),
            )

        data = collective_call(pack, comm=self.comm, label="Composite output values")
        gathered = self.comm.gather(data, root=0)
        try:
            collective_call(
                lambda: (
                    self._write_root(float(time), gathered)
                    if self.comm.rank == 0
                    else None
                ),
                comm=self.comm,
                label="Composite XDMF/HDF5 storage",
            )
        except Exception:
            self.failed = True
            raise
        self.last_time = float(time)

    def _write_root(self, time, parts):
        import h5py
        import json
        from ..elements.composite_tet import _SUBTETS

        ids = np.concatenate([p[0] for p in parts])
        permutation = np.argsort(ids)
        if not np.array_equal(ids[permutation], np.arange(len(ids))):
            raise ValueError("Composite output ownership must cover global DOFs once.")
        x = np.concatenate([p[1] for p in parts])[permutation]
        cells = np.concatenate([p[2] for p in parts])
        centers = x[cells[:, 4:]].mean(axis=1)
        display_x = np.vstack((x, centers))
        expanded = np.column_stack((cells, np.arange(len(cells)) + len(x)))
        topology = expanded[:, _SUBTETS].reshape(-1, 4)
        points = display_x[topology]
        negative = np.linalg.det((points[:, 1:] - points[:, :1]).transpose(0, 2, 1)) < 0
        topology[negative, 2:4] = topology[negative, 2:4][:, ::-1]
        signature = tuple((a[0], a[1], a[2].shape[1]) for a in parts[0][3])
        if any(
            tuple((a[0], a[1], a[2].shape[1]) for a in p[3]) != signature for p in parts
        ):
            raise ValueError("MPI output field selections differ.")
        if self.h5 is None:
            self.signature = signature
            self.geometry = display_x.copy()
            self.connectivity = cells.copy()
            self.h5 = h5py.File(self.h5_path, "w")
            self.h5.attrs["agentfem_schema"] = "agentfem.unified-xdmf"
            self.h5.attrs["schema_version"] = "0.1.0"
            self.h5.attrs["nodes_per_cell"] = 4
            self.h5.attrs["vtk_cell_type"] = 10
            self.h5.attrs["xdmf_topology_type"] = "Tetrahedron"
            self.h5.attrs["point_count"] = len(display_x)
            self.h5.attrs["cell_count"] = len(topology)
            self.h5.attrs["primary_field"] = "U"
            self.h5.attrs["geometry_dimension"] = 3
            self.h5.attrs["primary_storage_components"] = 3
            self.h5.attrs["warp_compatible"] = True
            self.h5.attrs["deformation_scale"] = 0.0
            self.h5.attrs["solver_element"] = "composite_tet10"
            self.h5.attrs["interpolation"] = "piecewise_linear_12_subtets_tied_centre"
            self.h5.attrs["assembly"] = "owned_cells_MPI_reverse_ghost"
            self.h5.attrs["writer"] = "collective_gather_to_root"
            g = self.h5.create_group("Mesh")
            for name, value in [
                ("Topology", topology),
                ("ReferenceGeometry", display_x),
                ("SourceCellDofs", cells),
                ("SourceCellIds", np.concatenate([p[4] for p in parts])),
                ("VisualizationParentCell", np.repeat(np.arange(len(cells)), 12)),
            ]:
                g.create_dataset(name, data=value)
        elif (
            signature != self.signature
            or not np.array_equal(cells, self.connectivity)
            or not np.array_equal(display_x, self.geometry)
        ):
            raise ValueError("Composite output geometry/layout changed between frames.")
        frame = f"Frames/{self.frames:04d}"
        group = self.h5.create_group(frame)
        group.attrs["time"] = time
        group.attrs["load_factor"] = time
        group["Geometry"] = self.h5["Mesh/ReferenceGeometry"]
        group.create_group("Node")
        group["Point"] = group["Node"]  # standard unified-reader compatibility
        group.create_group("Cell")
        grid = ET.SubElement(
            self.collection, "Grid", Name=f"Step{self.frames}", GridType="Uniform"
        )
        ET.SubElement(grid, "Time", Value=str(time))

        def item(parent, path, shape, integer=False):
            ET.SubElement(
                parent,
                "DataItem",
                Dimensions=" ".join(map(str, shape)),
                NumberType="Int" if integer else "Float",
                Precision="8",
                Format="HDF",
            ).text = f"{self.h5_path.name}:/{path}"

        top = ET.SubElement(
            grid,
            "Topology",
            TopologyType="Tetrahedron",
            NumberOfElements=str(len(topology)),
        )
        item(top, "Mesh/Topology", topology.shape, True)
        geo = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
        item(geo, "Mesh/ReferenceGeometry", display_x.shape)
        for index, (name, center, width) in enumerate(signature):
            values = np.concatenate([p[3][index][2] for p in parts])
            if center == "Node":
                values = values[permutation]
                group.create_dataset("Native/" + name, data=values, compression="gzip")
                values = np.vstack((values, values[cells[:, 4:]].mean(axis=1)))
            else:
                values = np.repeat(values, 12, axis=0)
            path = f"{frame}/{center}/{name}"
            dset = self.h5.create_dataset(path, data=values, compression="gzip")
            dset.attrs["processing"] = json.dumps(parts[0][3][index][3], sort_keys=True)
            attr = ET.SubElement(
                grid,
                "Attribute",
                Name=name,
                Center=center,
                AttributeType={1: "Scalar", 3: "Vector", 9: "Tensor"}.get(
                    width, "Matrix"
                ),
            )
            item(attr, path, values.shape)
        self.h5.flush()
        self.frames += 1
        self._publish(self.path.with_suffix(".partial.xdmf"))
