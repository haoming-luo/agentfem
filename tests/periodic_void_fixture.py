"""Real-geometry periodic cube with a centred spherical void.

This fixture deliberately keeps geometry generation outside AgentFEM's core.
Gmsh creates a first-order tetrahedral mesh with exactly matching opposite
faces; the returned source-node equations exercise the same public
``AbaqusPeriodicConstraint`` used for imported engineering meshes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentfem import constraints, mesh
from agentfem.mesh import abaqus


@dataclass(frozen=True)
class PeriodicVoidFixture:
    """Gmsh void cell plus exact source-coordinate periodic semantics."""

    domain: object
    cell_tags: object
    facet_tags: object
    nodes: abaqus.AbaqusNodeTable
    equations: abaqus.AbaqusEquationSet
    deformation_gradient: np.ndarray
    anchor_node: int
    reference_nodes: tuple[int, int, int]
    side_length: float
    void_radius: float
    periodic_pairing_error: float

    @property
    def cell_reference_volume(self) -> float:
        return float(self.side_length**3)

    @property
    def exact_solid_volume(self) -> float:
        return float(
            self.side_length**3 - 4.0 * np.pi * self.void_radius**3 / 3.0
        )

    def constraint(self, displacement):
        return constraints.abaqus_periodic_cell(
            displacement,
            nodes=self.nodes,
            equations=self.equations,
            deformation_gradient=self.deformation_gradient,
            anchor_node=self.anchor_node,
            reference_nodes=self.reference_nodes,
            tolerance=2.0e-9 * self.side_length,
            name="gmsh_spherical_void_periodic_cell",
        )


def periodic_spherical_void_cell(
    comm,
    *,
    side_length: float = 1.0,
    void_radius: float = 0.2,
    mesh_size: float = 0.18,
    stretch: float = 1.01,
    model_rank: int = 0,
) -> PeriodicVoidFixture:
    """Build a periodic tetrahedral cube containing a true spherical void.

    The void is geometric: no low-stiffness surrogate phase is present.  The
    macroscopic loading is isochoric uniaxial tension by default.
    """

    side_length = float(side_length)
    void_radius = float(void_radius)
    mesh_size = float(mesh_size)
    stretch = float(stretch)
    if not np.isfinite(side_length) or side_length <= 0.0:
        raise ValueError("side_length must be finite and positive.")
    if not np.isfinite(void_radius) or not 0.0 < void_radius < 0.5 * side_length:
        raise ValueError("void_radius must lie strictly inside half the cell size.")
    if not np.isfinite(mesh_size) or mesh_size <= 0.0:
        raise ValueError("mesh_size must be finite and positive.")
    if not np.isfinite(stretch) or stretch <= 0.0:
        raise ValueError("stretch must be finite and positive.")

    gmsh = mesh.require_gmsh()
    initialized_here = not gmsh.isInitialized()
    if initialized_here:
        gmsh.initialize()
    semantics = None
    try:
        if comm.rank == model_rank:
            gmsh.clear()
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.model.add("agentfem_periodic_spherical_void")
            box = gmsh.model.occ.addBox(
                0.0,
                0.0,
                0.0,
                side_length,
                side_length,
                side_length,
            )
            sphere = gmsh.model.occ.addSphere(
                0.5 * side_length,
                0.5 * side_length,
                0.5 * side_length,
                void_radius,
            )
            volumes, _ = gmsh.model.occ.cut(
                [(3, box)],
                [(3, sphere)],
                removeObject=True,
                removeTool=True,
            )
            gmsh.model.occ.synchronize()
            if len(volumes) != 1:
                raise RuntimeError("The spherical-void Boolean cut is not unique.")

            outer_faces, void_faces = _classify_faces(gmsh, side_length)
            for axis in range(3):
                transform = np.eye(4)
                transform[axis, 3] = side_length
                gmsh.model.mesh.setPeriodic(
                    2,
                    [outer_faces[(axis, 1)]],
                    [outer_faces[(axis, 0)]],
                    transform.reshape(-1).tolist(),
                )

            volume_tag = int(volumes[0][1])
            gmsh.model.addPhysicalGroup(3, [volume_tag], 1)
            gmsh.model.setPhysicalName(3, 1, "matrix")
            gmsh.model.addPhysicalGroup(2, list(outer_faces.values()), 10)
            gmsh.model.setPhysicalName(2, 10, "periodic_boundary")
            gmsh.model.addPhysicalGroup(2, void_faces, 20)
            gmsh.model.setPhysicalName(2, 20, "void_surface")
            gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
            gmsh.model.mesh.generate(3)
            semantics = _periodic_semantics(
                gmsh,
                outer_faces,
                side_length=side_length,
                tolerance=1.0e-9 * side_length,
            )

        semantics = comm.bcast(semantics, root=model_rank)
        imported = mesh.import_gmsh_model(
            gmsh.model,
            comm,
            model_rank=model_rank,
            gdim=3,
        )
    finally:
        if initialized_here:
            gmsh.finalize()

    lateral = 1.0 / np.sqrt(stretch)
    return PeriodicVoidFixture(
        domain=imported.domain,
        cell_tags=imported.cell_tags,
        facet_tags=imported.facet_tags,
        nodes=abaqus.AbaqusNodeTable(
            labels=np.asarray(semantics["labels"], dtype=np.int64),
            coordinates=np.asarray(semantics["coordinates"], dtype=float),
        ),
        equations=abaqus.AbaqusEquationSet(
            tuple(
                abaqus.LinearEquation(
                    tuple(
                        abaqus.EquationTerm(
                            int(node),
                            int(component),
                            float(coefficient),
                        )
                        for node, component, coefficient in terms
                    )
                )
                for terms in semantics["equations"]
            )
        ),
        deformation_gradient=np.diag((stretch, lateral, lateral)),
        anchor_node=int(semantics["anchor_node"]),
        reference_nodes=tuple(int(value) for value in semantics["reference_nodes"]),
        side_length=side_length,
        void_radius=void_radius,
        periodic_pairing_error=float(semantics["periodic_pairing_error"]),
    )


def _classify_faces(gmsh, side_length: float):
    tolerance = 1.0e-8 * side_length
    outer = {}
    void = []
    for _dimension, tag in gmsh.model.getEntities(2):
        center = np.asarray(gmsh.model.occ.getCenterOfMass(2, tag), dtype=float)
        selected = None
        for axis in range(3):
            if abs(center[axis]) <= tolerance:
                selected = (axis, 0)
                break
            if abs(center[axis] - side_length) <= tolerance:
                selected = (axis, 1)
                break
        if selected is None:
            void.append(int(tag))
        elif selected in outer:
            raise RuntimeError(f"Multiple faces were classified as {selected}.")
        else:
            outer[selected] = int(tag)
    expected = {(axis, side) for axis in range(3) for side in (0, 1)}
    if set(outer) != expected or not void:
        raise RuntimeError("Could not identify six outer faces and the void surface.")
    return outer, void


def _periodic_semantics(
    gmsh,
    faces,
    *,
    side_length: float,
    tolerance: float,
    origin=(0.0, 0.0, 0.0),
    periods=None,
):
    """Return source-node equations for one rectangular periodic lattice.

    ``side_length`` remains the backward-compatible cubic default.  ``origin``
    and ``periods`` let external benchmark fixtures reuse the exact same node
    graph for shifted or thin extruded cells.
    """

    origin = np.asarray(origin, dtype=float)
    periods = (
        np.full(3, float(side_length), dtype=float)
        if periods is None
        else np.asarray(periods, dtype=float)
    )
    if origin.shape != (3,) or periods.shape != (3,):
        raise ValueError("origin and periods must contain three coordinates.")
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(periods)):
        raise ValueError("origin and periods must be finite.")
    if np.any(periods <= 0.0):
        raise ValueError("periods must be positive.")
    node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes()
    coordinates = np.asarray(node_coordinates, dtype=float).reshape(-1, 3)
    labels = np.asarray(node_tags, dtype=np.int64)
    coordinate_to_label = {}
    coordinate_by_label = {}
    for label, coordinate in zip(labels, coordinates):
        key = _coordinate_key(coordinate, tolerance)
        if key in coordinate_to_label:
            raise RuntimeError("Gmsh produced duplicate coordinates for a first-order mesh.")
        coordinate_to_label[key] = int(label)
        coordinate_by_label[int(label)] = coordinate

    def label_at(coordinate):
        key = _coordinate_key(coordinate, tolerance)
        try:
            return coordinate_to_label[key]
        except KeyError as exc:
            raise RuntimeError(
                f"The periodic mesh has no node at {np.asarray(coordinate).tolist()}."
            ) from exc

    anchor = label_at(origin)
    references = tuple(
        label_at(origin + np.eye(3)[axis] * periods[axis]) for axis in range(3)
    )
    control_nodes = {anchor, *references}
    boundary_labels = sorted(
        int(label)
        for label, coordinate in zip(labels, coordinates)
        if np.any(np.isclose(coordinate, origin, rtol=0.0, atol=tolerance))
        or np.any(
            np.isclose(coordinate, origin + periods, rtol=0.0, atol=tolerance)
        )
    )
    equations = []
    for slave in boundary_labels:
        coordinate = coordinate_by_label[slave]
        active_axes = tuple(
            axis
            for axis in range(3)
            if abs(coordinate[axis] - (origin[axis] + periods[axis])) <= tolerance
        )
        if not active_axes or slave in control_nodes:
            continue
        wrapped = coordinate.copy()
        wrapped[list(active_axes)] = origin[list(active_axes)]
        base = label_at(wrapped)
        for component in (1, 2, 3):
            terms = [(slave, component, 1.0), (base, component, -1.0)]
            terms.extend(
                (references[axis], component, -1.0) for axis in active_axes
            )
            terms.append((anchor, component, float(len(active_axes))))
            equations.append(terms)

    pairing_error = 0.0
    for axis in range(3):
        selected_faces = faces[(axis, 1)]
        if np.isscalar(selected_faces):
            selected_faces = (int(selected_faces),)
        found_pair = False
        for selected_face in selected_faces:
            _master, slaves, masters, _transform = gmsh.model.mesh.getPeriodicNodes(
                2,
                int(selected_face),
                True,
            )
            if len(slaves) == 0:
                continue
            found_pair = True
            translation = np.eye(3)[axis] * periods[axis]
            pairing_error = max(
                pairing_error,
                max(
                    float(
                        np.linalg.norm(
                            coordinate_by_label[int(slave)]
                            - coordinate_by_label[int(master)]
                            - translation
                        )
                    )
                    for slave, master in zip(slaves, masters)
                ),
            )
        if not found_pair:
            raise RuntimeError("Gmsh did not retain periodic surface-node pairs.")

    selected_coordinates = [coordinate_by_label[label].tolist() for label in boundary_labels]
    return {
        "labels": boundary_labels,
        "coordinates": selected_coordinates,
        "equations": equations,
        "anchor_node": anchor,
        "reference_nodes": references,
        "periodic_pairing_error": pairing_error,
    }


def _coordinate_key(coordinate, tolerance: float):
    return tuple(
        np.rint(np.asarray(coordinate, dtype=float) / tolerance).astype(np.int64)
    )
