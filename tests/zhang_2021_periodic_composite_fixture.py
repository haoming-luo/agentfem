"""External finite-strain periodic-composite benchmark from Zhang et al. (2021).

The fixture contains no solver implementation.  It lowers the published unit
cell to AgentFEM's existing Gmsh import, named cell regions, affine-periodic
constraint, regional material, and accepted-result contracts.

Reference
---------
Zhang, Feng and Khandelwal, *A computational framework for homogenization and
multiscale stability analyses of nonlinear periodic materials*, International
Journal for Numerical Methods in Engineering (2021), Table 5.
https://doi.org/10.1002/nme.6802
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentfem import constitutive, constraints, mesh
from agentfem.mesh import abaqus

from periodic_void_fixture import _periodic_semantics


@dataclass(frozen=True)
class Zhang2021Table5Reference:
    """Published Table 5 response in its explicit column-major 2D ordering."""

    first_piola: np.ndarray
    effective_tangent: np.ndarray
    elastic_energy_density: float
    component_order: tuple[str, ...] = ("11", "21", "12", "22")

    def __post_init__(self) -> None:
        first_piola = np.asarray(self.first_piola, dtype=float)
        tangent = np.asarray(self.effective_tangent, dtype=float)
        if first_piola.shape != (4,) or tangent.shape != (4, 4):
            raise ValueError("Table 5 requires a four-vector and 4x4 tangent.")
        if not np.all(np.isfinite(first_piola)) or not np.all(np.isfinite(tangent)):
            raise ValueError("Table 5 tensors must be finite.")
        if not np.isfinite(self.elastic_energy_density):
            raise ValueError("Table 5 elastic energy must be finite.")
        object.__setattr__(self, "first_piola", first_piola)
        object.__setattr__(self, "effective_tangent", tangent)


TABLE5 = Zhang2021Table5Reference(
    first_piola=np.asarray((0.0128, 0.1893, 0.1953, 0.0598)),
    effective_tangent=np.asarray(
        (
            (26.1954, -0.6689, 0.3549, 8.3450),
            (-0.6689, 0.1601, 0.0503, -0.9698),
            (0.3549, 0.0503, 0.2038, 0.9365),
            (8.3450, -0.9698, 0.9365, 21.0161),
        )
    ),
    elastic_energy_density=2.423e-3,
)

# The external promotion oracle is deliberately fixed.  A caller may request
# a stricter comparison, but cannot make a failed benchmark pass by relaxing
# the contract used by the benchmark card and public documentation.
TABLE5_MAXIMUM_RELATIVE_TOLERANCE = 0.03


@dataclass(frozen=True)
class Zhang2021PeriodicCompositeFixture:
    """Thin-3D representation of the published plane-strain unit square."""

    domain: object
    cell_tags: object
    facet_tags: object
    nodes: abaqus.AbaqusNodeTable
    equations: abaqus.AbaqusEquationSet
    deformation_gradient: np.ndarray
    anchor_node: int
    reference_nodes: tuple[int, int, int]
    matrix_tag: int
    inclusion_tag: int
    thickness: float
    periodic_pairing_error: float

    @property
    def reference_cell_volume(self) -> float:
        return float(self.thickness)

    @property
    def matrix_moduli(self) -> tuple[float, float]:
        return 17.5, 8.0

    @property
    def matrix_young_poisson(self) -> tuple[float, float]:
        return young_poisson_from_bulk_shear(*self.matrix_moduli)

    def regions(self):
        return (
            mesh.cell_region(
                self.domain,
                self.cell_tags,
                tag=self.matrix_tag,
                name="matrix",
            ),
            mesh.cell_region(
                self.domain,
                self.cell_tags,
                tag=self.inclusion_tag,
                name="stiff_inclusions",
            ),
        )

    def materials(self):
        """Return the published matrix and a 100x-stiff elastic surrogate.

        The public stateful route requires one constitutive family per Step.
        A very large inclusion yield stress keeps the inclusion response elastic
        while preserving one finite-strain J2 state schema and transaction.
        """

        young, poisson = self.matrix_young_poisson
        matrix = constitutive.finite_strain_j2_logarithmic(
            young=young,
            poisson=poisson,
            yield_stress=0.45,
            hardening_modulus=0.1,
        )
        inclusion = constitutive.finite_strain_j2_logarithmic(
            young=100.0 * young,
            poisson=poisson,
            yield_stress=1.0e6,
            # The published inclusion is kept elastic by its very large yield
            # stress.  Its hardening modulus is consequently inactive; retain
            # the matrix value so the two regional materials differ only in
            # the parameters explicitly changed by the benchmark.
            hardening_modulus=0.1,
        )
        return matrix, inclusion

    def constraint(self, displacement):
        return constraints.abaqus_periodic_cell(
            displacement,
            nodes=self.nodes,
            equations=self.equations,
            deformation_gradient=self.deformation_gradient,
            anchor_node=self.anchor_node,
            reference_nodes=self.reference_nodes,
            tolerance=2.0e-9,
            name="zhang_2021_table5_periodic_cell",
        )


def young_poisson_from_bulk_shear(bulk: float, shear: float) -> tuple[float, float]:
    """Convert three-dimensional bulk/shear moduli to ``E, nu``."""

    bulk = float(bulk)
    shear = float(shear)
    if not np.isfinite(bulk) or not np.isfinite(shear) or min(bulk, shear) <= 0.0:
        raise ValueError("bulk and shear moduli must be finite and positive.")
    young = 9.0 * bulk * shear / (3.0 * bulk + shear)
    poisson = (3.0 * bulk - 2.0 * shear) / (2.0 * (3.0 * bulk + shear))
    return float(young), float(poisson)


def column_major_plane_components(tensor) -> np.ndarray:
    """Return ``[11, 21, 12, 22]`` exactly as reported in Table 5."""

    selected = np.asarray(tensor, dtype=float)
    if selected.shape not in {(2, 2), (3, 3)}:
        raise ValueError("tensor must be 2x2 or 3x3.")
    return selected[:2, :2].reshape(-1, order="F")


def assess_table5(
    *,
    first_piola,
    elastic_energy_density: float | None = None,
    effective_tangent=None,
    convergence_evidence: dict[str, bool] | None = None,
    relative_tolerance: float = TABLE5_MAXIMUM_RELATIVE_TOLERANCE,
) -> dict[str, object]:
    """Assess, but never over-promote, one numerical Table 5 candidate.

    A single stress comparison is diagnostic, not external verification.  The
    result becomes ``accepted`` only when stress, elastic energy, effective
    tangent, discretization/cell-size convergence, serial/MPI equivalence, and
    checkpoint/restart equivalence are all present and satisfy the declared
    contract.  The published 3 percent gate can be tightened, not relaxed.
    """

    tolerance = float(relative_tolerance)
    if (
        not np.isfinite(tolerance)
        or tolerance <= 0.0
        or tolerance > TABLE5_MAXIMUM_RELATIVE_TOLERANCE
    ):
        raise ValueError(
            "relative_tolerance must be finite, positive, and no greater "
            f"than the fixed {TABLE5_MAXIMUM_RELATIVE_TOLERANCE:.0%} "
            "external-promotion contract."
        )
    stress = column_major_plane_components(first_piola)
    stress_error = float(
        np.linalg.norm(stress - TABLE5.first_piola)
        / np.linalg.norm(TABLE5.first_piola)
    )
    energy_error = None
    if elastic_energy_density is not None:
        selected_energy = float(elastic_energy_density)
        if not np.isfinite(selected_energy):
            raise ValueError("elastic_energy_density must be finite.")
        energy_error = abs(
            selected_energy - TABLE5.elastic_energy_density
        ) / abs(TABLE5.elastic_energy_density)
    tangent_error = None
    if effective_tangent is not None:
        selected_tangent = np.asarray(effective_tangent, dtype=float)
        if selected_tangent.shape != (4, 4) or not np.all(np.isfinite(selected_tangent)):
            raise ValueError("effective_tangent must be one finite 4x4 matrix.")
        tangent_error = float(
            np.linalg.norm(selected_tangent - TABLE5.effective_tangent)
            / np.linalg.norm(TABLE5.effective_tangent)
        )
    convergence = {} if convergence_evidence is None else dict(convergence_evidence)
    invalid_boolean_evidence = tuple(
        name for name, value in convergence.items() if type(value) is not bool
    )
    if invalid_boolean_evidence:
        raise TypeError(
            "convergence_evidence values must be bool; invalid keys: "
            + ", ".join(sorted(invalid_boolean_evidence))
        )
    required_convergence = (
        "mesh_converged",
        "plane_strain_formulation_converged",
        "periodic_cell_size_invariant",
        "serial_mpi_equivalent",
        "restart_equivalent",
    )
    missing = []
    if energy_error is None:
        missing.append("published_elastic_energy")
    if tangent_error is None:
        missing.append("effective_tangent")
    missing.extend(name for name in required_convergence if not convergence.get(name, False))
    errors = tuple(
        value for value in (stress_error, energy_error, tangent_error) if value is not None
    )
    accepted = not missing and all(value <= tolerance for value in errors)
    failed = any(value > tolerance for value in errors)
    return {
        "schema": "agentfem.external-benchmark-assessment.v1",
        "benchmark": "Zhang-Feng-Khandelwal-2021-Table-5",
        "status": "accepted" if accepted else ("failed" if failed else "incomplete"),
        "accepted": accepted,
        "relative_tolerance": tolerance,
        "component_order": TABLE5.component_order,
        "first_piola": stress.tolist(),
        "first_piola_relative_l2_error": stress_error,
        "elastic_energy_relative_error": energy_error,
        "effective_tangent_relative_frobenius_error": tangent_error,
        "missing_evidence": tuple(missing),
        "convergence": convergence,
    }


def zhang_2021_periodic_composite(
    comm,
    *,
    mesh_size: float = 0.12,
    thickness: float = 0.10,
    shear: float = 0.10,
    model_rank: int = 0,
) -> Zhang2021PeriodicCompositeFixture:
    """Build the two-inclusion/one-void Table 5 cell as a thin 3D extrusion.

    The published problem is two-dimensional plane strain.  AgentFEM's current
    public finite-strain J2 provider is three-dimensional, so the benchmark is
    represented by an extruded periodic layer with ``F33 = 1``.  This mapping
    remains experimental until thickness, mesh, and element-formulation
    convergence demonstrate equivalence to the published mixed 2D element.
    """

    mesh_size = float(mesh_size)
    thickness = float(thickness)
    shear = float(shear)
    if not np.isfinite(mesh_size) or mesh_size <= 0.0:
        raise ValueError("mesh_size must be finite and positive.")
    if not np.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("thickness must be finite and positive.")
    if not np.isfinite(shear):
        raise ValueError("shear must be finite.")

    gmsh = mesh.require_gmsh()
    initialized_here = not gmsh.isInitialized()
    if initialized_here:
        gmsh.initialize()
    semantics = None
    try:
        if comm.rank == model_rank:
            gmsh.clear()
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.model.add("zhang_2021_periodic_composite")
            box = gmsh.model.occ.addBox(-0.5, -0.5, 0.0, 1.0, 1.0, thickness)
            cylinders = tuple(
                gmsh.model.occ.addCylinder(x, y, 0.0, 0.0, 0.0, thickness, 0.15)
                for x, y in ((-0.2, 0.2), (-0.2, -0.2), (0.2, 0.0))
            )
            _volumes, entity_maps = gmsh.model.occ.fragment(
                [(3, box)],
                [(3, tag) for tag in cylinders],
                removeObject=True,
                removeTool=True,
            )
            gmsh.model.occ.synchronize()
            inclusion_volumes = tuple(int(entity_maps[index][0][1]) for index in (1, 2))
            void_volume = int(entity_maps[3][0][1])
            # Recursive removal deletes the two orphan cap surfaces while
            # retaining the cylindrical wall shared with the matrix volume.
            # Leaving those caps in the Gmsh model creates source nodes that
            # have no DOLFINx volume-mesh dof and must therefore be rejected.
            gmsh.model.occ.remove([(3, void_volume)], recursive=True)
            gmsh.model.occ.synchronize()
            existing_volumes = {int(tag) for _dim, tag in gmsh.model.getEntities(3)}
            matrix_volumes = tuple(sorted(existing_volumes - set(inclusion_volumes)))
            if len(matrix_volumes) != 1 or not set(inclusion_volumes) <= existing_volumes:
                raise RuntimeError("Could not identify matrix and inclusion volumes.")

            periodic_faces, void_faces = _classify_periodic_faces(gmsh, thickness)
            periods = np.asarray((1.0, 1.0, thickness))
            for axis in range(3):
                transform = np.eye(4)
                transform[axis, 3] = periods[axis]
                gmsh.model.mesh.setPeriodic(
                    2,
                    list(periodic_faces[(axis, 1)]),
                    list(periodic_faces[(axis, 0)]),
                    transform.reshape(-1).tolist(),
                )

            gmsh.model.addPhysicalGroup(3, list(matrix_volumes), 1)
            gmsh.model.setPhysicalName(3, 1, "matrix")
            gmsh.model.addPhysicalGroup(3, list(inclusion_volumes), 2)
            gmsh.model.setPhysicalName(3, 2, "stiff_inclusions")
            outer_faces = sorted(
                {tag for tags in periodic_faces.values() for tag in tags}
            )
            gmsh.model.addPhysicalGroup(2, outer_faces, 10)
            gmsh.model.setPhysicalName(2, 10, "periodic_boundary")
            gmsh.model.addPhysicalGroup(2, list(void_faces), 20)
            gmsh.model.setPhysicalName(2, 20, "void_surface")
            gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
            gmsh.model.mesh.generate(3)
            semantics = _periodic_semantics(
                gmsh,
                periodic_faces,
                side_length=1.0,
                tolerance=1.0e-9,
                origin=(-0.5, -0.5, 0.0),
                periods=periods,
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

    deformation_gradient = np.eye(3)
    deformation_gradient[0, 1] = shear
    return Zhang2021PeriodicCompositeFixture(
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
        deformation_gradient=deformation_gradient,
        anchor_node=int(semantics["anchor_node"]),
        reference_nodes=tuple(int(value) for value in semantics["reference_nodes"]),
        matrix_tag=1,
        inclusion_tag=2,
        thickness=thickness,
        periodic_pairing_error=float(semantics["periodic_pairing_error"]),
    )


def _classify_periodic_faces(gmsh, thickness: float):
    bounds = ((-0.5, 0.5), (-0.5, 0.5), (0.0, thickness))
    tolerance = 2.0e-6 * max(1.0, thickness)
    periodic = {(axis, side): [] for axis in range(3) for side in (0, 1)}
    void_faces = []
    for _dimension, tag in gmsh.model.getEntities(2):
        upward = gmsh.model.getAdjacencies(2, tag)[0]
        if len(upward) == 0:
            continue
        box = np.asarray(gmsh.model.getBoundingBox(2, tag), dtype=float)
        selected = None
        for axis, (lower, upper) in enumerate(bounds):
            if abs(box[axis] - lower) <= tolerance and abs(box[axis + 3] - lower) <= tolerance:
                selected = (axis, 0)
                break
            if abs(box[axis] - upper) <= tolerance and abs(box[axis + 3] - upper) <= tolerance:
                selected = (axis, 1)
                break
        if selected is None:
            if len(upward) == 1:
                void_faces.append(int(tag))
        else:
            periodic[selected].append(int(tag))

    for axis in range(3):
        key = lambda tag: tuple(
            np.round(gmsh.model.occ.getCenterOfMass(2, tag), 12)
        )
        periodic[(axis, 0)].sort(key=key)
        periodic[(axis, 1)].sort(key=key)
        if not periodic[(axis, 0)] or len(periodic[(axis, 0)]) != len(periodic[(axis, 1)]):
            raise RuntimeError(f"Periodic face topology differs on axis {axis}.")
    if not void_faces:
        raise RuntimeError("Could not identify the circular void boundary.")
    return {key: tuple(value) for key, value in periodic.items()}, tuple(void_faces)


__all__ = [
    "TABLE5",
    "Zhang2021PeriodicCompositeFixture",
    "Zhang2021Table5Reference",
    "assess_table5",
    "column_major_plane_components",
    "young_poisson_from_bulk_shear",
    "zhang_2021_periodic_composite",
]
