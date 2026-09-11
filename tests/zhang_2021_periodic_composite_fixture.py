"""External finite-strain periodic-composite benchmark from Zhang et al. (2021).

The fixture contains no solver implementation.  It lowers the published unit
cell to AgentFEM's existing Gmsh import, named cell regions, affine-periodic
constraint, regional material, and accepted-result contracts.

Reference
---------
G. Zhang, N. Feng and K. Khandelwal, *A computational framework for
homogenization and multiscale stability analyses of nonlinear periodic
materials*, International Journal for Numerical Methods in Engineering (2021),
Table 5.
https://doi.org/10.1002/nme.6802
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentfem import constitutive, constraints, fields, mesh
from agentfem.mesh import abaqus

from periodic_void_fixture import _periodic_semantics


@dataclass(frozen=True)
class Zhang2021Table5Reference:
    """Published Table 5 response in its explicit column-major 2D ordering."""

    first_piola: np.ndarray
    effective_tangent: np.ndarray
    elastic_energy_density: float
    component_order: tuple[str, ...] = ("11", "21", "12", "22")
    published_q9_element_count: int = 2823

    def __post_init__(self) -> None:
        first_piola = np.asarray(self.first_piola, dtype=float)
        tangent = np.asarray(self.effective_tangent, dtype=float)
        if first_piola.shape != (4,) or tangent.shape != (4, 4):
            raise ValueError("Table 5 requires a four-vector and 4x4 tangent.")
        if not np.all(np.isfinite(first_piola)) or not np.all(np.isfinite(tangent)):
            raise ValueError("Table 5 tensors must be finite.")
        if not np.isfinite(self.elastic_energy_density):
            raise ValueError("Table 5 elastic energy must be finite.")
        if self.published_q9_element_count <= 0:
            raise ValueError("The published Q9 element count must be positive.")
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

# These are AgentFEM-owned comparison thresholds, not tolerances published by
# Zhang et al.  They are deliberately fixed: a caller may request a stricter
# comparison, but cannot make an out-of-tolerance candidate pass by relaxing
# the contract used by the benchmark card and public documentation.  Passing
# this numerical comparison is not, by itself, content-bound benchmark
# evidence or authority to promote a capability.
TABLE5_MAXIMUM_RELATIVE_TOLERANCE = 0.03
TABLE5_COMPONENT_ABSOLUTE_TOLERANCE = 6.0e-4


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
    element_order: int
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


@dataclass(frozen=True)
class Zhang2021PlaneStrainCompositeFixture:
    """Exact 2D geometry and Q2/DPC1 discretization preparation for Table 5.

    This fixture establishes only the published plane-strain geometry and its
    periodic finite-element topology.  It deliberately carries no claim that
    the nonlinear solve or the Table 5 response has been reproduced.
    """

    domain: object
    cell_tags: object
    facet_tags: object
    nodes: abaqus.AbaqusNodeTable
    equations: abaqus.AbaqusEquationSet
    deformation_gradient: np.ndarray
    anchor_node: int
    reference_nodes: tuple[int, int]
    matrix_tag: int
    inclusion_tag: int
    periodic_boundary_tag: int
    void_boundary_tag: int
    mesh_size: float
    element_order: int
    gmsh_element_name: str
    element_count: int
    nodes_per_element: int
    periodic_pair_counts: tuple[int, int]
    periodic_expected_pair_counts: tuple[int, int]
    periodic_pairing_error: float
    minimum_scaled_jacobian: float
    inclusion_surface_count: int
    void_curve_count: int

    @property
    def reference_cell_area(self) -> float:
        return 1.0

    @property
    def pressure_modes_per_cell(self) -> int:
        return 3

    @property
    def region_tags(self) -> dict[str, int]:
        return {
            "matrix": self.matrix_tag,
            "stiff_inclusions": self.inclusion_tag,
        }

    @property
    def boundary_tags(self) -> dict[str, int]:
        return {
            "periodic_boundary": self.periodic_boundary_tag,
            "void_boundary": self.void_boundary_tag,
        }

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
        """Return the same regional material definitions as the 3D diagnostic."""

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
            hardening_modulus=0.1,
        )
        return matrix, inclusion

    def mixed_field(self):
        """Create the published nine-displacement/three-pressure interpolation."""

        return fields.displacement_pressure(
            self.domain,
            displacement_degree=2,
            pressure_family="DPC",
            pressure_degree=1,
        )

    def constraint(self, displacement_pressure):
        """Create exact two-dimensional affine-periodic equations."""

        return constraints.abaqus_periodic_cell(
            displacement_pressure,
            nodes=self.nodes,
            equations=self.equations,
            deformation_gradient=self.deformation_gradient,
            anchor_node=self.anchor_node,
            reference_nodes=self.reference_nodes,
            tolerance=2.0e-9,
            name="zhang_2021_table5_plane_strain_periodic_cell",
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
    elastic_energy_semantics: str | None = None,
    effective_tangent=None,
    convergence_evidence: dict[str, bool] | None = None,
    relative_tolerance: float = TABLE5_MAXIMUM_RELATIVE_TOLERANCE,
) -> dict[str, object]:
    """Compare caller-supplied Table 5 data and completeness assertions.

    The arrays and every Boolean in ``convergence_evidence`` are supplied by
    the caller.  This function checks their shapes, finiteness, numerical
    agreement, and asserted completeness; it does not bind those assertions to
    archived artifacts, provenance, or an independently executed result.
    Consequently ``accepted`` means only that the supplied comparison passes.
    It is not authority to promote the benchmark or solver capability.  The
    AgentFEM-owned three-percent comparison threshold can be tightened, not
    relaxed.
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
            "comparison contract."
        )
    stress = column_major_plane_components(first_piola)
    stress_error = float(
        np.linalg.norm(stress - TABLE5.first_piola)
        / np.linalg.norm(TABLE5.first_piola)
    )
    stress_component_absolute_error = np.abs(stress - TABLE5.first_piola)
    stress_component_allowance = (
        TABLE5_COMPONENT_ABSOLUTE_TOLERANCE
        + tolerance * np.abs(TABLE5.first_piola)
    )
    stress_component_error_ratio = (
        stress_component_absolute_error / stress_component_allowance
    )
    stress_componentwise_passed = bool(
        np.all(stress_component_absolute_error <= stress_component_allowance)
    )
    energy_error = None
    if elastic_energy_density is None and elastic_energy_semantics is not None:
        raise ValueError(
            "elastic_energy_semantics requires elastic_energy_density."
        )
    if elastic_energy_density is not None:
        if elastic_energy_semantics != "primal_hencky_elastic_energy":
            raise ValueError(
                "Table 5 energy comparison requires the primal Hencky elastic "
                "energy. A condensed mixed or saddle-potential channel is not "
                "the published observable."
            )
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
        "load_increment_path_converged",
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
    accepted = (
        not missing
        and stress_componentwise_passed
        and all(value <= tolerance for value in errors)
    )
    failed = (
        not stress_componentwise_passed
        or any(value > tolerance for value in errors)
    )
    return {
        "schema": "agentfem.external-benchmark-assessment.v1",
        "benchmark": "Zhang-Feng-Khandelwal-2021-Table-5",
        "status": "accepted" if accepted else ("failed" if failed else "incomplete"),
        "accepted": accepted,
        "comparison_accepted": accepted,
        "evidence_authority": "caller_supplied_assertions_not_content_bound",
        "content_bound": False,
        "benchmark_promotion_authorized": False,
        "decision_scope": "numeric_comparison_and_caller_asserted_completeness",
        "relative_tolerance": tolerance,
        "component_absolute_tolerance": TABLE5_COMPONENT_ABSOLUTE_TOLERANCE,
        "tolerance_authority": "AgentFEM comparison contract; not published",
        "component_order": TABLE5.component_order,
        "first_piola": stress.tolist(),
        "first_piola_relative_l2_error": stress_error,
        "first_piola_component_absolute_error": (
            stress_component_absolute_error.tolist()
        ),
        "first_piola_component_allowance": stress_component_allowance.tolist(),
        "first_piola_component_error_ratio": stress_component_error_ratio.tolist(),
        "first_piola_componentwise_passed": stress_componentwise_passed,
        "elastic_energy_relative_error": energy_error,
        "elastic_energy_semantics": elastic_energy_semantics,
        "effective_tangent_relative_frobenius_error": tangent_error,
        "missing_evidence": tuple(missing),
        "convergence": convergence,
    }


def zhang_2021_plane_strain_composite(
    comm,
    *,
    mesh_size: float = 0.12,
    shear: float = 0.10,
    element_order: int = 2,
    model_rank: int = 0,
) -> Zhang2021PlaneStrainCompositeFixture:
    """Build the published 2D unit cell as a pure curved Q9 mesh.

    Gmsh first creates a periodic linear surface mesh using Frontal-Delaunay
    and the all-quadrilateral subdivision algorithm.  ``setOrder(2)`` then
    upgrades the topology while retaining the periodic high-order node graph.
    Any non-Q9 cell, invalid element, or missing periodic pairing is rejected
    before the model reaches a solver.
    """

    mesh_size = float(mesh_size)
    shear = float(shear)
    if not np.isfinite(mesh_size) or mesh_size <= 0.0:
        raise ValueError("mesh_size must be finite and positive.")
    if not np.isfinite(shear):
        raise ValueError("shear must be finite.")
    if isinstance(element_order, bool) or int(element_order) != element_order:
        raise ValueError("The exact plane-strain fixture requires element_order=2.")
    element_order = int(element_order)
    if element_order != 2:
        raise ValueError("The exact plane-strain fixture requires element_order=2.")
    if not 0 <= int(model_rank) < int(comm.size):
        raise ValueError("model_rank must identify one rank in the communicator.")

    gmsh = mesh.require_gmsh()
    initialized_here = not gmsh.isInitialized()
    if initialized_here:
        gmsh.initialize()
    semantics = None
    diagnostics = None
    previous_options = {}
    selected_options = {
        "General.Verbosity": 0.0,
        "Mesh.MeshSizeMin": mesh_size,
        "Mesh.MeshSizeMax": mesh_size,
        "Mesh.Algorithm": 6.0,
        "Mesh.RecombineAll": 0.0,
        "Mesh.SubdivisionAlgorithm": 1.0,
        "Mesh.SecondOrderIncomplete": 0.0,
        "Mesh.SecondOrderLinear": 0.0,
    }
    try:
        if comm.rank == model_rank:
            gmsh.clear()
            previous_options = {
                name: float(gmsh.option.getNumber(name))
                for name in selected_options
            }
            for name, value in selected_options.items():
                gmsh.option.setNumber(name, float(value))
            gmsh.model.add("zhang_2021_plane_strain_composite")

            square = gmsh.model.occ.addRectangle(-0.5, -0.5, 0.0, 1.0, 1.0)
            disks = tuple(
                gmsh.model.occ.addDisk(x, y, 0.0, 0.15, 0.15)
                for x, y in ((-0.2, 0.2), (-0.2, -0.2), (0.2, 0.0))
            )
            _surfaces, entity_maps = gmsh.model.occ.fragment(
                [(2, square)],
                [(2, tag) for tag in disks],
                removeObject=True,
                removeTool=True,
            )
            gmsh.model.occ.synchronize()
            mapped_disks = []
            for index in (1, 2, 3):
                mapped = tuple(
                    int(tag) for dim, tag in entity_maps[index] if int(dim) == 2
                )
                if len(mapped) != 1:
                    raise RuntimeError(
                        "The Zhang 2021 disks must remain three disjoint surfaces."
                    )
                mapped_disks.append(mapped[0])
            inclusion_surfaces = tuple(mapped_disks[:2])
            void_surface = int(mapped_disks[2])
            gmsh.model.occ.remove([(2, void_surface)], recursive=True)
            gmsh.model.occ.synchronize()

            existing_surfaces = {
                int(tag) for _dim, tag in gmsh.model.getEntities(2)
            }
            matrix_surfaces = tuple(
                sorted(existing_surfaces - set(inclusion_surfaces))
            )
            if (
                len(matrix_surfaces) != 1
                or not set(inclusion_surfaces) <= existing_surfaces
            ):
                raise RuntimeError(
                    "Could not identify one matrix and two inclusion surfaces."
                )

            periodic_curves, void_curves = _classify_periodic_curves_2d(gmsh)
            for axis in range(2):
                transform = np.eye(4)
                transform[axis, 3] = 1.0
                gmsh.model.mesh.setPeriodic(
                    1,
                    list(periodic_curves[(axis, 1)]),
                    list(periodic_curves[(axis, 0)]),
                    transform.reshape(-1).tolist(),
                )

            gmsh.model.addPhysicalGroup(2, list(matrix_surfaces), 1)
            gmsh.model.setPhysicalName(2, 1, "matrix")
            gmsh.model.addPhysicalGroup(2, list(inclusion_surfaces), 2)
            gmsh.model.setPhysicalName(2, 2, "stiff_inclusions")
            outer_curves = sorted(
                {tag for tags in periodic_curves.values() for tag in tags}
            )
            gmsh.model.addPhysicalGroup(1, outer_curves, 10)
            gmsh.model.setPhysicalName(1, 10, "periodic_boundary")
            gmsh.model.addPhysicalGroup(1, list(void_curves), 20)
            gmsh.model.setPhysicalName(1, 20, "void_boundary")

            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.setOrder(2)
            diagnostics = _quadrilateral9_mesh_diagnostics(gmsh)
            semantics = _periodic_semantics_2d(
                gmsh,
                periodic_curves,
                tolerance=1.0e-9,
            )
            semantics["inclusion_surface_count"] = len(inclusion_surfaces)
            semantics["void_curve_count"] = len(void_curves)

        semantics = comm.bcast(semantics, root=model_rank)
        diagnostics = comm.bcast(diagnostics, root=model_rank)
        imported = mesh.import_gmsh_model(
            gmsh.model,
            comm,
            model_rank=model_rank,
            gdim=2,
        )
    finally:
        if comm.rank == model_rank and gmsh.isInitialized():
            for name, value in previous_options.items():
                gmsh.option.setNumber(name, value)
        if initialized_here:
            gmsh.finalize()

    deformation_gradient = np.eye(2)
    deformation_gradient[0, 1] = shear
    return Zhang2021PlaneStrainCompositeFixture(
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
        reference_nodes=tuple(
            int(value) for value in semantics["reference_nodes"]
        ),
        matrix_tag=1,
        inclusion_tag=2,
        periodic_boundary_tag=10,
        void_boundary_tag=20,
        mesh_size=mesh_size,
        element_order=2,
        gmsh_element_name=str(diagnostics["element_name"]),
        element_count=int(diagnostics["element_count"]),
        nodes_per_element=int(diagnostics["nodes_per_element"]),
        periodic_pair_counts=tuple(
            int(value) for value in semantics["periodic_pair_counts"]
        ),
        periodic_expected_pair_counts=tuple(
            int(value) for value in semantics["periodic_expected_pair_counts"]
        ),
        periodic_pairing_error=float(semantics["periodic_pairing_error"]),
        minimum_scaled_jacobian=float(
            diagnostics["minimum_scaled_jacobian"]
        ),
        inclusion_surface_count=int(semantics["inclusion_surface_count"]),
        void_curve_count=int(semantics["void_curve_count"]),
    )


def zhang_2021_periodic_composite(
    comm,
    *,
    mesh_size: float = 0.12,
    thickness: float = 0.10,
    shear: float = 0.10,
    element_order: int = 1,
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
    if isinstance(element_order, bool) or int(element_order) != element_order:
        raise ValueError("element_order must be either 1 or 2.")
    element_order = int(element_order)
    if element_order not in {1, 2}:
        raise ValueError("element_order must be either 1 or 2.")
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
            if element_order == 2:
                # Preserve the periodic source-node graph at the same order as
                # the P2 displacement space.  Constraining only corner nodes
                # would leave boundary edge dofs non-periodic and invalidate
                # the mixed benchmark before the constitutive solve begins.
                gmsh.model.mesh.setOrder(2)
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
        element_order=element_order,
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


def _classify_periodic_curves_2d(gmsh):
    bounds = ((-0.5, 0.5), (-0.5, 0.5))
    # OpenCASCADE bounding boxes carry its geometric tolerance beyond the
    # analytical coordinate even though the curve itself is exact.
    tolerance = 2.0e-6
    periodic = {(axis, side): [] for axis in range(2) for side in (0, 1)}
    void_curves = []
    for _dimension, tag in gmsh.model.getEntities(1):
        upward = gmsh.model.getAdjacencies(1, tag)[0]
        if len(upward) == 0:
            continue
        box = np.asarray(gmsh.model.getBoundingBox(1, tag), dtype=float)
        selected = None
        for axis, (lower, upper) in enumerate(bounds):
            if (
                abs(box[axis] - lower) <= tolerance
                and abs(box[axis + 3] - lower) <= tolerance
            ):
                selected = (axis, 0)
                break
            if (
                abs(box[axis] - upper) <= tolerance
                and abs(box[axis + 3] - upper) <= tolerance
            ):
                selected = (axis, 1)
                break
        if selected is None:
            if len(upward) == 1:
                void_curves.append(int(tag))
        else:
            periodic[selected].append(int(tag))

    for axis in range(2):
        key = lambda tag: tuple(
            np.round(gmsh.model.occ.getCenterOfMass(1, tag), 12)
        )
        periodic[(axis, 0)].sort(key=key)
        periodic[(axis, 1)].sort(key=key)
        if (
            not periodic[(axis, 0)]
            or len(periodic[(axis, 0)]) != len(periodic[(axis, 1)])
        ):
            raise RuntimeError(f"Periodic curve topology differs on axis {axis}.")
    if not void_curves:
        raise RuntimeError("Could not identify the circular void boundary.")
    return (
        {key: tuple(value) for key, value in periodic.items()},
        tuple(sorted(void_curves)),
    )


def _quadrilateral9_mesh_diagnostics(gmsh) -> dict[str, object]:
    element_types, element_tags, element_nodes = gmsh.model.mesh.getElements(2)
    if not element_types:
        raise RuntimeError("Gmsh produced no two-dimensional cells.")
    names = []
    counts = []
    all_tags = []
    for element_type, tags, nodes in zip(
        element_types,
        element_tags,
        element_nodes,
    ):
        name, dimension, order, node_count, *_rest = (
            gmsh.model.mesh.getElementProperties(int(element_type))
        )
        names.append(str(name))
        count = int(np.asarray(tags).size)
        counts.append(count)
        all_tags.extend(int(value) for value in np.asarray(tags).reshape(-1))
        if (
            name != "Quadrilateral 9"
            or int(dimension) != 2
            or int(order) != 2
            or int(node_count) != 9
            or np.asarray(nodes).size != 9 * count
        ):
            raise RuntimeError(
                "The exact Zhang 2021 plane-strain fixture requires a pure "
                "Quadrilateral 9 mesh; observed "
                f"{name!r} with order={order} and nodes={node_count}."
            )
    if len(set(names)) != 1:
        raise RuntimeError(
            "The exact Zhang 2021 plane-strain fixture requires one Q9 cell type."
        )
    qualities = np.asarray(
        gmsh.model.mesh.getElementQualities(all_tags, "minSJ"),
        dtype=float,
    )
    if (
        qualities.size != sum(counts)
        or not np.all(np.isfinite(qualities))
        or float(np.min(qualities)) <= 0.0
    ):
        raise RuntimeError(
            "The exact Zhang 2021 plane-strain mesh contains an invalid Q9 cell."
        )
    return {
        "element_name": names[0],
        "element_count": sum(counts),
        "nodes_per_element": 9,
        "minimum_scaled_jacobian": float(np.min(qualities)),
    }


def _periodic_semantics_2d(gmsh, curves, *, tolerance: float):
    """Return exact two-component equations for the unit-square lattice."""

    origin = np.asarray((-0.5, -0.5, 0.0), dtype=float)
    periods = np.asarray((1.0, 1.0), dtype=float)
    node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes()
    coordinates = np.asarray(node_coordinates, dtype=float).reshape(-1, 3)
    labels = np.asarray(node_tags, dtype=np.int64)
    coordinate_to_label = {}
    coordinate_by_label = {}
    for label, coordinate in zip(labels, coordinates):
        key = tuple(
            np.rint(np.asarray(coordinate) / tolerance).astype(np.int64)
        )
        if key in coordinate_to_label:
            raise RuntimeError("Gmsh produced duplicate source-node coordinates.")
        coordinate_to_label[key] = int(label)
        coordinate_by_label[int(label)] = coordinate

    def label_at(coordinate):
        key = tuple(
            np.rint(np.asarray(coordinate) / tolerance).astype(np.int64)
        )
        try:
            return coordinate_to_label[key]
        except KeyError as exc:
            raise RuntimeError(
                f"The 2D periodic mesh has no node at {np.asarray(coordinate).tolist()}."
            ) from exc

    anchor = label_at(origin)
    references = tuple(
        label_at(origin + np.eye(3)[axis] * periods[axis])
        for axis in range(2)
    )
    control_nodes = {anchor, *references}
    boundary_labels = sorted(
        int(label)
        for label, coordinate in zip(labels, coordinates)
        if any(
            abs(coordinate[axis] - bound) <= tolerance
            for axis in range(2)
            for bound in (-0.5, 0.5)
        )
    )
    equations = []
    for slave in boundary_labels:
        coordinate = coordinate_by_label[slave]
        active_axes = tuple(
            axis
            for axis in range(2)
            if abs(coordinate[axis] - 0.5) <= tolerance
        )
        if not active_axes or slave in control_nodes:
            continue
        wrapped = coordinate.copy()
        wrapped[list(active_axes)] = -0.5
        base = label_at(wrapped)
        for component in (1, 2):
            terms = [(slave, component, 1.0), (base, component, -1.0)]
            terms.extend(
                (references[axis], component, -1.0)
                for axis in active_axes
            )
            terms.append((anchor, component, float(len(active_axes))))
            equations.append(terms)

    pairing_error = 0.0
    pair_counts = []
    expected_pair_counts = []
    for axis in range(2):
        selected_curves = curves[(axis, 1)]
        slave_labels = set()
        expected_slave_labels = {
            int(label)
            for label, coordinate in zip(labels, coordinates)
            if abs(coordinate[axis] - 0.5) <= tolerance
        }
        found_pair = False
        for selected_curve in selected_curves:
            _master, slaves, masters, _transform = (
                gmsh.model.mesh.getPeriodicNodes(1, int(selected_curve), True)
            )
            if len(slaves) == 0:
                continue
            found_pair = True
            translation = np.eye(3)[axis]
            for slave, master in zip(slaves, masters):
                slave = int(slave)
                master = int(master)
                slave_labels.add(slave)
                pairing_error = max(
                    pairing_error,
                    float(
                        np.linalg.norm(
                            coordinate_by_label[slave]
                            - coordinate_by_label[master]
                            - translation
                        )
                    ),
                )
        if not found_pair:
            raise RuntimeError(
                f"Gmsh did not retain periodic curve-node pairs on axis {axis}."
            )
        if slave_labels != expected_slave_labels:
            missing = sorted(expected_slave_labels - slave_labels)
            unexpected = sorted(slave_labels - expected_slave_labels)
            raise RuntimeError(
                "Gmsh periodic pairing does not cover the complete high-side "
                f"Q9 node set on axis {axis}: missing={missing}, "
                f"unexpected={unexpected}."
            )
        pair_counts.append(len(slave_labels))
        expected_pair_counts.append(len(expected_slave_labels))

    return {
        "labels": boundary_labels,
        "coordinates": [
            coordinate_by_label[label][:2].tolist() for label in boundary_labels
        ],
        "equations": equations,
        "anchor_node": anchor,
        "reference_nodes": references,
        "periodic_pair_counts": tuple(pair_counts),
        "periodic_expected_pair_counts": tuple(expected_pair_counts),
        "periodic_pairing_error": pairing_error,
        "geometric_dimension": 2,
    }


__all__ = [
    "TABLE5",
    "Zhang2021PlaneStrainCompositeFixture",
    "Zhang2021PeriodicCompositeFixture",
    "Zhang2021Table5Reference",
    "assess_table5",
    "column_major_plane_components",
    "young_poisson_from_bulk_shear",
    "zhang_2021_plane_strain_composite",
    "zhang_2021_periodic_composite",
]
