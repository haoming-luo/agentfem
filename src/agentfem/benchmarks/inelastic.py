"""External structural benchmarks for stateful inelastic solids.

The benchmark geometry is generated with NumPy/DOLFINx so the verification
path does not make Gmsh a mandatory runtime dependency.  The scientific
references remain external: the J2 problem follows the Comet-FEniCSx
plane-strain pressurised cylinder and the creep problem follows NAFEMS R0027
Test 7 as published in the Abaqus verification manual.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import basix.ufl
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np
import ufl

from .. import (
    constitutive,
    constraints,
    fields,
    mechanics,
    mesh,
    models,
    solvers,
    steps,
    studies,
)


@dataclass(frozen=True)
class InelasticStructuralBenchmark:
    """Compact, rank-independent evidence from one structural benchmark."""

    name: str
    mpi_ranks: int
    quantities: dict[str, float]
    tolerances: dict[str, float]

    @property
    def acceptable(self) -> bool:
        return all(
            np.isfinite(self.quantities[name])
            and self.quantities[name] <= limit
            for name, limit in self.tolerances.items()
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mpi_ranks": self.mpi_ranks,
            "acceptable": self.acceptable,
            "quantities": dict(self.quantities),
            "tolerances": dict(self.tolerances),
        }


def thick_cylinder_sector_mesh(
    *,
    inner_radius: float,
    outer_radius: float,
    thickness: float,
    radial_cells: int,
    angular_cells: int,
    comm=MPI.COMM_WORLD,
):
    """Create a one-layer 3D quarter-cylinder tetrahedral benchmark mesh."""

    inner = float(inner_radius)
    outer = float(outer_radius)
    height = float(thickness)
    nr = int(radial_cells)
    nt = int(angular_cells)
    if not 0.0 < inner < outer or height <= 0.0:
        raise ValueError("Cylinder radii and thickness must be positive with inner < outer.")
    if nr < 1 or nt < 2:
        raise ValueError("The cylinder sector needs radial_cells >= 1 and angular_cells >= 2.")

    if comm.rank == 0:
        radii = np.linspace(inner, outer, nr + 1)
        angles = np.linspace(0.0, 0.5 * pi, nt + 1)
        coordinates = np.asarray(
            [
                (radius * np.cos(angle), radius * np.sin(angle), z)
                for z in (0.0, height)
                for angle in angles
                for radius in radii
            ],
            dtype=float,
        )

        def node(i: int, j: int, k: int) -> int:
            return k * (nt + 1) * (nr + 1) + j * (nr + 1) + i

        cells: list[tuple[int, int, int, int]] = []
        for j in range(nt):
            for i in range(nr):
                a, b = node(i, j, 0), node(i + 1, j, 0)
                c, d = node(i, j + 1, 0), node(i + 1, j + 1, 0)
                e, f = node(i, j, 1), node(i + 1, j, 1)
                g, h = node(i, j + 1, 1), node(i + 1, j + 1, 1)
                cells.extend(
                    (
                        (a, b, d, h),
                        (a, d, c, h),
                        (a, c, g, h),
                        (a, g, e, h),
                        (a, e, f, h),
                        (a, f, b, h),
                    )
                )
        topology = np.asarray(cells, dtype=np.int64)
    else:
        coordinates = np.empty((0, 3), dtype=float)
        topology = np.empty((0, 4), dtype=np.int64)

    coordinate_element = ufl.Mesh(
        basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    )
    return dolfinx_mesh.create_mesh(comm, topology, coordinate_element, coordinates)


def j2_plane_strain_first_yield_pressure(
    *, inner_radius: float, outer_radius: float, poisson: float, yield_stress: float
) -> float:
    """Lamé plane-strain pressure at first Mises yield on the inner wall."""

    inner = float(inner_radius)
    outer = float(outer_radius)
    denominator = outer**2 - inner**2
    a_per_pressure = inner**2 / denominator
    b_over_inner_squared = outer**2 / denominator
    sigma_r = a_per_pressure - b_over_inner_squared
    sigma_theta = a_per_pressure + b_over_inner_squared
    sigma_z = 2.0 * float(poisson) * a_per_pressure
    mises_per_pressure = np.sqrt(
        0.5
        * (
            (sigma_theta - sigma_r) ** 2
            + (sigma_z - sigma_r) ** 2
            + (sigma_theta - sigma_z) ** 2
        )
    )
    return float(yield_stress) / float(mises_per_pressure)


def _cylinder_model(
    *,
    domain,
    inner_radius: float,
    thickness: float,
    study,
    degree: int = 2,
    name: str,
):
    model = models.create(study=study, mesh=domain, name=name)
    displacement = model.field(fields.displacement(domain, degree=degree))
    tolerance = 1.0e-10 * max(1.0, abs(float(inner_radius)))
    inner = mesh.boundary_region(
        domain,
        lambda x: np.isclose(
            np.sqrt(x[0] ** 2 + x[1] ** 2),
            inner_radius,
            rtol=0.0,
            atol=tolerance,
        ),
        name="inner_wall",
        tag=1,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=0.0, name="x_symmetry", tag=2),
        component=0,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="y", value=0.0, name="y_symmetry", tag=3),
        component=1,
    )
    # This is a 3D extrusion of the published plane-strain cross-section.
    # Constrain every axial displacement dof, including P2 edge dofs in the
    # interior, rather than constraining only the two end faces.
    model.constraint(
        constraints.component_dirichlet(
            displacement,
            2,
            marker=lambda x: np.ones(x.shape[1], dtype=bool),
            name="plane_strain_uz",
        )
    )
    return model, displacement, inner


def j2_thick_cylinder_benchmark(
    *,
    comm=MPI.COMM_WORLD,
    radial_cells: int = 4,
    angular_cells: int = 8,
    increments: int = 24,
) -> InelasticStructuralBenchmark:
    """Run the Comet-FEniCSx thick-cylinder first-yield benchmark."""

    inner_radius, outer_radius, thickness = 1.0, 1.3, 0.1
    young, poisson, yield_stress = 70.0e3, 0.3, 250.0
    tangent_modulus = young / 100.0
    hardening = young * tangent_modulus / (young - tangent_modulus)
    domain = thick_cylinder_sector_mesh(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        thickness=thickness,
        radial_cells=radial_cells,
        angular_cells=angular_cells,
        comm=comm,
    )
    model, displacement, inner = _cylinder_model(
        domain=domain,
        inner_radius=inner_radius,
        thickness=thickness,
        study=studies.nonlinear_static(physics="solid_mechanics", dimension=3),
        name="comet_j2_thick_cylinder",
    )
    material = model.material(
        constitutive.J2LinearIsotropicHardening(
            young=young,
            poisson=poisson,
            yield_stress=yield_stress,
            hardening_modulus=hardening,
        )
    )
    analytical = j2_plane_strain_first_yield_pressure(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        poisson=poisson,
        yield_stress=yield_stress,
    )
    # Cross first yield far enough to exercise the return mapping while
    # remaining below the pressure-controlled limit-load regime.
    target_pressure = 1.05 * analytical
    model.pressure(target_pressure, on=inner)
    step = mechanics.j2_plasticity_step(
        displacement=displacement,
        material=material,
        external_force=model.external_force(displacement),
        constraints=model.constraints,
        study=model.study,
        incrementation=steps.fixed(increments),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=30,
            line_search="backtracking",
            linear_solver=solvers.direct_solver(package="mumps"),
        ),
        progress=False,
        _experimental_distributed=True,
        name="comet_j2_thick_cylinder",
    )
    result = step.solve_result()
    first_plastic = next(
        item for item in step.accepted_increments if item.plastic_points > 0
    )
    previous = max(
        (
            item.load_factor
            for item in step.accepted_increments
            if item.load_factor < first_plastic.load_factor
        ),
        default=0.0,
    )
    lower = previous * target_pressure
    upper = first_plastic.load_factor * target_pressure
    bracket_error = max(lower - analytical, analytical - upper, 0.0) / analytical
    bracket_width = (upper - lower) / analytical
    dofmap = step.solution.function_space.dofmap
    owned = int(dofmap.index_map.size_local) * int(dofmap.index_map_bs)
    local_displacement = step.solution.x.array[:owned].reshape((-1, 3))
    maximum_displacement = float(
        comm.allreduce(
            np.max(np.linalg.norm(local_displacement, axis=1), initial=0.0),
            op=MPI.MAX,
        )
    )
    return InelasticStructuralBenchmark(
        name="comet_j2_thick_cylinder_first_yield",
        mpi_ranks=comm.size,
        quantities={
            "yield_bracket_error": float(bracket_error),
            "yield_bracket_width": float(bracket_width),
            "maximum_equivalent_plastic_strain": float(
                result.quantity("maximum_equivalent_plastic_strain")
            ),
            "maximum_displacement": maximum_displacement,
            "final_residual_norm": float(step.accepted_increments[-1].residual_norm),
        },
        tolerances={
            "yield_bracket_error": 0.03,
            "yield_bracket_width": 0.08,
            "final_residual_norm": 1.0e-7,
        },
    )


def power_law_creep_cylinder_stress(
    radius,
    *,
    inner_radius: float,
    outer_radius: float,
    pressure: float,
    stress_exponent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the NAFEMS R0027 Test 7 steady-state cylinder stresses.

    The solution is for an axisymmetric, plane-strain thick cylinder with a
    Mises secondary-creep law ``creep_rate = A * sigma**n``. It is independent
    of ``A`` once steady state is reached and is therefore a useful structural
    check of equilibrium, tensor flow, and global constitutive integration.
    """

    selected = np.asarray(radius, dtype=float)
    inner = float(inner_radius)
    outer = float(outer_radius)
    applied = float(pressure)
    exponent = float(stress_exponent)
    if not 0.0 < inner < outer:
        raise ValueError("Cylinder radii must satisfy 0 < inner_radius < outer_radius.")
    if applied <= 0.0 or exponent <= 0.0:
        raise ValueError("pressure and stress_exponent must be positive.")
    if (
        np.any(~np.isfinite(selected))
        or np.any(selected < inner)
        or np.any(selected > outer)
    ):
        raise ValueError("radius must remain inside the cylinder wall.")

    power = -2.0 / exponent
    radial_term = selected**power
    outer_term = outer**power
    denominator = inner**power - outer_term
    radial = -applied * (radial_term - outer_term) / denominator
    hoop = applied * (
        -(1.0 - 2.0 / exponent) * radial_term + outer_term
    ) / denominator
    axial = applied * (
        -(1.0 - 1.0 / exponent) * radial_term + outer_term
    ) / denominator
    return radial, hoop, axial


def creep_thick_cylinder_benchmark(
    *,
    comm=MPI.COMM_WORLD,
    radial_cells: int = 4,
    angular_cells: int = 8,
    increments: int = 40,
    duration: float = 1000.0,
    progress: object = False,
) -> InelasticStructuralBenchmark:
    """Run the NAFEMS R0027 Test 7 secondary-creep benchmark.

    The public reference is reproduced in the Abaqus verification manual as
    "Test 7: Axisymmetric -- pressurized cylinder, secondary creep". AgentFEM
    uses a thin three-dimensional quarter sector because its trusted global
    creep provider is deliberately three-dimensional; axial displacement is
    constrained everywhere to recover the published plane-strain assumption.
    """

    inner_radius, outer_radius, thickness = 100.0, 200.0, 1.0
    pressure, exponent = 200.0, 5.0
    selected_increments = int(increments)
    if selected_increments < 1:
        raise ValueError("increments must be a positive integer.")
    domain = thick_cylinder_sector_mesh(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        thickness=thickness,
        radial_cells=radial_cells,
        angular_cells=angular_cells,
        comm=comm,
    )
    model, displacement, inner = _cylinder_model(
        domain=domain,
        inner_radius=inner_radius,
        thickness=thickness,
        study=studies.creep_solid(),
        degree=2,
        name="nafems_r0027_test_7_creep_cylinder",
    )
    material = model.material(
        constitutive.isotropic_power_law(
            young=200.0e3,
            poisson=0.3,
            density=1.0,
            coefficient=3.125e-14,
            stress_exponent=exponent,
            reference_stress=1.0,
            reference_time=1.0,
            name="NAFEMS R0027 Test 7 secondary creep",
        )
    )
    model.pressure(pressure, on=inner)
    # Match the published two-step procedure: apply pressure elastically,
    # then hold it while creep redistributes the stress.  Two model registries
    # share the same live field because Study states physical analysis intent;
    # a creep Study must not pretend that its preload is linear static.
    preload_model = models.create(
        study=studies.static_solid(dimension=3),
        mesh=domain,
        name="nafems_r0027_elastic_preload",
    )
    preload_model.field(displacement)
    preload_model.material(material)
    for constraint in model.constraints:
        preload_model.constraint(constraint)
    for load in model.loads:
        preload_model.load(load)
    elastic_preload = preload_model.step(
        target=displacement,
        K=preload_model.stiffness(displacement, material),
        F=preload_model.external_force(displacement),
        constraints=preload_model.constraints,
        solver_options=solvers.direct_solver(package="mumps"),
        name="nafems_r0027_elastic_preload",
    )
    elastic_preload.solve()
    step = model.step(
        target=displacement,
        material=material,
        duration=float(duration),
        incrementation=steps.automatic(
            initial=1.0 / selected_increments,
            minimum=1.0e-6,
            maximum=max(
                1.0 / selected_increments,
                min(0.05, 2.0 / selected_increments),
            ),
            max_increments=max(500, 10 * selected_increments),
            max_cutbacks=12,
            growth_factor=1.25,
            fast_iterations=4,
            slow_iterations=12,
        ),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-7,
            maximum_iterations=30,
            line_search="backtracking",
            linear_solver=solvers.direct_solver(package="mumps"),
        ),
        progress=progress,
    )
    result = step.solve_result()

    stress = result.field("S_CELL")
    cell_map = domain.topology.index_map(domain.topology.dim)
    owned = int(cell_map.size_local)
    cells = np.arange(owned, dtype=np.int32)
    midpoints = dolfinx_mesh.compute_midpoints(domain, domain.topology.dim, cells)
    radius = np.linalg.norm(midpoints[:, :2], axis=1)
    values = np.empty((owned, 3, 3), dtype=float)
    block_size = int(stress.function_space.dofmap.bs)
    for cell in cells:
        dofs = stress.function_space.dofmap.cell_dofs(int(cell))
        if len(dofs) != 1 or block_size != 9:
            raise RuntimeError(
                "Creep benchmark requires one DG0 tensor block per cell."
            )
        start = int(dofs[0]) * block_size
        values[int(cell)] = stress.x.array[
            start : start + block_size
        ].reshape((3, 3))

    radial_direction = np.column_stack(
        (
            midpoints[:, 0] / radius,
            midpoints[:, 1] / radius,
            np.zeros(owned),
        )
    )
    hoop_direction = np.column_stack(
        (
            -midpoints[:, 1] / radius,
            midpoints[:, 0] / radius,
            np.zeros(owned),
        )
    )
    radial_stress = np.einsum(
        "ci,cij,cj->c", radial_direction, values, radial_direction
    )
    hoop_stress = np.einsum(
        "ci,cij,cj->c", hoop_direction, values, hoop_direction
    )
    axial_stress = values[:, 2, 2]
    exact = power_law_creep_cylinder_stress(
        radius,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        pressure=pressure,
        stress_exponent=exponent,
    )

    def global_relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
        numerator = comm.allreduce(
            float(np.sum((actual - expected) ** 2)), op=MPI.SUM
        )
        denominator = comm.allreduce(
            float(np.sum(expected**2)), op=MPI.SUM
        )
        return float(np.sqrt(numerator / denominator))

    quantities = {
        "radial_stress_relative_l2": global_relative_l2(radial_stress, exact[0]),
        "hoop_stress_relative_l2": global_relative_l2(hoop_stress, exact[1]),
        "axial_stress_relative_l2": global_relative_l2(axial_stress, exact[2]),
        "maximum_equivalent_creep_strain": float(
            result.quantity("maximum_equivalent_creep_strain")
        ),
        "final_residual_norm": float(step.accepted_increments[-1].residual_norm),
    }
    return InelasticStructuralBenchmark(
        name="nafems_r0027_test_7_secondary_creep",
        mpi_ranks=comm.size,
        quantities=quantities,
        tolerances={
            "radial_stress_relative_l2": 0.08,
            "hoop_stress_relative_l2": 0.08,
            "axial_stress_relative_l2": 0.08,
            "final_residual_norm": 1.0e-6,
        },
    )


__all__ = [
    "creep_thick_cylinder_benchmark",
    "InelasticStructuralBenchmark",
    "j2_plane_strain_first_yield_pressure",
    "j2_thick_cylinder_benchmark",
    "power_law_creep_cylinder_stress",
    "thick_cylinder_sector_mesh",
]
