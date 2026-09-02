"""Public three-dimensional linear-elasticity benchmarks.

The NAFEMS LE10 thick elliptical plate is a deliberately demanding solid-
mechanics check: it combines three-dimensional bending, symmetry and support
constraints, a surface pressure, quadratic displacement interpolation and a
stress-recovery quantity at point D.  The implementation follows the public
NAFEMS/Abaqus geometry and boundary-condition specification while keeping the
solve on AgentFEM's canonical ``model.step() -> SimulationResult`` path.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import basix
import basix.ufl
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np
import ufl

from .. import fields, mesh, models, results, studies
from ..constitutive import elasticity


@dataclass(frozen=True)
class Elasticity3DBenchmark:
    """Rank-independent evidence from a public 3D elasticity benchmark."""

    name: str
    reference: str
    mpi_ranks: int
    quantities: dict[str, float]
    tolerances: dict[str, float]
    extraction: dict[str, object]

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
            "reference": self.reference,
            "mpi_ranks": self.mpi_ranks,
            "acceptable": self.acceptable,
            "quantities": dict(self.quantities),
            "tolerances": dict(self.tolerances),
            "extraction": dict(self.extraction),
        }


def nafems_le10_mesh(
    *,
    radial_cells: int = 4,
    angular_cells: int = 12,
    thickness_cells: int = 2,
    geometry_degree: int = 2,
    comm=MPI.COMM_WORLD,
):
    """Create the quarter thick-elliptical-plate domain from NAFEMS LE10.

    The inner ellipse has semiaxes ``(2, 1)`` m, the outer ellipse has
    semiaxes ``(3.25, 2.75)`` m, and the thickness is ``0.6`` m.  A mapped
    hexahedral mesh keeps Gmsh optional and makes the public benchmark usable
    in clean runtime installations.
    """

    nr = int(radial_cells)
    nt = int(angular_cells)
    nz = int(thickness_cells)
    geometry_order = int(geometry_degree)
    if nr < 1 or nt < 2 or nz < 2:
        raise ValueError(
            "NAFEMS LE10 needs radial_cells >= 1, angular_cells >= 2, "
            "and thickness_cells >= 2."
        )
    if nz % 2:
        raise ValueError(
            "thickness_cells must be even so the prescribed mid-thickness "
            "support line is represented exactly."
        )
    if geometry_order not in {1, 2}:
        raise ValueError("geometry_degree must be 1 or 2.")

    if comm.rank == 0:
        radial = np.linspace(0.0, 1.0, geometry_order * nr + 1)
        angles = np.linspace(0.5 * pi, 0.0, geometry_order * nt + 1)
        elevations = np.linspace(0.0, 0.6, geometry_order * nz + 1)
        coordinates = np.asarray(
            [
                (
                    (2.0 + 1.25 * s) * np.cos(theta),
                    (1.0 + 1.75 * s) * np.sin(theta),
                    z,
                )
                for z in elevations
                for theta in angles
                for s in radial
            ],
            dtype=float,
        )

        radial_nodes = geometry_order * nr + 1
        angular_nodes = geometry_order * nt + 1

        def node(i: int, j: int, k: int) -> int:
            return k * angular_nodes * radial_nodes + j * radial_nodes + i

        coordinate_basis = basix.create_element(
            basix.ElementFamily.P,
            basix.CellType.hexahedron,
            geometry_order,
            basix.LagrangeVariant.equispaced,
        )
        local_offsets = np.rint(
            geometry_order * np.asarray(coordinate_basis.points)
        ).astype(np.int64)

        cells = []
        for k in range(nz):
            for j in range(nt):
                for i in range(nr):
                    base = geometry_order * np.asarray((i, j, k), dtype=np.int64)
                    cells.append(
                        tuple(
                            node(*(base + offset))
                            for offset in local_offsets
                        )
                    )
        topology = np.asarray(cells, dtype=np.int64)
    else:
        coordinates = np.empty((0, 3), dtype=float)
        topology = np.empty((0, (geometry_order + 1) ** 3), dtype=np.int64)

    coordinate_element = ufl.Mesh(
        basix.ufl.element(
            "Lagrange",
            "hexahedron",
            geometry_order,
            shape=(3,),
            lagrange_variant=basix.LagrangeVariant.equispaced,
        )
    )
    return dolfinx_mesh.create_mesh(
        comm, topology, coordinate_element, coordinates
    )


def nafems_le10_3d_benchmark(
    *,
    radial_cells: int = 4,
    angular_cells: int = 12,
    thickness_cells: int = 2,
    comm=MPI.COMM_WORLD,
    output=None,
) -> tuple[Elasticity3DBenchmark, object]:
    """Solve NAFEMS LE10 and return benchmark evidence plus SimulationResult.

    NAFEMS reports ``sigma_yy(D) = -5.38 MPa``.  The published comparison is
    a surface-node value obtained by element extrapolation and nodal
    averaging, not an integration-point value.  AgentFEM therefore reports
    both its default unsmoothed DG0 stress field in ``SimulationResult`` and
    an explicitly labelled continuous-P1 L2 recovery used only for the
    benchmark comparison.
    """

    domain = nafems_le10_mesh(
        radial_cells=radial_cells,
        angular_cells=angular_cells,
        thickness_cells=thickness_cells,
        geometry_degree=2,
        comm=comm,
    )
    study = studies.static_solid(dimension=3, name="NAFEMS LE10")
    model = models.create(study=study, mesh=domain, name="nafems_le10_3d")
    displacement = model.field(fields.displacement(domain, degree=2))
    material = elasticity.isotropic_elastic(
        young=210.0e9,
        poisson=0.3,
        density=7800.0,
        name="NAFEMS LE10 isotropic solid",
    )
    model.material(material)

    x_symmetry = mesh.face(
        domain, axis="x", value=0.0, name="AB_x_symmetry", tag=1
    )
    y_symmetry = mesh.face(
        domain, axis="y", value=0.0, name="DC_y_symmetry", tag=2
    )
    top = mesh.face(domain, axis="z", value=0.6, name="top_pressure", tag=3)
    tolerance = 2.0e-8
    outer = mesh.boundary_region(
        domain,
        lambda x: np.isclose(
            (x[0] / 3.25) ** 2 + (x[1] / 2.75) ** 2,
            1.0,
            rtol=0.0,
            atol=2.0e-8,
        ),
        name="BC_outer_support",
        tag=4,
    )
    mid_outer = lambda x: (
        np.isclose(x[2], 0.3, rtol=0.0, atol=tolerance)
        & np.isclose(
            (x[0] / 3.25) ** 2 + (x[1] / 2.75) ** 2,
            1.0,
            rtol=0.0,
            atol=tolerance,
        )
    )

    model.fix(displacement, on=x_symmetry, component=0, value=0.0)
    model.fix(displacement, on=y_symmetry, component=1, value=0.0)
    model.fix(displacement, on=outer, components=(0, 1), value=0.0)
    model.fix(
        displacement,
        location=mid_outer,
        component=2,
        value=0.0,
        name="EE_mid_thickness_uz",
    )
    model.traction((0.0, 0.0, -1.0e6), on=top, name="top_pressure")

    simulation = model.step(target=displacement).solve_result(output=output)
    stress_expression = elasticity.stress(
        displacement.value, material, study=study
    )
    recovered_sigma_yy = results.project(
        stress_expression[1, 1],
        domain=domain,
        family="Lagrange",
        degree=1,
        name="S22_NAFEMS_RECOVERED",
    )
    sigma_yy_d = float(
        results.probe(recovered_sigma_yy, at=(2.0, 0.0, 0.6))
    )
    displacement_d = np.asarray(
        results.probe(displacement.value, at=(2.0, 0.0, 0.6)),
        dtype=float,
    )
    reference = -5.38e6
    # Independent public Fino/FeenoX LE10 result on the original fine mesh.
    displacement_reference = np.asarray((-2.76353e-5, 0.0, -9.8182e-5))
    relative_stress_error = abs(sigma_yy_d - reference) / abs(reference)
    relative_displacement_error = np.linalg.norm(
        displacement_d - displacement_reference
    ) / np.linalg.norm(displacement_reference)
    force_error = float(
        simulation.quantities["relative_force_balance_error"].value
    )
    external_force = np.asarray(
        simulation.quantities["external_force_resultant"].value,
        dtype=float,
    )
    exact_loaded_area = 0.25 * pi * (3.25 * 2.75 - 2.0 * 1.0)
    loaded_area_error = abs(abs(external_force[2]) / 1.0e6 - exact_loaded_area) / (
        exact_loaded_area
    )
    strain_energy = float(simulation.quantities["strain_energy"].value)
    energy_error = abs(
        float(simulation.quantities["energy_balance_error"].value)
    ) / max(abs(strain_energy), np.finfo(float).eps)
    benchmark = Elasticity3DBenchmark(
        name="NAFEMS LE10 thick elliptical plate",
        reference=(
            "NAFEMS LE10; Abaqus Benchmarks Guide, Linear analysis of the "
            "NAFEMS LE10 thick plate"
        ),
        mpi_ranks=int(comm.size),
        quantities={
            "relative_sigma_yy_D_error": float(relative_stress_error),
            "relative_point_D_displacement_error": float(
                relative_displacement_error
            ),
            "relative_force_balance_error": force_error,
            "relative_energy_balance_error": float(energy_error),
            "relative_loaded_area_error": float(loaded_area_error),
            "sigma_yy_D_pa": sigma_yy_d,
            "u_x_D_m": float(displacement_d[0]),
            "u_z_D_m": float(displacement_d[2]),
        },
        tolerances={
            "relative_sigma_yy_D_error": 0.05,
            "relative_point_D_displacement_error": 0.03,
            "relative_force_balance_error": 1.0e-9,
            "relative_energy_balance_error": 1.0e-9,
            "relative_loaded_area_error": 5.0e-5,
        },
        extraction={
            "target": "sigma_yy at surface point D=(2,0,0.6)",
            "reference_pa": reference,
            "secondary_displacement_reference_m": tuple(
                float(value) for value in displacement_reference
            ),
            "reference_semantics": "element extrapolation and nodal averaging",
            "agentfem_comparison_semantics": "continuous-P1 global L2 recovery",
            "default_result_semantics": "unsmoothed DG0 cell average",
            "nodal_recovery_is_default": False,
        },
    )
    simulation.add_quantity(
        "nafems_sigma_yy_D",
        sigma_yy_d,
        unit="Pa",
        kind="benchmark",
        description="LE10-compatible recovered surface stress at point D.",
    )
    simulation.metadata["external_benchmark"] = benchmark.as_dict()
    return benchmark, simulation
