"""Bounded serial finite-J2 contact with one closed convex sharp rigid tool.

Run with an installed AgentFEM wheel and FEniCSx environment:

    python examples/finite_j2_closed_convex_contact.py --cells 1 --increments 8

The analytical bar comparison is valid only for this homogeneous elastic
regime, Poisson ratio zero, and the flat tool face selected throughout this
particular example. It does not validate arbitrary sharp-feature contact.
"""

from __future__ import annotations

import argparse
import json

from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np

from agentfem import boundary_models, constitutive, fields, mesh, models, solvers, steps, studies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=1)
    parser.add_argument("--increments", type=int, default=8)
    args = parser.parse_args()
    if args.cells < 1 or args.increments < 1:
        parser.error("--cells and --increments must be positive")

    domain = dolfinx_mesh.create_unit_cube(
        MPI.COMM_SELF, args.cells, args.cells, args.cells,
    )
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain)
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="clamped_left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="tool_side")
    model.clamp(displacement, on=left)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=0.0, yield_stress=1.0e9, hardening_modulus=100.0,
    ))

    x0, x1, y0, y1, z0, z1 = 1.005, 1.105, -0.5, 1.5, -0.5, 1.5
    tool = boundary_models.convex_polyhedral_rigid_surface(
        vertices=(
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ),
        triangles=(
            (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
            (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
        ),
        name="closed_box_tool",
    )
    motion = boundary_models.prescribed_rigid_motion(
        translation=(-0.008, 0.0, 0.0), reference_point=(x0, 0.0, 0.0),
    )
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        right,
        boundary_models.rigid_body(
            tool, motion_schedule=boundary_models.prescribed_rigid_motion_schedule(
                motion, start_time=0.0, end_time=1.0,
            ),
        ),
        penalty=500.0,
    ))

    step = model.step(
        target=displacement, material=material,
        incrementation=steps.fixed(args.increments),
        solver_options=solvers.newton(
            relative_tolerance=1e-8, absolute_tolerance=1e-9,
            maximum_iterations=25, line_search="backtracking",
        ),
        progress=False,
    )
    result = step.solve_result()
    coordinates = step.solution.function_space.tabulate_dof_coordinates()
    right_ux = step.solution.x.array.reshape((-1, 3))[
        np.isclose(coordinates[:, 0], 1.0), 0,
    ]
    lower, upper = 0.997, 1.0
    for _ in range(70):
        stretch = 0.5 * (lower + upper)
        residual = 1000.0 * np.log(stretch) / stretch + 500.0 * (stretch - 0.997)
        if residual < 0.0:
            lower = stretch
        else:
            upper = stretch
    expected = 0.5 * (lower + upper) - 1.0
    print(json.dumps({
        "status": result.status,
        "cells_per_axis": args.cells,
        "accepted_increments": args.increments,
        "right_displacement": float(np.mean(right_ux)),
        "analytical_right_displacement": expected,
        "maximum_right_displacement_spread": float(np.ptp(right_ux)),
        "contact_potential": result.quantity("rigid_contact_pair_contact_energy"),
        "tool_work": step.energy_history[-1].tool_work,
        "geometry": tool.summary()["kind"],
        "serial_reference_only": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
