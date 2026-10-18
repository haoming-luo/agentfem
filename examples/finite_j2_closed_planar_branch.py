# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded finite-J2 contact on one certified smooth face of a concave tool.

The reviewed L-prism is a tiny synthetic example, not a CAD import or an
industrial forming validation. The selected contact patch is deliberately
away from physical tool edges. A trial that leaves its certified single
master-plane region fails closed instead of using a sampled normal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import (
    boundary_models, constitutive, fields, mesh, models, results,
    solvers, steps, studies,
)


def reviewed_tool():
    polygon = np.asarray(
        ((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)), dtype=float,
    )
    front = np.column_stack((np.full(6, 1.10), polygon))
    back = front.copy()
    back[:, 0] += 0.5
    top = ((0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5))
    triangles = [tuple(reversed(face)) for face in top]
    triangles += [tuple(6 + index for index in face) for face in top]
    for first in range(6):
        second = (first + 1) % 6
        triangles.extend((
            (first, second, second + 6),
            (first, second + 6, first + 6),
        ))
    return boundary_models.closed_triangulated_rigid_solid(
        vertices=np.vstack((front, back)), triangles=triangles,
        reviewed_embedded=True, name="reviewed_synthetic_L_prism",
    )


def build_case(*, cells: int = 1, increments: int = 6, penalty: float = 5000.0):
    """Build the fixed-geometry fixture used by the example and sensitivity study."""

    if type(cells) is not int or cells < 1:
        raise ValueError("cells must be a positive integer")
    if type(increments) is not int or increments < 1:
        raise ValueError("increments must be a positive integer")
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("penalty must be finite and positive")
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("This bounded implicit-contact example is serial only.")

    domain = dolfinx_mesh.create_unit_cube(
        MPI.COMM_SELF, cells, cells, cells,
        cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    xyz = domain.geometry.x
    xyz[:, 0] = 1.05 + 0.10 * xyz[:, 0]
    xyz[:, 1] = 1.25 + 0.10 * xyz[:, 1]
    xyz[:, 2] = 0.45 + 0.10 * xyz[:, 2]
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True),
        mesh=domain, name="finite_j2_closed_planar_branch",
    )
    displacement = model.field(fields.displacement(domain))
    contact = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 1.05), name="contact",
    )
    support = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 1.15), name="support",
    )
    model.clamp(displacement, on=support)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=0.0,
        yield_stress=2.0, hardening_modulus=100.0,
    ))
    motion = boundary_models.prescribed_rigid_motion(
        translation=(-0.08, 0.0, 0.0), reference_point=(1.10, 1.3, 0.5),
    )
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        contact,
        boundary_models.rigid_body(
            reviewed_tool(),
            motion_schedule=boundary_models.prescribed_rigid_motion_schedule(
                motion, start_time=0.0, end_time=1.0,
            ),
        ),
        penalty=penalty,
    ))
    step = model.step(
        target=displacement, material=material,
        incrementation=steps.fixed(increments),
        contact_integration="closed_planar_branch_partition",
        solver_options=solvers.newton(
            relative_tolerance=1e-8, absolute_tolerance=1e-9,
            maximum_iterations=25, line_search="backtracking",
        ),
        progress=False,
    )
    return step, displacement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/closed_planar_branch"))
    parser.add_argument("--cells", type=int, default=1)
    parser.add_argument("--increments", type=int, default=6)
    parser.add_argument("--penalty", type=float, default=5000.0)
    args = parser.parse_args()
    step, displacement = build_case(
        cells=args.cells, increments=args.increments, penalty=args.penalty,
    )
    result = step.solve_result()
    args.output.mkdir(parents=True, exist_ok=True)
    with results.UnifiedXDMFTimeSeries(args.output / "fields.xdmf") as writer:
        writer.write_fields(1.0, displacement.value)
    results.export_accepted_contact_points(
        step, args.output / "contact", result=result, vtk_view=True,
    )
    result.write_manifest(args.output / "result.json")
    print(f"Status: {result.status}")
    print(f"Maximum PEEQ: {result.quantity('maximum_equivalent_plastic_strain'):.6g}")
    print(f"Tool work: {step.energy_history[-1].tool_work:.6g}")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
