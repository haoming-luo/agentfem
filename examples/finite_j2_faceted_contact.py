# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded finite-J2 contact with a coplanar triangulated rigid tool.

The faceted tool is a general geometry asset, not a plane-specific J2 law.
The optional prescribed motion shares the Step amplitude. This experimental
example does not validate curved/non-smooth dies, friction, periodic MPC or
distributed implicit contact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np

from agentfem import boundary_models, constitutive, fields, mesh, models, steps, studies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/finite_j2_faceted"))
    parser.add_argument("--tool-travel", type=float, default=0.0,
                        help="End-of-step x translation of the rigid tool")
    parser.add_argument("--tool-rotation", type=float, default=0.0,
                        help="End-of-step z-axis tool rotation in radians")
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Implicit faceted/J2 contact is serial only.")

    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True),
        mesh=domain,
        name="finite_j2_faceted_contact",
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right")
    model.clamp(displacement, on=left)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=0.0, yield_stress=2.0, hardening_modulus=100.0,
    ))
    tool_x = 1.005
    tool = boundary_models.triangulated_rigid_surface(
        vertices=((tool_x, -0.5, -0.5), (tool_x, -0.5, 1.5),
                  (tool_x, 1.5, 1.5), (tool_x, 1.5, -0.5)),
        triangles=((0, 1, 2), (0, 2, 3)),
        name="triangulated_tool",
    )
    motion = None
    if args.tool_travel != 0.0 or args.tool_rotation != 0.0:
        motion = boundary_models.prescribed_rigid_motion(
            translation=(args.tool_travel, 0.0, 0.0),
            rotation=(0.0, 0.0, args.tool_rotation),
            reference_point=(tool_x, 0.0, 0.0),
        )
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        right,
        boundary_models.rigid_body(
            tool,
            motion_schedule=(
                None if motion is None
                else boundary_models.prescribed_rigid_motion_schedule(
                    motion, start_time=0.0, end_time=1.0,
                )
            ),
        ),
        penalty=500.0,
    ))
    model.traction((10.0, 0.0, 0.0), on=right)
    step = model.step(target=displacement, material=material,
                      incrementation=steps.fixed(3), progress=False)
    args.output.mkdir(parents=True, exist_ok=True)
    result = step.solve_result(output=args.output / "fields.xdmf")
    result.write_manifest(args.output / "result.json")
    print(f"Status: {result.status}")
    print(f"Contact potential: {step.energy_history[-1].contact_potential:.6g}")
    if motion is not None:
        print(f"Tool work: {step.energy_history[-1].tool_work:.6g}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
