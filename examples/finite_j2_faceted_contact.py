# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded finite-J2 contact with a fixed, coplanar triangulated rigid tool.

The faceted tool is a general geometry asset, not a plane-specific J2 law.
This experimental example does not validate curved/non-smooth dies, tool
motion, friction, periodic MPC or distributed implicit contact.
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
        name="fixed_tool",
    )
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        right, boundary_models.rigid_body(tool), penalty=500.0,
    ))
    model.traction((10.0, 0.0, 0.0), on=right)
    step = model.step(target=displacement, material=material,
                      incrementation=steps.fixed(3), progress=False)
    args.output.mkdir(parents=True, exist_ok=True)
    result = step.solve_result(output=args.output / "fields.xdmf")
    result.write_manifest(args.output / "result.json")
    print(f"Status: {result.status}")
    print(f"Contact potential: {step.energy_history[-1].contact_potential:.6g}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
