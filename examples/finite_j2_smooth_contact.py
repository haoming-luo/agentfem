# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Experimental finite-J2 contact with one analytical curved rigid tool.

This small serial example exercises a sphere or cylinder with an exact
active-branch curvature tangent. It is not a validated forming simulation.
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
    parser.add_argument("--shape", choices=("sphere", "cylinder"), default="sphere")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/finite_j2_smooth_contact"),
    )
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Implicit curved-tool J2 contact is serial only.")

    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True),
        mesh=domain,
        name="finite_j2_smooth_contact",
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right")
    model.clamp(displacement, on=left)
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=1000.0,
            poisson=0.0,
            yield_stress=2.0,
            hardening_modulus=100.0,
        )
    )
    surface = (
        boundary_models.rigid_sphere(center=(1.6, 0.5, 0.5), radius=0.75)
        if args.shape == "sphere"
        else boundary_models.rigid_cylinder(
            axis_point=(1.6, 0.5, 0.0),
            axis_direction=(0.0, 0.0, 1.0),
            radius=0.7,
        )
    )
    model.add_boundary_model(
        boundary_models.rigid_contact_pair(
            right,
            boundary_models.rigid_body(
                surface,
                reference_point=(1.6, 0.5, 0.5),
            ),
            penalty=500.0,
        )
    )
    model.traction((10.0, 0.0, 0.0), on=right)
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(3),
        progress=False,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result = step.solve_result(output=args.output / "fields.xdmf")
    result.write_manifest(args.output / "result.json")
    print(f"Status: {result.status}")
    print(f"Active contact points: {step.contact_provider.evaluate().active_points}")
    print(f"Contact potential: {step.energy_history[-1].contact_potential:.6g}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
