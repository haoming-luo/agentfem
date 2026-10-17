# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded serial finite-J2 friction with a moving plane or convex rigid tool.

This is an installed-package workflow and a force/work/restart illustration,
not validation of arbitrary curved-tool friction or a forming process.
"""

from __future__ import annotations

import argparse
import json

from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np

from agentfem import (
    boundary_models, constitutive, fields, mesh, models, results, steps, studies,
)


def _surface(geometry: str):
    if geometry == "plane":
        return boundary_models.rigid_plane(
            point=(0.05, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
        )
    if geometry != "closed-convex":
        raise ValueError("geometry must be 'plane' or 'closed-convex'.")
    return boundary_models.convex_polyhedral_rigid_surface(
        vertices=(
            (0.05, -1.0, -1.0), (0.55, -1.0, -1.0),
            (0.55, 2.0, -1.0), (0.05, 2.0, -1.0),
            (0.05, -1.0, 2.0), (0.55, -1.0, 2.0),
            (0.55, 2.0, 2.0), (0.05, 2.0, 2.0),
        ),
        triangles=(
            (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
            (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
        ),
    )


def run_case(
    *, increments: int = 4, rotation: float = 0.15,
    geometry: str = "plane", contact_point_dir: str | None = None,
) -> dict[str, object]:
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("The moving-plane implicit friction example is serial only.")
    if increments < 1 or not np.isfinite(rotation):
        raise ValueError("increments must be positive and rotation must be finite.")
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1)
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True),
        mesh=domain, name="moving_plane_friction",
    )
    displacement = model.field(fields.displacement(domain))
    support = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="support")
    slave = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="tool_side")
    model.clamp(displacement, on=support)
    for component, value in enumerate((0.10, 0.0, 0.0)):
        model.fix(displacement, on=slave, component=component, value=value)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=0.0, yield_stress=1.0e9,
        hardening_modulus=100.0,
    ))
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.0, 0.40, 0.0), rotation=(0.0, 0.0, rotation),
        reference_point=(0.05, 0.0, 0.0),
    )
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        slave,
        boundary_models.rigid_body(
            _surface(geometry),
            motion_schedule=boundary_models.prescribed_rigid_motion_schedule(
                motion, start_time=0.0, end_time=1.0,
            ),
        ),
        penalty=500.0, friction_coefficient=0.4,
        tangential_penalty=100.0, name="moving_frictional_plane",
    ))
    step = model.step(
        target=displacement, material=material,
        incrementation=steps.fixed(increments), progress=False,
    )
    result = step.solve_result()
    dual = result.metadata["constraint_duals"][0]
    accepted = step.energy_history[-1]
    report = {
        "status": result.status,
        "accepted_increments": len(step.accepted_increments),
        "tool_force_and_moment": dual["force"],
        "tool_work": accepted.tool_work,
        "friction_dissipation": accepted.friction_dissipation,
        "mechanical_energy_residual_diagnostic": accepted.mechanical_energy_residual,
        "linearization": step.contact_provider.summary()["linearization"],
        "geometry": geometry,
        "scientifically_validated": False,
    }
    if contact_point_dir is not None:
        report["contact_point_artifacts"] = {
            name: str(path) for name, path in results.export_accepted_contact_points(
                step, contact_point_dir, result=result, vtk_view=True,
            ).items()
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--increments", type=int, default=4)
    parser.add_argument("--rotation", type=float, default=0.15)
    parser.add_argument("--geometry", choices=("plane", "closed-convex"), default="plane")
    parser.add_argument("--contact-points", metavar="DIRECTORY")
    options = parser.parse_args()
    print(json.dumps(run_case(
        increments=options.increments, rotation=options.rotation,
        geometry=options.geometry, contact_point_dir=options.contact_points,
    ), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
