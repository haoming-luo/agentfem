# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded serial finite-J2 example with independently moving rigid planes.

This demonstrates ordinary Model -> Step -> Result composition, not a sheet-
forming validation or a general-contact claim. The frictionless finite block
is clamped on one face; two planar tools follow separate prescribed motions.
"""

from __future__ import annotations

import argparse
import json

from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np

from agentfem import (
    boundary_models, constitutive, fields, mesh, models, solvers, steps, studies,
)


def run_case(*, cells: int = 2, increments: int = 4) -> dict[str, object]:
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("This bounded implicit two-tool example is serial only.")
    if cells < 1 or increments < 1:
        raise ValueError("cells and increments must be positive integers.")
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, cells, cells, cells)
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain,
        name="opposing_rigid_tools",
    )
    displacement = model.field(fields.displacement(domain))
    support = mesh.boundary(
        domain, lambda x: np.isclose(x[1], 0.0), name="support",
    )
    model.clamp(displacement, on=support)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=0.3, yield_stress=1.0e9,
        hardening_modulus=100.0,
    ))
    for name, boundary_x, tool_x, normal, travel in (
        ("left_die", 0.0, -0.005, (1.0, 0.0, 0.0), (0.018, 0.0, 0.0)),
        ("right_die", 1.0, 1.005, (-1.0, 0.0, 0.0), (-0.018, 0.0, 0.0)),
    ):
        region = mesh.boundary(
            domain,
            lambda x, value=boundary_x: np.isclose(x[0], value),
            name=name,
        )
        motion = boundary_models.prescribed_rigid_motion(
            translation=travel, reference_point=(tool_x, 0.0, 0.0),
        )
        model.add_boundary_model(boundary_models.rigid_contact_pair(
            region,
            boundary_models.rigid_body(
                boundary_models.rigid_plane(
                    point=(tool_x, 0.0, 0.0), normal=normal,
                ),
                reference_point=(tool_x, 0.0, 0.0),
                motion_schedule=boundary_models.prescribed_rigid_motion_schedule(
                    motion, start_time=0.0, end_time=1.0,
                ),
            ),
            penalty=5000.0, name=name,
        ))
    step = model.step(
        target=displacement, material=material,
        incrementation=steps.fixed(increments),
        solver_options=solvers.newton(
            relative_tolerance=1e-8, absolute_tolerance=1e-9,
            maximum_iterations=30, line_search="backtracking",
        ),
        progress=False,
    )
    result = step.solve_result()
    last = step.energy_history[-1]
    return {
        "status": result.status,
        "cell_count": int(domain.topology.index_map(3).size_local),
        "accepted_increments": len(step.accepted_increments),
        "tool_work_by_pair": dict(last.pair_tool_work),
        "contact_potential": last.contact_potential,
        "contact_resultants": {
            item["constraint_name"]: item["resultant"]
            for item in result.metadata["constraint_duals"]
        },
        "work_balance_available": result.metadata[
            "constraint_balance_contract"
        ]["work_balance_available"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=2)
    parser.add_argument("--increments", type=int, default=4)
    arguments = parser.parse_args()
    print(json.dumps(run_case(
        cells=arguments.cells, increments=arguments.increments,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
