# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded serial finite-J2 contact with a fixed frictional rigid plane.

This small active-contact fixture demonstrates the public Model -> Step ->
Result path, not general forming validation. The plane is fixed, the solid
has ordinary strong supports, and only one Coulomb pair is present.
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


def run_case(
    *, cells: int = 1, increments: int = 4, tangential_load: float = 100.0,
) -> dict[str, object]:
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("The fixed-plane implicit friction example is serial only.")
    if cells < 1 or increments < 1:
        raise ValueError("cells and increments must be positive integers.")
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, cells, cells, cells)
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True),
        mesh=domain, name="fixed_plane_friction",
    )
    displacement = model.field(fields.displacement(domain))
    support = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 1.0), name="support",
    )
    slave = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="contact_face",
    )
    model.clamp(displacement, on=support)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=0.0, yield_stress=1.0e9,
        hardening_modulus=100.0,
    ))
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        slave,
        boundary_models.rigid_body(boundary_models.rigid_plane(
            point=(0.05, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
        )),
        penalty=500.0, friction_coefficient=0.4,
        tangential_penalty=100.0, name="fixed_frictional_plane",
    ))
    model.traction((100.0, tangential_load, 0.0), on=slave)
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
    contact = step.contact_provider.current_point_diagnostics()
    return {
        "status": result.status,
        "cell_count": int(domain.topology.index_map(3).size_local),
        "accepted_increments": len(step.accepted_increments),
        "active_contact_points": contact["active_point_count"],
        "contact_potential": last.contact_potential,
        "friction_dissipation": last.friction_dissipation,
        "friction_separation_release": last.friction_separation_release,
        "force_on_structure": result.metadata["constraint_duals"][0]["force"],
        "energy_residual": last.mechanical_energy_residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=1)
    parser.add_argument("--increments", type=int, default=4)
    parser.add_argument("--tangential-load", type=float, default=100.0)
    options = parser.parse_args()
    print(json.dumps(run_case(
        cells=options.cells, increments=options.increments,
        tangential_load=options.tangential_load,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
