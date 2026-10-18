# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Small, dimensionless J2 example crossing a closed-convex tool feature.

The opt-in reference-face partition resolves a narrow active strip missed by
the default whole-face three-point rule. This is a serial software regression,
not a mesh/penalty-converged forming prediction or nonconvex CAD contact.

Run with an installed AgentFEM wheel and FEniCSx environment::

    python examples/finite_j2_convex_feature_partition.py --output /tmp/feature-case
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np

from agentfem import (
    boundary_models, constitutive, fields, mesh, models, results, solvers, steps,
    studies,
)


def _tool():
    x0, x1, y0, y1, z0, z1 = 0.985, 1.025, -1.0, 2.0, -1.0, 2.0
    return boundary_models.convex_polyhedral_rigid_surface(
        vertices=(
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ),
        triangles=(
            (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
            (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
        ),
    )


def run_case(output: Path) -> dict[str, object]:
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("The convex feature-partition example is serial only.")
    domain = dolfinx_mesh.create_unit_cube(
        MPI.COMM_SELF, 1, 1, 1, cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="clamp")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="tool_side")
    model.clamp(displacement, on=left)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=0.0, yield_stress=100.0,
        hardening_modulus=100.0,
    ))
    motion = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(
            translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.05),
            reference_point=(1.005, 0.5, 0.5),
        ),
        start_time=0.0, end_time=1.0,
    )
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        right,
        boundary_models.rigid_body(_tool(), motion_schedule=motion),
        penalty=200.0,
    ))
    step = model.step(
        target=displacement, material=material,
        contact_integration="convex_feature_partition",
        incrementation=steps.fixed(2),
        solver_options=solvers.newton(
            relative_tolerance=1e-8, absolute_tolerance=1e-9,
            maximum_iterations=25, line_search="basic",
        ),
        progress=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    result = step.solve_result(output=output / "fields.xdmf")
    result.write_manifest(output / "result.json")
    files = results.export_accepted_contact_points(step, output / "contact")
    diagnostics = step.contact_provider.current_point_diagnostics()
    return {
        "status": result.status,
        "scope": "serial_dimensionless_fixture_not_external_validation",
        "integration": "convex_feature_partition",
        "accepted_increments": len(step.accepted_increments),
        "split_slave_faces": diagnostics["split_slave_faces"],
        "active_feature_lines": diagnostics["active_feature_lines"],
        "contact_potential": result.quantity("rigid_contact_pair_contact_energy"),
        "tool_work": step.energy_history[-1].tool_work,
        "output": str(output),
        "contact_metadata": str(files["contact_point_metadata"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(run_case(options.output), sort_keys=True))


if __name__ == "__main__":
    main()
