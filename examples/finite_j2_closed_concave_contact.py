# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded serial J2 contact with a reviewed reentrant rigid-tool edge.

This manufactured unit-cube fixture tests ordinary ``model.step()`` and
accepted result output, not Abaqus equivalence or large-tool performance.
The explicit ``reviewed_embedded=True`` attestation applies only to the
constructed L-prism below; importing an arbitrary STL needs separate review.

Run from an installed AgentFEM environment::

    python examples/finite_j2_closed_concave_contact.py --increments 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np

from agentfem import (
    boundary_models, constitutive, fields, mesh, models, solvers, steps, studies,
)


def _reviewed_l_prism():
    polygon = np.asarray(
        ((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)), dtype=float,
    )
    polygon = 0.7 * polygon + np.asarray((-0.35, -0.51))
    front = np.column_stack((np.full(6, 1.005), polygon))
    back = front.copy()
    back[:, 0] += 0.5
    end_faces = ((0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5))
    triangles = [tuple(reversed(face)) for face in end_faces]
    triangles.extend(tuple(6 + vertex for vertex in face) for face in end_faces)
    for first in range(6):
        second = (first + 1) % 6
        triangles.extend(((first, second, second + 6),
                          (first, second + 6, first + 6)))
    return boundary_models.closed_triangulated_rigid_solid(
        vertices=np.vstack((front, back)), triangles=triangles,
        reviewed_embedded=True, name="reviewed_L_prism",
    )


def run_case(*, cells: int, increments: int, penalty: float, output: Path):
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("This bounded implicit closed-tool example is serial only.")
    if cells < 1 or increments < 1 or not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("Cells, increments and penalty must be positive and finite.")
    domain = dolfinx_mesh.create_unit_cube(
        MPI.COMM_SELF, cells, cells, cells,
        cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain,
        name="finite_j2_closed_concave_contact",
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right")
    model.clamp(displacement, on=left)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=0.0, yield_stress=0.05,
        hardening_modulus=100.0,
    ))
    tool = _reviewed_l_prism()
    motion = boundary_models.prescribed_rigid_motion(
        translation=(-0.055, 0.0, 0.0), reference_point=(1.005, 0.0, 0.0),
    )
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        right,
        boundary_models.rigid_body(
            tool,
            motion_schedule=boundary_models.prescribed_rigid_motion_schedule(
                motion, start_time=0.0, end_time=1.0,
            ),
        ),
        penalty=penalty,
    ))
    step = model.step(
        target=displacement, material=material,
        incrementation=steps.fixed(increments),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8, absolute_tolerance=1.0e-9,
            maximum_iterations=25, line_search="backtracking",
        ),
        progress=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    result = step.solve_result(output=output / "fields.xdmf")
    result.write_manifest(output / "result.json")
    samples = step.contact_provider.accepted_point_samples()
    foot = samples["master_closest_points"]
    # The L-prism's known reentrant edge has constant y=0.35, z=0.19.
    active_edge_count = int(np.count_nonzero(
        samples["active"]
        & np.isclose(foot[:, 1], 0.35, atol=1.0e-8)
        & np.isclose(foot[:, 2], 0.19, atol=1.0e-8)
    ))
    dual = result.metadata["constraint_duals"][0]
    return {
        "status": result.status,
        "cells_per_axis": cells,
        "accepted_increments": len(step.accepted_increments),
        "penalty": penalty,
        "active_reentrant_edge_points": active_edge_count,
        "contact_resultant": dual["resultant"],
        "maximum_equivalent_plastic_strain": result.quantity(
            "maximum_equivalent_plastic_strain"
        ),
        "tool_work": step.energy_history[-1].tool_work,
        "mechanical_energy_residual": (
            step.energy_history[-1].mechanical_energy_residual
        ),
        "output": str(output),
        "scope": "serial_manufactured_fixture_not_external_validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=1)
    parser.add_argument("--increments", type=int, default=24)
    parser.add_argument("--penalty", type=float, default=20.0)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/finite_j2_closed_concave"),
    )
    args = parser.parse_args()
    print(json.dumps(run_case(
        cells=args.cells, increments=args.increments,
        penalty=args.penalty, output=args.output,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
