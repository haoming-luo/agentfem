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


def run_case(
    *, cells: int, increments: int, penalty: float, output: Path,
    contact_quadrature_degree: int | None = None,
    line_search: str = "backtracking",
    linear_policy: str = "direct",
):
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("This bounded implicit closed-tool example is serial only.")
    if cells < 1 or increments < 1 or not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("Cells, increments and penalty must be positive and finite.")
    if line_search not in {"backtracking", "basic"}:
        raise ValueError("This example requires backtracking or basic line search.")
    if linear_policy not in {"direct", "gmres_gamg"}:
        raise ValueError("This example requires direct or gmres_gamg linear policy.")
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
    material_law = constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=0.0, yield_stress=0.05,
        hardening_modulus=100.0,
    )
    material = model.material(material_law)
    tool = _reviewed_l_prism()
    motion = boundary_models.prescribed_rigid_motion(
        translation=(-0.055, 0.0, 0.0), reference_point=(1.005, 0.0, 0.0),
    )
    contact_pair = boundary_models.rigid_contact_pair(
        right,
        boundary_models.rigid_body(
            tool,
            motion_schedule=boundary_models.prescribed_rigid_motion_schedule(
                motion, start_time=0.0, end_time=1.0,
            ),
        ),
        penalty=penalty,
    )
    model.add_boundary_model(contact_pair)
    incrementation = steps.fixed(increments)
    solver_options = solvers.newton(
        relative_tolerance=1.0e-8, absolute_tolerance=1.0e-9,
        maximum_iterations=25, line_search=line_search,
        linear_solver=(
            solvers.LinearSolverOptions(
                ksp_type="gmres", pc_type="gamg", rtol=1.0e-8, max_it=500,
            ) if linear_policy == "gmres_gamg" else None
        ),
    )
    step = model.step(
        target=displacement, material=material,
        incrementation=incrementation,
        **(
            {"contact_quadrature_degree": contact_quadrature_degree}
            if contact_quadrature_degree is not None else {}
        ),
        solver_options=solver_options,
        progress=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    result = step.solve_result(output=output / "fields.xdmf")
    result.add_scientific_inputs(
        mesh={"generator": "dolfinx_unit_cube_tetra", "cells_per_axis": cells},
        boundary={"left": "clamp_x_equal_zero", "right": "contact_x_equal_one"},
        material=material_law,
        contact=contact_pair,
        contact_integration={
            "method": "gauss",
            "triangle_degree": 2 if contact_quadrature_degree is None
            else int(contact_quadrature_degree),
            "measure": "reference_area",
        },
        incrementation=incrementation,
        solver=solver_options,
    )
    # This fixture is specifically used to audit accepted work/energy paths;
    # preserve their samples, not only their names and units.
    result.write_manifest(output / "result.json", include_histories=True)
    samples = step.contact_provider.accepted_point_samples()
    foot = samples["master_closest_points"]
    # The L-prism's known reentrant edge has constant y=0.35, z=0.19.
    active_edge_count = int(np.count_nonzero(
        samples["active"]
        & np.isclose(foot[:, 1], 0.35, atol=1.0e-8)
        & np.isclose(foot[:, 2], 0.19, atol=1.0e-8)
    ))
    dual = result.metadata["constraint_duals"][0]
    contact_diagnostics = result.metadata["rigid_contact_point_diagnostics"]
    return {
        "status": result.status,
        "cells_per_axis": cells,
        "accepted_increments": len(step.accepted_increments),
        "penalty": penalty,
        "line_search": line_search,
        "linear_policy": linear_policy,
        "contact_quadrature_degree": (
            2 if contact_quadrature_degree is None
            else contact_quadrature_degree
        ),
        "active_reentrant_edge_points": active_edge_count,
        "active_reference_area": contact_diagnostics["active_reference_area"],
        "maximum_penetration": contact_diagnostics["maximum_penetration"],
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
        "--line-search", choices=("backtracking", "basic"),
        default="backtracking",
        help="existing Newton policy; basic is for controlled sensitivity checks",
    )
    parser.add_argument(
        "--linear-policy", choices=("direct", "gmres_gamg"), default="direct",
        help="opt-in iterative solver for fixed-input timing and response checks",
    )
    parser.add_argument(
        "--contact-quadrature-degree", type=int, choices=(5, 7),
        help="opt-in positive tetrahedral face rule for sensitivity checks",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/finite_j2_closed_concave"),
    )
    args = parser.parse_args()
    print(json.dumps(run_case(
        cells=args.cells, increments=args.increments,
        penalty=args.penalty, output=args.output,
        contact_quadrature_degree=args.contact_quadrature_degree,
        line_search=args.line_search,
        linear_policy=args.linear_policy,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
