# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-input whole-Step screen for reviewed closed-tool sign methods.

This research benchmark includes finite-J2 Newton, material updates, and
contact assembly. It is not an independent scientific validation or a
cross-machine speed promise. Run from the source tree with ``PYTHONPATH=src``.
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter

from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np

from agentfem import (
    boundary_models, constitutive, fields, mesh, models, solvers, steps, studies,
)

from benchmark_closed_solid_projection import _refined_l_prism


def _tool(levels: int, sign_method: str):
    reference = _refined_l_prism(levels)
    vertices = reference.facets.vertices
    # A proper cyclic axis permutation maps the research prism to the same
    # physical L-tool used by the public manufactured example.
    positioned = np.column_stack((
        1.005 + 0.5 * vertices[:, 2],
        0.7 * vertices[:, 0] - 0.35,
        0.7 * vertices[:, 1] - 0.51,
    ))
    return boundary_models.closed_triangulated_rigid_solid(
        vertices=positioned, triangles=reference.facets.triangles,
        reviewed_embedded=True, name="benchmark_refined_L_prism",
        sign_method=sign_method,
    )


def _step(tool, *, cells: int, increments: int):
    domain = dolfinx_mesh.create_unit_cube(
        MPI.COMM_SELF, cells, cells, cells,
        cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right")
    model.clamp(displacement, on=left)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=0.0, yield_stress=0.05,
        hardening_modulus=100.0,
    ))
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
        penalty=20.0,
    ))
    return model.step(
        target=displacement, material=material,
        incrementation=steps.fixed(increments),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8, absolute_tolerance=1.0e-9,
            maximum_iterations=25, line_search="backtracking",
        ),
        progress=False,
    )


def _measure(tool, *, cells: int, increments: int, repeats: int):
    durations = []
    snapshots = []
    for _ in range(repeats):
        step = _step(tool, cells=cells, increments=increments)
        start = perf_counter()
        result = step.solve_result()
        durations.append(perf_counter() - start)
        if result.status != "completed":
            raise RuntimeError(f"Whole-Step screen did not complete: {result.status}")
        samples = step.contact_provider.accepted_point_samples()
        snapshots.append({
            "solution": step.solution.x.array.copy(),
            "state": step.response.state.committed_state_vectors().copy(),
            "tool_work": float(step.energy_history[-1].tool_work),
            "active_count": int(np.count_nonzero(samples["active"])),
            "increments": len(step.accepted_increments),
        })
    for snapshot in snapshots[1:]:
        np.testing.assert_allclose(snapshot["solution"], snapshots[0]["solution"])
        np.testing.assert_allclose(snapshot["state"], snapshots[0]["state"])
        assert snapshot["tool_work"] == snapshots[0]["tool_work"]
    return float(np.median(durations)), snapshots[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--cells", type=int, default=1)
    parser.add_argument("--increments", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1:
        parser.error("This candidate is serial-only")
    if args.levels < 0 or min(args.cells, args.increments, args.repeats) < 1:
        parser.error("levels must be nonnegative; other counts must be positive")
    reference = _tool(args.levels, "winding_reference")
    fast = _tool(args.levels, "angle_weighted_pseudonormal")
    slow_time, slow = _measure(
        reference, cells=args.cells, increments=args.increments,
        repeats=args.repeats,
    )
    fast_time, accelerated = _measure(
        fast, cells=args.cells, increments=args.increments,
        repeats=args.repeats,
    )
    np.testing.assert_allclose(accelerated["solution"], slow["solution"])
    np.testing.assert_allclose(accelerated["state"], slow["state"])
    assert accelerated["tool_work"] == slow["tool_work"]
    assert accelerated["active_count"] == slow["active_count"]
    assert accelerated["increments"] == slow["increments"]
    print(json.dumps({
        "scope": "fixed_input_serial_finite_j2_whole_step_screen",
        "facet_count": len(reference.facets.triangles),
        "cells_per_axis": args.cells,
        "increments": args.increments,
        "repeats": args.repeats,
        "reference_median_seconds": slow_time,
        "pseudonormal_median_seconds": fast_time,
        "speedup_ratio": slow_time / fast_time,
        "active_count": slow["active_count"],
        "tool_work": slow["tool_work"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
