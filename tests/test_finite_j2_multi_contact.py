"""Independent opposing rigid tools in the serial finite-J2 Procedure."""

from __future__ import annotations

from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np

from agentfem import (
    boundary_models, constitutive, fields, mesh, models, solvers, steps, studies,
)


def test_opposing_moving_planes_keep_distinct_force_work_and_state():
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2)
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    bottom = mesh.boundary(domain, lambda x: np.isclose(x[1], 0.0), name="bottom")
    model.clamp(displacement, on=bottom)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=0.3, yield_stress=1.0e9,
        hardening_modulus=100.0,
    ))
    for name, side, tool_x, normal, travel in (
        ("left_die", 0.0, -0.005, (1.0, 0.0, 0.0), (0.018, 0.0, 0.0)),
        ("right_die", 1.0, 1.005, (-1.0, 0.0, 0.0), (-0.018, 0.0, 0.0)),
    ):
        region = mesh.boundary(
            domain, lambda x, value=side: np.isclose(x[0], value), name=name,
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
        incrementation=steps.fixed(4),
        solver_options=solvers.newton(
            relative_tolerance=1e-8, absolute_tolerance=1e-9,
            maximum_iterations=30, line_search="backtracking",
        ),
        progress=False,
    )
    result = step.solve_result()
    assert result.status == "completed"
    assert len(step.contact_provider.providers) == 2
    names = ("left_die", "right_die")
    assert tuple(item["constraint_name"] for item in
                 result.metadata["constraint_duals"]) == names
    assert tuple(item["constraint"] for item in
                 result.metadata["rigid_contact_point_diagnostics"]) == names
    assert all(item["active_point_count"] > 0 for item in
               result.metadata["rigid_contact_point_diagnostics"])
    work = dict(step.energy_history[-1].pair_tool_work)
    assert all(work[name] > 0.0 for name in names)
    assert sum(work.values()) == step.energy_history[-1].tool_work
    forces = [
        np.asarray(item["resultant"], dtype=float)
        for item in result.metadata["constraint_duals"]
    ]
    assert forces[0][0] > 0.0 and forces[1][0] < 0.0
    # The clamped y=0 face can carry x reaction, so the two die forces need
    # not cancel one another. Balance the full residual instead.
    total_residual = np.sum(
        step._assembled_reaction_values().reshape(-1, 3), axis=0,
    )
    np.testing.assert_allclose(
        total_residual + forces[0] + forces[1], 0.0, atol=1e-7,
    )
