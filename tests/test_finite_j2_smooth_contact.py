"""Bounded 3D J2 with analytical curved rigid tools through the public Step."""

from __future__ import annotations

from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np
import pytest

from agentfem import (
    boundary_models,
    constitutive,
    fields,
    mesh,
    models,
    solvers,
    steps,
    studies,
)
from agentfem.step_providers import step_capability


def _curved_patch(surface, *, cells=1, motion=None,
                  cell_type=dolfinx_mesh.CellType.tetrahedron):
    domain = dolfinx_mesh.create_unit_cube(
        MPI.COMM_SELF, cells, cells, cells, cell_type=cell_type,
    )
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain
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
    schedule = (
        None
        if motion is None
        else boundary_models.prescribed_rigid_motion_schedule(
            motion,
            start_time=0.0,
            end_time=1.0,
        )
    )
    model.add_boundary_model(
        boundary_models.rigid_contact_pair(
            right,
            boundary_models.rigid_body(
                surface,
                reference_point=(1.6, 0.5, 0.5),
                motion_schedule=schedule,
            ),
            penalty=500.0,
        )
    )
    model.traction((10.0, 0.0, 0.0), on=right)
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(3),
        solver_options=solvers.newton(
            relative_tolerance=1e-8,
            absolute_tolerance=1e-9,
            maximum_iterations=25,
            line_search="backtracking",
        ),
        progress=False,
    )
    return model, displacement, material, step


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
@pytest.mark.parametrize(
    "cell_type", [dolfinx_mesh.CellType.tetrahedron, dolfinx_mesh.CellType.hexahedron],
)
def test_public_j2_step_assembles_curved_contact_force_and_curvature(
    kind, cell_type,
):
    surface = (
        boundary_models.rigid_sphere(center=(1.6, 0.5, 0.5), radius=0.75)
        if kind == "sphere"
        else boundary_models.rigid_cylinder(
            axis_point=(1.6, 0.5, 0),
            axis_direction=(0, 0, 1),
            radius=0.7,
        )
    )
    model, displacement, material, step = _curved_patch(
        surface, cell_type=cell_type,
    )
    assert step_capability(model, target=displacement, options={"material": material})[
        "supported"
    ]
    result = step.solve_result()
    assert result.status == "completed"
    contribution = step.contact_provider.evaluate()
    assert contribution.active_points > 0
    assert contribution.assembly.potential_energy > 0.0
    assert step.contact_provider.summary()["linearization"] == (
        "smooth_signed_gap_with_curvature_at_current_pose"
    )


def test_moving_sphere_pose_is_restored_after_failed_trial(monkeypatch):
    surface = boundary_models.rigid_sphere(center=(1.6, 0.5, 0.5), radius=0.75)
    motion = boundary_models.prescribed_rigid_motion(
        translation=(-0.01, 0.0, 0.0),
        rotation=(0.0, 0.1, 0.0),
        reference_point=(1.6, 0.5, 0.5),
    )
    _, _, _, step = _curved_patch(surface, motion=motion)
    step.solve(until=1 / 3)
    accepted_pose = float(step.contact_provider.motion_factor.value)
    accepted_u = step.solution.x.array.copy()
    accepted_material = step.response.state.committed_state_vectors().copy()
    generalized_force = step._contact_dual().force
    direction = motion.generalized_coordinate(1.0)
    delta = 1e-6
    try:
        step.contact_provider.update_motion(accepted_pose + delta)
        plus = step._contact_potential()
        step.contact_provider.update_motion(accepted_pose - delta)
        minus = step._contact_potential()
    finally:
        step.contact_provider.update_motion(accepted_pose)
    assert (plus - minus) / (2 * delta) == pytest.approx(
        float(np.dot(generalized_force, direction)), rel=2e-4,
    )

    def fail_energy():
        raise ValueError("injected smooth-contact energy failure")

    monkeypatch.setattr(step, "_contact_potential", fail_energy)
    with pytest.raises(RuntimeError, match="smooth-contact energy failure"):
        step.solve(until=2 / 3)
    assert float(step.contact_provider.motion_factor.value) == pytest.approx(
        accepted_pose,
    )
    np.testing.assert_allclose(step.solution.x.array, accepted_u)
    np.testing.assert_allclose(
        step.response.state.committed_state_vectors(),
        accepted_material,
    )
