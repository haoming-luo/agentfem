"""Bounded public J2 plus fixed triangulated-tool integration checks."""

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


def _patch(*, triangulated: bool, gap: float = 0.005, penalty=500.0,
           cells=1, constrain_contact=False, tilt=0.0,
           tool_translation=None, tool_rotation=(0.0, 0.0, 0.0),
           schedule_end=1.0, make_step=True, tool_y_max=1.5):
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, cells, cells, cells)
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True),
        mesh=domain,
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
    tool_motion = None
    if tool_translation is not None:
        tool_motion = boundary_models.prescribed_rigid_motion(
            translation=tool_translation,
            rotation=tool_rotation,
            reference_point=(1.0 + gap, 0.0, 0.0),
        )
    if triangulated:
        tool_x = 1.0 + gap
        x_at = lambda y: tool_x + tilt * y
        surface = boundary_models.triangulated_rigid_surface(
            vertices=(
                (x_at(-0.5), -0.5, -0.5),
                (x_at(-0.5), -0.5, 1.5),
                (x_at(tool_y_max), tool_y_max, 1.5),
                (x_at(tool_y_max), tool_y_max, -0.5),
            ),
            triangles=((0, 1, 2), (0, 2, 3)),
        )
        pair = boundary_models.rigid_contact_pair(
            right,
            boundary_models.rigid_body(
                surface,
                reference_point=(tool_x, 0, 0),
                motion_schedule=(
                    None if tool_motion is None
                    else boundary_models.prescribed_rigid_motion_schedule(
                        tool_motion, start_time=0, end_time=schedule_end,
                    )
                ),
            ),
            penalty=penalty,
        )
        model.add_boundary_model(pair)
    else:
        surface = boundary_models.rigid_plane(
            point=(1.0 + gap, 0, 0),
            normal=np.asarray((-1, tilt, 0)) / np.sqrt(1.0 + tilt * tilt),
        )
        model.rigid_obstacle_contact(
            on=right, penalty=penalty, surface=surface, motion=tool_motion,
        )
    if constrain_contact:
        model.fix(displacement, on=right, component=0, value=0.0)
    model.traction((10.0, 0.0, 0.0), on=right)
    if not make_step:
        return model, displacement, material, None
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


def test_non_normalized_tool_schedule_is_rejected_by_capability_and_step():
    model, displacement, material, _ = _patch(
        triangulated=True,
        tool_translation=(-0.006, 0.0, 0.0),
        schedule_end=2.0,
        make_step=False,
    )
    capability = step_capability(model, target=displacement, options={"material": material})
    assert not capability["supported"]
    assert any(
        item["message"].startswith("AFM-IMPLICIT-CONTACT-MOTION")
        for item in capability["rejection_reasons"]
    )
    with pytest.raises(NotImplementedError, match="AFM-IMPLICIT-CONTACT-MOTION"):
        model.step(target=displacement, material=material)


def test_active_free_tool_edge_fails_closed_and_preserves_accepted_state():
    _, _, _, step = _patch(
        triangulated=True,
        cells=2,
        tool_y_max=0.4,
        tool_translation=(-0.006, 0.0, 0.0),
    )
    baseline_u = step.solution.x.array.copy()
    baseline_state = step.response.state.committed_state_vectors().copy()
    with pytest.raises((ValueError, RuntimeError), match="AFM-IMPLICIT-CONTACT-BRANCH"):
        step.solve()
    assert step.accepted_load_factor == 0.0
    assert float(step.contact_provider.motion_factor.value) == 0.0
    np.testing.assert_allclose(step.solution.x.array, baseline_u)
    np.testing.assert_allclose(step.response.state.committed_state_vectors(),
                               baseline_state)
    assert not step.energy_history


@pytest.mark.parametrize("cells,penalty", [(1, 250.0), (1, 500.0), (1, 1000.0), (2, 500.0)])
def test_fixed_triangulated_plane_matches_analytic_plane_global_j2(cells, penalty):
    model, displacement, material, facet_step = _patch(
        triangulated=True, cells=cells, penalty=penalty,
    )
    capability = step_capability(model, target=displacement, options={"material": material})
    assert capability["supported"]
    facet_result = facet_step.solve_result()
    assert facet_result.status == "completed"
    assert facet_step.contact_provider.evaluate().active_points > 0
    plane_model, plane_displacement, plane_material, plane_step = _patch(
        triangulated=False, cells=cells, penalty=penalty,
    )
    plane_result = plane_step.solve_result()
    np.testing.assert_allclose(
        facet_step.solution.x.array,
        plane_step.solution.x.array,
        rtol=2e-4,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        facet_step.energy_history[-1].contact_potential,
        plane_step.energy_history[-1].contact_potential,
        rtol=1e-4,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        facet_step.response.state.committed_state_vectors(),
        plane_step.response.state.committed_state_vectors(),
        rtol=2e-4,
        atol=2e-6,
    )
    balance = facet_result.metadata["constraint_balance_contract"]
    assert balance["force_balance_available"]
    assert balance["work_balance_available"]
    assert plane_result.status == "completed"


@pytest.mark.parametrize("cells", [1, 2])
def test_tilted_fixed_tool_matches_analytical_plane_with_nonuniform_contact(cells):
    # The tilted gap creates a spatially varying active set and facet crossings.
    # This is a discrete two-implementation comparison, not an independent
    # curved-tool or experimentally measured validation.
    _, _, _, facet = _patch(triangulated=True, cells=cells, tilt=0.02)
    _, _, _, plane = _patch(triangulated=False, cells=cells, tilt=0.02)
    facet.solve()
    plane.solve()
    np.testing.assert_allclose(facet.solution.x.array, plane.solution.x.array,
                               rtol=3e-4, atol=3e-6)
    np.testing.assert_allclose(
        facet.response.state.committed_state_vectors(),
        plane.response.state.committed_state_vectors(),
        rtol=3e-4, atol=3e-6,
    )
    assert facet.energy_history[-1].contact_potential == pytest.approx(
        plane.energy_history[-1].contact_potential, rel=2e-4,
    )


@pytest.mark.parametrize("tool_rotation", [
    (0.0, 0.0, 0.0), (0.0, 0.0, 0.01), (0.0, 0.0, 0.05),
    (0.0, 0.0, 0.2),
])
def test_moving_faceted_tool_matches_analytical_plane_force_work_and_state(tool_rotation):
    options = dict(tool_translation=(-0.006, 0.0, 0.0),
                   tool_rotation=tool_rotation, cells=2)
    _, _, _, facet = _patch(triangulated=True, **options)
    _, _, _, plane = _patch(triangulated=False, **options)
    facet_result = facet.solve_result()
    plane_result = plane.solve_result()
    assert facet_result.status == plane_result.status == "completed"
    np.testing.assert_allclose(facet.solution.x.array, plane.solution.x.array,
                               rtol=3e-4, atol=3e-6)
    np.testing.assert_allclose(
        facet.response.state.committed_state_vectors(),
        plane.response.state.committed_state_vectors(),
        rtol=3e-4, atol=3e-6,
    )
    assert facet.energy_history[-1].tool_work == pytest.approx(
        plane.energy_history[-1].tool_work, rel=3e-4, abs=1e-8,
    )
    assert abs(facet.energy_history[-1].tool_work) > 1e-8
    assert facet.energy_history[-1].contact_potential == pytest.approx(
        plane.energy_history[-1].contact_potential, rel=3e-4, abs=1e-8,
    )
    facet_dual = facet_result.metadata["constraint_duals"][0]
    plane_dual = plane_result.metadata["constraint_duals"][0]
    balance = facet_result.metadata["constraint_balance_contract"]
    assert balance["force_balance_available"]
    assert balance["work_balance_available"]
    np.testing.assert_allclose(facet_dual["force"], plane_dual["force"],
                               rtol=3e-4, atol=1e-6)


@pytest.mark.parametrize("rotation", [0.01, 0.2])
def test_moving_facet_tool_potential_derivative_and_atomic_restart(
    tmp_path, monkeypatch, rotation,
):
    options = dict(triangulated=True, cells=2,
                   tool_translation=(-0.006, 0.0, 0.0),
                   tool_rotation=(0.0, 0.0, rotation))
    _, _, _, reference = _patch(**options)
    reference.solve()
    motion = reference.contact_provider.pair.rigid_body.motion_schedule.motion
    generalized_force = reference._contact_dual().force
    direction = motion.generalized_coordinate(1.0)
    delta = 1e-6
    try:
        reference.contact_provider.update_motion(1.0 - delta)
        before = reference._contact_potential()
    finally:
        reference.contact_provider.update_motion(1.0)
    derivative = (reference._contact_potential() - before) / delta
    assert derivative == pytest.approx(float(np.dot(generalized_force, direction)),
                                       rel=2e-4)

    _, _, _, partial = _patch(**options)
    partial.solve(until=1.0 / 3.0)
    saved = partial.save_checkpoint(tmp_path / "moving_facet")
    accepted_pose = float(partial.contact_provider.motion_factor.value)
    accepted_u = partial.solution.x.array.copy()
    accepted_state = partial.response.state.committed_state_vectors().copy()

    def fail_energy():
        raise ValueError("intentional moving facet energy failure")

    monkeypatch.setattr(partial, "_contact_potential", fail_energy)
    with pytest.raises(RuntimeError, match="intentional moving facet energy failure"):
        partial.solve(until=2.0 / 3.0)
    assert float(partial.contact_provider.motion_factor.value) == pytest.approx(accepted_pose)
    np.testing.assert_allclose(partial.solution.x.array, accepted_u)
    np.testing.assert_allclose(partial.response.state.committed_state_vectors(),
                               accepted_state)
    _, _, _, resumed = _patch(**options)
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(resumed.solution.x.array, reference.solution.x.array,
                               rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        resumed.response.state.committed_state_vectors(),
        reference.response.state.committed_state_vectors(),
        rtol=1e-8, atol=1e-10,
    )
    assert resumed.energy_history[-1].tool_work == pytest.approx(
        reference.energy_history[-1].tool_work, rel=1e-8,
    )


def test_facet_contact_rejects_shared_strong_dofs_early():
    # The operator currently has no contact-tangent lifting on constrained rows.
    with pytest.raises(NotImplementedError, match="AFM-IMPLICIT-CONTACT-BC"):
        _patch(triangulated=True, constrain_contact=True)


def test_facet_j2_checkpoint_and_failed_increment_are_atomic(tmp_path, monkeypatch):
    _, _, _, reference = _patch(triangulated=True)
    reference.solve()
    _, _, _, partial = _patch(triangulated=True)
    partial.solve(until=1.0 / 3.0)
    saved = partial.save_checkpoint(tmp_path / "facet_contact_restart")
    _, _, _, resumed = _patch(triangulated=True)
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(resumed.solution.x.array, reference.solution.x.array, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        resumed.response.state.committed_state_vectors(),
        reference.response.state.committed_state_vectors(),
        rtol=1e-8,
        atol=1e-10,
    )
    assert resumed.energy_history[-1].contact_potential == pytest.approx(
        reference.energy_history[-1].contact_potential, rel=1e-8,
    )
    _, _, _, changed = _patch(triangulated=True, penalty=600)
    with pytest.raises(ValueError, match="identity"):
        changed.load_checkpoint(saved)
    assert not changed.energy_history

    _, _, _, failed = _patch(triangulated=True)
    failed.solve(until=1.0 / 3.0)
    accepted_u = failed.solution.x.array.copy()
    accepted_material = failed.response.state.committed_state_vectors().copy()
    accepted_energy = tuple(failed.energy_history)

    def fail_energy():
        raise ValueError("intentional facet energy failure")

    monkeypatch.setattr(failed, "_contact_potential", fail_energy)
    with pytest.raises(RuntimeError, match="intentional facet energy failure"):
        failed.solve(until=2.0 / 3.0)
    assert failed.accepted_load_factor == pytest.approx(1.0 / 3.0)
    np.testing.assert_allclose(failed.solution.x.array, accepted_u)
    np.testing.assert_allclose(failed.response.state.committed_state_vectors(), accepted_material)
    assert tuple(failed.energy_history) == accepted_energy
