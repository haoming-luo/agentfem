from __future__ import annotations

import json

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import (
    boundary_models,
    checkpointing,
    constraints,
    constitutive,
    fields,
    loads,
    mesh,
    models,
    results,
    solvers,
    steps,
    studies,
)
from agentfem.step_providers import step_capability

from periodic_cube_fixture import periodic_unit_cube


def test_periodic_j2_rejects_registered_contact_in_capability_and_step():
    fixture = periodic_unit_cube(MPI.COMM_SELF)
    model = models.create(
        study=studies.nonlinear_static(physics="solid_mechanics", dimension=3),
        mesh=fixture.domain,
    )
    displacement = model.field(fields.displacement(fixture.domain))
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=200_000.0,
            poisson=0.3,
            yield_stress=200.0,
            hardening_modulus=2_000.0,
        )
    )
    periodicity = model.constraint(fixture.constraint(displacement))
    right = mesh.boundary(
        fixture.domain,
        lambda x: np.isclose(x[0], 1.0),
        name="periodic_contact_right",
    )
    model.rigid_obstacle_contact(
        on=right,
        penalty=1_000.0,
        surface=boundary_models.rigid_plane(
            point=(1.1, 0.0, 0.0),
            normal=(-1.0, 0.0, 0.0),
        ),
    )
    capability = step_capability(
        model,
        target=displacement,
        options={"material": material, "constraints": periodicity},
    )
    assert not capability["supported"]
    assert any(
        item["message"].startswith("AFM-J2-CONTACT-MPC")
        for item in capability["rejection_reasons"]
    )
    with pytest.raises(NotImplementedError, match="AFM-J2-CONTACT-MPC"):
        model.step(target=displacement, material=material, constraints=periodicity)


def _periodic_j2_contact_step(
    *, surface=None, motion=None, pair_count=1, opposed_pair=False,
    friction_coefficient=None, tangential_penalty=None, checkpoint=None,
    linear_solver=None, contact_integration="gauss",
):
    fixture = periodic_unit_cube(MPI.COMM_SELF)
    model = models.create(
        study=studies.nonlinear_static(physics="solid_mechanics", dimension=3),
        mesh=fixture.domain,
    )
    displacement = model.field(fields.displacement(fixture.domain))
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=200_000.0, poisson=0.3, yield_stress=200.0,
        hardening_modulus=2_000.0,
    ))
    periodicity = model.constraint(fixture.constraint(displacement))
    right = mesh.boundary(
        fixture.domain, lambda x: np.isclose(x[0], 1.0),
        name="periodic_contact_right",
    )
    if surface is None:
        surface = boundary_models.rigid_sphere(
            center=(1.6, 0.5, 0.5), radius=0.75,
        )
    for index in range(pair_count):
        pair = boundary_models.rigid_contact_pair(
            right,
            boundary_models.rigid_body(
                surface,
                reference_point=(1.6, 0.5, 0.5),
                motion_schedule=(
                    None if motion is None else
                    boundary_models.prescribed_rigid_motion_schedule(
                        motion, start_time=0.0, end_time=1.0,
                    )
                ),
            ),
            penalty=5_000.0 / pair_count,
            friction_coefficient=friction_coefficient,
            tangential_penalty=tangential_penalty,
            name=("rigid_contact_pair" if pair_count == 1 else f"periodic_tool_{index}"),
        )
        model.add_boundary_model(pair)
    if opposed_pair:
        left = mesh.boundary(
            fixture.domain, lambda x: np.isclose(x[0], 0.0),
            name="periodic_contact_left",
        )
        model.add_boundary_model(boundary_models.rigid_contact_pair(
            left,
            boundary_models.rigid_body(boundary_models.rigid_sphere(
                center=(-0.6, 0.5, 0.5), radius=0.75,
            )),
            penalty=5_000.0, name="left_sphere",
        ))
    capability = step_capability(
        model, target=displacement,
        options={
            "material": material, "constraints": periodicity,
            "contact_integration": contact_integration,
        },
    )
    assert capability["supported"]
    step = model.step(
        target=displacement, material=material, constraints=periodicity,
        contact_integration=contact_integration,
        incrementation=steps.fixed(3), progress=False,
        solver_options=solvers.newton(
            relative_tolerance=1e-8, absolute_tolerance=1e-9,
            maximum_iterations=25, line_search="backtracking",
            linear_solver=linear_solver,
        ),
        checkpoint=checkpoint,
    )
    return fixture, periodicity, step


def test_periodic_j2_fixed_sphere_uses_total_residual_and_mpc_reduction():
    from dolfinx import fem
    import dolfinx.fem.petsc as fem_petsc
    from petsc4py import PETSc

    fixture, periodicity, step = _periodic_j2_contact_step()
    assert len(step.full_space_operators) == 1
    result = step.solve_result()
    assert result.status == "completed"
    assert result.metadata["constraint_balance_contract"]["work_balance_available"]
    assert len(result.metadata["constraint_duals"]) == 2
    assert result.quantity("contact_potential") > 0.0
    contribution = step.contact_provider.evaluate()
    assert contribution.active_points > 0
    material_residual = fem_petsc.assemble_vector(fem.form(step.residual_form))
    material_residual.ghostUpdate(
        addv=PETSc.InsertMode.ADD,
        mode=PETSc.ScatterMode.REVERSE,
    )
    total_residual = step.assemble_full_residual()
    try:
        np.testing.assert_allclose(
            total_residual.array - material_residual.array,
            contribution.assembly.nodal_structural_residual.reshape(-1),
            rtol=1e-12, atol=1e-10,
        )
        transformation = periodicity.reduction(1.0).matrix(fixture.domain.comm)
        projected = transformation.createVecRight()
        try:
            transformation.multTranspose(total_residual, projected)
            assert projected.norm() < 1e-5
        finally:
            projected.destroy()
            transformation.destroy()
    finally:
        material_residual.destroy()
        total_residual.destroy()


def test_periodic_j2_smooth_plane_uses_the_same_contact_operator():
    surface = boundary_models.rigid_plane(
        point=(0.95, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
    )
    _, _, step = _periodic_j2_contact_step(surface=surface)
    result = step.solve_result()
    assert result.status == "completed"
    assert step.contact_provider.evaluate().active_points > 0
    assert result.quantity("contact_potential") > 0.0


def test_periodic_j2_convex_flat_face_matches_plane_total_mpc_operator():
    x0, x1, y0, y1, z0, z1 = 0.95, 2.0, -1.0, 2.0, -1.0, 2.0
    solid = boundary_models.convex_polyhedral_rigid_surface(
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
    plane = boundary_models.rigid_plane(
        point=(x0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
    )
    _, periodicity, convex_step = _periodic_j2_contact_step(surface=solid)
    _, _, planar_step = _periodic_j2_contact_step(surface=plane)
    convex_result = convex_step.solve_result()
    planar_result = planar_step.solve_result()
    assert convex_result.status == planar_result.status == "completed"
    feature_path = convex_result.metadata["rigid_contact_feature_path"]
    # All accepted stations were observed, but the Gauss contact integral is
    # still *not* certified. The release claim must remain failed.
    assert feature_path["complete"]
    assert feature_path["accepted_station_count"] == (
        len(convex_step.accepted_increments) + 1
    )
    assert feature_path["pairs"][0]["accepted_station_count"] == (
        len(convex_step.accepted_increments) + 1
    )
    assert not feature_path["pairs"][0]["all_observed_integrals_certified"]
    feature_claim = next(
        item for item in convex_result.verify("engineering").claims
        if item.name == "no_observed_uncertified_contact_feature_transition"
    )
    assert feature_claim.status == "passed"
    assert not feature_claim.actual["issues"]
    release_claim = next(
        item for item in convex_result.verify("release").claims
        if item.name == "polyhedral_contact_integral_certified"
    )
    assert release_claim.status == "failed"
    assert "rigid_contact_feature_path" not in planar_result.metadata
    np.testing.assert_allclose(
        convex_step.solution.x.array, planar_step.solution.x.array,
        rtol=1.0e-8, atol=1.0e-9,
    )
    assert convex_result.quantity("contact_potential") == pytest.approx(
        planar_result.quantity("contact_potential"), rel=1.0e-8,
    )
    residual = convex_step.assemble_full_residual()
    try:
        transformation = periodicity.reduction(1.0).matrix(
            convex_step.solution.function_space.mesh.comm,
        )
        projected = transformation.createVecRight()
        try:
            transformation.multTranspose(residual, projected)
            assert projected.norm() < 1.0e-5
        finally:
            projected.destroy()
            transformation.destroy()
    finally:
        residual.destroy()


def test_periodic_j2_fixed_contact_checkpoint_matches_continuous_run(tmp_path):
    _, _, reference = _periodic_j2_contact_step()
    reference.solve()

    _, _, partial = _periodic_j2_contact_step()
    partial.solve(until=1 / 3)
    checkpoint = partial.save_checkpoint(tmp_path / "periodic_contact_restart")

    _, _, resumed = _periodic_j2_contact_step()
    resumed.load_checkpoint(checkpoint)
    resumed.solve()
    np.testing.assert_allclose(
        resumed.solution.x.array, reference.solution.x.array,
        rtol=2e-8, atol=2e-10,
    )
    np.testing.assert_allclose(
        resumed.response.state.committed_state_vectors(),
        reference.response.state.committed_state_vectors(),
        rtol=2e-8, atol=2e-10,
    )
    assert resumed.constraint_dual_history == reference.constraint_dual_history
    assert resumed.checkpoint_capabilities().rank_count_portability == "unsupported"


def test_periodic_j2_contact_checkpoint_rejects_changed_tool(tmp_path):
    _, _, partial = _periodic_j2_contact_step()
    partial.solve(until=1 / 3)
    checkpoint = partial.save_checkpoint(tmp_path / "contact_tool_identity")
    changed_surface = boundary_models.rigid_sphere(
        center=(1.6, 0.5, 0.5), radius=0.7,
    )
    _, _, changed = _periodic_j2_contact_step(surface=changed_surface)
    with pytest.raises(ValueError, match="identity"):
        changed.load_checkpoint(checkpoint)
    assert changed.accepted_load_factor == 0.0


def test_periodic_j2_contact_projection_failure_restores_accepted_state(monkeypatch):
    _, _, step = _periodic_j2_contact_step()
    step.solve(until=1 / 3)
    accepted_u = step.solution.x.array.copy()
    accepted_material = step.response.state.committed_state_vectors().copy()
    accepted_duals = step.constraint_dual_history.snapshot_runtime_state()

    def fail_projection():
        raise ValueError("injected invalid current projection")

    monkeypatch.setattr(step.contact_provider, "evaluate", fail_projection)
    with pytest.raises(ValueError, match="invalid current projection"):
        step.solve(until=2 / 3)
    assert step.accepted_load_factor == pytest.approx(1 / 3)
    np.testing.assert_allclose(step.solution.x.array, accepted_u)
    np.testing.assert_allclose(
        step.response.state.committed_state_vectors(), accepted_material,
    )
    assert step.constraint_dual_history.snapshot_runtime_state() == accepted_duals
    monkeypatch.undo()
    step.solve(until=2 / 3)
    assert step.accepted_load_factor == pytest.approx(2 / 3)


def test_periodic_j2_contact_tangent_failure_restores_accepted_state(monkeypatch):
    _, _, step = _periodic_j2_contact_step()
    step.solve(until=1 / 3)
    accepted_u = step.solution.x.array.copy()
    accepted_material = step.response.state.committed_state_vectors().copy()

    def fail_tangent(_matrix, _contribution):
        raise ValueError("injected contact tangent failure")

    monkeypatch.setattr(step.contact_provider, "add_tangent", fail_tangent)
    with pytest.raises(ValueError, match="contact tangent failure"):
        step.solve(until=2 / 3)
    assert step.accepted_load_factor == pytest.approx(1 / 3)
    np.testing.assert_allclose(step.solution.x.array, accepted_u)
    np.testing.assert_allclose(
        step.response.state.committed_state_vectors(), accepted_material,
    )
    monkeypatch.undo()
    step.solve(until=2 / 3)
    assert step.accepted_load_factor == pytest.approx(2 / 3)


def test_periodic_j2_contact_tangent_matches_reduced_directional_difference():
    fixture, periodicity, step = _periodic_j2_contact_step()
    transformation = periodicity.reduction(1.0).matrix(fixture.domain.comm)
    direction = transformation.createVecRight()
    full_direction = transformation.createVecLeft()
    projected = transformation.createVecRight()
    full_residual = transformation.createVecLeft()
    reference = step.solution.x.array.copy()
    try:
        direction.array[:] = np.linspace(0.1, 0.3, direction.array.size)
        transformation.mult(direction, full_direction)
        tangent = step.contact_provider.evaluate()
        tangent_action = tangent.tangent_action(
            full_direction.array.reshape(-1, 3),
        ).reshape(-1)
        full_residual.array[:] = tangent_action
        transformation.multTranspose(full_residual, projected)
        expected = projected.array.copy()
        scale = 1e-6
        samples = []
        for sign in (1.0, -1.0):
            step.solution.x.array[:] = reference + sign * scale * full_direction.array
            step.solution.x.scatter_forward()
            full_residual.array[:] = (
                step.contact_provider.evaluate()
                .assembly.nodal_structural_residual.reshape(-1)
            )
            transformation.multTranspose(full_residual, projected)
            samples.append(projected.array.copy())
        np.testing.assert_allclose(
            (samples[0] - samples[1]) / (2 * scale), expected,
            rtol=1e-7, atol=1e-7,
        )
    finally:
        step.solution.x.array[:] = reference
        step.solution.x.scatter_forward()
        full_residual.destroy()
        projected.destroy()
        full_direction.destroy()
        direction.destroy()
        transformation.destroy()


def test_periodic_j2_fixed_triangle_uses_the_same_full_space_operator():
    surface = boundary_models.triangulated_rigid_surface(
        vertices=(
            (0.95, -1.0, -1.0), (0.95, -1.0, 2.0),
            (0.95, 2.0, 2.0), (0.95, 2.0, -1.0),
        ),
        triangles=((0, 1, 2), (0, 2, 3)),
    )
    _, _, step = _periodic_j2_contact_step(surface=surface)
    step.solve()
    assert step.contact_provider.evaluate().active_points > 0
    assert step.last_solve_info.increments[-1].converged


def test_periodic_j2_closed_convex_tool_preserves_planar_mpc_limit():
    x0, x1, y0, y1, z0, z1 = 0.95, 1.25, -1.0, 2.0, -1.0, 2.0
    surface = boundary_models.convex_polyhedral_rigid_surface(
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
    fixture, periodicity, step = _periodic_j2_contact_step(surface=surface)
    result = step.solve_result()
    assert result.status == "completed"
    assert step.contact_provider.evaluate().active_points > 0
    assert result.quantity("contact_potential") > 0.0
    total = step.assemble_full_residual()
    reduction = periodicity.reduction(1.0).matrix(fixture.domain.comm)
    reduced = reduction.createVecRight()
    try:
        reduction.multTranspose(total, reduced)
        assert reduced.norm() < 1.0e-5
    finally:
        reduced.destroy()
        reduction.destroy()
        total.destroy()

    plane = boundary_models.rigid_plane(
        point=(x0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
    )
    _, _, planar = _periodic_j2_contact_step(surface=plane)
    planar.solve()
    np.testing.assert_allclose(
        step.solution.x.array, planar.solution.x.array,
        rtol=1e-7, atol=1e-8,
    )


def _rotated_convex_periodic_tool():
    center = np.asarray((1.005, 0.5, 0.5))
    theta = 0.05
    rotation = np.asarray(((np.cos(theta), -np.sin(theta), 0),
                           (np.sin(theta), np.cos(theta), 0),
                           (0, 0, 1)))
    x0, x1, y0, y1, z0, z1 = 0.98, 1.03, -1.0, 2.0, -1.0, 2.0
    vertices = np.asarray((
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ))
    return boundary_models.convex_polyhedral_rigid_surface(
        vertices=center + (vertices - center) @ rotation.T,
        triangles=(
            (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
            (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
        ),
    )


def test_periodic_j2_convex_feature_partition_reduces_total_residual():
    fixture, periodicity, step = _periodic_j2_contact_step(
        surface=_rotated_convex_periodic_tool(),
        contact_integration="convex_feature_partition",
    )
    assert step.solve_result().status == "completed"
    assert step.contact_provider.current_point_diagnostics()["active_feature_lines"] > 0
    total = step.assemble_full_residual()
    reduction = periodicity.reduction(1.0).matrix(fixture.domain.comm)
    reduced = reduction.createVecRight()
    try:
        reduction.multTranspose(total, reduced)
        assert reduced.norm() < 1e-5
    finally:
        reduced.destroy()
        reduction.destroy()
        total.destroy()


def _periodic_concave_face_tool(*, wide_smooth_patch=False):
    x0, x1 = (0.95, 3.0) if wide_smooth_patch else (0.95, 1.25)
    polygon = np.asarray((
        (0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2),
    ), dtype=float)
    polygon = (
        4.0 * polygon - 2.0 if wide_smooth_patch
        else 1.75 * polygon - 0.5
    )
    front = np.column_stack((np.full(6, x0), polygon))
    back = np.column_stack((np.full(6, x1), polygon))
    top = ((0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5))
    triangles = [tuple(reversed(face)) for face in top]
    triangles += [tuple(6 + index for index in face) for face in top]
    for first in range(6):
        second = (first + 1) % 6
        triangles.extend(((first, second, second + 6),
                          (first, second + 6, first + 6)))
    return boundary_models.closed_triangulated_rigid_solid(
        vertices=np.vstack((front, back)), triangles=triangles,
        reviewed_embedded=True,
    )


def test_periodic_j2_reviewed_concave_tool_reduces_total_contact_residual():
    surface = _periodic_concave_face_tool()
    fixture, periodicity, step = _periodic_j2_contact_step(surface=surface)
    result = step.solve_result()
    assert result.status == "completed"
    assert result.quantity(
        f"{step.contact_provider.name}_sampled_mixed_feature_slave_faces"
    ) >= 0
    assert step.contact_provider.evaluate().active_points > 0
    total = step.assemble_full_residual()
    reduction = periodicity.reduction(1.0).matrix(fixture.domain.comm)
    reduced = reduction.createVecRight()
    try:
        reduction.multTranspose(total, reduced)
        assert reduced.norm() < 1.0e-5
    finally:
        reduced.destroy()
        reduction.destroy()
        total.destroy()
    plane = boundary_models.rigid_plane(
        point=(0.95, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
    )
    _, _, planar = _periodic_j2_contact_step(surface=plane)
    planar.solve()
    np.testing.assert_allclose(
        step.solution.x.array, planar.solution.x.array,
        rtol=1e-7, atol=1e-8,
    )


def test_periodic_j2_certified_closed_smooth_branch_reduces_total_residual(tmp_path):
    fixture, periodicity, step = _periodic_j2_contact_step(
        surface=_periodic_concave_face_tool(wide_smooth_patch=True),
        contact_integration="closed_planar_branch_partition",
    )
    result = step.solve_result()
    assert result.status == "completed"
    evidence = step.contact_provider.current_point_diagnostics()
    assert evidence["feature_transition_integral_certified"] is True
    assert evidence["active_point_count"] > 0
    total = step.assemble_full_residual()
    reduction = periodicity.reduction(1.0).matrix(fixture.domain.comm)
    reduced = reduction.createVecRight()
    try:
        reduction.multTranspose(total, reduced)
        assert reduced.norm() < 1.0e-5
    finally:
        reduced.destroy()
        reduction.destroy()
        total.destroy()

    plane = boundary_models.rigid_plane(
        point=(0.95, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
    )
    _, _, planar = _periodic_j2_contact_step(surface=plane)
    assert planar.solve_result().status == "completed"
    np.testing.assert_allclose(
        step.solution.x.array, planar.solution.x.array,
        rtol=1e-7, atol=1e-8,
    )

    _, _, partial = _periodic_j2_contact_step(
        surface=_periodic_concave_face_tool(wide_smooth_patch=True),
        contact_integration="closed_planar_branch_partition",
    )
    partial.solve(until=1.0 / 3.0)
    saved = partial.save_checkpoint(tmp_path / "periodic_closed_smooth_branch")
    _, _, resumed = _periodic_j2_contact_step(
        surface=_periodic_concave_face_tool(wide_smooth_patch=True),
        contact_integration="closed_planar_branch_partition",
    )
    resumed.load_checkpoint(saved)
    assert resumed.solve_result().status == "completed"
    np.testing.assert_allclose(
        resumed.solution.x.array, step.solution.x.array,
        rtol=1e-9, atol=1e-10,
    )


def test_periodic_j2_closed_smooth_branch_rejects_reentrant_edge():
    _, _, step = _periodic_j2_contact_step(
        surface=_active_reentrant_periodic_surface(),
        contact_integration="closed_planar_branch_partition",
    )
    with pytest.raises(NotImplementedError, match="AFM-CONTACT-CLOSED-PLANAR-BRANCH"):
        step.solve_result()
    assert step.accepted_load_factor == 0.0


def _active_reentrant_periodic_surface():
    """Put a reentrant line near, but not on, a periodic face quadrature point."""

    polygon = 0.7 * np.asarray((
        (0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2),
    ), dtype=float) + np.asarray((-0.34, -0.26))
    front = np.column_stack((np.full(6, 0.95), polygon))
    back = np.column_stack((np.full(6, 1.25), polygon))
    top = ((0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5))
    triangles = [tuple(reversed(face)) for face in top]
    triangles += [tuple(6 + index for index in face) for face in top]
    for first in range(6):
        second = (first + 1) % 6
        triangles.extend(((first, second, second + 6),
                          (first, second + 6, first + 6)))
    return boundary_models.closed_triangulated_rigid_solid(
        vertices=np.vstack((front, back)), triangles=triangles,
        reviewed_embedded=True,
    )


def test_periodic_j2_active_reentrant_edge_reduces_total_tangent():
    # A planar-limit test cannot verify the signed edge Hessian in the MPC.
    surface = _active_reentrant_periodic_surface()
    fixture, periodicity, step = _periodic_j2_contact_step(surface=surface)
    assert step.solve_result().status == "completed"
    _, record, response, _ = step.contact_provider._current_point_response()
    features = [
        surface._geometric_feature_projector(
            surface._facet_slots[int(identity)], barycentric,
        )[1]
        for identity, barycentric in zip(
            record.projection.entity_ids,
            record.projection.local_coordinates, strict=True,
        )
    ]
    assert any(active and feature == "edge" for active, feature in zip(
        response.active, features, strict=True,
    ))
    total = step.assemble_full_residual()
    reduction = periodicity.reduction(1.0).matrix(fixture.domain.comm)
    reduced = reduction.createVecRight()
    try:
        reduction.multTranspose(total, reduced)
        assert reduced.norm() < 1.0e-5
    finally:
        reduced.destroy()
        reduction.destroy()
        total.destroy()

    transformation = periodicity.reduction(1.0).matrix(fixture.domain.comm)
    direction = transformation.createVecRight()
    full_direction = transformation.createVecLeft()
    projected = transformation.createVecRight()
    full_residual = transformation.createVecLeft()
    accepted = step.solution.x.array.copy()
    try:
        direction.array[:] = np.linspace(0.1, 0.3, direction.array.size)
        transformation.mult(direction, full_direction)
        selected = step.contact_provider.evaluate()
        full_residual.array[:] = selected.tangent_action(
            full_direction.array.reshape(-1, 3),
        ).reshape(-1)
        transformation.multTranspose(full_residual, projected)
        expected = projected.array.copy()
        epsilon = 1.0e-7
        samples = []
        for sign in (1.0, -1.0):
            step.solution.x.array[:] = accepted + sign * epsilon * full_direction.array
            step.solution.x.scatter_forward()
            full_residual.array[:] = (
                step.contact_provider.evaluate()
                .assembly.nodal_structural_residual.reshape(-1)
            )
            transformation.multTranspose(full_residual, projected)
            samples.append(projected.array.copy())
        np.testing.assert_allclose(
            expected, (samples[0] - samples[1]) / (2.0 * epsilon),
            rtol=2.0e-4, atol=2.0e-5,
        )
    finally:
        step.solution.x.array[:] = accepted
        step.solution.x.scatter_forward()
        full_residual.destroy()
        projected.destroy()
        full_direction.destroy()
        direction.destroy()
        transformation.destroy()


def test_periodic_j2_opposed_active_tools_include_reentrant_edge_and_restart(
    tmp_path,
):
    surface = _active_reentrant_periodic_surface()
    fixture, periodicity, continuous = _periodic_j2_contact_step(
        surface=surface, opposed_pair=True,
    )
    result = continuous.solve_result()
    assert result.status == "completed"
    feature_path = result.metadata["rigid_contact_feature_path"]
    assert feature_path["complete"]
    assert feature_path["accepted_station_count"] == (
        len(continuous.accepted_increments) + 1
    )
    assert feature_path["pairs"][0]["maximum_active_edge_points"] > 0
    assert next(
        item for item in result.verify("engineering").claims
        if item.name == "no_observed_uncertified_contact_feature_transition"
    ).status == "failed"
    assert len(continuous.contact_providers) == 2
    assert all(provider.evaluate().active_points > 0
               for provider in continuous.contact_providers)
    provider = continuous.contact_providers[0]
    _, record, response, _ = provider._current_point_response()
    features = [
        surface._geometric_feature_projector(
            surface._facet_slots[int(identity)], barycentric,
        )[1]
        for identity, barycentric in zip(
            record.projection.entity_ids,
            record.projection.local_coordinates, strict=True,
        )
    ]
    assert any(active and feature == "edge" for active, feature in zip(
        response.active, features, strict=True,
    ))
    names = tuple(item.name for item in continuous.contact_providers)
    assert tuple(
        item["constraint_name"] for item in result.metadata["constraint_duals"]
        if item["role"] == "contact_constraint"
    ) == names
    reduction = periodicity.reduction(1.0).matrix(fixture.domain.comm)
    total = continuous.assemble_full_residual()
    reduced = reduction.createVecRight()
    try:
        reduction.multTranspose(total, reduced)
        assert reduced.norm() < 1.0e-5
    finally:
        reduced.destroy()
        total.destroy()
        reduction.destroy()

    _, _, partial = _periodic_j2_contact_step(
        surface=surface, opposed_pair=True,
    )
    partial.solve(until=2.0 / 3.0)
    saved = partial.save_checkpoint(tmp_path / "opposed_reentrant_tools")
    _, _, resumed = _periodic_j2_contact_step(
        surface=surface, opposed_pair=True,
    )
    resumed.load_checkpoint(saved)
    resumed_result = resumed.solve_result()
    assert resumed_result.status == "completed"
    assert resumed_result.metadata["rigid_contact_feature_path"] == feature_path
    np.testing.assert_allclose(
        resumed.solution.x.array, continuous.solution.x.array,
        rtol=1.0e-9, atol=1.0e-10,
    )
    np.testing.assert_allclose(
        resumed.state_transaction.response.state.committed_state_vectors(),
        continuous.state_transaction.response.state.committed_state_vectors(),
        rtol=1.0e-9, atol=1.0e-10,
    )


def test_periodic_polyhedral_feature_path_legacy_and_corrupt_restart(tmp_path):
    surface = _active_reentrant_periodic_surface()
    _, _, partial = _periodic_j2_contact_step(surface=surface)
    partial.solve(until=2.0 / 3.0)
    saved = partial.save_checkpoint(tmp_path / "polyhedral_feature_path")
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["accepted_observer_state"]["contact_features"]["complete"]

    damaged = json.loads(json.dumps(payload))
    damaged["accepted_observer_state"]["contact_features"]["pairs"][0][
        "constraint"
    ] = "wrong_pair"
    damaged_path = tmp_path / "damaged_feature_path.checkpoint.json"
    damaged_path.write_text(json.dumps(damaged), encoding="utf-8")
    _, _, rejected = _periodic_j2_contact_step(surface=surface)
    before = rejected.solution.x.array.copy()
    with pytest.raises(ValueError, match="pair coverage"):
        rejected.load_checkpoint(damaged_path)
    np.testing.assert_array_equal(rejected.solution.x.array, before)
    assert rejected.accepted_history_recorders["contact_features"].pairs == {}

    # Simulate a real older penalty checkpoint: its scientific identity and
    # observer state lacked this new ledger. It may resume, but its unobserved
    # earlier stations can never be reported as a complete contact path.
    legacy = json.loads(json.dumps(payload))
    legacy["identity"]["accepted_history_recorders"].pop("contact_features")
    legacy["accepted_observer_state"].pop("contact_features")
    legacy_path = tmp_path / "legacy_feature_path.checkpoint.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    changed_identity = json.loads(json.dumps(legacy))
    changed_identity["identity"]["step_name"] = "different_scientific_step"
    changed_path = tmp_path / "legacy_wrong_step.checkpoint.json"
    changed_path.write_text(json.dumps(changed_identity), encoding="utf-8")
    _, _, incompatible = _periodic_j2_contact_step(surface=surface)
    with pytest.raises(ValueError, match="material, state, mesh"):
        incompatible.load_checkpoint(changed_path)
    assert incompatible.accepted_history_recorders["contact_features"].pairs == {}
    _, _, resumed = _periodic_j2_contact_step(surface=surface)
    resumed.load_checkpoint(legacy_path)
    recorder = resumed.accepted_history_recorders["contact_features"]
    assert not recorder.complete
    assert recorder.accepted_station_count == len(resumed.accepted_increments) + 1
    result = resumed.solve_result()
    assert result.status == "completed"
    assert not result.metadata["rigid_contact_feature_path"]["complete"]
    claim = next(
        item for item in result.verify("engineering").claims
        if item.name == "no_observed_uncertified_contact_feature_transition"
    )
    assert claim.status == "failed"
    assert "accepted contact feature path is incomplete" in claim.actual["issues"]
    resaved = resumed.save_checkpoint(tmp_path / "resaved_incomplete_path")
    _, _, reloaded = _periodic_j2_contact_step(surface=surface)
    reloaded.load_checkpoint(resaved)
    assert not reloaded.accepted_history_recorders["contact_features"].complete


def test_periodic_polyhedral_feature_path_rolls_back_after_late_accept_failure(
    monkeypatch,
):
    _, _, step = _periodic_j2_contact_step(
        surface=_active_reentrant_periodic_surface(),
    )
    recorder = step.accepted_history_recorders["contact_features"]
    before = recorder.snapshot_runtime_state()
    initial_solution = step.solution.x.array.copy()
    original = step._capture_operator_duals

    def fail_after_feature_capture(factor):
        if factor > 0:
            raise RuntimeError("injected post-observer dual failure")
        return original(factor)

    monkeypatch.setattr(step, "_capture_operator_duals", fail_after_feature_capture)
    with pytest.raises(RuntimeError, match="post-observer dual failure"):
        step.solve()
    assert recorder.snapshot_runtime_state() == before
    assert step.accepted_load_factor == 0.0
    np.testing.assert_array_equal(step.solution.x.array, initial_solution)


@pytest.mark.parametrize(
    "with_friction, convex_partition",
    ((False, False), (False, True), (True, False)),
)
def test_periodic_j2_total_material_contact_jacobian(
    with_friction, convex_partition,
):
    import dolfinx.fem as fem
    import dolfinx.fem.petsc as fem_petsc

    options = {}
    if with_friction:
        surface = boundary_models.rigid_plane(
            point=(1.01, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
        )
        options = {
            "motion": boundary_models.prescribed_rigid_motion(
                translation=(0.0, 0.02, 0.0),
                reference_point=(1.6, 0.5, 0.5),
            ),
            "friction_coefficient": 0.2,
            "tangential_penalty": 1000.0,
        }
    elif convex_partition:
        surface = _rotated_convex_periodic_tool()
        options = {"contact_integration": "convex_feature_partition"}
    else:
        surface = _active_reentrant_periodic_surface()
    _, _, completed = _periodic_j2_contact_step(surface=surface, **options)
    completed.solve()
    fixture, periodicity, trial = _periodic_j2_contact_step(
        surface=surface, **options,
    )
    trial.solve(until=2.0 / 3.0)
    # Hold the fifth/previous accepted material state fixed while testing
    # the final increment. Perturbing the already committed final J2 state
    # at zero reload would measure a branch kink, not the Newton tangent.
    trial.solution.x.array[:] = completed.solution.x.array
    trial.solution.x.scatter_forward()
    if with_friction:
        trial.contact_provider.update_motion(1.0)
    trial.state_transaction.refresh_trial(
        start_factor=2.0 / 3.0, target_factor=1.0,
    )
    transformation = periodicity.reduction(1.0).matrix(fixture.domain.comm)
    direction = transformation.createVecRight()
    full_direction = transformation.createVecLeft()
    reduced = transformation.createVecRight()
    full_matrix = None
    reduced_matrix = None
    action = None
    try:
        direction.array[:] = np.linspace(0.1, 0.3, direction.array.size)
        transformation.mult(direction, full_direction)
        full_matrix = fem_petsc.assemble_matrix(fem.form(trial.jacobian_form))
        for operator in trial.full_space_operators:
            operator.add_tangent(full_matrix, operator.evaluate())
        full_matrix.assemble()
        reduced_matrix = full_matrix.PtAP(transformation)
        action = reduced_matrix.createVecLeft()
        reduced_matrix.mult(direction, action)
        expected = action.array.copy()

        def reduced_residual(offset):
            trial.solution.x.array[:] = (
                completed.solution.x.array + offset * full_direction.array
            )
            trial.solution.x.scatter_forward()
            trial.state_transaction.refresh_trial(
                start_factor=2.0 / 3.0, target_factor=1.0,
            )
            full = trial.assemble_full_residual()
            try:
                transformation.multTranspose(full, reduced)
                return reduced.array.copy()
            finally:
                full.destroy()

        epsilon = 1.0e-6
        numerical = (
            reduced_residual(epsilon) - reduced_residual(-epsilon)
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            expected, numerical, rtol=3.0e-3, atol=2.0e-4,
        )
    finally:
        trial.solution.x.array[:] = completed.solution.x.array
        trial.solution.x.scatter_forward()
        trial.state_transaction.rollback_increment(accepted_factor=2.0 / 3.0)
        if with_friction:
            trial.contact_provider.update_motion(2.0 / 3.0)
        if action is not None:
            action.destroy()
        if reduced_matrix is not None:
            reduced_matrix.destroy()
        if full_matrix is not None:
            full_matrix.destroy()
        reduced.destroy()
        full_direction.destroy()
        direction.destroy()
        transformation.destroy()


def _periodic_tool_motion():
    return boundary_models.prescribed_rigid_motion(
        translation=(-0.01, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        reference_point=(1.6, 0.5, 0.5),
    )


def test_periodic_j2_moving_plane_friction_commits_and_restarts(tmp_path):
    surface = boundary_models.rigid_plane(
        point=(1.01, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
    )
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.0, 0.02, 0.0),
        reference_point=(1.6, 0.5, 0.5),
    )
    options = dict(
        surface=surface, motion=motion,
        friction_coefficient=0.2, tangential_penalty=1000.0,
    )
    fixture, periodicity, continuous = _periodic_j2_contact_step(**options)
    result = continuous.solve_result()
    assert result.status == "completed"
    assert result.quantity("friction_dissipation") > 0.0
    assert result.metadata["affine_operator_path_work"]["status"] == "complete"
    assert result.quantity("rigid_contact_pair_path_work") != 0.0
    assert result.histories[
        "rigid_contact_pair_generalized_force"
    ].abscissa_unit is None
    assert result.histories[
        "rigid_contact_pair_generalized_coordinate"
    ].abscissa_unit is None
    point_diagnostics = result.metadata["rigid_contact_point_diagnostics"]
    assert point_diagnostics["location"] == "reference_boundary_quadrature"
    assert point_diagnostics["active_point_count"] > 0
    assert result.quantity("rigid_contact_pair_maximum_penetration") > 0.0
    assert continuous.contact_provider.friction_state.accepted is not None
    samples = continuous.contact_provider.accepted_point_samples()
    assert result.quantity("friction_dissipation") == pytest.approx(
        float(np.sum(
            samples["reference_weights"]
            * samples["cumulative_friction_dissipation_densities"]
        )),
        rel=1e-10,
    )
    assert continuous.operator_dual_history.complete(accepted_factor=1.0)
    reduction = periodicity.reduction(1.0).matrix(fixture.domain.comm)
    full = continuous.assemble_full_residual()
    reduced = reduction.createVecRight()
    reduction.multTranspose(full, reduced)
    assert reduced.norm() < 1.0e-5
    full.destroy()
    reduced.destroy()
    reduction.destroy()

    _, _, interrupted = _periodic_j2_contact_step(**options)
    interrupted.solve(until=1.0 / 3.0)
    # Differentiate the full contact Operator at a later current trial while
    # the preceding Coulomb State stays fixed. At the final accepted station
    # zero reloading would sit on a stick/slip cap kink.
    saved_solution = interrupted.solution.x.array.copy()
    interrupted.contact_provider.update_motion(1.0)
    interrupted.solution.x.array[:] = continuous.solution.x.array
    interrupted.solution.x.scatter_forward()
    reduction = periodicity.reduction(1.0).matrix(fixture.domain.comm)
    q = reduction.createVecRight()
    du = reduction.createVecLeft()
    residual = reduction.createVecLeft()
    reduced = reduction.createVecRight()
    try:
        q.array[:] = np.linspace(0.1, 0.3, q.array.size)
        reduction.mult(q, du)
        selected = interrupted.contact_provider.evaluate()
        residual.array[:] = selected.tangent_action(
            du.array.reshape(-1, 3),
        ).reshape(-1)
        reduction.multTranspose(residual, reduced)
        expected = reduced.array.copy()
        sampled = []
        for sign in (1.0, -1.0):
            interrupted.solution.x.array[:] = (
                continuous.solution.x.array + sign * 1e-7 * du.array
            )
            interrupted.solution.x.scatter_forward()
            residual.array[:] = (
                interrupted.contact_provider.evaluate()
                .nodal_structural_residual.reshape(-1)
            )
            reduction.multTranspose(residual, reduced)
            sampled.append(reduced.array.copy())
        np.testing.assert_allclose(
            (sampled[0] - sampled[1]) / (2e-7), expected,
            rtol=3e-4, atol=3e-5,
        )
    finally:
        interrupted.solution.x.array[:] = saved_solution
        interrupted.solution.x.scatter_forward()
        interrupted.contact_provider.update_motion(1.0 / 3.0)
        reduced.destroy()
        residual.destroy()
        du.destroy()
        q.destroy()
        reduction.destroy()
    saved = interrupted.save_checkpoint(tmp_path / "affine_friction")
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["schema"] == "agentfem.affine-stateful-checkpoint.v2"
    assert payload["operator_accepted_state"][0]["friction"] is not None
    changed_law = dict(options, friction_coefficient=0.25)
    _, _, incompatible = _periodic_j2_contact_step(**changed_law)
    with pytest.raises(ValueError, match="full-space Operator identity"):
        incompatible.load_checkpoint(saved)
    assert incompatible.contact_provider.friction_state.accepted is None
    damaged = dict(payload)
    damaged["operator_accepted_state"] = json.loads(
        json.dumps(payload["operator_accepted_state"])
    )
    damaged["operator_accepted_state"][0]["point_ids"][0] = -1
    damaged_path = saved.with_name("damaged_affine_friction.checkpoint.json")
    damaged_path.write_text(json.dumps(damaged), encoding="utf-8")
    _, _, rejected = _periodic_j2_contact_step(**options)
    before = rejected.solution.x.array.copy()
    with pytest.raises(ValueError, match="point IDs"):
        rejected.load_checkpoint(damaged_path)
    np.testing.assert_array_equal(rejected.solution.x.array, before)
    assert rejected.contact_provider.friction_state.accepted is None
    assert rejected.contact_provider.snapshot_motion() == pytest.approx(0.0)

    _, _, resumed = _periodic_j2_contact_step(**options)
    resumed.load_checkpoint(saved)
    resumed_result = resumed.solve_result()
    assert resumed_result.status == "completed"
    np.testing.assert_allclose(
        resumed.solution.x.array, continuous.solution.x.array,
        rtol=1e-9, atol=1e-10,
    )
    assert resumed.contact_provider.friction_state.accepted is not None
    assert resumed_result.quantity("friction_dissipation") == pytest.approx(
        result.quantity("friction_dissipation"), rel=1e-9,
    )


def test_periodic_j2_friction_rejects_symmetric_linear_solver_policy():
    surface = boundary_models.rigid_plane(
        point=(1.01, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="AFM-IMPLICIT-FRICTION-LINEAR"):
        _periodic_j2_contact_step(
            surface=surface,
            friction_coefficient=0.2,
            tangential_penalty=1000.0,
            linear_solver=solvers.LinearSolverOptions(
                ksp_type="cg", pc_type="jacobi",
            ),
        )


def test_periodic_j2_operator_state_protocol_rejects_unpaired_methods(
    monkeypatch, tmp_path,
):
    _, _, step = _periodic_j2_contact_step(
        surface=boundary_models.rigid_plane(
            point=(1.01, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
        ),
        friction_coefficient=0.2,
        tangential_penalty=1000.0,
    )
    initial_solution = step.solution.x.array.copy()
    with monkeypatch.context() as patch:
        patch.setattr(step.contact_provider, "restore_runtime_state", None)
        with pytest.raises(TypeError, match="paired runtime snapshot/restore"):
            step.solve()
    np.testing.assert_array_equal(step.solution.x.array, initial_solution)

    with monkeypatch.context() as patch:
        patch.setattr(step.contact_provider, "restore_accepted_state", None)
        with pytest.raises(TypeError, match="paired snapshot/restore"):
            step.save_checkpoint(tmp_path / "unpaired_operator")
    assert not list(tmp_path.glob("*.checkpoint.json"))


def test_periodic_j2_friction_gmres_matches_direct_linear_solve():
    options = dict(
        surface=boundary_models.rigid_plane(
            point=(1.01, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
        ),
        motion=boundary_models.prescribed_rigid_motion(
            translation=(0.0, 0.02, 0.0),
            reference_point=(1.6, 0.5, 0.5),
        ),
        friction_coefficient=0.2,
        tangential_penalty=1000.0,
    )
    _, _, direct = _periodic_j2_contact_step(**options)
    direct_result = direct.solve_result()
    _, _, iterative = _periodic_j2_contact_step(
        **options,
        linear_solver=solvers.LinearSolverOptions(
            ksp_type="gmres", pc_type="ilu", rtol=1e-10, max_it=200,
        ),
    )
    iterative_result = iterative.solve_result()
    assert direct_result.status == iterative_result.status == "completed"
    np.testing.assert_allclose(
        iterative.solution.x.array, direct.solution.x.array,
        rtol=1e-8, atol=1e-9,
    )
    assert iterative_result.quantity("friction_dissipation") == pytest.approx(
        direct_result.quantity("friction_dissipation"), rel=1e-8,
    )
    assert iterative_result.quantity("rigid_contact_pair_path_work") == pytest.approx(
        direct_result.quantity("rigid_contact_pair_path_work"), rel=1e-8,
    )


def test_periodic_j2_friction_public_result_writes_field_artifact(tmp_path):
    _, _, step = _periodic_j2_contact_step(
        surface=boundary_models.rigid_plane(
            point=(1.01, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
        ),
        motion=boundary_models.prescribed_rigid_motion(
            translation=(0.0, 0.02, 0.0),
            reference_point=(1.6, 0.5, 0.5),
        ),
        friction_coefficient=0.2,
        tangential_penalty=1000.0,
    )
    field_path = tmp_path / "fields.xdmf"
    result = step.solve_result(output=field_path, strict_output=True)
    assert result.status == "completed"
    assert field_path.exists()
    assert (tmp_path / "fields.h5").exists()
    assert result.quantity("friction_dissipation") > 0.0
    assert result.metadata["rigid_contact_point_diagnostics"][
        "active_point_count"
    ] > 0


def test_periodic_j2_friction_scheduled_checkpoint_retention_and_resume(tmp_path):
    surface = boundary_models.rigid_plane(
        point=(1.01, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
    )
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.0, 0.02, 0.0),
        reference_point=(1.6, 0.5, 0.5),
    )
    options = dict(
        surface=surface, motion=motion,
        friction_coefficient=0.2, tangential_penalty=1000.0,
    )
    policy = checkpointing.every(
        1, directory=tmp_path / "scheduled", keep_last=1,
    )
    _, _, continuous = _periodic_j2_contact_step(
        **options, checkpoint=policy,
    )
    reference_result = continuous.solve_result()
    assert reference_result.status == "completed"
    assert len(continuous.checkpoints) == 1
    saved = continuous.checkpoints[0].path
    assert saved.exists()
    assert len(list((tmp_path / "scheduled").glob("*.checkpoint.json"))) == 1
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["schema"] == "agentfem.affine-stateful-checkpoint.v2"

    _, _, resumed = _periodic_j2_contact_step(**options)
    resumed.load_checkpoint(saved)
    np.testing.assert_allclose(
        resumed.solution.x.array, continuous.solution.x.array,
        rtol=1e-9, atol=1e-10,
    )
    assert resumed.contact_provider.friction_state.accepted is not None
    samples = resumed.contact_provider.accepted_point_samples()
    assert float(np.sum(
        samples["reference_weights"]
        * samples["cumulative_friction_dissipation_densities"]
    )) == pytest.approx(
        reference_result.quantity("friction_dissipation"), rel=1e-9,
    )


def test_periodic_j2_friction_and_opposed_frictionless_tool_restart(tmp_path):
    surface = boundary_models.rigid_plane(
        point=(1.01, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
    )
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.0, 0.02, 0.0),
        reference_point=(1.6, 0.5, 0.5),
    )
    options = dict(
        surface=surface, motion=motion, opposed_pair=True,
        friction_coefficient=0.2, tangential_penalty=1000.0,
    )
    _, _, continuous = _periodic_j2_contact_step(**options)
    result = continuous.solve_result()
    assert result.status == "completed"
    assert len(continuous.contact_providers) == 2
    assert all(item.evaluate().active_points > 0
               for item in continuous.contact_providers)
    assert result.quantity("friction_dissipation") > 0.0
    point_diagnostics = result.metadata["rigid_contact_point_diagnostics"]
    assert tuple(item["constraint"] for item in point_diagnostics) == (
        "rigid_contact_pair", "left_sphere",
    )
    assert all(item["active_point_count"] > 0 for item in point_diagnostics)
    assert continuous.contact_providers[0].friction_state.accepted is not None
    assert continuous.contact_providers[1].friction_state is None

    _, _, partial = _periodic_j2_contact_step(**options)
    partial.solve(until=1.0 / 3.0)
    saved = partial.save_checkpoint(tmp_path / "mixed_contact_states")
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["schema"] == "agentfem.affine-stateful-checkpoint.v2"
    assert payload["operator_accepted_state"][0] is not None
    assert payload["operator_accepted_state"][1] is None
    damaged = json.loads(json.dumps(payload))
    damaged["operator_accepted_state"][1] = {"invalid": "frictionless_pair"}
    damaged_path = saved.with_name("mixed_contact_damaged.checkpoint.json")
    damaged_path.write_text(json.dumps(damaged), encoding="utf-8")
    _, _, rejected = _periodic_j2_contact_step(**options)
    rejected_u = rejected.solution.x.array.copy()
    with pytest.raises(ValueError, match="frictionless rigid pair"):
        rejected.load_checkpoint(damaged_path)
    np.testing.assert_array_equal(rejected.solution.x.array, rejected_u)
    assert rejected.contact_providers[0].friction_state.accepted is None
    assert rejected.contact_providers[1].friction_state is None
    assert rejected.contact_providers[0].snapshot_motion() == pytest.approx(0.0)

    _, _, resumed = _periodic_j2_contact_step(**options)
    resumed.load_checkpoint(saved)
    resumed_result = resumed.solve_result()
    np.testing.assert_allclose(
        resumed.solution.x.array, continuous.solution.x.array,
        rtol=1e-9, atol=1e-10,
    )
    assert resumed_result.quantity("friction_dissipation") == pytest.approx(
        result.quantity("friction_dissipation"), rel=1e-9,
    )


def test_periodic_mixed_contact_late_commit_failure_rolls_back_all_pairs(
    monkeypatch,
):
    surface = boundary_models.rigid_plane(
        point=(1.01, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
    )
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.0, 0.02, 0.0),
        reference_point=(1.6, 0.5, 0.5),
    )
    _, _, step = _periodic_j2_contact_step(
        surface=surface, motion=motion, opposed_pair=True,
        friction_coefficient=0.2, tangential_penalty=1000.0,
    )
    initial_solution = step.solution.x.array.copy()
    initial_material = step.state_transaction.response.state.committed_state_vectors().copy()

    def fail_second_pair_commit():
        raise RuntimeError("injected second pair commit failure")

    monkeypatch.setattr(
        step.contact_providers[1], "commit_accepted", fail_second_pair_commit,
    )
    with pytest.raises(RuntimeError, match="second pair commit failure"):
        step.solve()
    assert step.accepted_load_factor == 0.0
    np.testing.assert_array_equal(step.solution.x.array, initial_solution)
    np.testing.assert_array_equal(
        step.state_transaction.response.state.committed_state_vectors(),
        initial_material,
    )
    assert step.contact_providers[0].friction_state.accepted is None
    assert step.contact_providers[0].snapshot_motion() == pytest.approx(0.0)
    assert not step.accepted_increments


@pytest.mark.parametrize("tool_kind", ["triangle", "closed_concave"])
@pytest.mark.parametrize("rotation", [0.0, 0.02])
def test_periodic_moving_planar_face_friction_matches_analytical_plane(
    tool_kind, rotation,
):
    plane = boundary_models.rigid_plane(
        point=(0.95, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
    )
    if tool_kind == "triangle":
        face = boundary_models.triangulated_rigid_surface(
            vertices=(
                (0.95, -0.5, -0.5), (0.95, -0.5, 1.5),
                (0.95, 1.5, 1.5), (0.95, 1.5, -0.5),
            ),
            triangles=((0, 1, 2), (0, 2, 3)),
        )
    else:
        face = _periodic_concave_face_tool()
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.0, 0.02, 0.0),
        rotation=(0.0, 0.0, rotation),
        reference_point=(1.6, 0.5, 0.5),
    )
    options = dict(
        motion=motion, friction_coefficient=0.2,
        tangential_penalty=1000.0,
    )
    _, _, reference = _periodic_j2_contact_step(surface=plane, **options)
    reference_result = reference.solve_result()
    _, _, candidate = _periodic_j2_contact_step(surface=face, **options)
    candidate_result = candidate.solve_result()
    assert reference_result.status == candidate_result.status == "completed"
    np.testing.assert_allclose(
        candidate.solution.x.array, reference.solution.x.array,
        rtol=2e-7, atol=2e-8,
    )
    assert candidate_result.quantity("friction_dissipation") == pytest.approx(
        reference_result.quantity("friction_dissipation"), rel=2e-7,
    )
    assert candidate_result.quantity("rigid_contact_pair_path_work") == pytest.approx(
        reference_result.quantity("rigid_contact_pair_path_work"), rel=2e-7,
    )


def test_periodic_friction_post_commit_failure_restores_all_accepted_state(
    monkeypatch,
):
    surface = boundary_models.rigid_plane(
        point=(1.01, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
    )
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.0, 0.02, 0.0),
        reference_point=(1.6, 0.5, 0.5),
    )
    _, _, step = _periodic_j2_contact_step(
        surface=surface, motion=motion,
        friction_coefficient=0.2, tangential_penalty=1000.0,
    )
    before_u = step.solution.x.array.copy()
    before_material = step.state_transaction.response.state.committed_state_vectors().copy()
    before_kinematics = (
        step.contact_provider.friction_kinematics.accepted.slave_positions.copy()
    )
    original = step.contact_provider.commit_accepted

    def commit_then_fail():
        original()
        raise RuntimeError("injected affine friction post-commit failure")

    monkeypatch.setattr(step.contact_provider, "commit_accepted", commit_then_fail)
    with pytest.raises(RuntimeError, match="post-commit failure"):
        step.solve()
    assert step.accepted_load_factor == 0.0
    np.testing.assert_array_equal(step.solution.x.array, before_u)
    np.testing.assert_array_equal(
        step.state_transaction.response.state.committed_state_vectors(),
        before_material,
    )
    assert step.contact_provider.friction_state.accepted is None
    np.testing.assert_array_equal(
        step.contact_provider.friction_kinematics.accepted.slave_positions,
        before_kinematics,
    )
    assert step.contact_provider.snapshot_motion() == pytest.approx(0.0)
    assert not step.accepted_increments


def test_periodic_closed_sharp_edge_friction_rejects_without_partial_state():
    surface = _active_reentrant_periodic_surface()
    _, _, step = _periodic_j2_contact_step(
        surface=surface,
        friction_coefficient=0.2, tangential_penalty=1000.0,
    )
    before = step.solution.x.array.copy()
    before_material = step.state_transaction.response.state.committed_state_vectors().copy()
    with pytest.raises(ValueError, match="AFM-IMPLICIT-FRICTION-BRANCH"):
        step.solve()
    assert step.accepted_load_factor == 0.0
    np.testing.assert_array_equal(step.solution.x.array, before)
    np.testing.assert_array_equal(
        step.state_transaction.response.state.committed_state_vectors(),
        before_material,
    )
    assert step.contact_provider.friction_state.accepted is None


def test_periodic_j2_moving_tool_records_accepted_work_and_joint_restart(tmp_path):
    motion = _periodic_tool_motion()
    _, _, reference = _periodic_j2_contact_step(motion=motion)
    result = reference.solve_result()
    assert result.status == "completed"
    assert result.metadata["affine_operator_path_work"]["status"] == "complete"
    assert result.quantity("rigid_contact_pair_path_work") != 0.0
    assert result.metadata["constraint_balance_contract"]["work_balance_available"]
    assert reference.operator_dual_history.complete(accepted_factor=1.0)
    assert reference.contact_provider.motion_factor.value == pytest.approx(1.0)

    _, _, partial = _periodic_j2_contact_step(motion=motion)
    partial.solve(until=1 / 3)
    checkpoint = partial.save_checkpoint(tmp_path / "periodic_moving_contact")
    _, _, resumed = _periodic_j2_contact_step(motion=motion)
    resumed.load_checkpoint(checkpoint)
    assert resumed.contact_provider.motion_factor.value == pytest.approx(1 / 3)
    resumed.solve()
    np.testing.assert_allclose(
        resumed.solution.x.array, reference.solution.x.array,
        rtol=2e-8, atol=2e-10,
    )
    assert resumed.operator_dual_history.records == reference.operator_dual_history.records


def test_periodic_j2_two_moving_pairs_share_total_mpc_and_keep_named_work(tmp_path, monkeypatch):
    """Splitting one law checks additive MPC assembly, work and rollback."""

    motion = _periodic_tool_motion()
    _, _, single = _periodic_j2_contact_step(motion=motion)
    single_result = single.solve_result()
    _, _, double = _periodic_j2_contact_step(motion=motion, pair_count=2)
    result = double.solve_result()
    assert result.status == single_result.status == "completed"
    assert len(double.full_space_operators) == 2
    names = ("periodic_tool_0", "periodic_tool_1")
    np.testing.assert_allclose(
        double.solution.x.array, single.solution.x.array,
        rtol=2e-8, atol=2e-10,
    )
    assert result.quantity("contact_potential") == pytest.approx(
        single_result.quantity("contact_potential"), rel=1e-8,
    )
    assert tuple(item["constraint_name"] for item in
                 result.metadata["constraint_duals"] if item["constraint_name"] in names) == names
    assert result.metadata["constraint_balance_contract"]["work_balance_available"]
    assert result.metadata["affine_operator_path_work"]["status"] == "complete"
    assert sum(result.quantity(f"{name}_path_work") for name in names) == pytest.approx(
        single_result.quantity("rigid_contact_pair_path_work"), rel=1e-8,
    )

    _, _, partial = _periodic_j2_contact_step(motion=motion, pair_count=2)
    partial.solve(until=1 / 3)
    saved = partial.save_checkpoint(tmp_path / "periodic_two_moving_pairs")
    accepted = partial.solution.x.array.copy()
    duals = partial.operator_dual_history.snapshot_runtime_state()
    second = partial.contact_providers[1]

    def fail_projection():
        raise ValueError("injected second affine pair projection failure")

    monkeypatch.setattr(second, "evaluate", fail_projection)
    with pytest.raises(ValueError, match="second affine pair projection failure"):
        partial.solve(until=2 / 3)
    assert partial.accepted_load_factor == pytest.approx(1 / 3)
    np.testing.assert_allclose(partial.solution.x.array, accepted)
    assert partial.operator_dual_history.snapshot_runtime_state() == duals
    assert all(item.motion_factor.value == pytest.approx(1 / 3)
               for item in partial.contact_providers)

    _, _, resumed = _periodic_j2_contact_step(motion=motion, pair_count=2)
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(
        resumed.solution.x.array, double.solution.x.array,
        rtol=2e-8, atol=2e-10,
    )
    assert resumed.operator_dual_history.records == double.operator_dual_history.records


def test_periodic_j2_opposed_tools_reduce_the_total_residual():
    """Distinct opposed surfaces must both enter the exact MPC reduction."""

    fixture, periodicity, step = _periodic_j2_contact_step(opposed_pair=True)
    result = step.solve_result()
    assert result.status == "completed"
    assert len(step.full_space_operators) == 2
    contacts = {operator.name: operator.evaluate() for operator in step.full_space_operators}
    assert all(item.active_points > 0 for item in contacts.values())
    assert result.quantity("contact_potential") == pytest.approx(
        sum(item.assembly.potential_energy for item in contacts.values()),
    )
    duals = {
        item["constraint_name"]: item
        for item in result.metadata["constraint_duals"]
    }
    assert set(contacts).issubset(duals)
    assert duals["rigid_contact_pair"]["resultant"][0] < 0.0
    assert duals["left_sphere"]["resultant"][0] > 0.0
    total = step.assemble_full_residual()
    reduction = periodicity.reduction(1.0).matrix(fixture.domain.comm)
    reduced = reduction.createVecRight()
    try:
        reduction.multTranspose(total, reduced)
        assert reduced.norm() < 1.0e-5
    finally:
        reduced.destroy()
        reduction.destroy()
        total.destroy()


def test_periodic_j2_moving_tool_force_is_potential_conjugate():
    motion = _periodic_tool_motion()
    _, _, step = _periodic_j2_contact_step(motion=motion)
    step.solve()
    contact = step.contact_provider
    dual = contact.dual_evidence(step)
    final_energy = contact.evaluate().assembly.potential_energy
    increment = 1.0e-6
    try:
        contact.update_motion(1.0 - increment)
        previous_energy = contact.evaluate().assembly.potential_energy
    finally:
        contact.update_motion(1.0)
    path_direction = motion.generalized_coordinate(1.0)
    assert (final_energy - previous_energy) / increment == pytest.approx(
        float(np.dot(dual.force, path_direction)), rel=2.0e-4,
    )


def test_periodic_j2_moving_checkpoint_rejects_missing_work_path_atomically(tmp_path):
    motion = _periodic_tool_motion()
    _, _, partial = _periodic_j2_contact_step(motion=motion)
    partial.solve(until=1 / 3)
    checkpoint = partial.save_checkpoint(tmp_path / "moving_work_path")
    payload = json.loads(checkpoint.read_text())
    payload.pop("operator_dual_history")
    checkpoint.write_text(json.dumps(payload))

    _, _, resumed = _periodic_j2_contact_step(motion=motion)
    with pytest.raises(ValueError, match="accepted force--coordinate path"):
        resumed.load_checkpoint(checkpoint)
    assert resumed.accepted_load_factor == pytest.approx(0.0)
    assert resumed.contact_provider.motion_factor.value == pytest.approx(0.0)
    assert not resumed.operator_dual_history.records
    np.testing.assert_array_equal(resumed.solution.x.array, 0.0)


def test_periodic_j2_moving_tool_rolls_back_pose_after_projection_failure(monkeypatch):
    _, _, step = _periodic_j2_contact_step(motion=_periodic_tool_motion())
    step.solve(until=1 / 3)
    accepted = step.solution.x.array.copy()
    state = step.response.state.committed_state_vectors().copy()
    duals = step.operator_dual_history.snapshot_runtime_state()

    def fail_projection():
        raise ValueError("injected moving projection failure")

    monkeypatch.setattr(step.contact_provider, "evaluate", fail_projection)
    with pytest.raises(ValueError, match="injected moving projection failure"):
        step.solve(until=2 / 3)
    assert step.contact_provider.motion_factor.value == pytest.approx(1 / 3)
    assert step.accepted_load_factor == pytest.approx(1 / 3)
    np.testing.assert_allclose(step.solution.x.array, accepted)
    np.testing.assert_allclose(step.response.state.committed_state_vectors(), state)
    assert step.operator_dual_history.snapshot_runtime_state() == duals
    monkeypatch.undo()
    step.solve(until=2 / 3)
    assert step.contact_provider.motion_factor.value == pytest.approx(2 / 3)


def _integrate_material_path(material, deformation_gradient, increments):
    state = material.state_schema.initial_state()
    old_gradient = np.eye(3)
    response = None
    for index in range(1, increments + 1):
        factor = index / increments
        new_gradient = np.eye(3) + factor * (deformation_gradient - np.eye(3))
        response = material.update(
            constitutive.MaterialPointInput(
                deformation_gradient_old=old_gradient,
                deformation_gradient_new=new_gradient,
                time=factor,
                time_increment=1.0 / increments,
                properties=[],
                state_old=state,
                state_schema=material.state_schema,
            )
        )
        state = response.state_new.copy()
        old_gradient = new_gradient
    return response


def _integrate_gradient_history(material, coordinates, gradients):
    state = material.state_schema.initial_state()
    old_gradient = np.asarray(gradients[0], dtype=float)
    response = None
    for start, target, new_gradient in zip(
        coordinates[:-1],
        coordinates[1:],
        gradients[1:],
    ):
        response = material.update(
            constitutive.MaterialPointInput(
                deformation_gradient_old=old_gradient,
                deformation_gradient_new=np.asarray(new_gradient, dtype=float),
                time=float(target),
                time_increment=float(target - start),
                properties=[],
                state_old=state,
                state_schema=material.state_schema,
            )
        )
        state = response.state_new.copy()
        old_gradient = np.asarray(new_gradient, dtype=float)
    return response


def _material_first_piola_history(material, coordinates, gradients):
    state = material.state_schema.initial_state()
    old_gradient = np.asarray(gradients[0], dtype=float)
    first_piola = [np.zeros_like(old_gradient)]
    for start, target, new_gradient in zip(
        coordinates[:-1], coordinates[1:], gradients[1:]
    ):
        selected_gradient = np.asarray(new_gradient, dtype=float)
        response = material.update(
            constitutive.MaterialPointInput(
                deformation_gradient_old=old_gradient,
                deformation_gradient_new=selected_gradient,
                time=float(target),
                time_increment=float(target - start),
                properties=[],
                state_old=state,
                state_schema=material.state_schema,
            )
        )
        first_piola.append(
            np.linalg.det(selected_gradient)
            * response.cauchy_stress
            @ np.linalg.inv(selected_gradient).T
        )
        state = response.state_new.copy()
        old_gradient = selected_gradient
    return tuple(first_piola)


def _nonproportional_gradient_path(*, final_shear: float = 0.03):
    coordinates = (0.0, 0.35, 0.65, 1.0)
    loaded = np.diag((1.04, 1.0 / np.sqrt(1.04), 1.0 / np.sqrt(1.04)))
    unloaded = np.diag((1.01, 1.0 / np.sqrt(1.01), 1.0 / np.sqrt(1.01)))
    final = unloaded.copy()
    final[0, 1] = float(final_shear)
    gradients = (np.eye(3), loaded, unloaded, final)
    return coordinates, gradients


def _periodic_j2_step(
    *,
    stretch: float = 1.02,
    tangent_relative_step: float = 2.0e-6,
    relative_tolerance: float = 1.0e-8,
    checkpoint=None,
):
    fixture = periodic_unit_cube(MPI.COMM_SELF, stretch=stretch)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
        name="finite_strain_j2_periodic_restart",
    )
    displacement = model.field(fields.displacement(fixture.domain))
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=200_000.0,
            poisson=0.3,
            yield_stress=200.0,
            hardening_modulus=2_000.0,
            tangent_relative_step=tangent_relative_step,
        )
    )
    periodicity = model.constraint(fixture.constraint(displacement))
    step = model.step(
        target=displacement,
        material=material,
        constraints=periodicity,
        incrementation=steps.fixed(4),
        solver_options=solvers.newton(
            relative_tolerance=relative_tolerance,
            absolute_tolerance=1.0e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        checkpoint=checkpoint,
        progress=False,
    )
    return step


def _nonproportional_periodic_j2_step(*, incrementation):
    fixture = periodic_unit_cube(MPI.COMM_SELF)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
        name="finite_strain_j2_nonproportional_restart",
    )
    displacement = model.field(fields.displacement(fixture.domain))
    material = constitutive.finite_strain_j2_logarithmic(
        young=200_000.0,
        poisson=0.3,
        yield_stress=200.0,
        hardening_modulus=2_000.0,
    )
    material_record = model.material(material)
    coordinates, gradients = _nonproportional_gradient_path()
    path = constraints.deformation_gradient_path(
        coordinates=coordinates,
        gradients=gradients,
        name="tension_unload_shear",
    )
    periodicity = model.constraint(
        constraints.abaqus_periodic_cell(
            displacement,
            nodes=fixture.nodes,
            equations=fixture.equations,
            deformation_gradient_path=path,
            anchor_node=fixture.anchor_node,
            reference_nodes=fixture.reference_nodes,
        )
    )
    step = model.step(
        target=displacement,
        material=material_record,
        constraints=periodicity,
        incrementation=incrementation,
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        progress=False,
    )
    return step, material, path


def test_deformation_gradient_path_is_positive_and_fingerprint_complete():
    coordinates, gradients = _nonproportional_gradient_path()
    path = constraints.deformation_gradient_path(
        coordinates,
        gradients,
        name="tension_unload_shear",
    )

    np.testing.assert_allclose(path.at(0.0), np.eye(3))
    np.testing.assert_allclose(path.at(0.35), gradients[1])
    np.testing.assert_allclose(path.at(0.5), 0.5 * (gradients[1] + gradients[2]))
    assert path.summary()["coordinates"] == list(coordinates)
    assert len(path.summary()["fingerprint"]) == 64
    assert path.summary()["schema"] == "agentfem.deformation-gradient-path"

    near_identity = np.eye(3)
    near_identity[0, 0] += 5.0e-13
    canonical = constraints.deformation_gradient_path(
        (5.0e-13, 0.5, 1.0 + 5.0e-13),
        (near_identity, gradients[1], gradients[-1]),
    )
    assert canonical.coordinates[0] == 0.0
    assert canonical.coordinates[-1] == 1.0
    np.testing.assert_array_equal(canonical.at(0.0), np.eye(3))

    changed = list(gradients)
    changed[1] = np.asarray(changed[1]).copy()
    changed[1][0, 1] = 1.0e-3
    changed_path = constraints.deformation_gradient_path(coordinates, changed)
    assert changed_path.summary()["fingerprint"] != path.summary()["fingerprint"]

    with pytest.raises(ValueError, match="begin at the identity"):
        constraints.deformation_gradient_path(
            (0.0, 0.5, 1.0),
            (2.0 * np.eye(3), np.eye(3), np.eye(3)),
        )
    with pytest.raises(ValueError, match="at least one internal knot"):
        constraints.deformation_gradient_path(
            (0.0, 1.0),
            (np.eye(3), gradients[-1]),
        )
    endpoint_positive_but_segment_singular = np.diag((-2.0, -0.5, 1.0))
    assert np.linalg.det(endpoint_positive_but_segment_singular) > 0.0
    with pytest.raises(ValueError, match="positive determinant throughout segment"):
        constraints.deformation_gradient_path(
            (0.0, 0.5, 1.0),
            (
                np.eye(3),
                endpoint_positive_but_segment_singular,
                np.eye(3),
            ),
        )


def test_affine_constraint_legacy_positional_constructor_is_preserved():
    fixture = periodic_unit_cube(MPI.COMM_SELF)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
    )
    displacement = model.field(fields.displacement(fixture.domain))
    final_gradient = np.diag((1.01, 1.0 / np.sqrt(1.01), 1.0 / np.sqrt(1.01)))

    legacy = constraints.AbaqusPeriodicConstraint(
        displacement,
        fixture.nodes,
        fixture.equations,
        fixture.anchor_node,
        fixture.reference_nodes,
        final_gradient,
        None,
        2.0e-9,
        "legacy_positional",
    )
    keyword = constraints.abaqus_periodic_cell(
        displacement,
        nodes=fixture.nodes,
        equations=fixture.equations,
        anchor_node=fixture.anchor_node,
        reference_nodes=fixture.reference_nodes,
        deformation_gradient=final_gradient,
        tolerance=2.0e-9,
        name="legacy_positional",
    )

    assert legacy.deformation_gradient_path is None
    assert legacy.name == "legacy_positional"
    assert legacy.tolerance == pytest.approx(2.0e-9)
    assert legacy.scientific_identity() == keyword.scientific_identity()


def test_fixed_affine_path_cannot_skip_a_physical_gradient_knot():
    fixture = periodic_unit_cube(MPI.COMM_SELF)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
    )
    displacement = model.field(fields.displacement(fixture.domain))
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=200_000.0,
            poisson=0.3,
            yield_stress=200.0,
            hardening_modulus=2_000.0,
        )
    )
    coordinates, gradients = _nonproportional_gradient_path()
    periodicity = model.constraint(
        constraints.abaqus_periodic_cell(
            displacement,
            nodes=fixture.nodes,
            equations=fixture.equations,
            deformation_gradient_path=constraints.deformation_gradient_path(
                coordinates,
                gradients,
            ),
            anchor_node=fixture.anchor_node,
            reference_nodes=fixture.reference_nodes,
        )
    )
    step = model.step(
        target=displacement,
        material=material,
        constraints=periodicity,
        incrementation=steps.fixed(2),
        progress=False,
    )

    with pytest.raises(ValueError, match="deformation-path knot"):
        step.solve()


def test_public_affine_j2_consumes_unload_and_nonproportional_macro_path(tmp_path):
    fixture = periodic_unit_cube(MPI.COMM_SELF)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
        name="finite_strain_j2_nonproportional_path",
    )
    displacement = model.field(fields.displacement(fixture.domain))
    material = constitutive.finite_strain_j2_logarithmic(
        young=200_000.0,
        poisson=0.3,
        yield_stress=200.0,
        hardening_modulus=2_000.0,
    )
    material_record = model.material(material)
    coordinates, gradients = _nonproportional_gradient_path()
    path = constraints.deformation_gradient_path(
        coordinates,
        gradients,
        name="tension_unload_shear",
    )
    periodicity = model.constraint(
        constraints.abaqus_periodic_cell(
            displacement,
            nodes=fixture.nodes,
            equations=fixture.equations,
            deformation_gradient_path=path,
            anchor_node=fixture.anchor_node,
            reference_nodes=fixture.reference_nodes,
        )
    )
    step = model.step(
        target=displacement,
        material=material_record,
        constraints=periodicity,
        incrementation=steps.at(*coordinates[1:]),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        output_every=1,
        progress=False,
    )

    result = step.solve_result()
    expected = _integrate_gradient_history(material, coordinates, gradients)
    expected_first_piola = (
        np.linalg.det(gradients[-1])
        * expected.cauchy_stress
        @ np.linalg.inv(gradients[-1]).T
    )

    assert result.status == "completed"
    assert all(item.total_seconds > 0.0 for item in step.accepted_increments)
    assert all(item.linear_solve_calls >= 0 for item in step.accepted_increments)
    assert all(item.linear_iterations >= 0 for item in step.accepted_increments)
    assert all(
        len(item.linear_converged_reasons) == item.linear_solve_calls
        for item in step.accepted_increments
    )
    assert all(
        reason > 0
        for item in step.accepted_increments
        for reason in item.linear_converged_reasons
    )
    assert [snapshot.load_factor for snapshot in step.snapshots] == pytest.approx(
        coordinates
    )
    for snapshot, expected_gradient in zip(step.snapshots, gradients):
        np.testing.assert_allclose(
            periodicity.deformation_gradient_at(snapshot.load_factor),
            expected_gradient,
            rtol=0.0,
            atol=2.0e-14,
        )
    np.testing.assert_allclose(
        periodicity.measured_deformation_gradient(displacement),
        gradients[-1],
        rtol=0.0,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        step.response.state.committed_state_vectors(),
        np.broadcast_to(
            expected.state_new,
            step.response.state.committed_state_vectors().shape,
        ),
        rtol=8.0e-6,
        atol=5.0e-8,
    )
    np.testing.assert_allclose(
        step.response.first_piola_stress.owned_values,
        np.broadcast_to(
            expected_first_piola,
            step.response.first_piola_stress.owned_values.shape,
        ),
        rtol=8.0e-6,
        atol=8.0e-6,
    )
    piola_history = _material_first_piola_history(material, coordinates, gradients)
    expected_path_work = periodicity.reference_cell_volume * sum(
        0.5 * float(np.sum((left_P + right_P) * (right_F - left_F)))
        for left_P, right_P, left_F, right_F in zip(
            piola_history[:-1],
            piola_history[1:],
            gradients[:-1],
            gradients[1:],
        )
    )
    assert result.quantity("affine_constraint_path_work") == pytest.approx(
        expected_path_work,
        rel=8.0e-6,
        abs=8.0e-6,
    )
    incoming = result.histories["affine_path_generalized_reaction"].values
    outgoing = result.histories["affine_path_outgoing_generalized_reaction"].values
    assert result.histories[
        "affine_path_generalized_reaction"
    ].abscissa_unit is None
    assert result.histories[
        "affine_path_outgoing_generalized_reaction"
    ].abscissa_unit is None
    assert abs(float(incoming[1, 0] - outgoing[1, 0])) > 1.0e-3
    final_path_tangent = (gradients[-1] - gradients[-2]) / (
        coordinates[-1] - coordinates[-2]
    )
    assert result.quantity("affine_path_generalized_reaction") == pytest.approx(
        periodicity.reference_cell_volume
        * float(np.sum(expected_first_piola * final_path_tangent)),
        rel=8.0e-6,
        abs=8.0e-6,
    )
    assert result.metadata["constraint_balance_contract"]["work_balance_available"]
    assert (
        result.metadata["constraint_work_evidence"][0]["integration"]
        == "accepted_path_trapezoidal"
    )
    identity = periodicity.scientific_identity()
    assert (
        identity["deformation_gradient_path"]["fingerprint"]
        == path.summary()["fingerprint"]
    )

    checkpoint = step.save_checkpoint(tmp_path / "nonproportional_path")
    fixture_changed = periodic_unit_cube(MPI.COMM_SELF)
    changed_model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture_changed.domain,
    )
    changed_displacement = changed_model.field(
        fields.displacement(fixture_changed.domain)
    )
    changed_material = changed_model.material(material)
    changed_coordinates, changed_gradients = _nonproportional_gradient_path(
        final_shear=0.031
    )
    changed_periodicity = changed_model.constraint(
        constraints.abaqus_periodic_cell(
            changed_displacement,
            nodes=fixture_changed.nodes,
            equations=fixture_changed.equations,
            deformation_gradient_path=constraints.deformation_gradient_path(
                changed_coordinates,
                changed_gradients,
            ),
            anchor_node=fixture_changed.anchor_node,
            reference_nodes=fixture_changed.reference_nodes,
        )
    )
    changed_step = changed_model.step(
        target=changed_displacement,
        material=changed_material,
        constraints=changed_periodicity,
        incrementation=steps.at(*changed_coordinates[1:]),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        progress=False,
    )
    with pytest.raises(ValueError, match="deformation history differ"):
        changed_step.load_checkpoint(checkpoint)


def test_automatic_macro_path_hits_knots_and_restarts_at_internal_state(tmp_path):
    control = steps.automatic(
        initial=0.2,
        minimum=1.0e-4,
        maximum=0.2,
        max_increments=20,
        max_cutbacks=8,
        growth_factor=1.1,
    )
    reference, material, path = _nonproportional_periodic_j2_step(
        incrementation=control
    )
    reference.solve()
    accepted = tuple(float(item.load_factor) for item in reference.accepted_increments)
    for knot in path.coordinates[1:]:
        assert any(abs(value - knot) <= 1.0e-12 for value in accepted)

    oracle_coordinates = (0.0, *accepted)
    expected = _integrate_gradient_history(
        material,
        oracle_coordinates,
        tuple(path.at(value) for value in oracle_coordinates),
    )
    np.testing.assert_allclose(
        reference.response.state.committed_state_vectors(),
        np.broadcast_to(
            expected.state_new,
            reference.response.state.committed_state_vectors().shape,
        ),
        rtol=8.0e-6,
        atol=5.0e-8,
    )

    partial, _, _ = _nonproportional_periodic_j2_step(incrementation=control)
    partial.solve(until=0.65)
    assert partial.accepted_load_factor == pytest.approx(0.65)
    checkpoint = partial.save_checkpoint(tmp_path / "mid_path_restart")

    restarted, _, _ = _nonproportional_periodic_j2_step(incrementation=control)
    restarted.load_checkpoint(checkpoint)
    restarted.solve()
    np.testing.assert_allclose(
        restarted.solution.x.array,
        reference.solution.x.array,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        restarted.response.state.committed_state_vectors(),
        reference.response.state.committed_state_vectors(),
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        restarted.response.first_piola_stress.values,
        reference.response.first_piola_stress.values,
        rtol=0.0,
        atol=2.0e-11,
    )


def test_public_finite_strain_j2_periodic_cube_matches_material_point(tmp_path):
    fixture = periodic_unit_cube(MPI.COMM_SELF)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
        name="finite_strain_j2_periodic_cube",
    )
    displacement = model.field(fields.displacement(fixture.domain))
    material = constitutive.finite_strain_j2_logarithmic(
        young=200_000.0,
        poisson=0.3,
        yield_stress=200.0,
        hardening_modulus=2_000.0,
    )
    material_record = model.material(material)
    periodicity = model.constraint(fixture.constraint(displacement))
    reduction = periodicity.reduction()
    assert fixture.equations.summary()["equation_count"] == 48
    assert reduction.full_size == 81
    assert reduction.reduced_size == 21
    output = results.output_plan(
        tmp_path,
        field=results.field_output(
            "U",
            every=2,
            configuration="reference",
            backend="xdmf",
        ),
        requests=(results.periodic_cell_history(periodicity),),
        presentation=None,
        basename="finite_strain_j2_periodic",
    )

    capability = step_capability(
        model,
        target=displacement,
        options={
            "material": material_record,
            "constraints": periodicity,
            "checkpoint": checkpointing.every(
                2,
                directory=tmp_path / "checkpoint_contract",
            ),
        },
    )
    assert capability["supported"]
    assert capability["provider"]["name"] == "finite_strain_j2_affine_static"
    assert "checkpoint" in capability["provider"]["options"]["accepted"]
    wrapped_capability = step_capability(
        model,
        target=displacement,
        options={
            "material": material_record,
            "constraints": constraints.ConstraintSet(periodic=[periodicity]),
        },
    )
    assert wrapped_capability["supported"]
    assert wrapped_capability["provider"]["name"] == "finite_strain_j2_affine_static"
    natural_load = model.load(
        loads.LoadSet.create(
            loads.body_force(
                (0.0, 0.0, -1.0),
                domain=fixture.domain,
                target=displacement,
            )
        )
    )
    loaded_capability = step_capability(
        model,
        target=displacement,
        options={
            "material": material_record,
            "constraints": periodicity,
        },
    )
    assert not loaded_capability["supported"]
    assert model.loads.pop() is natural_load

    step = model.step(
        target=displacement,
        material=material_record,
        constraints=periodicity,
        incrementation=steps.fixed(4),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        output=output,
        progress=False,
    )
    result = step.solve_result()
    expected = _integrate_material_path(material, fixture.deformation_gradient, 4)
    expected_first_piola = (
        np.linalg.det(fixture.deformation_gradient)
        * expected.cauchy_stress
        @ np.linalg.inv(fixture.deformation_gradient).T
    )

    assert step.last_solve_info.converged
    assert len(step.last_solve_info.increments) == 4
    assert periodicity.mismatch() < 1.0e-10
    np.testing.assert_allclose(
        periodicity.measured_deformation_gradient(displacement),
        fixture.deformation_gradient,
        rtol=0.0,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        step.response.first_piola_stress.owned_values,
        np.broadcast_to(
            expected_first_piola,
            step.response.first_piola_stress.owned_values.shape,
        ),
        rtol=5.0e-6,
        atol=5.0e-6,
    )
    np.testing.assert_allclose(
        step.response.state.committed_state_vectors(),
        np.broadcast_to(
            expected.state_new,
            step.response.state.committed_state_vectors().shape,
        ),
        rtol=5.0e-6,
        atol=5.0e-8,
    )
    peeq = step.response.state.committed_state_vectors()[:, -2]
    assert np.ptp(peeq) < 1.0e-9
    assert np.mean(peeq) > 0.0

    assert {
        "Displacement",
        "F",
        "P",
        "S",
        "MISES",
        "SENER",
        "FP",
        "PEEQ",
        "PDENER",
    } <= set(result.fields)
    assert result.fields["PEEQ"].location == "quadrature_points"
    assert result.quantity("maximum_equivalent_plastic_strain") == pytest.approx(
        float(
            material.state_schema.unpack(expected.state_new)[
                "equivalent_plastic_strain"
            ]
        ),
        rel=5.0e-6,
    )
    assert result.quantity("plastic_dissipation") > 0.0
    assert result.quantity("homogenized_plastic_dissipation_density") == pytest.approx(
        material.yield_stress * result.quantity("maximum_equivalent_plastic_strain"),
        rel=5.0e-6,
    )
    np.testing.assert_allclose(
        result.histories["homogenized_first_piola_stress"].values[-1],
        expected_first_piola,
        rtol=5.0e-6,
        atol=5.0e-6,
    )
    assert result.quantity("maximum_hill_mandel_relative_error") < 1.0e-8
    expected_macro_reaction = periodicity.reference_cell_volume * float(
        np.sum(expected_first_piola * (fixture.deformation_gradient - np.eye(3)))
    )
    assert result.quantity("affine_path_generalized_reaction") == pytest.approx(
        expected_macro_reaction,
        rel=8.0e-6,
        abs=8.0e-6,
    )
    np.testing.assert_allclose(
        result.quantity("affine_constraint_force_resultant"),
        np.zeros(3),
        atol=2.0e-7,
    )
    balance_contract = result.metadata["constraint_balance_contract"]
    assert balance_contract["force_balance_available"]
    assert balance_contract["work_balance_available"]
    assert balance_contract["work_balance_gaps"] == ()
    assert balance_contract["provider_work"][0]["sample_count"] >= 2
    assert result.metadata["constraint_duals"][0]["source"] == (
        "exact_affine_reduction_full_residual_virtual_work"
    )
    assert result.metadata["affine_constraint_path_work"]["status"] == "complete"
    expected_path_work = periodicity.reference_cell_volume * float(
        np.sum(result.histories["hill_mandel_macroscopic_work_density"].values)
    )
    assert result.quantity("affine_constraint_path_work") == pytest.approx(
        expected_path_work,
        rel=8.0e-6,
        abs=8.0e-6,
    )
    assert result.metadata["output_plan"]["status"] == "completed"


def test_public_finite_strain_j2_periodic_checkpoint_restart_is_equivalent(
    tmp_path,
):
    reference = _periodic_j2_step()
    reference.solve()
    expected_solution = reference.solution.x.array.copy()
    expected_state = reference.response.state.committed_state_vectors().copy()
    expected_first_piola = reference.response.first_piola_stress.values.copy()

    partial = _periodic_j2_step()
    partial.solve(until=0.5)
    assert partial.state_transaction.accepted_factor == pytest.approx(0.5)
    checkpoint = partial.save_checkpoint(tmp_path / "finite_strain_j2_periodic_restart")
    assert checkpoint.is_file()

    restarted = _periodic_j2_step()
    restarted.load_checkpoint(checkpoint)
    restarted.solve()

    np.testing.assert_allclose(
        restarted.solution.x.array,
        expected_solution,
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        restarted.response.state.committed_state_vectors(),
        expected_state,
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        restarted.response.first_piola_stress.values,
        expected_first_piola,
        rtol=2.0e-8,
        atol=2.0e-8,
    )
    assert [
        item.load_factor for item in restarted.last_solve_info.increments
    ] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert restarted.constraint_dual_history == reference.constraint_dual_history
    assert restarted.state_transaction.accepted_factor == pytest.approx(1.0)


def test_affine_checkpoint_refuses_mutated_constraint_identity(tmp_path):
    partial = _periodic_j2_step()
    partial.solve(until=0.5)
    checkpoint = partial.save_checkpoint(tmp_path / "constraint_identity")

    mutated = _periodic_j2_step(stretch=1.021)
    original_solution = mutated.solution.x.array.copy()
    original_state = mutated.response.state.committed_state_vectors().copy()
    with pytest.raises(ValueError, match="differ from the current analysis"):
        mutated.load_checkpoint(checkpoint)

    np.testing.assert_array_equal(mutated.solution.x.array, original_solution)
    np.testing.assert_array_equal(
        mutated.response.state.committed_state_vectors(),
        original_state,
    )
    assert mutated.accepted_load_factor == pytest.approx(0.0)
    assert mutated.state_transaction.accepted_factor == pytest.approx(0.0)


def test_affine_checkpoint_rejects_corrupt_constraint_dual_history(tmp_path):
    source = _periodic_j2_step()
    source.solve(until=0.5)
    checkpoint = source.save_checkpoint(tmp_path / "corrupt_dual_history")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["constraint_dual_history"]["records"][-1]["force"] = [float("nan")]
    checkpoint.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    target = _periodic_j2_step()
    with pytest.raises(ValueError, match="Invalid affine constraint dual history"):
        target.load_checkpoint(checkpoint)
    assert target.accepted_load_factor == pytest.approx(0.0)
    assert target.constraint_dual_history.records == []


@pytest.mark.parametrize(
    "changed_options",
    (
        {"tangent_relative_step": 5.0e-6},
        {"relative_tolerance": 5.0e-8},
    ),
)
def test_affine_checkpoint_identity_covers_material_and_solver_numerics(
    tmp_path,
    changed_options,
):
    partial = _periodic_j2_step()
    partial.solve(until=0.5)
    checkpoint = partial.save_checkpoint(tmp_path / "numerical_identity")

    changed = _periodic_j2_step(**changed_options)
    with pytest.raises(ValueError, match="differ from the current analysis"):
        changed.load_checkpoint(checkpoint)


def test_failed_affine_checkpoint_load_restores_complete_runtime_state(tmp_path):
    source = _periodic_j2_step()
    source.solve(until=0.5)
    checkpoint = source.save_checkpoint(tmp_path / "atomic_restore")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["accepted_increments"][-1]["load_factor"] = 0.49
    checkpoint.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    target = _periodic_j2_step()
    target.solve(until=0.25)
    transaction = target.state_transaction
    solution = target.solution.x.array.copy()
    accepted_solution = transaction.accepted_solution.x.array.copy()
    transaction_state = transaction.snapshot_runtime_state()
    accepted_load_factor = target.accepted_load_factor
    accepted_increments = list(target.accepted_increments)
    attempted_increments = list(target.attempted_increments)
    next_increment_size = target.next_increment_size
    execution_events = list(target.execution_events)
    last_solve_info = target.last_solve_info
    snapshots = list(target.snapshots)
    checkpoints = list(target.checkpoints)

    with pytest.raises(
        ValueError,
        match="coordinate and accepted history disagree",
    ):
        target.load_checkpoint(checkpoint)

    np.testing.assert_array_equal(target.solution.x.array, solution)
    np.testing.assert_array_equal(
        transaction.accepted_solution.x.array,
        accepted_solution,
    )
    restored = transaction.snapshot_runtime_state()
    for name in (
        "first_piola_stress",
        "cauchy_stress",
        "tangent",
        "strain_energy_density",
        "deformation_gradient",
        "equivalent_stress",
    ):
        np.testing.assert_array_equal(restored[name], transaction_state[name])
    for name in transaction_state["committed_state"]:
        np.testing.assert_array_equal(
            restored["committed_state"][name],
            transaction_state["committed_state"][name],
        )
        np.testing.assert_array_equal(
            restored["trial_state"][name],
            transaction_state["trial_state"][name],
        )
    assert restored["accepted_factor"] == transaction_state["accepted_factor"]
    assert restored["last_plastic_points"] == transaction_state["last_plastic_points"]
    assert (
        restored["last_maximum_plastic_increment"]
        == transaction_state["last_maximum_plastic_increment"]
    )
    assert target.accepted_load_factor == accepted_load_factor
    assert target.accepted_increments == accepted_increments
    assert target.attempted_increments == attempted_increments
    assert target.next_increment_size == next_increment_size
    assert target.execution_events == execution_events
    assert target.last_solve_info is last_solve_info
    assert len(target.snapshots) == len(snapshots)
    assert all(
        restored_snapshot is original_snapshot
        for restored_snapshot, original_snapshot in zip(
            target.snapshots,
            snapshots,
            strict=True,
        )
    )
    assert target.checkpoints == checkpoints


class _FailingAcceptedObserver:
    """Fault injector implementing the accepted-observer rollback protocol."""

    def __init__(self):
        self.accepted = 0
        self.fail = True

    def snapshot_runtime_state(self):
        return {"accepted": self.accepted}

    def restore_runtime_state(self, state) -> None:
        self.accepted = int(state["accepted"])

    def reset(self, _snapshot) -> None:
        self.accepted = 0

    def accept(self, _snapshot) -> None:
        self.accepted += 1
        if self.fail:
            raise RuntimeError("injected accepted-observer failure")


def test_post_commit_observer_failure_restores_complete_accepted_boundary():
    step = _periodic_j2_step()
    observer = _FailingAcceptedObserver()
    step.accepted_observers = (observer,)
    initial_solution = step.solution.x.array.copy()
    initial_state = step.response.state.committed_state_vectors().copy()

    with pytest.raises(RuntimeError, match="accepted-observer failure"):
        step.solve()

    np.testing.assert_array_equal(step.solution.x.array, initial_solution)
    np.testing.assert_array_equal(
        step.state_transaction.accepted_solution.x.array,
        initial_solution,
    )
    np.testing.assert_array_equal(
        step.response.state.committed_state_vectors(),
        initial_state,
    )
    np.testing.assert_array_equal(
        step.response.state.trial_state_vectors(),
        initial_state,
    )
    assert observer.accepted == 0
    assert step.accepted_load_factor == pytest.approx(0.0)
    assert step.state_transaction.accepted_factor == pytest.approx(0.0)
    assert step.accepted_increments == []
    assert step.attempted_increments == []

    observer.fail = False
    step.solve()
    assert step.accepted_load_factor == pytest.approx(1.0)
    assert step.state_transaction.accepted_factor == pytest.approx(1.0)
    assert observer.accepted == 4


def test_post_commit_snapshot_failure_restores_complete_accepted_boundary():
    reference = _periodic_j2_step()
    reference.solve(until=0.75)
    step = _periodic_j2_step()
    snapshot_field_factory = step.snapshot_field_factory

    def fail_after_initial_boundary():
        if np.max(np.abs(step.solution.x.array), initial=0.0) > 1.0e-14:
            raise RuntimeError("injected accepted-snapshot failure")
        return snapshot_field_factory()

    step.snapshot_field_factory = fail_after_initial_boundary
    with pytest.raises(RuntimeError, match="accepted-snapshot failure"):
        step.solve()

    np.testing.assert_allclose(
        step.solution.x.array,
        reference.solution.x.array,
        rtol=2.0e-9,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        step.response.state.committed_state_vectors(),
        reference.response.state.committed_state_vectors(),
        rtol=2.0e-9,
        atol=2.0e-11,
    )
    assert step.accepted_load_factor == pytest.approx(0.75)
    assert step.state_transaction.accepted_factor == pytest.approx(0.75)
    assert [item.load_factor for item in step.accepted_increments] == pytest.approx(
        [0.25, 0.5, 0.75]
    )

    step.snapshot_field_factory = snapshot_field_factory
    step.solve()
    assert step.accepted_load_factor == pytest.approx(1.0)


def test_post_commit_checkpoint_failure_restores_complete_accepted_boundary(
    tmp_path,
):
    policy = checkpointing.every(1, directory=tmp_path / "scheduled")
    step = _periodic_j2_step(checkpoint=policy)
    initial_solution = step.solution.x.array.copy()
    initial_state = step.response.state.committed_state_vectors().copy()
    save_checkpoint = step.save_checkpoint

    def fail_checkpoint(*_args, **_kwargs):
        raise OSError("injected checkpoint failure")

    step.save_checkpoint = fail_checkpoint
    with pytest.raises(OSError, match="checkpoint failure"):
        step.solve()

    np.testing.assert_array_equal(step.solution.x.array, initial_solution)
    np.testing.assert_array_equal(
        step.state_transaction.accepted_solution.x.array,
        initial_solution,
    )
    np.testing.assert_array_equal(
        step.response.state.committed_state_vectors(),
        initial_state,
    )
    assert step.accepted_load_factor == pytest.approx(0.0)
    assert step.state_transaction.accepted_factor == pytest.approx(0.0)
    assert step.accepted_increments == []
    assert step.attempted_increments == []
    assert step.checkpoints == []

    step.save_checkpoint = save_checkpoint
    step.solve()
    assert step.accepted_load_factor == pytest.approx(1.0)
    assert len(step.checkpoints) == 4
