"""Bounded public J2 plus fixed and moving triangulated-tool checks."""

from __future__ import annotations

import json

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
           cells=1, constrain_contact=False, contact_value=0.0, tilt=0.0,
           tool_translation=None, tool_rotation=(0.0, 0.0, 0.0),
           schedule_end=1.0, make_step=True, tool_y_max=1.5,
           increments=3, applied_traction=10.0, yield_stress=2.0,
           convex_tool=False, convex_y_min=-0.5):
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
            yield_stress=yield_stress,
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
        if convex_tool:
            assert tilt == 0.0
            x0, x1, y0, y1, z0, z1 = (
                tool_x, tool_x + 0.1, convex_y_min, 1.5, -0.5, 1.5,
            )
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
        else:
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
        model.fix(displacement, on=right, component=0, value=contact_value)
    if applied_traction != 0.0:
        model.traction((applied_traction, 0.0, 0.0), on=right)
    if not make_step:
        return model, displacement, material, None
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(increments),
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


def test_closed_convex_tool_uses_ordinary_j2_step_and_matches_planar_limit():
    _, _, _, planar = _patch(
        triangulated=True, yield_stress=1.0e9,
        tool_translation=(-0.008, 0, 0), applied_traction=0.0,
    )
    model, displacement, material, sharp = _patch(
        triangulated=True, convex_tool=True, yield_stress=1.0e9,
        tool_translation=(-0.008, 0, 0), applied_traction=0.0,
    )
    capability = step_capability(model, target=displacement, options={"material": material})
    assert capability["supported"]
    planar_result = planar.solve_result()
    sharp_result = sharp.solve_result()
    assert planar_result.status == sharp_result.status == "completed"
    assert sharp.contact_provider.summary()["linearization"] == (
        "closed_convex_polyhedron_piecewise_affine_semismooth"
    )
    np.testing.assert_allclose(
        sharp.solution.x.array, planar.solution.x.array,
        rtol=1e-7, atol=1e-8,
    )
    assert sharp.energy_history[-1].tool_work == pytest.approx(
        planar.energy_history[-1].tool_work, rel=1e-7, abs=1e-9,
    )


def test_convex_contact_matches_independent_hencky_bar_penalty_equilibrium():
    _, _, _, step = _patch(
        triangulated=True, convex_tool=True, yield_stress=1.0e9,
        tool_translation=(-0.008, 0, 0), applied_traction=0.0,
        increments=8,
    )
    result = step.solve_result()
    assert result.status == "completed"
    # Homogeneous uniaxial compression with nu=0 has P=E log(lambda)/lambda.
    # The tool ends at x=0.997, and one-sided reference-area penalty contact
    # contributes k(lambda-0.997). This scalar equation is independent of FE.
    lower, upper = 0.997, 1.0
    for _ in range(70):
        stretch = 0.5 * (lower + upper)
        residual = 1000.0 * np.log(stretch) / stretch + 500.0 * (stretch - 0.997)
        if residual < 0.0:
            lower = stretch
        else:
            upper = stretch
    expected_stretch = 0.5 * (lower + upper)
    coordinates = step.solution.function_space.tabulate_dof_coordinates()
    right = np.isclose(coordinates[:, 0], 1.0)
    right_ux = step.solution.x.array.reshape((-1, 3))[right, 0]
    np.testing.assert_allclose(
        right_ux, expected_stretch - 1.0, rtol=1e-5, atol=1e-8,
    )
    contribution = step.contact_provider.evaluate()
    np.testing.assert_allclose(
        contribution.potential_energy,
        250.0 * (expected_stretch - 0.997) ** 2,
        rtol=1e-5, atol=1e-8,
    )


def test_convex_tool_tracks_noncoplanar_feature_during_incremental_solve(tmp_path):
    def build():
        return _patch(
            triangulated=True, convex_tool=True, convex_y_min=0.15,
            yield_stress=1.0e9, tool_translation=(-0.06, 0, 0),
            applied_traction=0.0, increments=12,
        )[3]

    step = build()
    step.solve(until=0.25)
    _evaluation, early_record, early_response, _factor = (
        step.contact_provider._current_point_response()
    )
    early = early_record.projection
    step.solve()
    _evaluation, late_record, late_response, _factor = (
        step.contact_provider._current_point_response()
    )
    late = late_record.projection
    assert step.last_solve_info.increments[-1].converged
    switching = (
        early_response.active & late_response.active
        & (early.entity_ids != late.entity_ids)
        & (np.sum(early.normals * late.normals, axis=1) < 0.99)
    )
    assert np.any(switching)
    assert step.contact_provider.evaluate().active_points > 0
    assert np.isfinite(step.solution.x.array).all()

    partial = build()
    partial.solve(until=0.25)
    saved = partial.save_checkpoint(tmp_path / "convex_feature_switch")
    resumed = build()
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(
        resumed.solution.x.array, step.solution.x.array, rtol=1e-9, atol=1e-10,
    )
    np.testing.assert_allclose(
        resumed.response.state.committed_state_vectors(),
        step.response.state.committed_state_vectors(), rtol=1e-9, atol=1e-10,
    )
    assert resumed.energy_history[-1].tool_work == pytest.approx(
        step.energy_history[-1].tool_work, rel=1e-9,
    )


def test_two_implicit_tools_add_residual_tangent_and_keep_separate_evidence():
    model, displacement, material, _ = _patch(
        triangulated=True, make_step=False, yield_stress=1.0e9,
    )
    first = model.boundary_models[0]
    model.add_boundary_model(
        boundary_models.rigid_contact_pair(
            first.slave_boundary,
            boundary_models.rigid_body(first.rigid_body.surface),
            penalty=500.0,
            name="second_tool",
        )
    )
    capability = step_capability(
        model, target=displacement, options={"material": material},
    )
    assert capability["supported"]
    combined = model.step(
        target=displacement, material=material,
        incrementation=steps.fixed(3), progress=False,
    )
    result = combined.solve_result()
    assert result.status == "completed"
    assert tuple(
        item["constraint_name"] for item in result.metadata["constraint_duals"]
    ) == (first.name, "second_tool")
    assert len(result.metadata["constraint_work_evidence"]) == 2
    assert result.metadata["constraint_balance_contract"]["force_balance_available"]
    assert combined._contact_potential() == pytest.approx(sum(
        item.evaluate().potential_energy
        for item in combined.contact_provider.providers
    ))

    _, _, _, doubled = _patch(
        triangulated=True, penalty=1000.0, yield_stress=1.0e9,
    )
    doubled.solve()
    np.testing.assert_allclose(
        combined.solution.x.array, doubled.solution.x.array,
        rtol=2e-4, atol=2e-6,
    )
    accepted = combined.solution.x.array.copy()
    direction = np.random.default_rng(117).normal(size=accepted.shape)
    constrained = np.unique(np.concatenate([
        np.asarray(bc.dof_indices()[0], dtype=np.int64) for bc in combined.bcs
    ]))
    direction[constrained] = 0.0
    direction /= np.linalg.norm(direction)

    def rhs_at(offset):
        combined.solution.x.array[:] = accepted + offset * direction
        combined.solution.x.scatter_forward()
        combined._update_response(start_factor=1.0, target_factor=1.0)
        rhs, _ = combined._correction_rhs()
        try:
            return rhs.array.copy()
        finally:
            rhs.destroy()

    try:
        rhs_at(0.0)
        matrix = combined._assemble_tangent()
        from petsc4py import PETSc

        vector = PETSc.Vec().createWithArray(direction.copy(), comm=MPI.COMM_SELF)
        action = matrix.createVecLeft()
        matrix.mult(vector, action)
        expected = action.array.copy()
        vector.destroy()
        action.destroy()
        epsilon = 1.0e-6
        combined.solution.x.array[:] = accepted
        combined.solution.x.scatter_forward()
        contact_action = combined.contact_provider.evaluate().tangent_action(
            direction.reshape(-1, 3)
        ).reshape(-1)
        contact_residuals = []
        for sign in (1.0, -1.0):
            combined.solution.x.array[:] = accepted + sign * epsilon * direction
            combined.solution.x.scatter_forward()
            contact_residuals.append(
                combined.contact_provider.evaluate().nodal_structural_residual.reshape(-1)
            )
        np.testing.assert_allclose(
            contact_action,
            (contact_residuals[0] - contact_residuals[1]) / (2.0 * epsilon),
            rtol=1.0e-4, atol=1.0e-5,
        )
        numerical = -(rhs_at(epsilon) - rhs_at(-epsilon)) / (2.0 * epsilon)
        np.testing.assert_allclose(
            expected, numerical, rtol=2.0e-3, atol=2.0e-5,
        )
    finally:
        combined.solution.x.array[:] = accepted
        combined.solution.x.scatter_forward()
        combined._update_response(start_factor=1.0, target_factor=1.0)


def test_two_moving_tools_keep_independent_work_and_atomic_restart(tmp_path, monkeypatch):
    def build():
        model, displacement, material, _ = _patch(
            triangulated=True, make_step=False,
            tool_translation=(-0.008, 0.0, 0.0), applied_traction=0.0,
        )
        first = model.boundary_models[0]
        second_motion = boundary_models.prescribed_rigid_motion(
            translation=(-0.012, 0.0, 0.0),
            reference_point=(1.005, 0.0, 0.0),
        )
        model.add_boundary_model(boundary_models.rigid_contact_pair(
            first.slave_boundary,
            boundary_models.rigid_body(
                first.rigid_body.surface,
                reference_point=(1.005, 0.0, 0.0),
                motion_schedule=boundary_models.prescribed_rigid_motion_schedule(
                    second_motion, start_time=0.0, end_time=1.0,
                ),
            ),
            penalty=500.0,
            name="second_tool",
        ))
        return model.step(
            target=displacement, material=material,
            incrementation=steps.fixed(3), progress=False,
        )

    reference = build()
    result = reference.solve_result()
    assert result.status == "completed"
    assert np.max(
        reference.response.state.committed[
            "equivalent_plastic_strain"
        ].function.x.array
    ) > 0.0
    names = tuple(item.name for item in reference.contact_provider.providers)
    assert names == ("rigid_contact_pair", "second_tool")
    work = dict(reference.energy_history[-1].pair_tool_work)
    assert set(work) == set(names)
    assert all(value > 0.0 for value in work.values())
    assert sum(work.values()) == pytest.approx(reference.energy_history[-1].tool_work)
    assert tuple(item["constraint_name"] for item in
                 result.metadata["constraint_work_evidence"]) == names

    partial = build()
    partial.solve(until=1.0 / 3.0)
    saved = partial.save_checkpoint(tmp_path / "two_tools")
    accepted_u = partial.solution.x.array.copy()
    accepted_state = partial.response.state.committed_state_vectors().copy()
    accepted_work = partial.energy_history[-1].pair_tool_work

    record_energy = partial._record_energy

    def reject_energy(*args, **kwargs):
        record_energy(*args, **kwargs)
        raise ValueError("intentional multi-tool energy rejection")

    monkeypatch.setattr(partial, "_record_energy", reject_energy)
    with pytest.raises(RuntimeError, match="intentional multi-tool energy rejection"):
        partial.solve(until=2.0 / 3.0)
    np.testing.assert_allclose(partial.solution.x.array, accepted_u)
    np.testing.assert_allclose(
        partial.response.state.committed_state_vectors(), accepted_state,
    )
    assert partial.energy_history[-1].pair_tool_work == accepted_work
    assert partial.contact_provider.snapshot_motion() == tuple(
        (name, 1.0 / 3.0) for name in names
    )

    resumed = build()
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(
        resumed.solution.x.array, reference.solution.x.array,
        rtol=1e-8, atol=1e-10,
    )
    np.testing.assert_allclose(
        resumed.response.state.committed_state_vectors(),
        reference.response.state.committed_state_vectors(),
        rtol=1e-8, atol=1e-10,
    )
    assert dict(resumed.energy_history[-1].pair_tool_work) == pytest.approx(work)

    with np.load(saved, allow_pickle=False) as archive:
        damaged = {name: archive[name].copy() for name in archive.files}
    frames = json.loads(str(damaged["energy_history"]))
    frames[-1]["pair_tool_work"]["second_tool"] += 1.0
    damaged["energy_history"] = json.dumps(frames)
    corrupt = tmp_path / "corrupt_two_tools.npz"
    np.savez_compressed(corrupt, **damaged)
    rejected = build()
    with pytest.raises(ValueError, match="per-tool work must sum"):
        rejected.load_checkpoint(corrupt)
    assert rejected.accepted_load_factor == 0.0
    assert not rejected.energy_history


def test_two_implicit_tools_reject_duplicate_identity_before_lowering():
    model, displacement, material, _ = _patch(
        triangulated=True, make_step=False,
    )
    first = model.boundary_models[0]
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        first.slave_boundary,
        boundary_models.rigid_body(first.rigid_body.surface),
        penalty=500.0, name=first.name,
    ))
    capability = step_capability(
        model, target=displacement, options={"material": material},
    )
    assert not capability["supported"]
    assert any(
        item["message"].startswith("AFM-IMPLICIT-CONTACT-MULTI")
        for item in capability["rejection_reasons"]
    )


def test_implicit_facet_friction_is_rejected_outside_fixed_plane_gate():
    model, displacement, material, _ = _patch(
        triangulated=True, make_step=False,
    )
    first = model.boundary_models[0]
    model.boundary_models[:] = [
        boundary_models.rigid_contact_pair(
            first.slave_boundary, first.rigid_body,
            penalty=500.0, friction_coefficient=0.3,
            tangential_penalty=200.0,
        )
    ]
    capability = step_capability(
        model, target=displacement, options={"material": material},
    )
    assert not capability["supported"]
    assert any(
        item["message"].startswith("AFM-IMPLICIT-CONTACT-FRICTION")
        for item in capability["rejection_reasons"]
    )


def test_native_facet_result_rf_uses_the_full_residual_once():
    _, _, _, step = _patch(triangulated=True, applied_traction=10.0)
    result = step.solve_result()
    full_reaction = result.fields["RF"].field.x.array.reshape((-1, 3)).sum(axis=0)
    contact_force = np.asarray(result.metadata["constraint_duals"][0]["resultant"])
    np.testing.assert_allclose(
        full_reaction + contact_force + np.array((10.0, 0.0, 0.0)),
        0.0, rtol=0.0, atol=1.0e-7,
    )
    assert np.linalg.norm(contact_force) > 0.0
    _, _, _, analytic_step = _patch(
        triangulated=False, applied_traction=10.0,
    )
    analytic = analytic_step.solve_result()
    np.testing.assert_allclose(
        result.fields["RF"].field.x.array,
        analytic.fields["RF"].field.x.array,
        rtol=2.0e-4, atol=2.0e-6,
    )


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
    (0.0, 0.0, 0.2), (0.03, 0.04, 0.05),
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


def test_prescribed_tool_motion_drives_j2_without_a_separate_traction():
    options = dict(tool_translation=(-0.02, 0.0, 0.0), cells=2,
                   applied_traction=0.0)
    _, _, _, facet = _patch(triangulated=True, **options)
    _, _, _, plane = _patch(triangulated=False, **options)
    assert facet.solve_result().status == plane.solve_result().status == "completed"
    assert np.max(np.abs(facet.solution.x.array)) > 1e-4
    assert abs(facet.energy_history[-1].tool_work) > 1e-6
    np.testing.assert_allclose(facet.solution.x.array, plane.solution.x.array,
                               rtol=3e-4, atol=3e-6)
    assert facet.energy_history[-1].tool_work == pytest.approx(
        plane.energy_history[-1].tool_work, rel=3e-4, abs=1e-8,
    )


def test_moving_faceted_tool_global_work_refines_in_step_and_mesh():
    """Check integral observables, not a pointwise maximum, on one planar fixture."""

    def observables(*, cells, increments):
        _, _, _, step = _patch(
            triangulated=True,
            cells=cells,
            increments=increments,
            tool_translation=(-0.006, 0.0, 0.0),
            tool_rotation=(0.0, 0.0, 0.2),
        )
        assert step.solve_result().status == "completed"
        frame = step.energy_history[-1]
        return np.asarray((frame.tool_work, frame.contact_potential))

    time_scan = np.asarray([
        observables(cells=2, increments=count) for count in (3, 6, 12)
    ])
    mesh_scan = np.asarray([
        observables(cells=count, increments=6) for count in (1, 2, 3, 4)
    ])
    assert np.all(np.isfinite(time_scan)) and np.all(np.isfinite(mesh_scan))
    assert np.all(np.abs(time_scan[2] - time_scan[1]) <
                  np.abs(time_scan[1] - time_scan[0]))
    assert np.all(np.abs(mesh_scan[2] - mesh_scan[1]) <
                  np.abs(mesh_scan[1] - mesh_scan[0]))
    assert np.all(np.abs(mesh_scan[3] - mesh_scan[2]) <
                  np.abs(mesh_scan[2] - mesh_scan[1]))


def test_moving_faceted_tool_penalty_scan_changes_physics_explicitly():
    work = []
    potential = []
    for penalty in (250.0, 500.0, 1000.0, 2000.0):
        _, _, _, step = _patch(
            triangulated=True,
            cells=2,
            increments=6,
            penalty=penalty,
            tool_translation=(-0.006, 0.0, 0.0),
            tool_rotation=(0.0, 0.0, 0.2),
        )
        assert step.solve_result().status == "completed"
        frame = step.energy_history[-1]
        work.append(frame.tool_work)
        potential.append(frame.contact_potential)
    assert np.all(np.diff(work) > 0.0)
    assert np.all(np.diff(potential) < 0.0)


@pytest.mark.parametrize("rotation", [
    (0.0, 0.0, 0.01), (0.0, 0.0, 0.2), (0.03, 0.04, 0.05),
])
def test_moving_facet_tool_potential_derivative_and_atomic_restart(
    tmp_path, monkeypatch, rotation,
):
    options = dict(triangulated=True, cells=2,
                   tool_translation=(-0.006, 0.0, 0.0),
                   tool_rotation=rotation)
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

    record_energy = partial._record_energy

    def fail_energy(*args, **kwargs):
        record_energy(*args, **kwargs)
        raise ValueError("intentional moving facet energy failure")

    monkeypatch.setattr(partial, "_record_energy", fail_energy)
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


def test_facet_contact_lifts_shared_strong_dofs_and_keeps_dirichlet_rows():
    _, _, _, step = _patch(
        triangulated=True, constrain_contact=True, contact_value=0.002,
        tool_translation=(-0.02, 0.0, 0.0), tilt=0.01,
        applied_traction=0.0,
    )
    assert step.contact_constrained_dofs
    result = step.solve_result()
    assert result.status == "completed"
    assert step.contact_provider.evaluate().active_points > 0
    assert np.isfinite(step.energy_history[-1].contact_potential)
    evidence = result.metadata["rigid_contact_point_diagnostics"]
    assert evidence["location"] == "reference_boundary_quadrature"
    assert evidence["pressure_measure"] == "penalty_force_per_reference_area_not_cauchy_pressure"
    assert evidence["active_point_count"] > 0
    assert evidence["maximum_pressure"] > 0.0
    assert evidence["maximum_penetration"] > 0.0
    # DOLFINx strong rows must remain identity after the native contact COO
    # tangent is inserted, including active contact on prescribed components.
    step._correction_rhs()[0].destroy()
    matrix = step._assemble_tangent()
    for dof in step.contact_constrained_dofs:
        columns, values = matrix.getRow(int(dof))
        nonzero = np.abs(values) > 1.0e-12
        np.testing.assert_array_equal(columns[nonzero], [dof])
        np.testing.assert_allclose(values[nonzero], [1.0])
    # A prescribed correction on the contact trace must affect the free
    # residual through the *same* contact tangent used in Newton assembly.
    # Compare against a fresh nonlinear contact evaluation, not just the COO.
    accepted = step.solution.x.array.copy()
    try:
        trial = accepted.copy()
        trial[list(step.contact_constrained_dofs)] -= 1.0e-7
        step.solution.x.array[:] = trial
        step.solution.x.scatter_forward()
        off = step.contact_provider.evaluate()
        delta = np.zeros_like(trial)
        for bc in step.bcs:
            bc.set(delta, x0=trial, alpha=1.0)
        assert np.linalg.norm(delta) > 0.0
        step.solution.x.array[:] = trial + delta
        step.solution.x.scatter_forward()
        on = step.contact_provider.evaluate()
        predicted = (
            off.assembly.nodal_structural_residual
            + off.tangent_action(delta.reshape(-1, 3))
        )
        np.testing.assert_allclose(
            on.assembly.nodal_structural_residual,
            predicted,
            rtol=1.0e-7,
            atol=1.0e-9,
        )
    finally:
        step.solution.x.array[:] = accepted
        step.solution.x.scatter_forward()


def test_total_j2_contact_newton_tangent_matches_fixed_old_state_difference():
    """Differentiate the assembled *total* residual, not contact in isolation."""

    _, _, _, step = _patch(
        triangulated=True, cells=2, tilt=0.01,
        tool_translation=(-0.006, 0.0, 0.0), applied_traction=0.0,
    )
    step.solve()
    assert step.contact_provider.evaluate().active_points > 0
    accepted = step.solution.x.array.copy()
    constrained = np.unique(np.concatenate([
        np.asarray(bc.dof_indices()[0], dtype=np.int64) for bc in step.bcs
    ]))
    direction = np.random.default_rng(20261018).normal(size=accepted.shape)
    direction[constrained] = 0.0
    direction /= np.linalg.norm(direction)

    def rhs_at(offset):
        step.solution.x.array[:] = accepted + offset * direction
        step.solution.x.scatter_forward()
        step._update_response(start_factor=1.0, target_factor=1.0)
        rhs, _ = step._correction_rhs()
        try:
            return rhs.array.copy()
        finally:
            rhs.destroy()

    try:
        rhs_at(0.0)
        tangent = step._assemble_tangent()
        from petsc4py import PETSc

        trial = PETSc.Vec().createWithArray(direction.copy(), comm=MPI.COMM_SELF)
        action = tangent.createVecLeft()
        tangent.mult(trial, action)
        expected = action.array.copy()
        trial.destroy()
        action.destroy()
        epsilon = 1.0e-6
        numerical = -(rhs_at(epsilon) - rhs_at(-epsilon)) / (2.0 * epsilon)
        np.testing.assert_allclose(
            expected, numerical, rtol=2.0e-3, atol=2.0e-5,
        )
    finally:
        step.solution.x.array[:] = accepted
        step.solution.x.scatter_forward()
        step._update_response(start_factor=1.0, target_factor=1.0)


def test_shared_contact_dirichlet_restart_preserves_tool_work(tmp_path):
    options = dict(
        triangulated=True, constrain_contact=True, contact_value=0.002,
        tool_translation=(-0.02, 0.0, 0.0), tilt=0.01,
        applied_traction=0.0,
    )
    _, _, _, reference = _patch(**options)
    reference.solve()
    _, _, _, partial = _patch(**options)
    partial.solve(until=1.0 / 3.0)
    saved = partial.save_checkpoint(tmp_path / "shared_contact_dirichlet")
    _, _, _, resumed = _patch(**options)
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(resumed.solution.x.array, reference.solution.x.array,
                               rtol=1.0e-9, atol=1.0e-11)
    np.testing.assert_allclose(
        resumed.response.state.committed_state_vectors(),
        reference.response.state.committed_state_vectors(),
        rtol=1.0e-9, atol=1.0e-11,
    )
    assert resumed.energy_history[-1].tool_work == pytest.approx(
        reference.energy_history[-1].tool_work, rel=1.0e-9,
    )
    assert resumed.energy_history[-1].prescribed_motion_work == pytest.approx(
        reference.energy_history[-1].prescribed_motion_work, rel=1.0e-9,
    )
    _, _, _, changed = _patch(**{**options, "contact_value": 0.003})
    with pytest.raises(ValueError, match="identity"):
        changed.load_checkpoint(saved)
    assert not changed.energy_history


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

    record_energy = failed._record_energy

    def fail_energy(*args, **kwargs):
        record_energy(*args, **kwargs)
        raise ValueError("intentional facet energy failure")

    monkeypatch.setattr(failed, "_record_energy", fail_energy)
    with pytest.raises(RuntimeError, match="intentional facet energy failure"):
        failed.solve(until=2.0 / 3.0)
    assert failed.accepted_load_factor == pytest.approx(1.0 / 3.0)
    np.testing.assert_allclose(failed.solution.x.array, accepted_u)
    np.testing.assert_allclose(failed.response.state.committed_state_vectors(), accepted_material)
    assert tuple(failed.energy_history) == accepted_energy
