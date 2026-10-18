"""Bounded public J2 plus fixed and moving triangulated-tool checks."""

from __future__ import annotations

import json
from pathlib import Path
import runpy

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
           convex_tool=False, convex_y_min=-0.5, closed_concave_tool=False,
           closed_concave_extent=1.75, closed_concave_thickness=0.1,
           closed_concave_corner=None,
           closed_sign_method="winding_reference",
           friction_coefficient=None, tangential_penalty=None,
           regional_material=False, contact_quadrature_degree=None,
           hexahedral=False, displacement_degree=1, linear_solver=None,
           clamp_left=True, left_traction=0.0,
           initial_penetration_limit=None):
    domain = dolfinx_mesh.create_unit_cube(
        MPI.COMM_SELF, cells, cells, cells,
        cell_type=(
            dolfinx_mesh.CellType.hexahedron if hexahedral
            else dolfinx_mesh.CellType.tetrahedron
        ),
    )
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(
        domain, degree=displacement_degree,
    ))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right")
    if clamp_left:
        model.clamp(displacement, on=left)
    else:
        model.fix(displacement, on=left, component=1, value=0.0)
        model.fix(displacement, on=left, component=2, value=0.0)
    if left_traction:
        model.traction((left_traction, 0.0, 0.0), on=left)
    if regional_material:
        regions = mesh.partition_cells(
            domain,
            left=lambda x: x[0] <= 0.5,
            right=lambda x: x[0] > 0.5,
        )
        model.material(
            constitutive.finite_strain_j2_logarithmic(
                young=1000.0, poisson=0.0,
                yield_stress=yield_stress, hardening_modulus=100.0,
            ),
            region=regions.left,
        )
        model.material(
            constitutive.finite_strain_j2_logarithmic(
                young=1500.0, poisson=0.0,
                yield_stress=1.5 * yield_stress, hardening_modulus=150.0,
            ),
            region=regions.right,
        )
        material = None  # The ordinary Step lowers the complete regional map.
    else:
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
        if closed_concave_tool:
            assert not convex_tool and tilt == 0.0
            # The reentrant corner is beyond the contacted unit face. This
            # tests ordinary Step lowering against the planar limit without
            # pretending that the edge branch is globally smooth.
            polygon = np.asarray((
                (0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2),
            ), dtype=float)
            polygon = closed_concave_extent * polygon - 0.5
            if closed_concave_corner is not None:
                polygon += (
                    np.asarray(closed_concave_corner, dtype=float)
                    - (closed_concave_extent - 0.5)
                )
            front = np.column_stack((
                np.full(6, tool_x), polygon[:, 0], polygon[:, 1],
            ))
            back = front.copy()
            back[:, 0] += closed_concave_thickness
            top = ((0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5))
            triangles = [tuple(reversed(face)) for face in top]
            triangles += [tuple(6 + index for index in face) for face in top]
            for first in range(6):
                second = (first + 1) % 6
                triangles.extend(((first, second, second + 6),
                                  (first, second + 6, first + 6)))
            surface = boundary_models.closed_triangulated_rigid_solid(
                vertices=np.vstack((front, back)), triangles=triangles,
                reviewed_embedded=True,
                name="reviewed_L_prism_tool",
                sign_method=closed_sign_method,
            )
        elif convex_tool:
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
            friction_coefficient=friction_coefficient,
            tangential_penalty=tangential_penalty,
            initial_penetration_limit=initial_penetration_limit,
        )
        model.add_boundary_model(pair)
    else:
        surface = boundary_models.rigid_plane(
            point=(1.0 + gap, 0, 0),
            normal=np.asarray((-1, tilt, 0)) / np.sqrt(1.0 + tilt * tilt),
        )
        if initial_penetration_limit is None:
            model.rigid_obstacle_contact(
                on=right, penalty=penalty, surface=surface, motion=tool_motion,
            )
        else:
            model.add_boundary_model(boundary_models.rigid_contact_pair(
                right, boundary_models.rigid_body(surface),
                penalty=penalty,
                initial_penetration_limit=initial_penetration_limit,
            ))
    if constrain_contact:
        model.fix(displacement, on=right, component=0, value=contact_value)
    if applied_traction != 0.0:
        model.traction((applied_traction, 0.0, 0.0), on=right)
    if not make_step:
        return model, displacement, material, None
    step_options = {} if material is None else {"material": material}
    if contact_quadrature_degree is not None:
        step_options["contact_quadrature_degree"] = contact_quadrature_degree
    step = model.step(
        target=displacement,
        **step_options,
        incrementation=steps.fixed(increments),
        solver_options=solvers.newton(
            relative_tolerance=1e-8,
            absolute_tolerance=1e-9,
            maximum_iterations=25,
            line_search="backtracking",
            linear_solver=linear_solver,
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
    assert sharp_result.quantity(
        f"{sharp.contact_provider.name}_initial_maximum_penetration"
    ) == 0.0
    assert sharp_result.metadata["rigid_contact_point_diagnostics"][
        "initial_contact"
    ]["policy"] == "report"


def test_faceted_initial_overlap_limit_rejects_before_finite_j2_newton():
    model, displacement, material, _ = _patch(
        triangulated=True, gap=-0.05,
        initial_penetration_limit=0.01,
        applied_traction=0.0, make_step=False,
    )
    capability = step_capability(
        model, target=displacement, options={"material": material},
    )
    assert not capability["supported"]
    assert any(
        item["message"].startswith("AFM-CONTACT-INITIAL-BOUND")
        for item in capability["rejection_reasons"]
    )
    with pytest.raises(NotImplementedError, match="AFM-CONTACT-INITIAL-BOUND"):
        _patch(
            triangulated=True, gap=-0.05,
            initial_penetration_limit=0.01,
            applied_traction=0.0,
        )


def test_certified_plane_initial_overlap_limit_reaches_ordinary_j2_result():
    model, displacement, material, _ = _patch(
        triangulated=False, gap=0.005,
        initial_penetration_limit=0.01,
        applied_traction=0.0, make_step=False,
    )
    capability = step_capability(
        model, target=displacement, options={"material": material},
    )
    assert capability["supported"], capability["rejection_reasons"]
    with pytest.raises(ValueError, match="AFM-CONTACT-INITIAL-OVERLAP"):
        _patch(
            triangulated=False, gap=-0.05,
            initial_penetration_limit=0.01,
            applied_traction=0.0,
        )
    _, _, _, step = _patch(
        triangulated=False, gap=0.005,
        initial_penetration_limit=0.01,
        applied_traction=0.0, yield_stress=1.0e9,
    )
    result = step.solve_result()
    assert result.status == "completed"
    name = step.contact_provider.name
    assert result.quantity(f"{name}_initial_maximum_penetration_bound") == 0.0
    assert result.metadata["rigid_contact_point_diagnostics"][
        "initial_contact"
    ]["face_wise_bound_certified"] is True


def test_quadratic_plane_initial_overlap_limit_is_not_misadvertised():
    model, displacement, material, _ = _patch(
        triangulated=False, initial_penetration_limit=0.01,
        displacement_degree=2, make_step=False,
    )
    capability = step_capability(
        model, target=displacement, options={"material": material},
    )
    assert not capability["supported"]
    assert any(
        item["message"].startswith("AFM-CONTACT-INITIAL-BOUND")
        for item in capability["rejection_reasons"]
    )
    with pytest.raises(NotImplementedError, match="AFM-CONTACT-INITIAL-BOUND"):
        model.step(target=displacement, material=material)


def test_reviewed_closed_concave_tool_uses_ordinary_j2_step_and_restart(tmp_path):
    options = dict(
        triangulated=True, closed_concave_tool=True,
        yield_stress=1.0e9, tool_translation=(-0.008, 0.0, 0.0),
        applied_traction=0.0,
    )
    model, displacement, material, step = _patch(**options)
    capability = step_capability(
        model, target=displacement, options={"material": material},
    )
    assert capability["supported"], capability["rejection_reasons"]
    result = step.solve_result()
    assert result.status == "completed"
    assert step.contact_provider.summary()["search"]["kind"] == (
        "closed_triangulated_rigid_solid"
    )
    planar = _patch(
        triangulated=True, yield_stress=1.0e9,
        tool_translation=(-0.008, 0.0, 0.0), applied_traction=0.0,
    )[3]
    assert planar.solve_result().status == "completed"
    np.testing.assert_allclose(
        step.solution.x.array, planar.solution.x.array,
        rtol=1e-7, atol=1e-8,
    )
    assert step.energy_history[-1].tool_work == pytest.approx(
        planar.energy_history[-1].tool_work, rel=1e-7, abs=1e-9,
    )
    saved = step.save_checkpoint(tmp_path / "closed_concave")
    same = _patch(**options)[3]
    same.load_checkpoint(saved)
    np.testing.assert_allclose(same.solution.x.array, step.solution.x.array)
    assert same.accepted_load_factor == pytest.approx(1.0)
    different_geometry = _patch(
        **{**options, "closed_concave_extent": 1.7},
    )[3]
    with pytest.raises(ValueError, match="identity"):
        different_geometry.load_checkpoint(saved)


def test_reviewed_concave_edge_is_active_in_global_plastic_step_and_restart(
    tmp_path, monkeypatch,
):
    # Place the reentrant corner near, but not exactly on, one boundary
    # quadrature point. A thick tool prevents its back face from masking the
    # edge branch. This is an active-feature FE test, not an industrial tool
    # validation or a mesh/penalty convergence study.
    options = dict(
        triangulated=True, closed_concave_tool=True,
        closed_concave_extent=0.7, closed_concave_corner=(0.35, 0.19),
        closed_concave_thickness=0.5, gap=0.005,
        tool_translation=(-0.055, 0.0, 0.0), penalty=20.0,
        applied_traction=0.0, increments=6, yield_stress=0.05,
    )
    reference = _patch(**options)[3]
    result = reference.solve_result()
    assert result.status == "completed"
    provider = reference.contact_provider
    _, record, response, _ = provider._current_point_response()
    surface = provider.pair.rigid_body.surface
    features = [
        surface._geometric_feature_projector(
            surface._facet_slots[int(identity)], barycentric,
        )[1]
        for identity, barycentric in zip(
            record.projection.entity_ids,
            record.projection.local_coordinates,
            strict=True,
        )
    ]
    assert any(active and feature == "edge" for active, feature in zip(
        response.active, features, strict=True,
    ))
    feature_counts = result.metadata[
        "rigid_contact_point_diagnostics"
    ]["active_feature_counts"]
    sampled = result.quantity(
        f"{provider.name}_sampled_mixed_feature_slave_faces"
    )
    assert sampled >= 0
    assert result.metadata["rigid_contact_point_diagnostics"][
        "feature_transition_integral_certified"
    ] is False
    assert feature_counts["edge"] >= 1
    assert sum(feature_counts.values()) == int(np.count_nonzero(response.active))
    assert np.max(reference.response.state.committed[
        "equivalent_plastic_strain"
    ].function.x.array) > 0.0
    assert reference.energy_history[-1].tool_work > 0.0
    accepted = reference.solution.x.array.copy()
    direction = np.random.default_rng(503).normal(size=accepted.shape)
    epsilon = 1.0e-7
    selected = provider.evaluate()
    try:
        reference.solution.x.array[:] = accepted + epsilon * direction
        plus = provider.evaluate()
        reference.solution.x.array[:] = accepted - epsilon * direction
        minus = provider.evaluate()
    finally:
        reference.solution.x.array[:] = accepted
    np.testing.assert_allclose(
        selected.tangent_action(direction.reshape((-1, 3))),
        (plus.nodal_structural_residual - minus.nodal_structural_residual)
        / (2.0 * epsilon),
        rtol=1.0e-4, atol=1.0e-5,
    )
    # Check the *assembled* material-plus-contact Newton operator with the
    # fifth accepted state fixed. The final accepted state is unsuitable as
    # the old state: zero reloading there lies on an elastic/plastic kink.
    from petsc4py import PETSc

    trial = _patch(**options)[3]
    trial.solve(until=5.0 / 6.0)
    trial.contact_provider.update_motion(1.0)
    trial.solution.x.array[:] = accepted
    trial.solution.x.scatter_forward()
    constrained = np.unique(np.concatenate([
        np.asarray(bc.dof_indices()[0], dtype=np.int64)
        for bc in trial.bcs
    ]))
    direction[constrained] = 0.0
    direction /= np.linalg.norm(direction)

    def rhs_at(offset):
        trial.solution.x.array[:] = accepted + offset * direction
        trial.solution.x.scatter_forward()
        trial._update_response(start_factor=5.0 / 6.0, target_factor=1.0)
        rhs, _ = trial._correction_rhs()
        try:
            return rhs.array.copy()
        finally:
            rhs.destroy()

    try:
        rhs_at(0.0)
        matrix = trial._assemble_tangent()
        vector = PETSc.Vec().createWithArray(direction.copy(), comm=MPI.COMM_SELF)
        action = matrix.createVecLeft()
        matrix.mult(vector, action)
        expected = action.array.copy()
        vector.destroy()
        action.destroy()
        numerical = -(rhs_at(1.0e-6) - rhs_at(-1.0e-6)) / (2.0e-6)
        np.testing.assert_allclose(
            expected, numerical, rtol=3.0e-3, atol=2.0e-5,
        )
    finally:
        trial.solution.x.array[:] = accepted
        trial.solution.x.scatter_forward()
        trial.response.rollback()

    partial = _patch(**options)[3]
    partial.solve(until=0.5)
    saved = partial.save_checkpoint(tmp_path / "active_concave_edge")
    accepted_u = partial.solution.x.array.copy()
    accepted_state = partial.response.state.committed_state_vectors().copy()
    accepted_work = partial.energy_history[-1].tool_work
    record_energy = partial._record_energy

    def reject_after_recording(*args, **kwargs):
        record_energy(*args, **kwargs)
        raise ValueError("intentional active-edge acceptance rejection")

    monkeypatch.setattr(partial, "_record_energy", reject_after_recording)
    with pytest.raises(RuntimeError, match="active-edge acceptance rejection"):
        partial.solve(until=2.0 / 3.0)
    assert partial.accepted_load_factor == pytest.approx(0.5)
    np.testing.assert_array_equal(partial.solution.x.array, accepted_u)
    np.testing.assert_array_equal(
        partial.response.state.committed_state_vectors(), accepted_state,
    )
    assert partial.energy_history[-1].tool_work == accepted_work
    assert partial.contact_provider.snapshot_motion() == pytest.approx(0.5)

    resumed = _patch(**options)[3]
    resumed.load_checkpoint(saved)
    assert resumed.solve_result().status == "completed"
    np.testing.assert_allclose(
        resumed.solution.x.array, reference.solution.x.array,
        rtol=1.0e-8, atol=1.0e-10,
    )
    np.testing.assert_allclose(
        resumed.response.state.committed_state_vectors(),
        reference.response.state.committed_state_vectors(),
        rtol=1.0e-8, atol=1.0e-10,
    )
    assert resumed.energy_history[-1].tool_work == pytest.approx(
        reference.energy_history[-1].tool_work,
    )


def test_fast_closed_sign_preserves_active_edge_plastic_step_result(tmp_path):
    options = dict(
        triangulated=True, closed_concave_tool=True,
        closed_concave_extent=0.7, closed_concave_corner=(0.35, 0.19),
        closed_concave_thickness=0.5, gap=0.005,
        tool_translation=(-0.055, 0.0, 0.0), penalty=20.0,
        applied_traction=0.0, increments=6, yield_stress=0.05,
    )
    baseline = _patch(**options)[3]
    fast = _patch(**options, closed_sign_method="angle_weighted_pseudonormal")[3]
    assert baseline.solve_result().status == fast.solve_result().status == "completed"
    np.testing.assert_allclose(fast.solution.x.array, baseline.solution.x.array)
    np.testing.assert_allclose(
        fast.response.state.committed_state_vectors(),
        baseline.response.state.committed_state_vectors(),
    )
    assert fast.energy_history[-1].tool_work == pytest.approx(
        baseline.energy_history[-1].tool_work,
    )
    saved = baseline.save_checkpoint(tmp_path / "reference_sign")
    with pytest.raises(ValueError, match="identity"):
        _patch(**options, closed_sign_method="angle_weighted_pseudonormal")[3].load_checkpoint(saved)


def test_active_reentrant_edge_moving_tool_force_moment_are_work_conjugate():
    # The normal force at a physical edge is not the planar-face branch.
    # Verify the tool moment by an independent point-traction integral, then
    # compare generalized work against a fixed-displacement energy derivative.
    step = _patch(
        triangulated=True, closed_concave_tool=True,
        closed_concave_extent=0.7, closed_concave_corner=(0.35, 0.19),
        closed_concave_thickness=0.5, gap=0.005,
        tool_translation=(-0.055, 0.0, 0.0),
        tool_rotation=(0.0, 0.0, 0.01),
        penalty=20.0, applied_traction=0.0, increments=6,
        yield_stress=0.05,
    )[3]
    assert step.solve_result().status == "completed"
    provider = step.contact_provider
    samples = provider.accepted_point_samples()
    surface = provider.pair.rigid_body.surface
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

    motion = provider.pair.rigid_body.motion_schedule.motion
    current_reference = np.asarray(motion.state(1.0)["reference_point"])
    point_force = (
        samples["reference_weights"] * samples["normal_pressures"]
    )[:, None] * samples["surface_normals"]
    independent_force = np.sum(point_force, axis=0)
    independent_moment = np.sum(np.cross(
        samples["master_closest_points"] - current_reference,
        point_force,
    ), axis=0)
    dual = step._contact_dual()
    np.testing.assert_allclose(dual.force[:3], independent_force, atol=1.0e-11)
    np.testing.assert_allclose(dual.force[3:], independent_moment, atol=1.0e-11)

    delta = 1.0e-6
    potential = step._contact_potential()
    try:
        provider.update_motion(1.0 - delta)
        preceding_potential = step._contact_potential()
    finally:
        provider.update_motion(1.0)
    work_derivative = float(np.dot(
        dual.force, motion.generalized_coordinate(1.0),
    ))
    assert (potential - preceding_potential) / delta == pytest.approx(
        work_derivative, rel=2.0e-4,
    )


def test_mixed_concave_and_plane_tools_keep_distinct_work_and_restart(tmp_path):
    def build():
        model, displacement, material, _ = _patch(
            triangulated=True, closed_concave_tool=True,
            closed_concave_extent=0.7, closed_concave_corner=(0.35, 0.19),
            closed_concave_thickness=0.5, gap=0.005,
            tool_translation=(-0.055, 0.0, 0.0), penalty=20.0,
            applied_traction=0.0, increments=6, yield_stress=0.05,
            make_step=False,
        )
        plane = boundary_models.rigid_plane(
            point=(1.02, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
        )
        motion = boundary_models.prescribed_rigid_motion(
            translation=(-0.04, 0.0, 0.0),
            reference_point=(1.02, 0.0, 0.0),
        )
        model.add_boundary_model(boundary_models.rigid_contact_pair(
            model.boundary_models[0].slave_boundary,
            boundary_models.rigid_body(
                plane,
                motion_schedule=boundary_models.prescribed_rigid_motion_schedule(
                    motion, start_time=0.0, end_time=1.0,
                ),
            ),
            penalty=5.0, name="independent_flat_tool",
        ))
        return model.step(
            target=displacement, material=material,
            incrementation=steps.fixed(6), progress=False,
        )

    continuous = build()
    result = continuous.solve_result()
    assert result.status == "completed"
    providers = continuous.contact_provider.providers
    names = tuple(provider.name for provider in providers)
    assert names == ("rigid_contact_pair", "independent_flat_tool")
    assert all(np.count_nonzero(provider.accepted_point_samples()["active"]) > 0
               for provider in providers)
    pair_work = dict(continuous.energy_history[-1].pair_tool_work)
    assert set(pair_work) == set(names)
    assert sum(pair_work.values()) == pytest.approx(
        continuous.energy_history[-1].tool_work,
    )
    assert tuple(item["constraint_name"] for item in
                 result.metadata["constraint_duals"]) == names
    accepted = continuous.solution.x.array.copy()
    direction = np.random.default_rng(20261018).normal(size=accepted.shape)
    scale = 1.0e-7
    selected = continuous.contact_provider.evaluate()
    try:
        continuous.solution.x.array[:] = accepted + scale * direction
        plus = continuous.contact_provider.evaluate()
        continuous.solution.x.array[:] = accepted - scale * direction
        minus = continuous.contact_provider.evaluate()
    finally:
        continuous.solution.x.array[:] = accepted
    np.testing.assert_allclose(
        selected.tangent_action(direction.reshape(-1, 3)),
        (plus.nodal_structural_residual - minus.nodal_structural_residual)
        / (2.0 * scale),
        rtol=2.0e-4, atol=2.0e-5,
    )

    interrupted = build()
    interrupted.solve(until=0.5)
    checkpoint = interrupted.save_checkpoint(tmp_path / "mixed_closed_flat")
    resumed = build()
    resumed.load_checkpoint(checkpoint)
    assert resumed.solve_result().status == "completed"
    np.testing.assert_allclose(
        resumed.solution.x.array, continuous.solution.x.array,
    )
    np.testing.assert_allclose(
        resumed.response.state.committed_state_vectors(),
        continuous.response.state.committed_state_vectors(),
    )
    assert dict(resumed.energy_history[-1].pair_tool_work) == pytest.approx(pair_work)


def test_closed_concave_example_exports_accepted_public_result(tmp_path):
    script = Path(__file__).resolve().parents[1] / (
        "examples/finite_j2_closed_concave_contact.py"
    )
    run_case = runpy.run_path(str(script))["run_case"]
    summary = run_case(cells=1, increments=6, penalty=20.0, output=tmp_path)
    assert summary["status"] == "completed"
    assert summary["contact_quadrature_degree"] == 2
    assert summary["line_search"] == "backtracking"
    assert summary["active_reentrant_edge_points"] == 1
    assert summary["active_reference_area"] > 0.0
    assert summary["maximum_penetration"] > 0.0
    assert summary["maximum_equivalent_plastic_strain"] > 0.0
    assert (tmp_path / "fields.xdmf").is_file()
    assert (tmp_path / "result.json").is_file()
    manifest = json.loads((tmp_path / "result.json").read_text())
    inputs = manifest["scientific_inputs"]
    assert inputs["complete"] and not inputs["missing"]
    assert inputs["record"]["contact"]["value"]["law"]["penalty"] == 20.0
    assert inputs["record"]["contact_integration"]["triangle_degree"] == 2
    load_histories = [
        record for record in manifest["history_records"]
        if record["abscissa_name"] == "load_factor"
    ]
    assert load_histories
    assert all(record["abscissa_unit"] is None for record in load_histories)

    refined_output = tmp_path / "refined_face_rule"
    refined = run_case(
        cells=1, increments=6, penalty=20.0, output=refined_output,
        contact_quadrature_degree=7,
    )
    assert refined["status"] == "completed"
    assert refined["contact_quadrature_degree"] == 7
    assert refined["maximum_penetration"] > 0.0
    assert (refined_output / "fields.xdmf").is_file()
    assert (refined_output / "result.json").is_file()
    refined_inputs = json.loads((refined_output / "result.json").read_text())[
        "scientific_inputs"
    ]
    assert refined_inputs["record"]["contact_integration"]["triangle_degree"] == 7
    assert refined_inputs["fingerprint"] != inputs["fingerprint"]

    full_newton = run_case(
        cells=1, increments=6, penalty=20.0,
        output=tmp_path / "basic_newton", line_search="basic",
    )
    assert full_newton["status"] == "completed"
    assert full_newton["line_search"] == "basic"


def test_closed_concave_example_iterative_linear_policy_matches_direct(tmp_path):
    script = Path(__file__).resolve().parents[1] / (
        "examples/finite_j2_closed_concave_contact.py"
    )
    run_case = runpy.run_path(str(script))["run_case"]
    options = dict(cells=1, increments=6, penalty=20.0, line_search="basic")
    direct = run_case(**options, linear_policy="direct", output=tmp_path / "direct")
    iterative = run_case(
        **options, linear_policy="gmres_gamg", output=tmp_path / "iterative",
    )
    assert direct["status"] == iterative["status"] == "completed"
    assert direct["accepted_increments"] == iterative["accepted_increments"] == 6
    for quantity in (
        "contact_resultant", "maximum_penetration",
        "maximum_equivalent_plastic_strain", "tool_work",
        "mechanical_energy_residual",
    ):
        np.testing.assert_allclose(
            direct[quantity], iterative[quantity], rtol=1e-6, atol=1e-9,
        )
    direct_result = json.loads((tmp_path / "direct" / "result.json").read_text())
    iterative_result = json.loads((tmp_path / "iterative" / "result.json").read_text())
    assert (direct_result["scientific_inputs"]["fingerprint"]
            != iterative_result["scientific_inputs"]["fingerprint"])
    policy = iterative_result["scientific_inputs"]["record"]["solver"]["value"]
    assert policy["linear_solver"]["ksp_type"] == "gmres"
    assert policy["linear_solver"]["pc_type"] == "gamg"
    direct_histories = {item["name"]: item for item in direct_result["history_records"]}
    iterative_histories = {
        item["name"]: item for item in iterative_result["history_records"]
    }
    for name in (
        "tool_work", "contact_potential", "plastic_dissipation",
        "mechanical_energy_residual",
    ):
        reference = direct_histories[name]
        candidate = iterative_histories[name]
        assert reference["abscissa"] == candidate["abscissa"]
        np.testing.assert_allclose(
            reference["values"], candidate["values"], rtol=1e-6, atol=1e-9,
        )


@pytest.mark.parametrize("tool_rotation", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.01)])
@pytest.mark.parametrize(
    "closed_sign_method", ["winding_reference", "angle_weighted_pseudonormal"],
)
def test_reviewed_closed_concave_smooth_face_friction_matches_convex_limit_and_restart(
    tmp_path, tool_rotation, closed_sign_method,
):
    options = dict(
        triangulated=True, yield_stress=2.0,
        tool_translation=(-0.016, 0.012, 0.0),
        tool_rotation=tool_rotation,
        closed_sign_method=closed_sign_method,
        applied_traction=0.0, increments=4,
        friction_coefficient=0.2, tangential_penalty=100.0,
    )

    def build(*, closed):
        return _patch(**options, closed_concave_tool=closed,
                      convex_tool=not closed)[3]

    reference = build(closed=False)
    reference.solve()
    continuous = build(closed=True)
    assert continuous.solve_result().status == "completed"
    assert continuous.energy_history[-1].friction_dissipation > 0.0
    np.testing.assert_allclose(
        continuous.solution.x.array, reference.solution.x.array,
        rtol=1e-8, atol=1e-10,
    )
    assert continuous.energy_history[-1].friction_dissipation == pytest.approx(
        reference.energy_history[-1].friction_dissipation, rel=1e-8,
    )
    for name in (
        "stored_energy", "plastic_dissipation", "contact_potential",
        "tool_work", "friction_dissipation", "mechanical_energy_residual",
    ):
        np.testing.assert_allclose(
            [getattr(frame, name) for frame in continuous.energy_history],
            [getattr(frame, name) for frame in reference.energy_history],
            rtol=1e-8, atol=1e-10,
        )

    accepted = continuous.solution.x.array.copy()
    # Differentiate a genuine trial against the earlier accepted friction
    # State. Reloading at the final accepted state lies on a Coulomb cap kink.
    trial = build(closed=True)
    trial.solve(until=0.5)
    trial.contact_provider.update_motion(1.0)
    trial.solution.x.array[:] = accepted
    direction = np.random.default_rng(20261018).normal(size=accepted.shape)
    scale = 1e-7
    selected = trial.contact_provider.evaluate()
    try:
        trial.solution.x.array[:] = accepted + scale * direction
        plus = trial.contact_provider.evaluate()
        trial.solution.x.array[:] = accepted - scale * direction
        minus = trial.contact_provider.evaluate()
    finally:
        trial.solution.x.array[:] = accepted
    np.testing.assert_allclose(
        selected.tangent_action(direction.reshape(-1, 3)),
        (plus.nodal_structural_residual - minus.nodal_structural_residual)
        / (2 * scale),
        rtol=3e-4, atol=3e-5,
    )

    interrupted = build(closed=True)
    interrupted.solve(until=0.5)
    checkpoint = interrupted.save_checkpoint(tmp_path / "closed_face_friction")
    resumed = build(closed=True)
    resumed.load_checkpoint(checkpoint)
    resumed.solve()
    np.testing.assert_allclose(resumed.solution.x.array, accepted, rtol=1e-9, atol=1e-10)
    assert resumed.energy_history[-1].friction_dissipation == pytest.approx(
        continuous.energy_history[-1].friction_dissipation, rel=1e-9,
    )


def test_reviewed_closed_concave_active_edge_friction_fails_closed():
    _, _, _, step = _patch(
        triangulated=True, closed_concave_tool=True,
        closed_concave_extent=0.7, closed_concave_corner=(0.35, 0.19),
        closed_concave_thickness=0.5, gap=0.005,
        penalty=20.0, increments=6, applied_traction=0.0,
        tool_translation=(-0.055, 0.0, 0.0), yield_stress=0.05,
        friction_coefficient=0.2, tangential_penalty=100.0,
    )
    with pytest.raises(ValueError, match="AFM-IMPLICIT-FRICTION-BRANCH"):
        step.solve()
    accepted_factor = step.accepted_load_factor
    assert step.contact_provider.friction_state.trial is None
    assert step.contact_provider.friction_kinematics.trial is None
    reference = _patch(
        triangulated=True, closed_concave_tool=True,
        closed_concave_extent=0.7, closed_concave_corner=(0.35, 0.19),
        closed_concave_thickness=0.5, gap=0.005,
        penalty=20.0, increments=6, applied_traction=0.0,
        tool_translation=(-0.055, 0.0, 0.0), yield_stress=0.05,
        friction_coefficient=0.2, tangential_penalty=100.0,
    )[3]
    if accepted_factor > 0.0:
        reference.solve(until=accepted_factor)
    np.testing.assert_allclose(step.solution.x.array, reference.solution.x.array)
    np.testing.assert_allclose(
        step.response.state.committed_state_vectors(),
        reference.response.state.committed_state_vectors(),
    )


@pytest.mark.parametrize("cells", [1, 2])
@pytest.mark.parametrize("penalty", [500.0, 1000.0, 2000.0])
def test_closed_concave_face_contact_balances_independent_applied_force(
    cells, penalty,
):
    # Published finite-sliding verification suites check that an imposed
    # pressure load is balanced by the contact resultant. This separate
    # three-dimensional manufactured case uses a free x translation, with
    # tangential left-face restraints only, so no hidden x support can carry
    # the prescribed load. It is not an Abaqus same-input comparison.
    imposed = 0.5
    _, _, _, step = _patch(
        triangulated=True, closed_concave_tool=True, cells=cells,
        gap=-0.005, penalty=penalty, yield_stress=1.0e9,
        clamp_left=False, left_traction=imposed,
        applied_traction=0.0, increments=3,
    )
    assert step.solve_result().status == "completed"
    reaction = step.contact_provider.evaluate().contact_force_on_structure
    np.testing.assert_allclose(reaction, (-imposed, 0.0, 0.0),
                               rtol=1e-7, atol=1e-7)
    samples = step.contact_provider.accepted_point_samples()
    assert np.all(samples["active"])
    np.testing.assert_allclose(
        np.average(samples["penetrations"], weights=samples["reference_weights"]),
        imposed / penalty, rtol=1e-7, atol=1e-9,
    )


def test_refined_contact_quadrature_is_explicit_and_checkpoint_identified(tmp_path):
    options = dict(
        triangulated=True, convex_tool=True, cells=2, hexahedral=True,
        yield_stress=1.0e9, tool_translation=(-0.008, 0.0, 0.0),
        applied_traction=0.0, increments=3,
    )
    default = _patch(**options)[3]
    refined = _patch(**options, contact_quadrature_degree=5)[3]
    assert default.contact_provider.adapter.trace.point_count == 16
    assert refined.contact_provider.adapter.trace.point_count == 36
    assert default.contact_provider.summary()["trace"]["quadrature_rule"] != (
        refined.contact_provider.summary()["trace"]["quadrature_rule"]
    )
    default.solve()
    refined.solve()
    np.testing.assert_allclose(
        refined.solution.x.array, default.solution.x.array,
        rtol=1.0e-7, atol=1.0e-9,
    )
    assert refined.energy_history[-1].tool_work == pytest.approx(
        default.energy_history[-1].tool_work, rel=1.0e-7,
    )
    saved = refined.save_checkpoint(tmp_path / "refined_contact")
    changed = _patch(**options)[3]
    with pytest.raises(ValueError, match="identity"):
        changed.load_checkpoint(saved)


def test_quadratic_contact_quadrature_checkpoint_rejects_rule_change(tmp_path):
    options = dict(
        triangulated=True, convex_tool=True, cells=1, hexahedral=True,
        displacement_degree=2, yield_stress=1.0e9,
        tool_translation=(-0.008, 0.0, 0.0), applied_traction=0.0,
        increments=2,
    )
    refined = _patch(**options, contact_quadrature_degree=7)[3]
    assert refined.contact_provider.adapter.trace.point_count == 16
    assert refined.quadrature_degree == 4
    refined.solve()
    assert refined.accepted_load_factor == pytest.approx(1.0)
    saved = refined.save_checkpoint(tmp_path / "quadratic_refined_contact")
    same = _patch(**options, contact_quadrature_degree=7)[3]
    same.load_checkpoint(saved)
    np.testing.assert_allclose(same.solution.x.array, refined.solution.x.array)
    changed = _patch(**options)[3]
    assert changed.contact_provider.adapter.trace.point_count == 9
    with pytest.raises(ValueError, match="identity"):
        changed.load_checkpoint(saved)


def test_quadratic_hexahedron_rejects_underintegrated_bulk_rule():
    model, displacement, material, _ = _patch(
        triangulated=True, cells=1, hexahedral=True,
        displacement_degree=2, make_step=False,
    )
    capability = step_capability(
        model, target=displacement,
        options={"material": material, "quadrature_degree": 2},
    )
    assert not capability["supported"]
    assert any(
        item["message"].startswith("AFM-J2-QUADRATURE")
        for item in capability["rejection_reasons"]
    )
    with pytest.raises(NotImplementedError, match="AFM-J2-QUADRATURE"):
        model.step(
            target=displacement, material=material,
            quadrature_degree=2,
        )


def test_refined_tetra_contact_quadrature_keeps_planar_solution_and_restart_identity(
    tmp_path,
):
    model, displacement, material, _ = _patch(
        triangulated=True, make_step=False,
    )
    capability = step_capability(
        model, target=displacement,
        options={"material": material, "contact_quadrature_degree": 5},
    )
    assert capability["supported"]
    options = dict(
        triangulated=True, yield_stress=1.0e9,
        tool_translation=(-0.008, 0.0, 0.0), applied_traction=0.0,
        increments=3,
    )
    default = _patch(**options)[3]
    refined = _patch(**options, contact_quadrature_degree=5)[3]
    assert refined.contact_provider.adapter.trace.point_count > (
        default.contact_provider.adapter.trace.point_count
    )
    assert "triangle_basix_degree_5" in (
        refined.contact_provider.adapter.quadrature_rule
    )
    assert default.solve_result().status == refined.solve_result().status == "completed"
    np.testing.assert_allclose(
        refined.solution.x.array, default.solution.x.array,
        rtol=1.0e-7, atol=1.0e-9,
    )
    saved = refined.save_checkpoint(tmp_path / "refined_tetra_contact")
    same = _patch(**options, contact_quadrature_degree=5)[3]
    same.load_checkpoint(saved)
    np.testing.assert_allclose(same.solution.x.array, refined.solution.x.array)
    with pytest.raises(ValueError, match="identity"):
        _patch(**options)[3].load_checkpoint(saved)
    with pytest.raises(ValueError, match="identity"):
        _patch(**options, contact_quadrature_degree=7)[3].load_checkpoint(saved)


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


@pytest.mark.parametrize("yield_stress", [1.0e9, 2.0])
def test_convex_tool_tracks_noncoplanar_feature_during_incremental_solve(
    tmp_path, yield_stress,
):
    def build():
        return _patch(
            triangulated=True, convex_tool=True, convex_y_min=0.15,
            yield_stress=yield_stress, tool_translation=(-0.06, 0, 0),
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
    if yield_stress < 1.0e8:
        assert np.max(
            step.response.state.committed["equivalent_plastic_strain"].function.x.array
        ) > 0.0

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


def test_rigid_modes_gamg_preserves_plastic_feature_switch_state(tmp_path):
    options = dict(
        triangulated=True, convex_tool=True, convex_y_min=0.15,
        yield_stress=2.0, tool_translation=(-0.06, 0, 0),
        applied_traction=0.0, increments=12,
    )
    direct = _patch(**options)[3]
    iterative_options = {
        **options,
        "linear_solver": solvers.LinearSolverOptions(
            ksp_type="gmres", pc_type="gamg", rtol=1.0e-8, max_it=500,
        ),
    }
    iterative = _patch(**iterative_options)[3]
    assert direct.solve_result().status == "completed"
    assert iterative.solve_result().status == "completed"
    assert direct.summary()["linear_preconditioner_near_nullspace"] is None
    assert (
        iterative.summary()["linear_preconditioner_near_nullspace"]
        == "reference_rigid_body_3d"
    )
    assert np.max(
        iterative.response.state.committed[
            "equivalent_plastic_strain"
        ].function.x.array
    ) > 0.0
    np.testing.assert_allclose(
        iterative.solution.x.array, direct.solution.x.array,
        rtol=1.0e-7, atol=1.0e-9,
    )
    np.testing.assert_allclose(
        iterative.response.state.committed_state_vectors(),
        direct.response.state.committed_state_vectors(),
        rtol=1.0e-7, atol=1.0e-9,
    )
    assert iterative.energy_history[-1].tool_work == pytest.approx(
        direct.energy_history[-1].tool_work, rel=1.0e-7,
    )
    partial = _patch(**iterative_options)[3]
    partial.solve(until=0.25)
    saved = partial.save_checkpoint(tmp_path / "plastic_gamg_feature_switch")
    resumed = _patch(**iterative_options)[3]
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(
        resumed.solution.x.array, iterative.solution.x.array,
        rtol=1.0e-8, atol=1.0e-10,
    )
    np.testing.assert_allclose(
        resumed.response.state.committed_state_vectors(),
        iterative.response.state.committed_state_vectors(),
        rtol=1.0e-8, atol=1.0e-10,
    )


def test_convex_tool_friction_feature_switch_keeps_plastic_and_tangential_state(
    tmp_path,
):
    def build():
        return _patch(
            triangulated=True, convex_tool=True, convex_y_min=0.15,
            yield_stress=2.0, tool_translation=(-0.06, 0.0, 0.0),
            applied_traction=0.0, increments=12,
            friction_coefficient=0.2, tangential_penalty=100.0,
        )[3]

    reference = build()
    reference.solve(until=0.25)
    _evaluation, early_record, early_response, _factor = (
        reference.contact_provider._current_point_response()
    )
    reference.solve()
    _evaluation, late_record, late_response, _factor = (
        reference.contact_provider._current_point_response()
    )
    early = early_record.projection
    late = late_record.projection
    switching = (
        early_response.active & late_response.active
        & (early.entity_ids != late.entity_ids)
        & (np.sum(early.normals * late.normals, axis=1) < 0.99)
    )
    assert np.any(switching)
    assert reference.energy_history[-1].friction_dissipation > 0.0
    assert reference.contact_provider.friction_state.accepted is not None
    assert np.max(
        reference.response.state.committed["equivalent_plastic_strain"].function.x.array
    ) > 0.0

    partial = build()
    partial.solve(until=0.25)
    saved = partial.save_checkpoint(tmp_path / "convex_friction_switch")
    resumed = build()
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(
        resumed.solution.x.array, reference.solution.x.array, rtol=1e-9, atol=1e-10,
    )
    np.testing.assert_allclose(
        resumed.response.state.committed_state_vectors(),
        reference.response.state.committed_state_vectors(), rtol=1e-9, atol=1e-10,
    )
    assert resumed.energy_history[-1].friction_dissipation == pytest.approx(
        reference.energy_history[-1].friction_dissipation, rel=1e-9,
    )


def test_regional_j2_materials_share_moving_convex_friction_transaction(tmp_path):
    options = dict(
        triangulated=True, convex_tool=True, cells=2,
        regional_material=True, yield_stress=2.0,
        tool_translation=(-0.016, 0.012, 0.0),
        applied_traction=0.0, increments=4,
        friction_coefficient=0.2, tangential_penalty=100.0,
    )

    def build():
        model, displacement, material, step = _patch(**options)
        assert material is None
        capability = step_capability(model, target=displacement)
        assert capability["supported"]
        assert isinstance(step.material, constitutive.QuadratureMaterialMap)
        assert set(step.material.cell_regions) == {1, 2}
        return step

    continuous = build()
    continuous.solve()
    assert continuous.energy_history[-1].tool_work > 0.0
    assert continuous.energy_history[-1].friction_dissipation > 0.0
    assert np.max(
        continuous.response.state.committed[
            "equivalent_plastic_strain"
        ].function.x.array
    ) > 0.0

    interrupted = build()
    interrupted.solve(until=0.5)
    checkpoint = interrupted.save_checkpoint(tmp_path / "regional_convex_contact")
    resumed = build()
    resumed.load_checkpoint(checkpoint)
    resumed.solve()
    np.testing.assert_allclose(
        resumed.solution.x.array, continuous.solution.x.array,
        rtol=1.0e-9, atol=1.0e-10,
    )
    np.testing.assert_allclose(
        resumed.response.state.committed_state_vectors(),
        continuous.response.state.committed_state_vectors(),
        rtol=1.0e-9, atol=1.0e-10,
    )
    assert resumed.energy_history[-1].friction_dissipation == pytest.approx(
        continuous.energy_history[-1].friction_dissipation, rel=1.0e-9,
    )
    changed = _patch(**{**options, "yield_stress": 2.5})[3]
    with pytest.raises(ValueError, match="identity"):
        changed.load_checkpoint(checkpoint)
    assert not changed.energy_history


def test_two_moving_convex_friction_tools_keep_separate_work_and_restart(
    tmp_path, monkeypatch,
):
    def build():
        model, displacement, material, _ = _patch(
            triangulated=True, convex_tool=True, make_step=False,
            tool_translation=(-0.018, 0.06, 0.0), applied_traction=0.0,
            friction_coefficient=0.2, tangential_penalty=100.0,
        )
        first = model.boundary_models[0]
        second_motion = boundary_models.prescribed_rigid_motion(
            translation=(-0.015, 0.05, 0.0),
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
            penalty=500.0, friction_coefficient=0.2,
            tangential_penalty=100.0, name="second_convex_tool",
        ))
        return model.step(
            target=displacement, material=material,
            incrementation=steps.fixed(4), progress=False,
        )

    continuous = build()
    result = continuous.solve_result()
    assert result.status == "completed"
    names = tuple(item.name for item in continuous.contact_provider.providers)
    assert names == ("rigid_contact_pair", "second_convex_tool")
    assert all(item.friction_state.accepted is not None
               for item in continuous.contact_provider.providers)
    pair_work = dict(continuous.energy_history[-1].pair_tool_work)
    assert set(pair_work) == set(names)
    assert sum(pair_work.values()) == pytest.approx(
        continuous.energy_history[-1].tool_work,
    )
    assert continuous.energy_history[-1].friction_dissipation > 0.0
    for provider, dual in zip(
        continuous.contact_provider.providers,
        result.metadata["constraint_duals"], strict=True,
    ):
        samples = provider.accepted_point_samples()
        assert dual["constraint_name"] == provider.name
        resultant = np.sum(samples["reference_weights"][:, None] * (
            samples["normal_pressures"][:, None] * samples["surface_normals"]
            + samples["surface_friction_tractions"]
        ), axis=0)
        np.testing.assert_allclose(resultant, dual["resultant"], atol=1.0e-9)

    partial = build()
    partial.solve(until=0.5)
    saved = partial.save_checkpoint(tmp_path / "two_moving_convex_friction")
    accepted_half = partial.contact_provider.snapshot_accepted_state()
    second = partial.contact_provider.providers[1]
    original_commit = second.commit_accepted

    def fail_after_second_commit():
        original_commit()
        raise RuntimeError("injected second convex-tool commit failure")

    monkeypatch.setattr(second, "commit_accepted", fail_after_second_commit)
    with pytest.raises(RuntimeError, match="second convex-tool commit failure"):
        partial.solve()
    assert partial.accepted_load_factor == pytest.approx(0.5)
    assert partial.contact_provider.snapshot_accepted_state() == accepted_half
    resumed = build()
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(
        resumed.solution.x.array, continuous.solution.x.array,
        rtol=1.0e-9, atol=1.0e-10,
    )
    np.testing.assert_allclose(
        resumed.response.state.committed_state_vectors(),
        continuous.response.state.committed_state_vectors(),
        rtol=1.0e-9, atol=1.0e-10,
    )
    assert resumed.energy_history[-1].tool_work == pytest.approx(
        continuous.energy_history[-1].tool_work, rel=1.0e-9,
    )
    assert resumed.energy_history[-1].friction_dissipation == pytest.approx(
        continuous.energy_history[-1].friction_dissipation, rel=1.0e-9,
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


def test_implicit_coplanar_facet_friction_uses_the_public_j2_step():
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
    assert capability["supported"]
    step = model.step(
        target=displacement, material=material,
        incrementation=steps.fixed(2), progress=False,
    )
    result = step.solve_result()
    assert result.status == "completed"
    assert step.contact_provider.friction_state.accepted is not None
    assert step.energy_history[-1].friction_dissipation >= 0.0


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


@pytest.mark.parametrize("friction_coefficient,tangential_penalty", [
    (None, None), (0.3, 200.0),
])
def test_active_free_tool_edge_fails_closed_and_preserves_accepted_state(
    friction_coefficient, tangential_penalty,
):
    _, _, _, step = _patch(
        triangulated=True,
        cells=2,
        tool_y_max=0.4,
        tool_translation=(-0.006, 0.0, 0.0),
        friction_coefficient=friction_coefficient,
        tangential_penalty=tangential_penalty,
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
    if friction_coefficient is not None:
        assert step.contact_provider.friction_state.accepted is None
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


@pytest.mark.parametrize(
    ("displacement_degree", "hexahedral"), [(1, False), (2, True)],
)
def test_facet_contact_lifts_shared_strong_dofs_and_keeps_dirichlet_rows(
    displacement_degree, hexahedral,
):
    _, _, _, step = _patch(
        triangulated=True, constrain_contact=True, contact_value=0.002,
        tool_translation=(-0.02, 0.0, 0.0), tilt=0.01,
        applied_traction=0.0,
        displacement_degree=displacement_degree, hexahedral=hexahedral,
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
