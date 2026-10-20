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
    amplitudes,
    boundary_models,
    constitutive,
    fields,
    mesh,
    models,
    results,
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
           initial_penetration_limit=None,
           transverse_boundary_fixed=False, load_amplitude=None,
           increment_control=None, normal_augmentation=None):
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
    if transverse_boundary_fixed:
        model.fix(displacement, on=left, component=0, value=0.0)
        exterior = mesh.boundary(
            domain, lambda x: np.ones(x.shape[1], dtype=bool),
            name="transverse_fixed_exterior",
        )
        model.fix(displacement, on=exterior, component=1, value=0.0)
        model.fix(displacement, on=exterior, component=2, value=0.0)
    elif clamp_left:
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
                right, boundary_models.rigid_body(
                    surface,
                    motion_schedule=(
                        None if tool_motion is None
                        else boundary_models.prescribed_rigid_motion_schedule(
                            tool_motion, start_time=0, end_time=schedule_end,
                        )
                    ),
                ),
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
    if load_amplitude is not None:
        step_options["amplitude"] = load_amplitude
    if contact_quadrature_degree is not None:
        step_options["contact_quadrature_degree"] = contact_quadrature_degree
    if normal_augmentation is not None:
        step_options["normal_augmentation"] = normal_augmentation
    step = model.step(
        target=displacement,
        **step_options,
        incrementation=(
            steps.fixed(increments)
            if increment_control is None else increment_control
        ),
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


def _small_single_branch_j2_step(
    *, closed: bool, flat_seam: bool = False, second_plane: bool = False,
    active_second_plane: bool = False, tool_rotation=(0.0, 0.0, 0.0),
    open_sheet: bool = False, integration_override: str | None = None,
    initial_penetration_limit: float | None = None,
    open_sheet_x: float = 1.10,
    open_sheet_competitor: bool = False,
    open_sheet_competitor_kind: str = "old_refinement",
    radial_sphere: bool = False,
    radial_cylinder: bool = False,
    radial_integration: str | None = None,
    radial_motion: tuple[float, float, float] | None = None,
    radial_mesh_divisions: int = 1,
):
    if sum((closed, open_sheet, radial_sphere, radial_cylinder)) > 1:
        raise ValueError("Choose one reviewed rigid geometry.")
    domain = dolfinx_mesh.create_unit_cube(
        MPI.COMM_SELF,
        radial_mesh_divisions, radial_mesh_divisions, radial_mesh_divisions,
        cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    coordinates = domain.geometry.x
    coordinates[:, 0] = 1.05 + 0.10 * coordinates[:, 0]
    coordinates[:, 1] = (1.45 if flat_seam else 1.25) + 0.10 * coordinates[:, 1]
    coordinates[:, 2] = 0.45 + 0.10 * coordinates[:, 2]
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    contact = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 1.05), name="contact",
    )
    support = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 1.15), name="support",
    )
    model.clamp(displacement, on=support)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=0.0,
        yield_stress=2.0, hardening_modulus=100.0,
    ))
    if closed:
        polygon = np.asarray(
            ((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)), dtype=float,
        )
        front = np.column_stack((np.full(6, 1.10), polygon))
        back = front.copy()
        back[:, 0] += 0.5
        top = ((0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5))
        triangles = [tuple(reversed(face)) for face in top]
        triangles += [tuple(6 + index for index in face) for face in top]
        for first in range(6):
            second = (first + 1) % 6
            triangles.extend(((first, second, second + 6),
                              (first, second + 6, first + 6)))
        surface = boundary_models.closed_triangulated_rigid_solid(
            vertices=np.vstack((front, back)), triangles=triangles,
            reviewed_embedded=True, name="reviewed_L_tool",
        )
        mode = "closed_planar_branch_partition"
    elif radial_sphere:
        surface = boundary_models.rigid_sphere(
            center=(1.149, 1.30, 0.50), radius=0.1,
            name="small_circular_patch_tool",
        )
        mode = "radial_active_partition" if radial_integration is None else radial_integration
    elif radial_cylinder:
        surface = boundary_models.rigid_cylinder(
            axis_point=(1.149, 1.30, 0.50),
            axis_direction=(0., 1., 0.), radius=0.1,
            axial_interval=(-0.1, 0.1),
            name="small_cylindrical_patch_tool",
        )
        mode = "radial_active_partition" if radial_integration is None else radial_integration
    elif open_sheet:
        vertices = (
            (open_sheet_x, 0., -1.), (open_sheet_x, 0., 1.),
            (open_sheet_x, 2., 1.), (open_sheet_x, 2., -1.),
        )
        triangles = ((0, 1, 2), (0, 2, 3))
        if open_sheet_competitor:
            if open_sheet_competitor_kind == "finite_vertex":
                competitor_vertices = (
                    (open_sheet_x - 0.3, 1.42, 0.5),
                    (open_sheet_x - 0.1, 1.42, 0.5),
                    (open_sheet_x - 0.3, 1.90, 0.5),
                )
            else:
                competitor_vertices = (
                    (open_sheet_x - 0.1, 1.45, 0.45),
                    (open_sheet_x - 0.1, 1.45, 0.55),
                    (open_sheet_x - 0.1, 1.50, 0.45),
                )
            vertices += competitor_vertices
            triangles += ((4, 5, 6),)
        surface = boundary_models.triangulated_rigid_surface(
            vertices=vertices,
            triangles=triangles,
            name="reviewed_open_sheet_tool",
        )
        mode = "reviewed_planar_branch_partition"
    else:
        surface = boundary_models.rigid_plane(
            point=(1.10, 1.3, 0.5), normal=(-1.0, 0.0, 0.0),
        )
        mode = "planar_active_partition"
    if radial_sphere or radial_cylinder:
        motion = None if radial_motion is None else boundary_models.prescribed_rigid_motion(
            translation=radial_motion, reference_point=(1.149, 1.30, 0.50),
        )
    else:
        motion = boundary_models.prescribed_rigid_motion(
            translation=(-0.08, 0.0, 0.0), rotation=tool_rotation,
            reference_point=(1.10, 1.3, 0.5),
        )
    primary = boundary_models.rigid_contact_pair(
        contact,
        boundary_models.rigid_body(
            surface, motion_schedule=(
                None if motion is None else boundary_models.prescribed_rigid_motion_schedule(
                    motion, start_time=0.0, end_time=1.0,
                )
            ),
        ),
        penalty=5000.0,
        initial_penetration_limit=initial_penetration_limit,
    )
    model.add_boundary_model(primary)
    integration = mode if integration_override is None else integration_override
    if second_plane or active_second_plane:
        if active_second_plane:
            second_boundary = mesh.boundary(
                domain, lambda x: np.isclose(x[1], 1.35),
                name="transverse_contact",
            )
            second_surface = boundary_models.rigid_plane(
                point=(1.10, 1.37, 0.5), normal=(0., -1., 0.),
            )
            second_motion = boundary_models.prescribed_rigid_motion_schedule(
                boundary_models.prescribed_rigid_motion(
                    translation=(0., -0.04, 0.),
                    reference_point=(1.10, 1.37, 0.5),
                ),
                start_time=0.0, end_time=1.0,
            )
            second_name = "transverse_plane"
        else:
            second_boundary = contact
            second_surface = boundary_models.rigid_plane(
                point=(0., 0., 0.), normal=(1., 0., 0.),
            )
            second_motion = None
            second_name = "far_plane"
        model.add_boundary_model(boundary_models.rigid_contact_pair(
            second_boundary,
            boundary_models.rigid_body(
                second_surface, motion_schedule=second_motion,
            ),
            penalty=5000.0, name=second_name,
        ))
        integration = {
            primary.name: mode,
            second_name: "planar_active_partition",
        }
    assert step_capability(
        model, target=displacement,
        options={"material": material, "contact_integration": integration},
    )["supported"]
    step = model.step(
        target=displacement, material=material,
        incrementation=steps.fixed(6), contact_integration=integration,
        solver_options=solvers.newton(
            relative_tolerance=1e-8, absolute_tolerance=1e-9,
            maximum_iterations=25, line_search="backtracking",
        ),
        progress=False,
    )
    return step, displacement


def test_reviewed_open_sheet_initial_overlap_is_certified_in_ordinary_step():
    step, _ = _small_single_branch_j2_step(
        closed=False, open_sheet=True, flat_seam=True,
        initial_penetration_limit=0.01,
    )
    initial = step.contact_provider.current_point_diagnostics()["initial_contact"]
    assert initial["face_wise_bound_certified"] is True
    assert initial["bound_method"] == "certified_reviewed_planar_branch_nodal"
    assert initial["maximum_penetration_bound"] == 0.0
    assert initial["bound_sampled_point_count"] > 0
    assert step.solve_result().status == "completed"

    with pytest.raises(ValueError, match="AFM-CONTACT-INITIAL-OVERLAP"):
        _small_single_branch_j2_step(
            closed=False, open_sheet=True, flat_seam=True,
            open_sheet_x=1.04, initial_penetration_limit=0.005,
        )


def test_reviewed_open_sheet_planar_branch_matches_analytical_plane_global_j2(tmp_path):
    sheet, sheet_displacement = _small_single_branch_j2_step(
        closed=False, open_sheet=True, flat_seam=True,
    )
    plane, plane_displacement = _small_single_branch_j2_step(
        closed=False, flat_seam=True,
    )
    sheet_result = sheet.solve_result()
    assert sheet_result.status == "completed"
    assert plane.solve_result().status == "completed"
    np.testing.assert_allclose(
        sheet_displacement.value.x.array, plane_displacement.value.x.array,
        rtol=1e-8, atol=1e-9,
    )
    assert sheet.energy_history[-1].tool_work == pytest.approx(
        plane.energy_history[-1].tool_work, rel=1e-8,
    )
    evidence = sheet.contact_provider.current_point_diagnostics()
    assert evidence["feature_transition_integral_certified"] is True
    assert evidence["whole_face_planar_branch"]["certified"] is True
    assert evidence["active_point_count"] > 0
    feature_path = sheet_result.metadata["rigid_contact_feature_path"]
    assert feature_path["complete"] is True
    assert feature_path["pairs"][0]["all_observed_integrals_certified"] is True

    interrupted, _ = _small_single_branch_j2_step(
        closed=False, open_sheet=True, flat_seam=True,
    )
    interrupted.solve(until=0.5)
    saved = interrupted.save_checkpoint(tmp_path / "open_sheet")
    restored, _ = _small_single_branch_j2_step(
        closed=False, open_sheet=True, flat_seam=True,
    )
    restored.load_checkpoint(saved)
    assert restored.solve_result().status == "completed"
    np.testing.assert_allclose(
        restored.solution.x.array, sheet.solution.x.array, rtol=1e-9, atol=1e-10,
    )
    wrong_mode, _ = _small_single_branch_j2_step(
        closed=False, open_sheet=True, flat_seam=True,
        integration_override="gauss",
    )
    with pytest.raises(ValueError, match="identity|contact|constraint"):
        wrong_mode.load_checkpoint(saved)


def test_bounded_open_sheet_proof_refinement_matches_plane_in_global_j2():
    sheet, sheet_displacement = _small_single_branch_j2_step(
        closed=False, open_sheet=True, open_sheet_competitor=True,
    )
    plane, plane_displacement = _small_single_branch_j2_step(closed=False)
    provider = sheet.contact_provider
    current = provider.adapter.current_nodal_positions(sheet_displacement.value)
    face_nodes = provider.adapter.trace.node_ids.reshape(-1, 3, 3)[:, 0, :]
    from agentfem.boundary_models.contact_planar_branch_bound import (
        certify_reviewed_planar_branch,
    )

    coarse = certify_reviewed_planar_branch(
        current[face_nodes], provider.surface, projector=provider.projector,
        maximum_refinement_levels=0,
    )
    assert not coarse.certified
    assert coarse.status == "another_master_feature_can_compete"
    assert sheet.solve_result().status == "completed"
    assert plane.solve_result().status == "completed"
    np.testing.assert_allclose(
        sheet_displacement.value.x.array, plane_displacement.value.x.array,
        rtol=1e-8, atol=1e-9,
    )
    assert sheet.energy_history[-1].tool_work == pytest.approx(
        plane.energy_history[-1].tool_work, rel=1e-8,
    )
    assert sheet.contact_provider.current_point_diagnostics()[
        "whole_face_planar_branch"
    ]["certified"] is True


def test_finite_vertex_competitor_certificate_matches_plane_in_global_j2(monkeypatch):
    sheet, sheet_displacement = _small_single_branch_j2_step(
        closed=False, open_sheet=True, open_sheet_competitor=True,
        open_sheet_competitor_kind="finite_vertex",
    )
    provider = sheet.contact_provider
    current = provider.adapter.current_nodal_positions(sheet_displacement.value)
    face_nodes = provider.adapter.trace.node_ids.reshape(-1, 3, 3)[:, 0, :]
    from agentfem.boundary_models import contact_planar_branch_bound as branch_bound

    coarse = branch_bound.certify_reviewed_planar_branch(
        current[face_nodes], provider.surface, projector=provider.projector,
        maximum_refinement_levels=0,
    )
    assert coarse.certified
    with monkeypatch.context() as disabled:
        disabled.setattr(
            branch_bound,
            "_quadratic_finite_feature_distance_difference_lower_bound",
            lambda *_args, **_kwargs: None,
        )
        old_coarse = branch_bound.certify_reviewed_planar_branch(
            current[face_nodes], provider.surface, projector=provider.projector,
            maximum_refinement_levels=0,
        )
    assert not old_coarse.certified
    assert old_coarse.status == "another_master_feature_can_compete"

    plane, plane_displacement = _small_single_branch_j2_step(closed=False)
    assert sheet.solve_result().status == "completed"
    assert plane.solve_result().status == "completed"
    np.testing.assert_allclose(
        sheet_displacement.value.x.array, plane_displacement.value.x.array,
        rtol=1e-8, atol=1e-9,
    )
    assert sheet.energy_history[-1].tool_work == pytest.approx(
        plane.energy_history[-1].tool_work, rel=1e-8,
    )
    assert provider.current_point_diagnostics()[
        "whole_face_planar_branch"
    ]["certified"] is True


def test_small_sphere_patch_missed_by_default_gauss_enters_global_j2(tmp_path):
    radial, radial_displacement = _small_single_branch_j2_step(
        closed=False, radial_sphere=True, initial_penetration_limit=0.0011,
    )
    sampled, sampled_displacement = _small_single_branch_j2_step(
        closed=False, radial_sphere=True, radial_integration="gauss",
    )
    assert radial.contact_provider.current_point_diagnostics()["active_point_count"] > 0
    initial = radial.contact_provider.current_point_diagnostics()["initial_contact"]
    assert initial["face_wise_bound_certified"] is True
    assert initial["bound_method"] == "exact_radial_face_extrema"
    assert initial["maximum_penetration_bound"] == pytest.approx(0.001, abs=1e-12)
    integration_policy = radial.contact_provider.summary()["integration_policy"]
    assert integration_policy["quadrature_orders"] == (12, 24)
    assert integration_policy["error_status"] == "paired_order_screen_not_rigorous_error_bound"
    assert sampled.contact_provider.current_point_diagnostics()["active_point_count"] == 0
    radial_result = radial.solve_result()
    assert radial_result.status == "completed"
    assert sampled.solve_result().status == "completed"
    assert np.max(np.abs(radial_displacement.value.x.array)) > 1.0e-8
    np.testing.assert_allclose(sampled_displacement.value.x.array, 0.0, atol=1.0e-11)
    assert radial.energy_history[-1].contact_potential > 0.0
    assert radial_result.metadata["constraint_balance_contract"]["force_balance_available"]
    assert radial_result.metadata["constraint_balance_contract"]["work_balance_available"]
    exported = results.export_accepted_contact_points(
        radial, tmp_path / "radial_samples", result=radial_result, vtk_view=True,
    )
    metadata = json.loads(exported["contact_point_metadata"].read_text())
    assert metadata["pairs"][0]["point_identity_scope"] == "accepted_snapshot_only"
    assert metadata["pressure_measure"] == (
        "penalty_force_per_reference_area_not_cauchy_pressure"
    )
    assert exported["contact_point_view"].is_file()
    with pytest.raises(ValueError, match="AFM-CONTACT-INITIAL-OVERLAP"):
        _small_single_branch_j2_step(
            closed=False, radial_sphere=True, initial_penetration_limit=0.0009,
        )


def test_radial_sphere_contact_tangent_matches_current_directional_difference():
    step, _ = _small_single_branch_j2_step(closed=False, radial_sphere=True)
    assert step.solve_result().status == "completed"
    provider = step.contact_provider
    selected = provider.evaluate()
    base = step.solution.x.array.copy()
    direction = np.random.default_rng(20261019).normal(size=base.shape)
    h = 1.0e-7
    try:
        step.solution.x.array[:] = base + h * direction
        plus = provider.evaluate()
        step.solution.x.array[:] = base - h * direction
        minus = provider.evaluate()
    finally:
        step.solution.x.array[:] = base
    np.testing.assert_allclose(
        selected.tangent_action(direction.reshape((-1, 3))),
        (plus.nodal_structural_residual - minus.nodal_structural_residual) / (2 * h),
        rtol=2e-4, atol=2e-6,
    )


def test_moving_sphere_partition_restarts_with_the_same_accepted_j2_state(tmp_path):
    motion = (0.0, 0.015, 0.0)
    continuous, _ = _small_single_branch_j2_step(
        closed=False, radial_sphere=True, radial_motion=motion,
    )
    assert continuous.solve_result().status == "completed"
    interrupted, _ = _small_single_branch_j2_step(
        closed=False, radial_sphere=True, radial_motion=motion,
    )
    interrupted.solve(until=0.5)
    checkpoint = interrupted.save_checkpoint(tmp_path / "moving_sphere")
    restored, _ = _small_single_branch_j2_step(
        closed=False, radial_sphere=True, radial_motion=motion,
    )
    restored.load_checkpoint(checkpoint)
    assert restored.solve_result().status == "completed"
    np.testing.assert_allclose(
        restored.solution.x.array, continuous.solution.x.array,
        rtol=1e-9, atol=1e-10,
    )
    assert restored.energy_history[-1].contact_potential == pytest.approx(
        continuous.energy_history[-1].contact_potential, rel=1e-9,
    )
    assert restored.energy_history[-1].tool_work == pytest.approx(
        continuous.energy_history[-1].tool_work, rel=1e-9, abs=1e-12,
    )
    wrong_mode, _ = _small_single_branch_j2_step(
        closed=False, radial_sphere=True, radial_motion=motion,
        radial_integration="gauss",
    )
    with pytest.raises(ValueError, match="identity|contact|constraint"):
        wrong_mode.load_checkpoint(checkpoint)


def test_small_cylinder_patch_enters_global_j2_without_sampled_false_open():
    radial, radial_displacement = _small_single_branch_j2_step(
        closed=False, radial_cylinder=True,
    )
    sampled, _ = _small_single_branch_j2_step(
        closed=False, radial_cylinder=True, radial_integration="gauss",
    )
    assert radial.contact_provider.current_point_diagnostics()["active_point_count"] > 0
    assert sampled.contact_provider.current_point_diagnostics()["active_point_count"] == 0
    assert radial.solve_result().status == "completed"
    assert np.max(np.abs(radial_displacement.value.x.array)) > 1.0e-8
    assert radial.energy_history[-1].contact_potential > 0.0


@pytest.mark.parametrize("divisions", [2, 4])
def test_radial_sphere_partition_on_refined_real_tetrahedral_mesh(divisions):
    step, displacement = _small_single_branch_j2_step(
        closed=False, radial_sphere=True, radial_mesh_divisions=divisions,
    )
    result = step.solve_result()
    assert result.status == "completed"
    assert np.all(np.isfinite(displacement.value.x.array))
    assert np.max(np.abs(displacement.value.x.array)) > 0.0
    assert step.energy_history[-1].contact_potential > 0.0
    assert result.metadata["constraint_balance_contract"]["force_balance_available"]


def test_radial_sphere_and_moving_plane_share_one_total_j2_step(tmp_path):
    options = dict(
        closed=False, radial_sphere=True, active_second_plane=True,
        radial_motion=(-0.015, 0.0, 0.0),
    )
    step, displacement = _small_single_branch_j2_step(**options)
    result = step.solve_result()
    assert result.status == "completed"
    assert np.all(np.isfinite(displacement.value.x.array))
    assert all(
        item.current_point_diagnostics()["active_point_count"] > 0
        for item in step.contact_provider.providers
    )
    work = dict(step.energy_history[-1].pair_tool_work)
    assert set(work) == {"rigid_contact_pair", "transverse_plane"}
    assert all(value > 0.0 for value in work.values())
    assert step.energy_history[-1].tool_work == pytest.approx(sum(work.values()))
    interrupted, _ = _small_single_branch_j2_step(**options)
    interrupted.solve(until=0.5)
    saved = interrupted.save_checkpoint(tmp_path / "two_moving_tools")
    restored, _ = _small_single_branch_j2_step(**options)
    restored.load_checkpoint(saved)
    assert restored.solve_result().status == "completed"
    np.testing.assert_allclose(
        restored.solution.x.array, step.solution.x.array, rtol=1e-9, atol=1e-10,
    )
    np.testing.assert_allclose(
        restored.response.state.committed_state_vectors(),
        step.response.state.committed_state_vectors(),
        rtol=1e-9, atol=1e-10,
    )
    assert dict(restored.energy_history[-1].pair_tool_work) == pytest.approx(work)


def test_reviewed_open_sheet_branch_refuses_positive_area_beyond_master_edge():
    step, displacement = _small_single_branch_j2_step(
        closed=False, open_sheet=True,
    )
    displacement.value.x.array.reshape((-1, 3))[:, 1] += 2.0
    with pytest.raises(NotImplementedError, match="AFM-CONTACT-REVIEWED-PLANAR-BRANCH"):
        step.contact_provider.evaluate()
    assert step.accepted_load_factor == 0.0


def test_reviewed_open_sheet_rotation_and_two_active_tools_match_plane_limit():
    rotation = (0.0, 0.0, 0.01)
    sheet, sheet_displacement = _small_single_branch_j2_step(
        closed=False, open_sheet=True, tool_rotation=rotation,
        active_second_plane=True,
    )
    plane, plane_displacement = _small_single_branch_j2_step(
        closed=False, tool_rotation=rotation, active_second_plane=True,
    )
    assert sheet.solve_result().status == "completed"
    assert plane.solve_result().status == "completed"
    np.testing.assert_allclose(
        sheet_displacement.value.x.array, plane_displacement.value.x.array,
        rtol=1e-8, atol=1e-9,
    )
    sheet_work = dict(sheet.energy_history[-1].pair_tool_work)
    plane_work = dict(plane.energy_history[-1].pair_tool_work)
    assert set(sheet_work) == {"rigid_contact_pair", "transverse_plane"}
    assert all(value > 0.0 for value in sheet_work.values())
    assert sheet_work == pytest.approx(plane_work, rel=1e-8, abs=1e-10)


def test_reviewed_open_sheet_contact_tangent_matches_current_directional_difference():
    step, _ = _small_single_branch_j2_step(closed=False, open_sheet=True)
    assert step.solve_result().status == "completed"
    provider = step.contact_provider
    selected = provider.evaluate()
    base = step.solution.x.array.copy()
    direction = np.random.default_rng(746).normal(size=base.shape)
    h = 1.0e-7
    try:
        step.solution.x.array[:] = base + h * direction
        plus = provider.evaluate()
        step.solution.x.array[:] = base - h * direction
        minus = provider.evaluate()
    finally:
        step.solution.x.array[:] = base
    np.testing.assert_allclose(
        selected.tangent_action(direction.reshape((-1, 3))),
        (plus.nodal_structural_residual - minus.nodal_structural_residual) / (2 * h),
        rtol=1e-5, atol=1e-6,
    )


def test_certified_closed_planar_branch_matches_analytical_plane_global_j2(tmp_path):
    closed_step, closed_displacement = _small_single_branch_j2_step(closed=True)
    plane_step, plane_displacement = _small_single_branch_j2_step(closed=False)
    closed_result = closed_step.solve_result()
    plane_result = plane_step.solve_result()
    assert closed_result.status == plane_result.status
    assert closed_result.quantity("maximum_equivalent_plastic_strain") > 0
    assert closed_result.quantity("maximum_equivalent_plastic_strain") == pytest.approx(
        plane_result.quantity("maximum_equivalent_plastic_strain"), rel=1e-8,
    )
    np.testing.assert_allclose(
        closed_displacement.value.x.array, plane_displacement.value.x.array,
        rtol=1e-8, atol=1e-9,
    )
    assert closed_step.energy_history[-1].tool_work > 0
    np.testing.assert_allclose(
        closed_step.energy_history[-1].tool_work,
        plane_step.energy_history[-1].tool_work,
        rtol=1e-8, atol=1e-9,
    )
    evidence = closed_step.contact_provider.current_point_diagnostics()
    assert evidence["feature_transition_integral_certified"] is True
    assert evidence["whole_face_planar_branch"]["certified"] is True
    assert evidence["active_point_count"] > 0
    exported = results.export_accepted_contact_points(
        closed_step, tmp_path / "raw_contact", result=closed_result,
        vtk_view=True,
    )
    metadata = json.loads(exported["contact_point_metadata"].read_text())
    assert metadata["pairs"][0]["point_identity_scope"] == "accepted_snapshot_only"
    assert exported["contact_point_view"].exists()


def test_closed_planar_branch_global_j2_crosses_only_flat_tool_triangulation_seam():
    closed_step, closed_displacement = _small_single_branch_j2_step(
        closed=True, flat_seam=True,
    )
    plane_step, plane_displacement = _small_single_branch_j2_step(
        closed=False, flat_seam=True,
    )
    closed_result = closed_step.solve_result()
    plane_result = plane_step.solve_result()
    assert closed_result.status == "completed"
    assert plane_result.status == "completed"
    np.testing.assert_allclose(
        closed_displacement.value.x.array, plane_displacement.value.x.array,
        rtol=1e-8, atol=1e-9,
    )
    assert closed_result.quantity("maximum_equivalent_plastic_strain") == pytest.approx(
        plane_result.quantity("maximum_equivalent_plastic_strain"), rel=1e-8,
    )
    evidence = closed_step.contact_provider.current_point_diagnostics()
    assert evidence["feature_transition_integral_certified"] is True


def test_closed_planar_branch_rotating_tool_matches_analytical_plane_global_j2():
    rotation = (0.0, 0.0, 0.01)
    closed_step, closed_displacement = _small_single_branch_j2_step(
        closed=True, tool_rotation=rotation,
    )
    plane_step, plane_displacement = _small_single_branch_j2_step(
        closed=False, tool_rotation=rotation,
    )
    assert closed_step.solve_result().status == "completed"
    assert plane_step.solve_result().status == "completed"
    np.testing.assert_allclose(
        closed_displacement.value.x.array, plane_displacement.value.x.array,
        rtol=1e-8, atol=1e-9,
    )
    assert closed_step.energy_history[-1].tool_work == pytest.approx(
        plane_step.energy_history[-1].tool_work, rel=1e-8, abs=1e-10,
    )
    assert closed_step.contact_provider.current_point_diagnostics()[
        "feature_transition_integral_certified"
    ] is True


def test_named_mixed_contact_integration_modes_keep_independent_pair_identity():
    single, single_displacement = _small_single_branch_j2_step(closed=True)
    mixed, mixed_displacement = _small_single_branch_j2_step(
        closed=True, second_plane=True,
    )
    assert single.solve_result().status == "completed"
    mixed_result = mixed.solve_result()
    assert mixed_result.status == "completed"
    np.testing.assert_allclose(
        mixed_displacement.value.x.array, single_displacement.value.x.array,
        rtol=1e-8, atol=1e-9,
    )
    modes = {
        item.name: item.contact_integration
        for item in mixed.contact_provider.providers
    }
    assert modes == {
        single.contact_provider.name: "closed_planar_branch_partition",
        "far_plane": "planar_active_partition",
    }
    assert tuple(
        item["constraint_name"] for item in mixed_result.metadata["constraint_duals"]
    ) == tuple(modes)


def test_named_mixed_contact_modes_are_checkpointed_as_one_accepted_path(tmp_path):
    full, _ = _small_single_branch_j2_step(closed=True, second_plane=True)
    assert full.solve_result().status == "completed"
    partial, _ = _small_single_branch_j2_step(closed=True, second_plane=True)
    partial.solve(until=0.5)
    saved = partial.save_checkpoint(tmp_path / "mixed_modes")
    resumed, _ = _small_single_branch_j2_step(closed=True, second_plane=True)
    resumed.load_checkpoint(saved)
    assert resumed.solve_result().status == "completed"
    np.testing.assert_allclose(
        resumed.solution.x.array, full.solution.x.array, rtol=1e-9, atol=1e-10,
    )
    assert resumed.energy_history[-1].tool_work == pytest.approx(
        full.energy_history[-1].tool_work, rel=1e-9,
    )
    wrong_pairs, _ = _small_single_branch_j2_step(closed=True)
    with pytest.raises(ValueError, match="identity|contact|constraint"):
        wrong_pairs.load_checkpoint(saved)


def test_named_mixed_contact_modes_can_both_work_and_restart(tmp_path):
    def build():
        return _small_single_branch_j2_step(
            closed=True, active_second_plane=True,
        )[0]

    full = build()
    result = full.solve_result()
    assert result.status == "completed"
    providers = full.contact_provider.providers
    assert tuple(item.name for item in providers) == (
        "rigid_contact_pair", "transverse_plane",
    )
    assert all(
        item.current_point_diagnostics()["active_point_count"] > 0
        for item in providers
    )
    work = dict(full.energy_history[-1].pair_tool_work)
    assert set(work) == {item.name for item in providers}
    assert all(value > 0.0 for value in work.values())
    assert sum(work.values()) == pytest.approx(full.energy_history[-1].tool_work)
    assert tuple(
        item["constraint_name"] for item in result.metadata["constraint_duals"]
    ) == tuple(work)
    for provider, dual in zip(
        providers, result.metadata["constraint_duals"], strict=True,
    ):
        samples = provider.accepted_point_samples()
        independent_force = np.sum(
            samples["reference_weights"][:, None]
            * samples["normal_pressures"][:, None]
            * samples["surface_normals"],
            axis=0,
        )
        np.testing.assert_allclose(
            dual["resultant"], independent_force, rtol=1e-9, atol=1e-10,
        )
    assert providers[0].current_point_diagnostics()[
        "feature_transition_integral_certified"
    ] is True

    partial = build()
    partial.solve(until=0.5)
    saved = partial.save_checkpoint(tmp_path / "two_active_mixed_modes")
    resumed = build()
    resumed.load_checkpoint(saved)
    assert resumed.solve_result().status == "completed"
    np.testing.assert_allclose(
        resumed.solution.x.array, full.solution.x.array, rtol=1e-9, atol=1e-10,
    )
    assert dict(resumed.energy_history[-1].pair_tool_work) == pytest.approx(work)


def test_closed_planar_branch_mode_rejects_current_edge_transition():
    step, displacement = _small_single_branch_j2_step(closed=True)
    displacement.value.x.array.reshape((-1, 3))[:, 1] = -0.30
    displacement.value.x.array.reshape((-1, 3))[:, 2] = 0.55
    with pytest.raises(NotImplementedError, match="AFM-CONTACT-CLOSED-PLANAR-BRANCH"):
        step.contact_provider.evaluate()
    assert step.accepted_load_factor == 0.0


def test_closed_planar_branch_mode_rejects_friction_before_step_construction():
    model, displacement, material, _ = _patch(
        triangulated=True, closed_concave_tool=True,
        friction_coefficient=0.2, tangential_penalty=100.0,
        make_step=False,
    )
    capability = step_capability(
        model, target=displacement,
        options={
            "material": material,
            "contact_integration": "closed_planar_branch_partition",
        },
    )
    assert not capability["supported"]
    assert any(
        item["message"].startswith("AFM-CONTACT-PARTITION")
        for item in capability["rejection_reasons"]
    )
    with pytest.raises(NotImplementedError, match="AFM-CONTACT-PARTITION"):
        model.step(
            target=displacement, material=material,
            contact_integration="closed_planar_branch_partition",
        )


def test_closed_planar_branch_contact_tangent_matches_current_directional_difference():
    step, _ = _small_single_branch_j2_step(closed=True)
    assert step.solve_result().status == "completed"
    provider = step.contact_provider
    selected = provider.evaluate()
    base = step.solution.x.array.copy()
    direction = np.random.default_rng(744).normal(size=base.shape)
    h = 1.0e-7
    try:
        step.solution.x.array[:] = base + h * direction
        plus = provider.evaluate()
        step.solution.x.array[:] = base - h * direction
        minus = provider.evaluate()
    finally:
        step.solution.x.array[:] = base
    np.testing.assert_allclose(
        selected.tangent_action(direction.reshape((-1, 3))),
        (plus.nodal_structural_residual - minus.nodal_structural_residual) / (2 * h),
        rtol=1e-5, atol=1e-6,
    )


def test_closed_planar_branch_restart_preserves_plastic_contact_path(tmp_path):
    full, _ = _small_single_branch_j2_step(closed=True)
    assert full.solve_result().status == "completed"
    partial, _ = _small_single_branch_j2_step(closed=True)
    partial.solve(until=0.5)
    saved = partial.save_checkpoint(tmp_path / "closed_planar_branch")
    resumed, _ = _small_single_branch_j2_step(closed=True)
    resumed.load_checkpoint(saved)
    assert resumed.solve_result().status == "completed"
    np.testing.assert_allclose(
        resumed.solution.x.array, full.solution.x.array,
        rtol=1e-9, atol=1e-10,
    )
    assert resumed.energy_history[-1].tool_work == pytest.approx(
        full.energy_history[-1].tool_work, rel=1e-9, abs=1e-10,
    )
    assert resumed.contact_feature_path == full.contact_feature_path


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


@pytest.mark.parametrize("closed_concave_tool", [False, True])
def test_closed_tool_signed_distance_initial_bound_reaches_ordinary_step(
    closed_concave_tool,
):
    options = dict(
        triangulated=True,
        convex_tool=not closed_concave_tool,
        closed_concave_tool=closed_concave_tool,
        initial_penetration_limit=0.1,
        applied_traction=0.0,
    )
    model, displacement, material, _ = _patch(**options, make_step=False)
    capability = step_capability(
        model, target=displacement, options={"material": material},
    )
    assert capability["supported"], capability["rejection_reasons"]
    step = _patch(**options)[3]
    initial = step.contact_provider.current_point_diagnostics()["initial_contact"]
    assert initial["face_wise_bound_certified"] is True
    assert initial["bound_method"] == (
        "adaptive_closed_signed_distance_lipschitz"
        if closed_concave_tool
        else "convex_support_plane_or_adaptive_lipschitz"
    )
    assert initial["maximum_penetration_bound"] <= 0.1
    assert initial["bound_sampled_point_count"] > 0


def test_closed_tool_excess_initial_overlap_rejects_before_newton():
    with pytest.raises(ValueError, match="AFM-CONTACT-INITIAL-OVERLAP"):
        _patch(
            triangulated=True, convex_tool=True, gap=-0.05,
            initial_penetration_limit=0.01,
            applied_traction=0.0,
        )


def test_near_touching_convex_tool_uses_exact_initial_support_bound():
    _, _, _, step = _patch(
        triangulated=True, convex_tool=True, gap=0.0,
        initial_penetration_limit=1.0e-9,
        tool_translation=(-0.008, 0.0, 0.0),
        applied_traction=0.0, yield_stress=1.0e9,
    )
    initial = step.contact_provider.current_point_diagnostics()["initial_contact"]
    assert initial["face_wise_bound_certified"] is True
    assert initial["bound_method"] == "convex_support_plane_or_adaptive_lipschitz"
    assert initial["bound_refinement_depth"] == 0
    assert initial["maximum_penetration_bound"] <= 1.0e-9
    assert step.solve_result().status == "completed"


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
    assert same.contact_feature_path == step.contact_feature_path
    with np.load(saved, allow_pickle=False) as archive:
        damaged = {name: archive[name].copy() for name in archive.files}
    feature_path = json.loads(str(damaged["contact_feature_path"]))
    feature_path[next(iter(feature_path))]["accepted_station_count"] += 1
    damaged["contact_feature_path"] = json.dumps(feature_path)
    damaged_path = tmp_path / "damaged_contact_path.npz"
    np.savez(damaged_path, **damaged)
    with pytest.raises(ValueError, match="accepted station disagree"):
        same.load_checkpoint(damaged_path)
    assert same.contact_feature_path == step.contact_feature_path
    np.testing.assert_allclose(same.solution.x.array, step.solution.x.array)
    legacy = {
        name: value for name, value in damaged.items()
        if name not in {"contact_feature_path", "contact_feature_path_complete"}
    }
    legacy_path = tmp_path / "older_contact_checkpoint.npz"
    np.savez(legacy_path, **legacy)
    older = _patch(**options)[3]
    older.load_checkpoint(legacy_path)
    assert not older.contact_feature_path_complete
    assert older.contact_feature_path == {}
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
    for feature in ("face", "edge", "vertex"):
        assert result.quantity(
            f"{provider.name}_active_{feature}_points"
        ) == feature_counts[feature]
    quality = result.verify("engineering")
    feature_claim = next(
        claim for claim in quality.claims
        if claim.name == "no_observed_uncertified_contact_feature_transition"
    )
    assert feature_claim.status == "failed"
    assert any("edge/vertex contact" in issue for issue in feature_claim.actual["issues"])
    path = result.metadata["rigid_contact_feature_path"]
    assert path["complete"]
    assert path["accepted_station_count"] == 7
    assert path["pairs"][0]["maximum_active_edge_points"] >= feature_counts["edge"]
    assert not quality.acceptable
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
    accepted_feature_path = dict(partial.contact_feature_path)
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
    assert partial.contact_feature_path == accepted_feature_path
    assert partial.contact_provider.snapshot_motion() == pytest.approx(0.5)

    monkeypatch.setattr(partial, "_record_energy", record_energy)
    record_features = partial._record_accepted_contact_features

    def reject_after_feature_observation(factor):
        record_features(factor)
        if factor > 0.5 + 1e-12:
            raise ValueError("intentional feature-ledger acceptance rejection")

    monkeypatch.setattr(
        partial, "_record_accepted_contact_features",
        reject_after_feature_observation,
    )
    with pytest.raises(RuntimeError, match="feature-ledger acceptance rejection"):
        partial.solve(until=2.0 / 3.0)
    assert partial.accepted_load_factor == pytest.approx(0.5)
    assert partial.contact_feature_path == accepted_feature_path
    np.testing.assert_array_equal(partial.solution.x.array, accepted_u)

    resumed = _patch(**options)[3]
    resumed.load_checkpoint(saved)
    assert resumed.contact_feature_path == accepted_feature_path
    assert resumed.solve_result().status == "completed"
    assert resumed.contact_feature_path == reference.contact_feature_path
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
    with np.load(saved, allow_pickle=False) as archive:
        older_payload = {
            name: archive[name].copy() for name in archive.files
            if name not in {"contact_feature_path", "contact_feature_path_complete"}
        }
    older_checkpoint = tmp_path / "older_partial_contact_path.npz"
    np.savez(older_checkpoint, **older_payload)
    older = _patch(**options)[3]
    older.load_checkpoint(older_checkpoint)
    older_result = older.solve_result()
    assert not older_result.metadata["rigid_contact_feature_path"]["complete"]
    older_claim = next(
        claim for claim in older_result.verify("engineering").claims
        if claim.name == "no_observed_uncertified_contact_feature_transition"
    )
    assert "accepted contact feature path is incomplete" in older_claim.actual["issues"]


def test_standard_polyhedral_feature_diagnostic_failure_rolls_back(monkeypatch):
    step = _patch(
        triangulated=True, convex_tool=True, increments=3,
        tool_translation=(-0.008, 0.0, 0.0), applied_traction=0.0,
        yield_stress=1.0e9,
    )[3]
    step.solve(until=1.0 / 3.0)
    accepted_u = step.solution.x.array.copy()
    accepted_state = step.response.state.committed_state_vectors().copy()
    accepted_path = dict(step.contact_feature_path)
    accepted_energy = tuple(step.energy_history)

    original = step.contact_provider.current_point_diagnostics

    def missing_feature_count():
        sample = original()
        sample.pop("sampled_mixed_feature_slave_faces")
        return sample

    monkeypatch.setattr(
        step.contact_provider, "current_point_diagnostics", missing_feature_count,
    )
    with pytest.raises(RuntimeError, match="feature diagnostics are incomplete"):
        step.solve(until=2.0 / 3.0)
    assert step.accepted_load_factor == pytest.approx(1.0 / 3.0)
    np.testing.assert_array_equal(step.solution.x.array, accepted_u)
    np.testing.assert_array_equal(
        step.response.state.committed_state_vectors(), accepted_state,
    )
    assert step.contact_feature_path == accepted_path
    assert tuple(step.energy_history) == accepted_energy


def test_standard_backtracking_does_not_recompute_accepted_trial(monkeypatch):
    step = _patch(
        triangulated=True, convex_tool=True, increments=3,
        tool_translation=(-0.008, 0.0, 0.0), applied_traction=0.0,
        yield_stress=1.0e9,
    )[3]
    original_increment = step._solve_increment
    original_update = step._update_response
    original_residual = step._correction_rhs
    inside_increment = False
    last_evaluation = None
    last_residual = None
    line_trials = 0
    original_search = step._line_search

    def audited_update(*, start_factor, target_factor):
        nonlocal last_evaluation
        if inside_increment:
            current = (
                start_factor, target_factor, step.solution.x.array.copy(),
            )
            if last_evaluation is not None:
                assert not (
                    current[:2] == last_evaluation[:2]
                    and np.array_equal(current[2], last_evaluation[2])
                ), "accepted Newton trial was evaluated twice"
            last_evaluation = current
        return original_update(
            start_factor=start_factor, target_factor=target_factor,
        )

    def audited_search(*args, **kwargs):
        nonlocal line_trials
        result = original_search(*args, **kwargs)
        if result[1] is not None:
            line_trials += 1
        return result

    def audited_residual():
        nonlocal last_residual
        if inside_increment:
            current = step.solution.x.array.copy()
            if last_residual is not None:
                assert not np.array_equal(
                    current, last_residual,
                ), "accepted Newton residual was assembled twice"
            last_residual = current
        return original_residual()

    def audited_increment(**kwargs):
        nonlocal inside_increment, last_evaluation, last_residual
        inside_increment = True
        last_evaluation = None
        last_residual = None
        try:
            return original_increment(**kwargs)
        finally:
            inside_increment = False

    monkeypatch.setattr(step, "_solve_increment", audited_increment)
    monkeypatch.setattr(step, "_update_response", audited_update)
    monkeypatch.setattr(step, "_correction_rhs", audited_residual)
    monkeypatch.setattr(step, "_line_search", audited_search)
    step.solve()
    assert line_trials > 0
    assert step.accepted_load_factor == pytest.approx(1.0)


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
    assert summary["engineering_contact_feature_status"] == "failed"
    assert any(
        "edge/vertex contact" in issue
        for issue in summary["engineering_contact_feature_issues"]
    )
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
        contact_quadrature_degree=7, audit_quadrature=True,
        audit_subdivision=True,
    )
    assert refined["status"] == "completed"
    assert refined["contact_quadrature_degree"] == 7
    assert refined["maximum_penetration"] > 0.0
    assert refined["fixed_configuration_quadrature_audit"]["method"] == (
        "fixed_configuration_positive_rule_sensitivity"
    )
    assert refined["fixed_configuration_subdivision_audit"]["method"] == (
        "fixed_configuration_uniform_reference_subdivision"
    )
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


def test_fixed_configuration_contact_quadrature_audit_is_read_only():
    _, displacement, _, step = _patch(
        triangulated=True, closed_concave_tool=True,
        closed_concave_extent=0.7, closed_concave_thickness=0.5,
        closed_concave_corner=(0.35, 0.19), gap=0.005,
        tool_translation=(-0.055, 0.0, 0.0),
        penalty=20.0, increments=3, applied_traction=0.0,
    )
    result = step.solve_result()
    provider = step.contact_provider
    assert result.metadata["rigid_contact_point_diagnostics"][
        "sampled_mixed_feature_slave_faces"
    ] > 0
    quality = result.verify("engineering")
    assert next(
        item for item in quality.claims
        if item.name == "no_observed_uncertified_contact_feature_transition"
    ).status == "failed"
    original_displacement = displacement.value.x.array.copy()
    original_factor = float(provider.motion_factor.value)
    original_force = provider.evaluate().contact_force_on_structure.copy()
    audit = provider.quadrature_sensitivity((None, 9, 11))
    assert audit["method"] == "fixed_configuration_positive_rule_sensitivity"
    assert audit["certifies_feature_transition_integral"] is False
    assert [item["degree"] for item in audit["evaluations"]] == [None, 9, 11]
    assert [item["sample_count"] for item in audit["evaluations"]] == [
        6, 38, 56,
    ]
    np.testing.assert_allclose(
        audit["evaluations"][0]["contact_force_on_structure"],
        original_force,
    )
    np.testing.assert_array_equal(displacement.value.x.array, original_displacement)
    assert float(provider.motion_factor.value) == original_factor
    np.testing.assert_array_equal(
        provider.evaluate().contact_force_on_structure, original_force,
    )
    subdivision = provider.reference_subdivision_sensitivity((0, 1, 2))
    assert subdivision["certifies_feature_transition_integral"] is False
    assert [item["sample_count"] for item in subdivision["evaluations"]] == [
        6, 24, 96,
    ]
    np.testing.assert_allclose(
        subdivision["evaluations"][0]["contact_force_on_structure"],
        original_force,
    )
    np.testing.assert_array_equal(displacement.value.x.array, original_displacement)
    enclosure = provider.closed_penalty_energy_enclosure(
        maximum_depth=2, maximum_cells_per_face=16,
    )
    assert enclosure["method"] == (
        "fixed_configuration_closed_penalty_energy_enclosure"
    )
    assert enclosure["returns_contact_residual_or_tangent"] is False
    assert enclosure["certifies_feature_transition_integral"] is False
    assert 0.0 <= enclosure["lower_bound"] <= enclosure["upper_bound"]
    assert len(enclosure["faces"]) == 2
    np.testing.assert_array_equal(displacement.value.x.array, original_displacement)
    assert float(provider.motion_factor.value) == original_factor
    with pytest.raises(ValueError, match="supported rules"):
        provider.quadrature_sensitivity((9, True))
    with pytest.raises(ValueError, match="unique"):
        provider.quadrature_sensitivity((9, 9))


def test_contact_quadrature_audit_refuses_frictional_state():
    _, _, _, step = _patch(
        triangulated=True, friction_coefficient=0.2, tangential_penalty=100.0,
    )
    with pytest.raises(NotImplementedError, match="AFM-CONTACT-QUADRATURE-AUDIT"):
        step.contact_provider.quadrature_sensitivity()


def test_closed_energy_enclosure_refuses_nondefault_face_trace():
    _, _, _, step = _patch(
        triangulated=True, closed_concave_tool=True,
        closed_concave_extent=0.7, closed_concave_thickness=0.5,
        closed_concave_corner=(0.35, 0.19),
        contact_quadrature_degree=7, applied_traction=0.0,
    )
    with pytest.raises(NotImplementedError, match="AFM-CONTACT-ENERGY-ENCLOSURE"):
        step.contact_provider.closed_penalty_energy_enclosure()


@pytest.mark.parametrize("cells", [1, 2])
@pytest.mark.parametrize("penalty", [1_000.0, 10_000.0])
def test_elastic_regime_plane_contact_approaches_independent_series_spring(
    cells, penalty,
):
    # A 1 x 1 x 1 bar with nu=0 and an elastic tangent E is in series with
    # the rigid-plane penalty compliance 1/k. The small-indentation limit is
    # F=delta/(1/E+1/k). This is a physical oracle, not a second AgentFEM
    # contact implementation; use a small but nonzero finite-strain path.
    indentation = 1.0e-4
    young = 1000.0
    _, _, _, step = _patch(
        triangulated=True, gap=0.0,
        tool_translation=(-indentation, 0.0, 0.0),
        penalty=penalty, yield_stress=1.0e9,
        applied_traction=0.0, increments=4, cells=cells,
    )
    result = step.solve_result()
    force = -step.contact_provider.evaluate().contact_force_on_structure[0]
    expected = indentation / (1.0 / young + 1.0 / penalty)
    assert result.status == "completed"
    assert force == pytest.approx(expected, rel=5.0e-4)
    assert result.quantity("maximum_equivalent_plastic_strain") == pytest.approx(0.0)


@pytest.mark.parametrize("cells", [1, 2])
def test_plastic_contact_matches_independent_hencky_return_and_penalty(cells):
    # Fixed transverse displacement gives the exact homogeneous deformation
    # F=diag(lambda, 1, 1). For nu=0, the trial Kirchhoff stress has
    # q_trial=E*abs(log(lambda)); linear J2 return uses
    # dgamma=(q_trial-Y)/(3*mu+H). The axial first Piola stress is
    # [E*log(lambda)+E*dgamma]/lambda in compression. Equilibrium with a
    # reference-area plane penalty supplies an independent scalar oracle.
    young, yield_stress, hardening = 1000.0, 0.1, 100.0
    penalty, indentation = 1000.0, 0.002
    _, _, _, step = _patch(
        triangulated=True, gap=0.0,
        tool_translation=(-indentation, 0.0, 0.0),
        penalty=penalty, yield_stress=yield_stress,
        applied_traction=0.0, increments=8, cells=cells,
        transverse_boundary_fixed=True,
    )
    result = step.solve_result()
    assert result.status == "completed"

    tool_x = 1.0 - indentation
    lower, upper = tool_x, 1.0
    for _ in range(70):
        stretch = 0.5 * (lower + upper)
        logarithmic_strain = np.log(stretch)
        q_trial = -young * logarithmic_strain
        plastic_increment = max(
            0.0, (q_trial - yield_stress) / (1.5 * young + hardening),
        )
        piola = young * (logarithmic_strain + plastic_increment) / stretch
        residual = piola + penalty * (stretch - tool_x)
        if residual < 0.0:
            lower = stretch
        else:
            upper = stretch
    expected_stretch = 0.5 * (lower + upper)
    expected_plastic_increment = max(
        0.0,
        (-young * np.log(expected_stretch) - yield_stress)
        / (1.5 * young + hardening),
    )
    coordinates = step.solution.function_space.tabulate_dof_coordinates()
    right = np.isclose(coordinates[:, 0], 1.0)
    right_ux = step.solution.x.array.reshape((-1, 3))[right, 0]
    np.testing.assert_allclose(
        right_ux, expected_stretch - 1.0, rtol=1e-5, atol=1e-8,
    )
    force = -step.contact_provider.evaluate().contact_force_on_structure[0]
    assert force == pytest.approx(
        penalty * (expected_stretch - tool_x), rel=1e-5,
    )
    assert result.quantity("maximum_equivalent_plastic_strain") == pytest.approx(
        expected_plastic_increment, rel=1e-5,
    )


def test_plastic_contact_unloads_and_reloads_without_losing_history(tmp_path):
    indentation = 0.002
    young, yield_stress, hardening, penalty = 1000.0, 0.1, 100.0, 1000.0
    path = amplitudes.tabular(
        (0.0, 0.5, 0.75, 1.0), (0.0, 1.0, 0.0, 1.0),
        name="indent_release_reindent",
    )

    # Independent one-dimensional plane-strain Hencky/J2 return map. The
    # signed axial plastic log strain a gives Fp=diag(exp(a),exp(-a/2),
    # exp(-a/2)). With nu=0: q_trial=E*abs(log(lambda)-3*a/2),
    # da=sign(trial)*dp, and Pxx=E*(log(lambda)-a_new)/lambda.
    # A fresh scalar equilibrium solve follows each accepted tool station;
    # old a and PEEQ stay fixed during its trial iterations.
    axial_plastic, peeq = 0.0, 0.0
    reference = {}
    for index in range(1, 9):
        station = index / 8
        tool_x = 1.0 - indentation * path(station)

        def response(
            stretch, *, old_axial=axial_plastic, old_peeq=peeq,
            tool_position=tool_x,
        ):
            log_strain = np.log(stretch)
            trial = young * (log_strain - 1.5 * old_axial)
            increment = max(
                0.0,
                (abs(trial) - yield_stress - hardening * old_peeq)
                / (1.5 * young + hardening),
            )
            new_axial = old_axial + np.sign(trial) * increment
            new_peeq = old_peeq + increment
            piola = young * (log_strain - new_axial) / stretch
            contact_force = penalty * max(0.0, stretch - tool_position)
            return piola + contact_force, new_axial, new_peeq, contact_force

        lower, upper = 0.99, 1.01
        for _ in range(70):
            middle = 0.5 * (lower + upper)
            if response(middle)[0] < 0.0:
                lower = middle
            else:
                upper = middle
        stretch = 0.5 * (lower + upper)
        _, axial_plastic, peeq, force = response(stretch)
        reference[station] = (stretch - 1.0, peeq, force)

    options = dict(
        triangulated=True, gap=0.0,
        tool_translation=(-indentation, 0.0, 0.0),
        penalty=penalty, yield_stress=yield_stress,
        applied_traction=0.0, increments=8,
        transverse_boundary_fixed=True, load_amplitude=path,
    )
    _, _, _, step = _patch(**options)
    coordinates = step.solution.function_space.tabulate_dof_coordinates()
    right = np.isclose(coordinates[:, 0], 1.0)

    step.solve(until=0.5)
    first_displacement = float(np.mean(
        step.solution.x.array.reshape((-1, 3))[right, 0],
    ))
    first_force = -step.contact_provider.evaluate().contact_force_on_structure[0]
    first_peeq = float(np.max(
        step.response.state.committed["equivalent_plastic_strain"].function.x.array,
    ))
    assert first_force > 0.0 and first_peeq > 0.0
    first_expected = reference[0.5]
    assert first_displacement == pytest.approx(first_expected[0], rel=1e-5)
    assert first_peeq == pytest.approx(first_expected[1], rel=1e-5)
    assert first_force == pytest.approx(first_expected[2], rel=1e-5)
    checkpoint = step.save_checkpoint(tmp_path / "accepted_before_release")

    step.solve(until=0.75)
    released_displacement = float(np.mean(
        step.solution.x.array.reshape((-1, 3))[right, 0],
    ))
    released_force = -step.contact_provider.evaluate().contact_force_on_structure[0]
    assert released_force == pytest.approx(0.0, abs=1e-9)
    assert released_displacement == pytest.approx(
        reference[0.75][0], rel=1e-5, abs=1e-8,
    )
    assert np.max(step.response.state.committed[
        "equivalent_plastic_strain"
    ].function.x.array) == pytest.approx(reference[0.75][1], rel=1e-5)

    step.solve()
    final_force = -step.contact_provider.evaluate().contact_force_on_structure[0]
    final_displacement = float(np.mean(
        step.solution.x.array.reshape((-1, 3))[right, 0],
    ))
    assert final_force == pytest.approx(reference[1.0][2], rel=1e-5)
    assert final_displacement == pytest.approx(reference[1.0][0], rel=1e-5)
    assert np.max(step.response.state.committed[
        "equivalent_plastic_strain"
    ].function.x.array) == pytest.approx(reference[1.0][1], rel=1e-5)

    resumed = _patch(**options)[3]
    resumed.load_checkpoint(checkpoint)
    resumed.solve(until=0.75)
    resumed.solve()
    np.testing.assert_allclose(
        resumed.solution.x.array, step.solution.x.array,
        rtol=1e-9, atol=1e-10,
    )
    np.testing.assert_allclose(
        resumed.response.state.committed[
            "equivalent_plastic_strain"
        ].function.x.array,
        step.response.state.committed[
            "equivalent_plastic_strain"
        ].function.x.array,
        rtol=1e-9, atol=1e-10,
    )
    assert resumed.energy_history[-1].tool_work == pytest.approx(
        step.energy_history[-1].tool_work, rel=1e-9, abs=1e-10,
    )


def test_tabular_contact_amplitude_knots_are_mandatory_accepted_stations():
    path = amplitudes.tabular(
        (0.0, 0.5, 0.75, 1.0), (0.0, 1.0, 0.0, 1.0),
        name="contact_reversal",
    )
    options = dict(
        triangulated=True, gap=0.0,
        tool_translation=(-0.002, 0.0, 0.0),
        penalty=1000.0, yield_stress=0.1,
        applied_traction=0.0, transverse_boundary_fixed=True,
        load_amplitude=path,
    )
    with pytest.raises(ValueError, match="tabular amplitude knot"):
        _patch(**options, increments=3)
    _, _, _, step = _patch(
        **options,
        increment_control=steps.automatic(
            initial=0.7, maximum=0.7, minimum=0.01,
        ),
    )
    assert step.required_load_factors == (0.5, 0.75)
    step.solve()
    accepted = tuple(item.load_factor for item in step.accepted_increments)
    assert any(abs(value - 0.5) <= 1e-12 for value in accepted)
    assert any(abs(value - 0.75) <= 1e-12 for value in accepted)


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
    incomplete = step_capability(
        model, target=displacement,
        options={
            "material": material,
            "contact_integration": {first.name: "gauss"},
        },
    )
    assert not incomplete["supported"]
    assert any(
        item["message"].startswith("AFM-CONTACT-INTEGRATION")
        for item in incomplete["rejection_reasons"]
    )
    wrong_geometry = step_capability(
        model, target=displacement,
        options={
            "material": material,
            "contact_integration": {
                first.name: "planar_active_partition",
                "second_tool": "gauss",
            },
        },
    )
    assert not wrong_geometry["supported"]
    assert any(
        item["message"].startswith("AFM-CONTACT-PARTITION")
        for item in wrong_geometry["rejection_reasons"]
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
