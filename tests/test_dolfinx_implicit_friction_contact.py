"""Private planar friction Operator checks; public Step admission is separate."""

from __future__ import annotations

import basix.ufl
import json
import xml.etree.ElementTree as ET
from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np
import pytest

from agentfem import boundary_models, constitutive, fields, models, results, solvers, steps, studies
from agentfem import mesh as agent_mesh
from agentfem.boundary_models.dolfinx_implicit_facet_contact import (
    dolfinx_implicit_rigid_contact,
)


def _rigid_surface(*, geometry="plane"):
    if geometry == "plane":
        return boundary_models.rigid_plane(
            point=(0.05, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
        )
    if geometry == "triangulated_plane":
        return boundary_models.triangulated_rigid_surface(
            vertices=(
                (0.05, -1.0, -1.0), (0.05, -1.0, 2.0),
                (0.05, 2.0, 2.0), (0.05, 2.0, -1.0),
            ),
            triangles=((0, 1, 2), (0, 2, 3)),
        )
    if geometry != "closed_convex":
        raise ValueError("Unknown friction test geometry.")
    return boundary_models.convex_polyhedral_rigid_surface(
        vertices=(
            (0.05, -1.0, -1.0), (0.55, -1.0, -1.0),
            (0.55, 2.0, -1.0), (0.05, 2.0, -1.0),
            (0.05, -1.0, 2.0), (0.55, -1.0, 2.0),
            (0.55, 2.0, 2.0), (0.05, 2.0, 2.0),
        ),
        triangles=(
            (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
            (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
        ),
    )


def _operator(*, motion=None, geometry="plane"):
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    space = fem.functionspace(
        domain,
        basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,)),
    )
    displacement = fem.Function(space)
    facets = mesh.locate_entities_boundary(
        domain, 2, lambda x: np.isclose(x[0], 0.0),
    )
    tags = mesh.meshtags(
        domain, 2, facets, np.full(len(facets), 7, dtype=np.int32),
    )
    slave = agent_mesh.tagged_boundary_region(domain, tags, tag=7, name="slave")
    pair = boundary_models.rigid_contact_pair(
        slave,
        boundary_models.rigid_body(
            _rigid_surface(geometry=geometry),
            motion_schedule=(
                None if motion is None else
                boundary_models.prescribed_rigid_motion_schedule(
                    motion, start_time=0.0, end_time=1.0,
                )
            ),
        ),
        penalty=200.0, friction_coefficient=0.4,
        tangential_penalty=100.0,
    )
    return displacement, dolfinx_implicit_rigid_contact(pair, displacement)


@pytest.mark.parametrize(
    ("normal_motion", "expected_force"),
    [(0.0, (-10.0, 4.0, 0.0)), (0.01, (-8.0, 3.2, 0.0))],
)
def test_translating_plane_friction_uses_relative_tool_motion_and_tangent(
    normal_motion, expected_force,
):
    motion = boundary_models.prescribed_rigid_motion(
        translation=(normal_motion, 0.2, 0.0),
    )
    displacement, operator = _operator(motion=motion)
    displacement.x.array.reshape(-1, 3)[:, 0] = 0.10
    operator.update_motion(1.0)
    base = displacement.x.array.copy()
    contribution = operator.evaluate()
    assert contribution.friction_tangent.sliding_points > 0
    np.testing.assert_allclose(
        contribution.contact_force_on_structure, expected_force,
        atol=1e-10,
    )
    direction = np.random.default_rng(444).normal(size=base.shape)
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-7
    displacement.x.array[:] = base + epsilon * direction
    plus = operator.evaluate().nodal_structural_residual
    displacement.x.array[:] = base - epsilon * direction
    minus = operator.evaluate().nodal_structural_residual
    displacement.x.array[:] = base
    np.testing.assert_allclose(
        contribution.tangent_action(direction.reshape(-1, 3)),
        (plus - minus) / (2 * epsilon), rtol=2e-5, atol=2e-6,
    )
    operator.commit_accepted()
    assert operator.friction_kinematics.accepted.motion_factor == 1.0
    assert operator.evaluate().friction_assembly.dissipation_increment == pytest.approx(0.0)
    accepted = operator.snapshot_accepted_state()
    assert accepted["kinematics"]["motion_factor"] == 1.0
    operator.restore_accepted_state(accepted)
    operator.validate_accepted_configuration()
    corrupted = json.loads(json.dumps(accepted))
    corrupted["kinematics"]["motion_factor"] = 0.5
    with pytest.raises(ValueError, match="checkpoint rigid pose"):
        operator.restore_accepted_state(corrupted)
    assert operator.snapshot_accepted_state() == accepted
    operator.update_motion(0.5)
    with pytest.raises(ValueError, match="accepted rigid pose"):
        operator.validate_accepted_configuration()


def test_coplanar_triangle_tool_matches_analytical_plane_friction_operator():
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.0, 0.2, 0.0),
    )
    plane_displacement, plane = _operator(motion=motion)
    facet_displacement, facet = _operator(
        motion=motion, geometry="triangulated_plane",
    )
    for displacement, operator in (
        (plane_displacement, plane), (facet_displacement, facet),
    ):
        displacement.x.array.reshape(-1, 3)[:, 0] = 0.10
        operator.update_motion(1.0)
    expected = plane.evaluate()
    actual = facet.evaluate()
    assert actual.friction_tangent.sliding_points > 0
    np.testing.assert_allclose(
        actual.nodal_structural_residual,
        expected.nodal_structural_residual, rtol=0.0, atol=1e-10,
    )
    direction = np.random.default_rng(2409).normal(size=facet_displacement.x.array.shape)
    direction /= np.linalg.norm(direction)
    np.testing.assert_allclose(
        actual.tangent_action(direction.reshape(-1, 3)),
        expected.tangent_action(direction.reshape(-1, 3)),
        rtol=0.0, atol=1e-10,
    )
    base = facet_displacement.x.array.copy()
    epsilon = 1.0e-7
    facet_displacement.x.array[:] = base + epsilon * direction
    plus = facet.evaluate().nodal_structural_residual
    facet_displacement.x.array[:] = base - epsilon * direction
    minus = facet.evaluate().nodal_structural_residual
    facet_displacement.x.array[:] = base
    np.testing.assert_allclose(
        actual.tangent_action(direction.reshape(-1, 3)),
        (plus - minus) / (2 * epsilon), rtol=2e-5, atol=2e-6,
    )


def _moving_friction_patch_step(
    *, increments=1, rotation=(0.0, 0.0, 0.0), geometry="plane",
):
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1)
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain)
    displacement = model.field(fields.displacement(domain))
    right = agent_mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="clamp")
    left = agent_mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="slave")
    model.clamp(displacement, on=right)
    model.fix(displacement, on=left, component=0, value=0.10)
    model.fix(displacement, on=left, component=1, value=0.0)
    model.fix(displacement, on=left, component=2, value=0.0)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=0.0, yield_stress=1.0e9, hardening_modulus=100.0,
    ))
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.0, 0.40, 0.0), rotation=rotation,
        reference_point=(0.05, 0.0, 0.0),
    )
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        left,
        boundary_models.rigid_body(
            _rigid_surface(geometry=geometry),
            motion_schedule=boundary_models.prescribed_rigid_motion_schedule(
                motion, start_time=0.0, end_time=1.0,
            ),
        ),
        penalty=500.0, friction_coefficient=0.4,
        tangential_penalty=100.0, name="moving_frictional_plane",
    ))
    return model.step(
        target=displacement, material=material,
        incrementation=steps.fixed(increments), progress=False,
    )


def test_moving_planar_friction_public_step_force_work_and_restart(tmp_path):
    reference = _moving_friction_patch_step(increments=2)
    result = reference.solve_result()
    assert result.status == "completed"
    dual = result.metadata["constraint_duals"][0]
    np.testing.assert_allclose(dual["resultant"], (-25.0, 10.0, 0.0), atol=1e-8)
    assert reference.energy_history[-1].friction_dissipation > 0.0
    assert reference.energy_history[-1].tool_work > 0.0
    assert reference.contact_provider.friction_kinematics.accepted.motion_factor == 1.0

    partial = _moving_friction_patch_step(increments=2)
    partial.solve(until=0.5)
    saved = partial.save_checkpoint(tmp_path / "moving_friction")
    resumed = _moving_friction_patch_step(increments=2)
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(
        resumed.solution.x.array, reference.solution.x.array, atol=1e-9,
    )
    assert resumed.energy_history[-1].tool_work == pytest.approx(
        reference.energy_history[-1].tool_work,
    )
    assert resumed.energy_history[-1].friction_dissipation == pytest.approx(
        reference.energy_history[-1].friction_dissipation,
    )


def test_rotating_planar_friction_public_step_work_and_restart(tmp_path):
    rotation = (0.0, 0.0, 0.15)
    reference = _moving_friction_patch_step(increments=4, rotation=rotation)
    result = reference.solve_result()
    assert result.status == "completed"
    assert reference.energy_history[-1].friction_dissipation > 0.0
    assert np.isfinite(reference.energy_history[-1].tool_work)
    assert reference.contact_provider.friction_kinematics.accepted.motion_factor == 1.0
    assert reference.contact_provider.summary()["linearization"] == (
        "moving_plane_normal_plus_coulomb_semismooth_return_map"
    )
    dual = result.metadata["constraint_duals"][0]
    generalized_force = np.asarray(dual["force"], dtype=float)
    assert generalized_force.shape == (6,)
    assert np.isfinite(generalized_force).all()
    assert abs(generalized_force[-1]) > 0.0
    assert result.metadata["constraint_work_evidence"][0]["complete"]

    partial = _moving_friction_patch_step(increments=4, rotation=rotation)
    partial.solve(until=0.5)
    saved = partial.save_checkpoint(tmp_path / "rotating_friction")
    resumed = _moving_friction_patch_step(increments=4, rotation=rotation)
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(
        resumed.solution.x.array, reference.solution.x.array, atol=1e-9,
    )
    assert resumed.energy_history[-1].tool_work == pytest.approx(
        reference.energy_history[-1].tool_work,
    )
    assert resumed.energy_history[-1].friction_dissipation == pytest.approx(
        reference.energy_history[-1].friction_dissipation,
    )


def test_closed_convex_planar_face_matches_rotating_plane_global_path(tmp_path):
    kwargs = {"increments": 4, "rotation": (0.0, 0.0, 0.15)}
    plane = _moving_friction_patch_step(**kwargs)
    convex = _moving_friction_patch_step(**kwargs, geometry="closed_convex")
    plane_result = plane.solve_result()
    convex_result = convex.solve_result()
    assert plane_result.status == convex_result.status == "completed"
    np.testing.assert_allclose(convex.solution.x.array, plane.solution.x.array, atol=1e-9)
    for name in ("tool_work", "friction_dissipation", "contact_potential"):
        assert getattr(convex.energy_history[-1], name) == pytest.approx(
            getattr(plane.energy_history[-1], name), rel=1e-8, abs=1e-9,
        )
    np.testing.assert_allclose(
        convex_result.metadata["constraint_duals"][0]["force"],
        plane_result.metadata["constraint_duals"][0]["force"], atol=1e-8,
    )
    partial = _moving_friction_patch_step(**kwargs, geometry="closed_convex")
    partial.solve(until=0.5)
    saved = partial.save_checkpoint(tmp_path / "convex_friction")
    resumed = _moving_friction_patch_step(**kwargs, geometry="closed_convex")
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(resumed.solution.x.array, convex.solution.x.array, atol=1e-9)
    assert resumed.energy_history[-1].tool_work == pytest.approx(
        convex.energy_history[-1].tool_work,
    )
    assert resumed.energy_history[-1].friction_dissipation == pytest.approx(
        convex.energy_history[-1].friction_dissipation,
    )


def test_rotating_coplanar_triangle_tool_matches_plane_and_restarts(tmp_path):
    kwargs = {"increments": 4, "rotation": (0.0, 0.0, 0.15)}
    plane = _moving_friction_patch_step(**kwargs)
    facet = _moving_friction_patch_step(**kwargs, geometry="triangulated_plane")
    plane_result = plane.solve_result()
    facet_result = facet.solve_result()
    assert plane_result.status == facet_result.status == "completed"
    np.testing.assert_allclose(facet.solution.x.array, plane.solution.x.array, atol=1e-9)
    for name in ("tool_work", "friction_dissipation", "contact_potential"):
        assert getattr(facet.energy_history[-1], name) == pytest.approx(
            getattr(plane.energy_history[-1], name), rel=1e-8, abs=1e-9,
        )
    np.testing.assert_allclose(
        facet_result.metadata["constraint_duals"][0]["force"],
        plane_result.metadata["constraint_duals"][0]["force"], atol=1e-8,
    )
    partial = _moving_friction_patch_step(**kwargs, geometry="triangulated_plane")
    partial.solve(until=0.5)
    saved = partial.save_checkpoint(tmp_path / "facet_friction")
    resumed = _moving_friction_patch_step(**kwargs, geometry="triangulated_plane")
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(resumed.solution.x.array, facet.solution.x.array, atol=1e-9)
    assert resumed.energy_history[-1].tool_work == pytest.approx(
        facet.energy_history[-1].tool_work,
    )
    assert resumed.energy_history[-1].friction_dissipation == pytest.approx(
        facet.energy_history[-1].friction_dissipation,
    )


def test_closed_convex_rotating_face_friction_tangent_and_stale_pose():
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.0, 0.40, 0.0), rotation=(0.0, 0.0, 0.15),
        reference_point=(0.05, 0.0, 0.0),
    )
    displacement, operator = _operator(motion=motion, geometry="closed_convex")
    displacement.x.array.reshape(-1, 3)[:, 0] = 0.10
    operator.update_motion(0.5)
    operator.commit_accepted()
    operator.update_motion(1.0)
    base = displacement.x.array.copy()
    contribution = operator.evaluate()
    assert contribution.friction_tangent.sliding_points > 0
    direction = np.random.default_rng(443).normal(size=base.shape)
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-7
    displacement.x.array[:] = base + epsilon * direction
    plus = operator.evaluate().nodal_structural_residual
    displacement.x.array[:] = base - epsilon * direction
    minus = operator.evaluate().nodal_structural_residual
    displacement.x.array[:] = base
    np.testing.assert_allclose(
        contribution.tangent_action(direction.reshape(-1, 3)),
        (plus - minus) / (2 * epsilon), rtol=2e-5, atol=2e-6,
    )


def test_accepted_contact_point_export_is_raw_and_reconstructs_tool_resultant(tmp_path):
    step = _moving_friction_patch_step(
        increments=4, rotation=(0.0, 0.0, 0.15), geometry="closed_convex",
    )
    result = step.solve_result()
    destination = tmp_path / "contact_samples"
    paths = results.export_accepted_contact_points(
        step, destination, units={"length": "m", "force": "N"},
        result=result, vtk_view=True,
    )
    assert result.artifacts["accepted_contact_points"] == paths["contact_points"]
    assert result.fields["rigid_contact_normal_pressure_raw"].location == (
        "reference_boundary_quadrature"
    )
    assert result.fields["rigid_contact_friction_traction_raw"].processing[
        "npz_key"
    ] == "surface_friction_tractions"
    with np.load(paths["contact_points"], allow_pickle=False) as archive:
        samples = {name: archive[name] for name in archive.files}
    metadata = json.loads(paths["contact_point_metadata"].read_text())
    assert metadata["schema"] == "agentfem.accepted-contact-points.v1"
    assert metadata["smoothing"] == "none"
    assert metadata["complete"] and not metadata["restart_archive"]
    assert metadata["units"] == {"length": "m", "force": "N"}
    assert metadata["vtk_view"] == "current_slave_positions_vertex_only"
    assert result.artifacts["accepted_contact_points_vtp"] == paths[
        "contact_point_view"
    ]
    assert samples["point_ids"].size == samples["reference_weights"].size
    vtk_root = ET.parse(paths["contact_point_view"]).getroot()
    assert vtk_root.attrib["type"] == "PolyData"
    vtk_piece = vtk_root.find("./PolyData/Piece")
    assert int(vtk_piece.attrib["NumberOfPoints"]) == len(samples["point_ids"])
    assert int(vtk_piece.attrib["NumberOfVerts"]) == len(samples["point_ids"])
    assert vtk_piece.find("./PointData/DataArray[@Name='normal_pressures']") is not None
    assert vtk_piece.find("./PointData/DataArray[@Name='pair_index']") is not None
    assert np.any(samples["active"])
    assert np.all(samples["normal_pressures"] >= 0.0)
    assert np.all(samples["cumulative_friction_dissipation_densities"] >= 0.0)
    resultant = np.sum(
        samples["reference_weights"][:, None] * (
            samples["normal_pressures"][:, None] * samples["surface_normals"]
            + samples["surface_friction_tractions"]
        ), axis=0,
    )
    np.testing.assert_allclose(
        resultant, result.metadata["constraint_duals"][0]["resultant"], atol=1e-9,
    )
    with pytest.raises(FileExistsError, match="fresh output"):
        results.export_accepted_contact_points(step, destination)
    step.solution.x.array[0] += 0.1
    with pytest.raises(RuntimeError, match="fully accepted"):
        results.export_accepted_contact_points(step, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_contact_point_vtp_is_readable_as_unsmoothed_vertices(tmp_path):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy

    step = _moving_friction_patch_step(
        increments=4, rotation=(0.0, 0.0, 0.15), geometry="closed_convex",
    )
    step.solve()
    paths = results.export_accepted_contact_points(
        step, tmp_path / "vtk_samples", vtk_view=True,
    )
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(paths["contact_point_view"]))
    reader.Update()
    output = reader.GetOutput()
    with np.load(paths["contact_points"], allow_pickle=False) as archive:
        assert output.GetNumberOfPoints() == len(archive["point_ids"])
        assert output.GetNumberOfVerts() == len(archive["point_ids"])
        np.testing.assert_allclose(
            vtk_to_numpy(output.GetPoints().GetData()), archive["slave_positions"],
        )
        np.testing.assert_allclose(
            vtk_to_numpy(output.GetPointData().GetArray("normal_pressures")),
            archive["normal_pressures"],
        )


def test_contact_point_export_rejects_nonfinite_provider_data_before_writing(
    tmp_path, monkeypatch,
):
    step = _moving_friction_patch_step(increments=2)
    step.solve()
    with pytest.raises(TypeError, match="AFM-CONTACT-EXPORT-VIEW"):
        results.export_accepted_contact_points(
            step, tmp_path / "bad_view_option", vtk_view="yes",
        )
    assert not (tmp_path / "bad_view_option").exists()
    from agentfem.results import contact_points as contact_export

    def fail_view(_arrays):
        raise RuntimeError("injected VTP preparation failure")

    with monkeypatch.context() as patch:
        patch.setattr(contact_export, "_contact_points_vtp", fail_view)
        with pytest.raises(RuntimeError, match="VTP preparation failure"):
            results.export_accepted_contact_points(
                step, tmp_path / "bad_view_data", vtk_view=True,
            )
    assert not (tmp_path / "bad_view_data").exists()
    provider = step.contact_provider
    original = provider.accepted_point_samples

    def corrupt():
        sample = original()
        pressure = sample["normal_pressures"].copy()
        pressure[0] = np.nan
        sample["normal_pressures"] = pressure
        return sample

    monkeypatch.setattr(provider, "accepted_point_samples", corrupt)
    destination = tmp_path / "invalid_samples"
    with pytest.raises(ValueError, match="AFM-CONTACT-EXPORT-ARRAY"):
        results.export_accepted_contact_points(step, destination)
    assert not destination.exists()


@pytest.mark.parametrize("tool_travel", [0.03, 0.40])
def test_rotating_planar_friction_uses_current_master_point_tangent(tool_travel):
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0, tool_travel, 0), rotation=(0, 0, 0.15),
        reference_point=(0.05, 0, 0),
    )
    displacement, operator = _operator(motion=motion)
    displacement.x.array.reshape(-1, 3)[:, 0] = 0.10
    operator.update_motion(0.5)
    operator.commit_accepted()
    operator.update_motion(1.0)
    base = displacement.x.array.copy()
    contribution = operator.evaluate()
    assert contribution.active_points > 0
    if tool_travel < 0.1:
        assert contribution.friction_tangent.sticking_points > 0
    else:
        assert contribution.friction_tangent.sliding_points > 0
    direction = np.random.default_rng(442).normal(size=base.shape)
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-7
    displacement.x.array[:] = base + epsilon * direction
    plus = operator.evaluate().nodal_structural_residual
    displacement.x.array[:] = base - epsilon * direction
    minus = operator.evaluate().nodal_structural_residual
    displacement.x.array[:] = base
    np.testing.assert_allclose(
        contribution.tangent_action(direction.reshape(-1, 3)),
        (plus - minus) / (2 * epsilon), rtol=2e-5, atol=2e-6,
    )
    assert operator.friction_kinematics.accepted.motion_factor == 0.5


def test_rotating_planar_friction_rejects_a_projection_from_another_pose():
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0, 0.2, 0), rotation=(0, 0, 0.15),
        reference_point=(0.05, 0, 0),
    )
    displacement, operator = _operator(motion=motion)
    displacement.x.array.reshape(-1, 3)[:, 0] = 0.10
    operator.update_motion(0.5)
    evaluation, record, response, _factor = operator._current_point_response()
    friction = operator.pair.friction.evaluate(
        point_ids=record.point_ids, normal_response=response,
        relative_displacement_increment=np.zeros_like(evaluation.query_points),
    )
    from agentfem.boundary_models.implicit_friction import linearize_planar_penalty_friction

    with pytest.raises(ValueError, match="stale for the current rigid pose"):
        linearize_planar_penalty_friction(
            evaluation, record, response, friction, operator.pair.friction,
            operator.surface, motion=motion, current_motion_factor=1.0,
        )


@pytest.mark.parametrize("tangential", [0.02, 0.10])
def test_fixed_plane_friction_backend_stick_slip_tangent(tangential):
    displacement, operator = _operator()
    accepted = displacement.x.array.copy()
    displacement.x.array.reshape(-1, 3)[:, 0] = 0.10
    displacement.x.array.reshape(-1, 3)[:, 1] = tangential
    base = displacement.x.array.copy()
    contribution = operator.evaluate()
    assert contribution.active_points > 0
    assert contribution.friction_assembly is not None
    assert contribution.friction_tangent is not None
    assert operator.friction_state.accepted is None
    if tangential < 0.05:
        assert contribution.friction_tangent.sticking_points > 0
    else:
        assert contribution.friction_tangent.sliding_points > 0
    direction = np.random.default_rng(102).normal(size=base.shape)
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-7
    displacement.x.array[:] = base + epsilon * direction
    plus = operator.evaluate().nodal_structural_residual
    displacement.x.array[:] = base - epsilon * direction
    minus = operator.evaluate().nodal_structural_residual
    displacement.x.array[:] = base
    np.testing.assert_allclose(
        contribution.tangent_action(direction.reshape(-1, 3)),
        (plus - minus) / (2.0 * epsilon),
        rtol=2.0e-5, atol=2.0e-6,
    )
    assert np.allclose(displacement.x.array, base)
    assert operator.friction_state.accepted is None
    assert not np.array_equal(accepted, base)


def test_fixed_plane_friction_accept_restore_and_corrupt_state_rejection():
    displacement, operator = _operator()
    displacement.x.array.reshape(-1, 3)[:, 0] = 0.10
    displacement.x.array.reshape(-1, 3)[:, 1] = 0.10
    initial = operator.snapshot_accepted_state()
    first = operator.evaluate()
    assert first.friction_assembly.dissipation_increment > 0.0
    assert operator.friction_state.accepted is None
    operator.commit_accepted()
    accepted = operator.snapshot_accepted_state()
    assert accepted != initial
    assert json.loads(json.dumps(accepted)) == accepted
    assert operator.friction_state.accepted is not None
    runtime = operator.snapshot_runtime_state()
    assert runtime[0] is operator.friction_state.accepted
    assert runtime[1] is operator.friction_kinematics.accepted
    assert operator.evaluate().friction_assembly.dissipation_increment == pytest.approx(0.0)
    displacement.x.array.reshape(-1, 3)[:, 1] += 0.04
    assert operator.evaluate().friction_assembly.dissipation_increment > 0.0
    operator.restore_accepted_state(accepted)
    assert operator.snapshot_accepted_state() == accepted
    damaged = json.loads(json.dumps(accepted))
    damaged["point_ids"][0] = -1
    with pytest.raises(ValueError, match="point IDs"):
        operator.restore_accepted_state(damaged)
    assert operator.snapshot_accepted_state() == accepted


def _frictional_j2_step(
    *, tangential_load: float, linear_solver=None, pair_count: int = 1,
    first_convex: bool = False,
):
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain)
    displacement = model.field(fields.displacement(domain))
    right = agent_mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="clamp")
    left = agent_mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="slave")
    model.clamp(displacement, on=right)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=0.0, yield_stress=1.0e9, hardening_modulus=100.0,
    ))
    if pair_count not in {1, 2}:
        raise ValueError("This fixture tests one or two additive rigid pairs.")
    for index in range(pair_count):
        geometry = "closed_convex" if first_convex and index == 0 else "plane"
        pair = boundary_models.rigid_contact_pair(
            left,
            boundary_models.rigid_body(_rigid_surface(geometry=geometry)),
            penalty=500.0 / pair_count, friction_coefficient=0.4,
            tangential_penalty=100.0 / pair_count,
            name=(
                "fixed_frictional_plane" if pair_count == 1
                else f"fixed_frictional_plane_{index}"
            ),
        )
        model.add_boundary_model(pair)
    model.traction((100.0, tangential_load, 0.0), on=left)
    step = model.step(
        target=displacement, material=material, incrementation=steps.fixed(2),
        solver_options=solvers.newton(
            relative_tolerance=1e-8, absolute_tolerance=1e-9,
            maximum_iterations=30, line_search="backtracking",
            linear_solver=(solvers.direct_solver() if linear_solver is None else linear_solver),
        ),
        progress=False,
    )
    return step


def test_mixed_convex_and_planar_friction_pairs_keep_separate_identity(tmp_path):
    reference = _frictional_j2_step(tangential_load=100.0)
    mixed = _frictional_j2_step(
        tangential_load=100.0, pair_count=2, first_convex=True,
    )
    reference_result = reference.solve_result()
    mixed_result = mixed.solve_result()
    assert reference_result.status == mixed_result.status == "completed"
    np.testing.assert_allclose(
        mixed.solution.x.array, reference.solution.x.array, rtol=1e-8, atol=1e-9,
    )
    assert len(mixed_result.metadata["constraint_duals"]) == 2
    assert mixed_result.metadata["constraint_duals"][0]["constraint_name"] != (
        mixed_result.metadata["constraint_duals"][1]["constraint_name"]
    )
    assert mixed.energy_history[-1].friction_dissipation == pytest.approx(
        reference.energy_history[-1].friction_dissipation, rel=1e-8,
    )
    paths = results.export_accepted_contact_points(mixed, tmp_path / "mixed_points")
    with np.load(paths["contact_points"], allow_pickle=False) as archive:
        pair_index = archive["pair_index"]
        assert set(pair_index.tolist()) == {0, 1}
        assert len(archive["point_ids"]) == len(pair_index)
    metadata = json.loads(paths["contact_point_metadata"].read_text())
    assert [item["constraint_name"] for item in metadata["pairs"]] == [
        item["constraint_name"] for item in mixed_result.metadata["constraint_duals"]
    ]


def test_public_friction_rejects_symmetry_only_linear_solver():
    with pytest.raises(ValueError, match="AFM-IMPLICIT-FRICTION-LINEAR"):
        _frictional_j2_step(
            tangential_load=15.0,
            linear_solver=solvers.LinearSolverOptions(ksp_type="cg", pc_type="jacobi"),
        )


@pytest.mark.parametrize(
    ("shear_displacement", "expected_shear_force", "expected_dissipation"),
    [(0.02, -2.0, 0.0), (0.20, -10.0, 1.0)],
)
@pytest.mark.parametrize("geometry", ["plane", "closed_convex"])
def test_prescribed_friction_patch_matches_independent_force_and_energy(
    shear_displacement, expected_shear_force, expected_dissipation, geometry,
):
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1)
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain)
    displacement = model.field(fields.displacement(domain))
    right = agent_mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="clamp")
    left = agent_mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="slave")
    model.clamp(displacement, on=right)
    model.fix(displacement, on=left, component=0, value=0.10)
    model.fix(displacement, on=left, component=1, value=shear_displacement)
    model.fix(displacement, on=left, component=2, value=0.0)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=0.0, yield_stress=1.0e9, hardening_modulus=100.0,
    ))
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        left,
        boundary_models.rigid_body(_rigid_surface(geometry=geometry)),
        penalty=500.0, friction_coefficient=0.4,
        tangential_penalty=100.0, name="patch_plane",
    ))
    step = model.step(
        target=displacement, material=material,
        incrementation=steps.fixed(1), progress=False,
    )
    result = step.solve_result()
    force = np.asarray(result.metadata["constraint_duals"][0]["force"])
    np.testing.assert_allclose(
        force, (-25.0, expected_shear_force, 0.0),
        rtol=1e-12, atol=1e-10,
    )
    strong_reaction = np.sum(
        result.fields["RF"].field.x.array.reshape((-1, 3)), axis=0
    )
    np.testing.assert_allclose(
        strong_reaction + force, 0.0, rtol=0.0, atol=1e-8,
    )
    assert step.energy_history[-1].friction_dissipation == pytest.approx(
        expected_dissipation, rel=1e-12, abs=1e-10,
    )
    assert step.energy_history[-1].contact_potential == pytest.approx(
        0.625 + 0.5 * expected_shear_force ** 2 / 100.0,
        rel=1e-12, abs=1e-10,
    )


@pytest.mark.parametrize("tangential_load", [1.0, 100.0])
def test_public_fixed_plane_j2_friction_accepts_and_restarts(tmp_path, tangential_load):
    reference = _frictional_j2_step(tangential_load=tangential_load)
    result = reference.solve_result()
    assert len(reference.energy_history) == 2
    assert reference.contact_provider.current_point_diagnostics()["active_point_count"] > 0
    assert reference.contact_provider.friction_state.accepted is not None
    assert reference.energy_history[-1].friction_dissipation >= 0.0
    if tangential_load == 100.0:
        assert reference.energy_history[-1].friction_dissipation > 0.0
    assert reference.energy_history[-1].internal_energy == pytest.approx(
        reference.energy_history[-1].stored_energy
        + reference.energy_history[-1].plastic_dissipation
        + reference.energy_history[-1].contact_potential
        + reference.energy_history[-1].friction_dissipation
        + reference.energy_history[-1].friction_separation_release
    )

    partial = _frictional_j2_step(tangential_load=tangential_load)
    partial.solve(until=0.5)
    path = partial.save_checkpoint(tmp_path / "friction.npz")
    restarted = _frictional_j2_step(tangential_load=tangential_load)
    restarted.load_checkpoint(path)
    restarted.solve()
    np.testing.assert_allclose(
        restarted.solution.x.array, reference.solution.x.array,
        rtol=1e-8, atol=1e-9,
    )
    assert restarted.energy_history[-1].friction_dissipation == pytest.approx(
        reference.energy_history[-1].friction_dissipation, rel=1e-8, abs=1e-9,
    )
    assert "friction_dissipation" in result.histories
    assert "friction_separation_release" in result.histories
    assert result.metadata["mechanical_energy_ledger"]["friction_separation_release"] == (
        "reported_numerical_loss_not_coulomb_dissipation"
    )


def test_two_named_frictional_pairs_sum_and_restart_atomically(tmp_path, monkeypatch):
    single = _frictional_j2_step(tangential_load=100.0)
    single.solve()
    double = _frictional_j2_step(tangential_load=100.0, pair_count=2)
    result = double.solve_result()
    assert result.status == "completed"
    np.testing.assert_allclose(
        double.solution.x.array, single.solution.x.array,
        rtol=1e-8, atol=1e-9,
    )
    assert double.energy_history[-1].friction_dissipation == pytest.approx(
        single.energy_history[-1].friction_dissipation, rel=1e-8, abs=1e-9,
    )
    names = tuple(item.name for item in double.contact_provider.providers)
    assert names == ("fixed_frictional_plane_0", "fixed_frictional_plane_1")
    assert tuple(
        item["constraint_name"] for item in result.metadata["constraint_duals"]
    ) == names
    accepted = double.contact_provider.snapshot_accepted_state()
    assert set(accepted["pairs"]) == set(names)
    assert all(accepted["pairs"][name] is not None for name in names)

    partial = _frictional_j2_step(tangential_load=100.0, pair_count=2)
    partial.solve(until=0.5)
    saved = partial.save_checkpoint(tmp_path / "two_frictional_pairs.npz")
    accepted_half = partial.contact_provider.snapshot_accepted_state()
    second = partial.contact_provider.providers[1]
    commit_second = second.commit_accepted

    def reject_after_second_commit():
        commit_second()
        raise RuntimeError("injected second-pair commit failure")

    monkeypatch.setattr(second, "commit_accepted", reject_after_second_commit)
    with pytest.raises(RuntimeError, match="second-pair commit failure"):
        partial.solve()
    assert partial.accepted_load_factor == pytest.approx(0.5)
    assert partial.contact_provider.snapshot_accepted_state() == accepted_half

    resumed = _frictional_j2_step(tangential_load=100.0, pair_count=2)
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(
        resumed.solution.x.array, double.solution.x.array,
        rtol=1e-8, atol=1e-9,
    )
    assert resumed.contact_provider.snapshot_accepted_state() == accepted
    with np.load(saved, allow_pickle=False) as archive:
        damaged = {name: archive[name].copy() for name in archive.files}
    state = json.loads(str(damaged["contact_state"]))
    state["pairs"].pop(names[1])
    damaged["contact_state"] = json.dumps(state)
    corrupt = tmp_path / "missing_frictional_pair.npz"
    np.savez_compressed(corrupt, **damaged)
    rejected = _frictional_j2_step(tangential_load=100.0, pair_count=2)
    initial_state = rejected.contact_provider.snapshot_accepted_state()
    with pytest.raises(ValueError, match="pair friction State identity"):
        rejected.load_checkpoint(corrupt)
    assert rejected.accepted_load_factor == 0.0
    assert rejected.contact_provider.snapshot_accepted_state() == initial_state


def test_public_friction_failure_rolls_back_joint_state(monkeypatch):
    step = _frictional_j2_step(tangential_load=100.0)
    step.solve(until=0.5)
    accepted_solution = step.solution.x.array.copy()
    accepted_friction = step.contact_provider.snapshot_accepted_state()
    accepted_energy = tuple(step.energy_history)
    commit = step.contact_provider.commit_accepted

    def fail_after_contact_commit():
        commit()
        raise RuntimeError("injected post-contact-commit failure")

    monkeypatch.setattr(step.contact_provider, "commit_accepted", fail_after_contact_commit)
    with pytest.raises(RuntimeError, match="injected post-contact-commit failure"):
        step.solve()
    assert step.accepted_load_factor == pytest.approx(0.5)
    np.testing.assert_array_equal(step.solution.x.array, accepted_solution)
    assert step.contact_provider.snapshot_accepted_state() == accepted_friction
    assert tuple(step.energy_history) == accepted_energy


def test_public_friction_checkpoint_corruption_is_atomic(tmp_path):
    partial = _frictional_j2_step(tangential_load=100.0)
    partial.solve(until=0.5)
    path = partial.save_checkpoint(tmp_path / "friction.npz")
    with np.load(path, allow_pickle=False) as archive:
        damaged = {name: archive[name].copy() for name in archive.files}
    state = json.loads(str(damaged["contact_state"]))
    state["point_ids"][0] = -1
    damaged["contact_state"] = json.dumps(state)
    corrupt = tmp_path / "corrupt_friction.npz"
    np.savez_compressed(corrupt, **damaged)
    rejected = _frictional_j2_step(tangential_load=100.0)
    initial_contact = rejected.contact_provider.snapshot_accepted_state()
    with pytest.raises(ValueError, match="point IDs"):
        rejected.load_checkpoint(corrupt)
    assert rejected.accepted_load_factor == 0.0
    assert not rejected.energy_history
    assert rejected.contact_provider.snapshot_accepted_state() == initial_contact
    state = json.loads(str(damaged["contact_state"]))
    state["point_ids"][0] = partial.contact_provider.adapter.trace.point_ids[0].item()
    state["kinematics"]["slave_positions"][0][1] += 0.05
    damaged["contact_state"] = json.dumps(state)
    inconsistent = tmp_path / "inconsistent_friction.npz"
    np.savez_compressed(inconsistent, **damaged)
    with pytest.raises(ValueError, match="accepted kinematics differ"):
        rejected.load_checkpoint(inconsistent)
    assert rejected.accepted_load_factor == 0.0
    assert rejected.contact_provider.snapshot_accepted_state() == initial_contact
