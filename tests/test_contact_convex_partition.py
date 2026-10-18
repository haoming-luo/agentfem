"""Bounded convex feature-partition checks against an independent oracle."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from agentfem import boundary_models
from agentfem.boundary_models.contact_convex_partition import (
    _unique_support_planes,
    partition_convex_p1_trace,
)
from agentfem.boundary_models.implicit_facet_contact import (
    linearize_piecewise_planar_contact,
)
from agentfem.boundary_models.dolfinx_implicit_facet_contact import (
    dolfinx_implicit_rigid_contact,
)


_SCRIPT = Path(__file__).resolve().parents[1] / "tools/verify_two_plane_contact_integration.py"
_oracle = runpy.run_path(str(_SCRIPT))["two_plane_triangle_contact"]


def _box(*, xmin=0.25, xmax=0.65, ymin=-10.0, ymax=10.0):
    zmin, zmax = -10.0, 10.0
    vertices = (
        (xmin, ymin, zmin), (xmax, ymin, zmin),
        (xmax, ymax, zmin), (xmin, ymax, zmin),
        (xmin, ymin, zmax), (xmax, ymin, zmax),
        (xmax, ymax, zmax), (xmin, ymax, zmax),
    )
    triangles = (
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
        (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
    )
    return boundary_models.convex_polyhedral_rigid_surface(
        vertices=vertices, triangles=triangles, name="two_plane_box",
    )


def _trace():
    return boundary_models.ContactTrace(
        point_ids=(0, 1, 2),
        node_ids=((0, 1, 2),) * 3,
        shape_values=((2 / 3, 1 / 6, 1 / 6),
                      (1 / 6, 2 / 3, 1 / 6),
                      (1 / 6, 1 / 6, 2 / 3)),
        weights=(1 / 6, 1 / 6, 1 / 6),
    )


def _evaluate(surface, positions, *, penalty=7.0):
    selected = partition_convex_p1_trace(
        _trace(), positions, surface, penalty=penalty,
    )
    evaluation = selected.trace.evaluate(positions)
    record = boundary_models.ContactProjectionRecord(
        selected.trace.point_ids, surface.project(evaluation.query_points),
    )
    assert record.projection.all_valid
    response = boundary_models.frictionless_penalty_contact_law(penalty).evaluate(
        record.projection,
    )
    bulk = linearize_piecewise_planar_contact(
        evaluation, record, response, surface,
    )
    tangent = np.zeros((len(positions) * 3, len(positions) * 3))
    np.add.at(tangent, (bulk.tangent_rows, bulk.tangent_columns), bulk.tangent_values)
    np.add.at(
        tangent,
        (selected.tangent_rows, selected.tangent_columns), selected.tangent_values,
    )
    return selected, bulk, tangent


@pytest.mark.parametrize("width, center", ((0.05, 0.3), (0.2, 0.45)))
def test_convex_box_feature_partition_matches_independent_oracle(width, center):
    surface = _box(xmin=center - width, xmax=center + width)
    positions = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    selected, bulk, tangent = _evaluate(surface, positions)
    assert selected.split_faces == selected.active_feature_lines == 1
    gaps = np.asarray((
        (-center - width, 1 - center - width, -center - width),
        (center - width, center - 1 - width, center - width),
    ))
    oracle = _oracle(gaps, ((1, 0, 0), (-1, 0, 0)), penalty=7.0)
    assert bulk.potential_energy == pytest.approx(oracle["potential"], rel=1e-12)
    np.testing.assert_allclose(
        bulk.nodal_structural_residual, oracle["nodal_residual"], atol=1e-12,
    )
    np.testing.assert_allclose(tangent, oracle["nodal_tangent"], atol=1e-12)
    if width == 0.05:
        assert selected.trace.point_count > _trace().point_count


def test_convex_partition_rejects_three_dominant_features():
    surface = boundary_models.convex_polyhedral_rigid_surface(
        vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
        triangles=((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
    )
    positions = np.asarray(((0.05, 0.3, 0.3),
                            (0.3, 0.05, 0.3),
                            (0.3, 0.3, 0.05)))
    with pytest.raises(NotImplementedError, match="more than two dominant"):
        partition_convex_p1_trace(_trace(), positions, surface, penalty=7)


def test_distinct_support_planes_are_not_merged_by_query_tolerance():
    from types import SimpleNamespace

    facets = SimpleNamespace(
        facet_normals=np.asarray(((1.0, 0.0, 0.0),
                                  (np.cos(1e-7), np.sin(1e-7), 0.0))),
        vertices=np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
        tolerance=1e-3,
        scale=1.0,
    )
    surface = SimpleNamespace(facets=facets, _offsets=np.asarray((0.0, 0.0)))
    normals, _ = _unique_support_planes(surface)
    assert len(normals) == 2


def test_convex_partition_single_feature_and_open_limits():
    positions = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    active_surface = _box(xmin=-10, xmax=0.05)
    selected, contribution, _ = _evaluate(active_surface, positions)
    assert selected.split_faces == selected.active_feature_lines == 0
    assert contribution.potential_energy == pytest.approx(
        7.0 * (0.05**3 / 6 - 0.05**4 / 24), rel=1e-12,
    )
    open_surface = _box(xmin=2, xmax=3)
    selected, contribution, _ = _evaluate(open_surface, positions)
    assert selected.split_faces == selected.active_feature_lines == 0
    assert contribution.potential_energy == 0.0
    assert selected.trace.point_count == _trace().point_count


def test_convex_partition_rejects_third_plane_tied_along_feature_line():
    # On this slave triangle y=0. The lower-y support plane is exactly as
    # close as both x faces on the compressed x=0.45 switch but dominates no
    # positive area. Silently keeping only the two x normals would be false.
    surface = _box(xmin=0.25, xmax=0.65, ymin=-0.2)
    positions = np.asarray(((0, 0, 0), (1, 0, 0), (0, 0, 1)), dtype=float)
    with pytest.raises(ValueError, match="third support plane ties"):
        partition_convex_p1_trace(_trace(), positions, surface, penalty=7)


def test_real_dolfinx_convex_feature_operator_captures_strip_and_full_tangent():
    basix_ufl = pytest.importorskip("basix.ufl")
    fem = pytest.importorskip("dolfinx.fem")
    mesh = pytest.importorskip("dolfinx.mesh")
    mpi = pytest.importorskip("mpi4py.MPI")
    agent_mesh = pytest.importorskip("agentfem.mesh")
    domain = mesh.create_unit_cube(
        mpi.COMM_SELF, 1, 1, 1, cell_type=mesh.CellType.tetrahedron,
    )
    space = fem.functionspace(
        domain, basix_ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,)),
    )
    displacement = fem.Function(space)
    displacement.x.array.reshape(-1, 3)[:, 0] = space.tabulate_dof_coordinates()[:, 1]
    facets = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], 0))
    tags = mesh.meshtags(domain, 2, facets, np.full(len(facets), 7, dtype=np.int32))
    slave = agent_mesh.tagged_boundary_region(domain, tags, tag=7, name="slave")
    surface = _box(xmin=0.41, xmax=0.49)
    pair = boundary_models.rigid_contact_pair(
        slave, boundary_models.rigid_body(surface), penalty=7.0,
    )
    clipped = dolfinx_implicit_rigid_contact(
        pair, displacement, contact_integration="convex_feature_partition",
    )
    ordinary = dolfinx_implicit_rigid_contact(pair, displacement)
    value = clipped.evaluate()
    assert value.potential_energy == pytest.approx(7.0 * 0.04**3 / 3, rel=1e-11)
    assert ordinary.evaluate().potential_energy == 0.0
    assert abs(value.contact_force_on_structure[0]) < 1e-11
    evidence = clipped.current_point_diagnostics()
    assert evidence["split_slave_faces"] > 0
    assert evidence["active_feature_lines"] > 0
    base = displacement.x.array.copy()
    direction = np.random.default_rng(529).normal(size=base.shape)
    h = 1e-7
    displacement.x.array[:] = base + h * direction
    plus = clipped.evaluate()
    displacement.x.array[:] = base - h * direction
    minus = clipped.evaluate()
    displacement.x.array[:] = base
    np.testing.assert_allclose(
        value.tangent_action(direction.reshape(-1, 3)),
        (plus.nodal_structural_residual - minus.nodal_structural_residual) / (2 * h),
        rtol=2e-7, atol=2e-8,
    )


def test_convex_partition_rejects_projection_from_a_different_feature(monkeypatch):
    from dataclasses import replace

    basix_ufl = pytest.importorskip("basix.ufl")
    fem = pytest.importorskip("dolfinx.fem")
    mesh = pytest.importorskip("dolfinx.mesh")
    mpi = pytest.importorskip("mpi4py.MPI")
    agent_mesh = pytest.importorskip("agentfem.mesh")
    from agentfem.boundary_models.convex_polyhedron import ConvexPolyhedralRigidSurface

    domain = mesh.create_unit_cube(
        mpi.COMM_SELF, 1, 1, 1, cell_type=mesh.CellType.tetrahedron,
    )
    space = fem.functionspace(
        domain, basix_ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,)),
    )
    displacement = fem.Function(space)
    displacement.x.array.reshape(-1, 3)[:, 0] = space.tabulate_dof_coordinates()[:, 1]
    facets = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], 0))
    tags = mesh.meshtags(domain, 2, facets, np.full(len(facets), 7, dtype=np.int32))
    slave = agent_mesh.tagged_boundary_region(domain, tags, tag=7, name="slave")
    surface = _box(xmin=0.41, xmax=0.49)
    pair = boundary_models.rigid_contact_pair(
        slave, boundary_models.rigid_body(surface), penalty=7.0,
    )
    contact = dolfinx_implicit_rigid_contact(
        pair, displacement, contact_integration="convex_feature_partition",
    )
    original = ConvexPolyhedralRigidSurface.project

    def wrong_feature(self, points, **kwargs):
        projection = original(self, points, **kwargs)
        if self is not surface:
            return projection
        normals = projection.normals.copy()
        index = int(np.flatnonzero(projection.signed_gaps < 0)[0])
        normals[index] = (0.0, 1.0, 0.0)
        closest = projection.query_points - projection.signed_gaps[:, None] * normals
        return replace(projection, normals=normals, closest_points=closest)

    monkeypatch.setattr(ConvexPolyhedralRigidSurface, "project", wrong_feature)
    with pytest.raises(ValueError, match="disagrees with the selected support-plane"):
        contact.evaluate()


def _ordinary_convex_step(integration="convex_feature_partition"):
    mesh = pytest.importorskip("dolfinx.mesh")
    mpi = pytest.importorskip("mpi4py.MPI")
    from agentfem import constitutive, fields, mesh as agent_mesh, models, solvers, steps, studies
    from agentfem.step_providers import step_capability

    domain = mesh.create_unit_cube(mpi.COMM_SELF, 1, 1, 1)
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    left = agent_mesh.boundary(domain, lambda x: np.isclose(x[0], 0), name="left")
    right = agent_mesh.boundary(domain, lambda x: np.isclose(x[0], 1), name="right")
    model.clamp(displacement, on=left)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000, poisson=0, yield_stress=100, hardening_modulus=100,
    ))
    motion = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(
            translation=(0, 0, 0), rotation=(0, 0, 0.05),
            reference_point=(1.005, 0.5, 0.5),
        ),
        start_time=0, end_time=1,
    )
    pair = boundary_models.rigid_contact_pair(
        right, boundary_models.rigid_body(
            _box(xmin=0.985, xmax=1.025), motion_schedule=motion,
        ), penalty=200,
    )
    model.add_boundary_model(pair)
    options = {"material": material, "contact_integration": integration}
    assert step_capability(model, target=displacement, options=options)["supported"]
    step = model.step(
        target=displacement, **options, incrementation=steps.fixed(2),
        solver_options=solvers.newton(
            relative_tolerance=1e-8, absolute_tolerance=1e-9,
            maximum_iterations=25, line_search="basic",
        ), progress=False,
    )
    return step


def test_ordinary_j2_step_admits_convex_feature_partition_and_restart(tmp_path):
    continuous = _ordinary_convex_step()
    assert continuous.solve_result().status == "completed"
    assert continuous.contact_provider.summary()["contact_integration"] == (
        "convex_feature_partition"
    )
    assert continuous.contact_provider.evaluate().potential_energy > 0
    assert continuous.contact_provider.current_point_diagnostics()["active_feature_lines"] > 0
    from agentfem import results

    outputs = results.export_accepted_contact_points(continuous, tmp_path / "contact")
    metadata = json.loads(outputs["contact_point_metadata"].read_text())
    assert metadata["pairs"][0]["point_identity_scope"] == "accepted_snapshot_only"
    partial = _ordinary_convex_step()
    partial.solve(until=0.5)
    checkpoint = partial.save_checkpoint(tmp_path / "two_plane_contact")
    resumed = _ordinary_convex_step()
    resumed.load_checkpoint(checkpoint)
    assert resumed.solve_result().status == "completed"
    np.testing.assert_allclose(
        resumed.solution.x.array, continuous.solution.x.array,
        rtol=1e-12, atol=1e-12,
    )
    assert resumed.energy_history[-1].tool_work == pytest.approx(
        continuous.energy_history[-1].tool_work,
    )
    mismatched = _ordinary_convex_step("gauss")
    with pytest.raises(ValueError, match="identity"):
        mismatched.load_checkpoint(checkpoint)
    assert mismatched.accepted_load_factor == 0.0


def test_convex_support_planes_are_prepared_once_per_operator(monkeypatch):
    import agentfem.boundary_models.dolfinx_implicit_facet_contact as backend

    original = backend._unique_support_planes
    calls = []

    def prepare(surface):
        calls.append(surface.geometry_fingerprint)
        return original(surface)

    monkeypatch.setattr(backend, "_unique_support_planes", prepare)
    step = _ordinary_convex_step()
    assert len(calls) == 1
    for _ in range(3):
        assert step.contact_provider.evaluate().active_points > 0
    assert len(calls) == 1


def test_convex_feature_trial_failure_restores_accepted_material_and_work(monkeypatch):
    step = _ordinary_convex_step()
    step.solve(until=0.5)
    accepted_displacement = step.solution.x.array.copy()
    accepted_material = step.response.state.committed_state_vectors().copy()
    accepted_work = step.energy_history[-1].tool_work
    original = step.contact_provider.evaluate

    def fail_once():
        raise ValueError("injected feature projection disagreement")

    monkeypatch.setattr(step.contact_provider, "evaluate", fail_once)
    with pytest.raises(RuntimeError, match="feature projection disagreement"):
        step.solve()
    assert step.accepted_load_factor == pytest.approx(0.5)
    np.testing.assert_array_equal(step.solution.x.array, accepted_displacement)
    np.testing.assert_array_equal(
        step.response.state.committed_state_vectors(), accepted_material,
    )
    assert step.energy_history[-1].tool_work == accepted_work
    monkeypatch.setattr(step.contact_provider, "evaluate", original)
    assert step.solve_result().status == "completed"


def test_total_j2_convex_feature_newton_matrix_matches_fixed_old_state_difference():
    from mpi4py import MPI
    from petsc4py import PETSc

    step = _ordinary_convex_step()
    step.solve()
    assert step.contact_provider.current_point_diagnostics()["active_feature_lines"] > 0
    accepted = step.solution.x.array.copy()
    constrained = np.unique(np.concatenate([
        np.asarray(bc.dof_indices()[0], dtype=np.int64) for bc in step.bcs
    ]))
    direction = np.random.default_rng(409).normal(size=accepted.shape)
    direction[constrained] = 0
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
        rhs_at(0)
        matrix = step._assemble_tangent()
        trial = PETSc.Vec().createWithArray(direction.copy(), comm=MPI.COMM_SELF)
        action = matrix.createVecLeft()
        matrix.mult(trial, action)
        expected = action.array.copy()
        trial.destroy()
        action.destroy()
        h = 1e-6
        numerical = -(rhs_at(h) - rhs_at(-h)) / (2 * h)
        np.testing.assert_allclose(expected, numerical, rtol=2e-4, atol=1e-6)
    finally:
        step.solution.x.array[:] = accepted
        step.solution.x.scatter_forward()
        step._update_response(start_factor=1.0, target_factor=1.0)


def test_moving_convex_feature_tool_force_moment_match_potential_derivative():
    step = _ordinary_convex_step()
    step.solve()
    provider = step.contact_provider
    assert provider.current_point_diagnostics()["active_feature_lines"] > 0
    motion = provider.pair.rigid_body.motion_schedule.motion
    dual = step._contact_dual()
    assert dual.force.shape == (6,)
    h = 1e-6
    try:
        provider.update_motion(1.0 - h)
        before = step._contact_potential()
    finally:
        provider.update_motion(1.0)
    derivative = (step._contact_potential() - before) / h
    assert derivative == pytest.approx(
        float(dual.force @ motion.generalized_coordinate(1.0)), rel=2e-4,
    )
