"""Bounded convex feature-partition checks against an independent oracle."""

from __future__ import annotations

import json
from itertools import permutations
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


@pytest.mark.parametrize(
    "point_ids, shape_values",
    (
        (
            (0, 1, 2),
            ((1 / 3, 1 / 3, 1 / 3),) * 3,
        ),
        (
            (0, 1, 3),
            ((2 / 3, 1 / 6, 1 / 6),
             (1 / 6, 2 / 3, 1 / 6),
             (1 / 6, 1 / 6, 2 / 3)),
        ),
    ),
)
def test_convex_partition_rejects_nondefault_three_point_trace(point_ids, shape_values):
    trace = boundary_models.ContactTrace(
        point_ids=point_ids,
        node_ids=((0, 1, 2),) * 3,
        shape_values=shape_values,
        weights=(1 / 6, 1 / 6, 1 / 6),
    )
    positions = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    with pytest.raises(NotImplementedError, match="default three-point P1"):
        partition_convex_p1_trace(trace, positions, _box(), penalty=7)


def test_convex_partition_accepts_reordered_default_rule():
    trace = boundary_models.ContactTrace(
        point_ids=(3, 4, 5),
        node_ids=((0, 1, 2),) * 3,
        shape_values=((1 / 6, 1 / 6, 2 / 3),
                      (2 / 3, 1 / 6, 1 / 6),
                      (1 / 6, 2 / 3, 1 / 6)),
        weights=(1 / 6, 1 / 6, 1 / 6),
    )
    positions = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    selected = partition_convex_p1_trace(trace, positions, _box(), penalty=7)
    assert selected.trace.point_count >= 3


def _evaluate(surface, positions, *, penalty=7.0, motion=None, factor=0.0):
    selected = partition_convex_p1_trace(
        _trace(), positions, surface, penalty=penalty,
        motion=motion, factor=factor,
    )
    evaluation = selected.trace.evaluate(positions)
    record = boundary_models.ContactProjectionRecord(
        selected.trace.point_ids,
        surface.project(evaluation.query_points, motion=motion, factor=factor),
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


def test_convex_corner_three_features_matches_exact_energy_and_full_tangent():
    surface = boundary_models.convex_polyhedral_rigid_surface(
        vertices=((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)),
        triangles=((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
    )
    positions = np.asarray(((0.05, 0.3, 0.3),
                            (0.3, 0.05, 0.3),
                            (0.3, 0.3, 0.05)))
    selected, contribution, tangent = _evaluate(surface, positions)
    assert selected.split_faces == 1
    assert selected.active_feature_lines == 3
    # On this slave triangle g=-0.3+0.25 max(lambda_0,lambda_1,lambda_2).
    # For a uniform barycentric point, E[max]=11/18 and E[max²]=85/216;
    # the reference triangle has area 1/2.
    assert contribution.potential_energy == pytest.approx(
        7.0 * 1981.0 / 345600.0, rel=1e-12,
    )
    # Symmetry makes each normal carry one third of the integrated penalty
    # traction. This checks force independently of the tangent difference.
    np.testing.assert_allclose(
        contribution.nodal_structural_residual.sum(axis=0),
        7.0 * (0.3 - 0.25 * 11.0 / 18.0) / 6.0,
        rtol=1e-12,
    )
    rng = np.random.default_rng(112)
    for _ in range(5):
        direction = rng.normal(size=positions.shape)
        direction /= np.linalg.norm(direction)
        h = 1e-7
        plus = _evaluate(surface, positions + h * direction)[1]
        minus = _evaluate(surface, positions - h * direction)[1]
        numerical = (
            plus.nodal_structural_residual - minus.nodal_structural_residual
        ) / (2 * h)
        np.testing.assert_allclose(
            tangent @ direction.ravel(), numerical.ravel(), rtol=2e-7, atol=2e-8,
        )


def test_convex_partition_still_rejects_four_dominant_features():
    surface = _box(xmin=0.1, xmax=0.9, ymin=0.1, ymax=0.9)
    positions = np.asarray(((0, 0, 0), (2, 0, 0), (0, 2, 0)), dtype=float)
    with pytest.raises(NotImplementedError, match="more than three dominant"):
        partition_convex_p1_trace(_trace(), positions, surface, penalty=7)


def test_convex_corner_perturbed_three_feature_tangent():
    surface = boundary_models.convex_polyhedral_rigid_surface(
        vertices=((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)),
        triangles=((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
    )
    reference = np.asarray(((0.05, 0.3, 0.3),
                            (0.3, 0.05, 0.3),
                            (0.3, 0.3, 0.05)))
    rng = np.random.default_rng(239)
    for _ in range(5):
        positions = reference + 0.02 * rng.normal(size=reference.shape)
        selected, contribution, tangent = _evaluate(surface, positions)
        assert selected.active_feature_lines == 3
        assert np.isfinite(contribution.potential_energy)
        direction = rng.normal(size=positions.shape)
        direction /= np.linalg.norm(direction)
        h = 1e-7
        plus = _evaluate(surface, positions + h * direction)[1]
        minus = _evaluate(surface, positions - h * direction)[1]
        numerical = (
            plus.nodal_structural_residual - minus.nodal_structural_residual
        ) / (2 * h)
        np.testing.assert_allclose(
            tangent @ direction.ravel(), numerical.ravel(), rtol=2e-7, atol=2e-8,
        )


def test_three_feature_corner_is_rigid_motion_covariant():
    surface = boundary_models.convex_polyhedral_rigid_surface(
        vertices=((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)),
        triangles=((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
    )
    positions = np.asarray(((0.05, 0.3, 0.3),
                            (0.3, 0.05, 0.3),
                            (0.3, 0.3, 0.05)))
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.1, -0.2, 0.05), rotation=(0.05, -0.03, 0.1),
        reference_point=(0, 0, 0),
    )
    pose = motion.state(1.0)
    rotation = np.asarray(pose["rotation_matrix"])
    transformed = positions @ rotation.T + np.asarray(pose["reference_point"])
    original_partition, original, _ = _evaluate(surface, positions)
    moved_partition, moved, _ = _evaluate(
        surface, transformed, motion=motion, factor=1.0,
    )
    assert original_partition.active_feature_lines == 3
    assert moved_partition.active_feature_lines == 3
    assert moved.potential_energy == pytest.approx(original.potential_energy, rel=1e-12)
    np.testing.assert_allclose(
        moved.nodal_structural_residual,
        original.nodal_structural_residual @ rotation.T,
        rtol=1e-12, atol=1e-12,
    )


def test_three_feature_corner_is_slave_vertex_order_invariant():
    surface = boundary_models.convex_polyhedral_rigid_surface(
        vertices=((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)),
        triangles=((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
    )
    positions = np.asarray(((0.05, 0.3, 0.3),
                            (0.3, 0.05, 0.3),
                            (0.3, 0.3, 0.05)))
    _, original, _ = _evaluate(surface, positions)
    for order in permutations(range(3)):
        _, changed, _ = _evaluate(surface, positions[list(order)])
        assert changed.potential_energy == pytest.approx(original.potential_energy)
        np.testing.assert_allclose(
            changed.nodal_structural_residual,
            original.nodal_structural_residual[list(order)],
            atol=1e-12,
        )


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


def test_real_dolfinx_convex_corner_three_feature_operator_tangent():
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
    displacement.x.array.reshape(-1, 3)[:, 0] = 0.2
    facets = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[0], 0))
    tags = mesh.meshtags(domain, 2, facets, np.full(len(facets), 7, dtype=np.int32))
    slave = agent_mesh.tagged_boundary_region(domain, tags, tag=7, name="slave")
    surface = boundary_models.convex_polyhedral_rigid_surface(
        # Offset the corner so no feature switch is coincident with the
        # DOLFINx triangulation's diagonal; that case has no two-sided tangent.
        vertices=((0, -0.04, 0.03), (3, -0.04, 0.03),
                  (0, 2.96, 0.03), (0, -0.04, 3.03)),
        triangles=((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
    )
    pair = boundary_models.rigid_contact_pair(
        slave, boundary_models.rigid_body(surface), penalty=7.0,
    )
    contact = dolfinx_implicit_rigid_contact(
        pair, displacement, contact_integration="convex_feature_partition",
    )
    value = contact.evaluate()
    assert value.potential_energy > 0
    assert contact.current_point_diagnostics()["active_feature_lines"] >= 3
    base = displacement.x.array.copy()
    direction = np.random.default_rng(531).normal(size=base.shape)
    h = 1e-7
    displacement.x.array[:] = base + h * direction
    plus = contact.evaluate()
    displacement.x.array[:] = base - h * direction
    minus = contact.evaluate()
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


def _ordinary_convex_step(
    integration="convex_feature_partition", *, three_feature_corner=False,
):
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
    surface = _box(xmin=0.985, xmax=1.025)
    if three_feature_corner:
        surface = boundary_models.convex_polyhedral_rigid_surface(
            vertices=((0.8, -0.04, 0.03), (3.8, -0.04, 0.03),
                      (0.8, 2.96, 0.03), (0.8, -0.04, 3.03)),
            triangles=((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
        )
    pair = boundary_models.rigid_contact_pair(
        right, boundary_models.rigid_body(surface, motion_schedule=motion),
        penalty=200,
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
    result = continuous.solve_result()
    assert result.status == "completed"
    assert next(
        item for item in result.verify("release").claims
        if item.name == "polyhedral_contact_integral_certified"
    ).status == "passed"
    assert continuous.contact_provider.summary()["contact_integration"] == (
        "convex_feature_partition"
    )
    assert continuous.contact_provider.evaluate().potential_energy > 0
    diagnostics = continuous.contact_provider.current_point_diagnostics()
    assert diagnostics["active_feature_lines"] > 0
    assert diagnostics["sampled_mixed_feature_slave_faces"] > 0
    assert diagnostics["feature_transition_integral_certified"] is True
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
    assert mismatched.contact_provider.current_point_diagnostics()[
        "feature_transition_integral_certified"
    ] is False
    with pytest.raises(ValueError, match="identity"):
        mismatched.load_checkpoint(checkpoint)
    assert mismatched.accepted_load_factor == 0.0


def test_ordinary_j2_step_accepts_three_feature_corner_and_checkpoint(tmp_path):
    continuous = _ordinary_convex_step(three_feature_corner=True)
    assert continuous.solve_result().status == "completed"
    diagnostics = continuous.contact_provider.current_point_diagnostics()
    assert diagnostics["active_feature_lines"] >= 3
    assert continuous.contact_provider.evaluate().potential_energy > 0
    partial = _ordinary_convex_step(three_feature_corner=True)
    partial.solve(until=0.5)
    checkpoint = partial.save_checkpoint(tmp_path / "three_feature_corner")
    resumed = _ordinary_convex_step(three_feature_corner=True)
    resumed.load_checkpoint(checkpoint)
    assert resumed.solve_result().status == "completed"
    np.testing.assert_allclose(
        resumed.solution.x.array, continuous.solution.x.array,
        rtol=1e-12, atol=1e-12,
    )
    assert resumed.energy_history[-1].tool_work == pytest.approx(
        continuous.energy_history[-1].tool_work,
    )


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


@pytest.mark.parametrize("three_feature_corner", (False, True))
def test_convex_feature_trial_failure_restores_accepted_material_and_work(
    monkeypatch, three_feature_corner,
):
    step = _ordinary_convex_step(three_feature_corner=three_feature_corner)
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


@pytest.mark.parametrize("three_feature_corner", (False, True))
def test_total_j2_convex_feature_newton_matrix_matches_fixed_old_state_difference(
    three_feature_corner,
):
    from mpi4py import MPI
    from petsc4py import PETSc

    step = _ordinary_convex_step(three_feature_corner=three_feature_corner)
    step.solve()
    minimum_lines = 3 if three_feature_corner else 1
    assert (step.contact_provider.current_point_diagnostics()["active_feature_lines"]
            >= minimum_lines)
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


@pytest.mark.parametrize("three_feature_corner", (False, True))
def test_moving_convex_feature_tool_force_moment_match_potential_derivative(
    three_feature_corner,
):
    step = _ordinary_convex_step(three_feature_corner=three_feature_corner)
    step.solve()
    provider = step.contact_provider
    minimum_lines = 3 if three_feature_corner else 1
    assert provider.current_point_diagnostics()["active_feature_lines"] >= minimum_lines
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
