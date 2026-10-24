from __future__ import annotations

import basix.ufl
from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np
import pytest

from agentfem import boundary_models
from agentfem import mesh as agent_mesh
from agentfem.boundary_models.dolfinx_implicit_facet_contact import (
    dolfinx_implicit_facet_contact,
    dolfinx_implicit_rigid_contact,
)


def _model(*, cell_type=mesh.CellType.tetrahedron):
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1, cell_type=cell_type)
    space = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,)))
    displacement = fem.Function(space, name="Displacement")
    facets = mesh.locate_entities_boundary(
        domain, 2, lambda x: np.isclose(x[0], 0.0)
    )
    tags = mesh.meshtags(domain, 2, facets, np.full(len(facets), 7, dtype=np.int32))
    slave = agent_mesh.tagged_boundary_region(domain, tags, tag=7, name="slave")
    tool = boundary_models.triangulated_rigid_surface(
        vertices=((0.05, 0, 0), (0.05, 0, 1), (0.05, 1, 1), (0.05, 1, 0)),
        triangles=((0, 1, 2), (0, 2, 3)),
    )
    pair = boundary_models.rigid_contact_pair(
        slave,
        boundary_models.rigid_body(tool, reference_point=(0.05, 0, 0)),
        penalty=200,
    )
    operator = dolfinx_implicit_facet_contact(pair, displacement)
    return displacement, operator


@pytest.mark.parametrize("cell_type", [mesh.CellType.tetrahedron, mesh.CellType.hexahedron])
@pytest.mark.parametrize("cells", [1, 2])
@pytest.mark.parametrize("quadrature_degree", [5, 11])
def test_quadratic_contact_trace_matches_polynomial_force_moment_and_work(
    cell_type, cells, quadrature_degree,
):
    """Independent nonuniform trace oracle, not a global spatial-convergence proof.

    On the reference square x=0, g=-.01*(1+y^2+z^2), k=200, so
    p=2*(1+y^2+z^2). Reference-area integration gives F_x=10/3,
    M_y=-M_z=11/6. For virtual u_x=yz, R.v=-1 and v.K.v=200/9.
    These polynomial integrals do not use the contact assembly as an oracle.
    """
    domain = mesh.create_unit_cube(MPI.COMM_SELF, cells, cells, cells, cell_type=cell_type)
    space = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 2, shape=(3,)))
    u = fem.Function(space)
    left = agent_mesh.boundary(domain, lambda x: np.isclose(x[0], 0.), name="polynomial_patch")
    pair = boundary_models.rigid_contact_pair(
        left, boundary_models.rigid_body(boundary_models.rigid_plane(
            point=(-.02, 0., 0.), normal=(1., 0., 0.))),
        penalty=200., initial_penetration_limit=0.,
    )
    operator = dolfinx_implicit_rigid_contact(
        pair, u, contact_quadrature_degree=quadrature_degree,
    )
    reference = space.tabulate_dof_coordinates()
    nodal = u.x.array.reshape(-1, 3)
    nodal[:, 0] = -.03 - .01 * (reference[:, 1]**2 + reference[:, 2]**2)
    contribution = operator.evaluate()
    np.testing.assert_allclose(contribution.contact_force_on_structure, (10./3., 0., 0.), atol=2e-12)
    assert contribution.potential_energy == pytest.approx(
        .01 * (1. + 4./3. + 2./5. + 2./9.), rel=2e-13, abs=2e-14,
    )
    nodal_force = -contribution.nodal_structural_residual
    np.testing.assert_allclose(nodal_force.sum(axis=0), (10./3., 0., 0.), atol=2e-12)
    np.testing.assert_allclose(np.cross(reference + nodal, nodal_force).sum(axis=0),
                               (0., 11./6., -11./6.), atol=2e-12)
    direction = np.zeros_like(nodal)
    direction[:, 0] = reference[:, 1] * reference[:, 2]
    assert np.sum(contribution.nodal_structural_residual * direction) == pytest.approx(-1., abs=2e-12)
    assert np.sum(contribution.tangent_action(direction) * direction) == pytest.approx(200./9., abs=2e-11)
    operator.commit_accepted()
    samples = operator.accepted_point_samples()
    points = samples["reference_positions"]
    np.testing.assert_allclose(points[:, 0], 0., atol=1e-14)
    np.testing.assert_allclose(samples["normal_pressures"],
                               2. * (1. + points[:, 1]**2 + points[:, 2]**2), atol=2e-12)


def test_dolfinx_implicit_facet_tangent_matches_current_residual_difference():
    displacement, operator = _model()
    base = displacement.x.array.copy()
    base.reshape((-1, 3))[:, 0] = 0.1
    displacement.x.array[:] = base
    contribution = operator.evaluate()
    assert contribution.active_points > 0
    assert contribution.assembly.potential_energy > 0
    assert operator.summary()["execution_scope"] == "serial_frictionless_reference_area"
    rng = np.random.default_rng(231)
    direction = rng.normal(size=base.shape).reshape((-1, 3))
    step = 1e-7
    displacement.x.array[:] = base + step * direction.reshape(-1)
    plus = operator.evaluate()
    displacement.x.array[:] = base - step * direction.reshape(-1)
    minus = operator.evaluate()
    displacement.x.array[:] = base
    difference = (
        plus.assembly.nodal_structural_residual
        - minus.assembly.nodal_structural_residual
    ) / (2 * step)
    np.testing.assert_allclose(contribution.tangent_action(direction), difference, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        np.sum(contribution.assembly.nodal_structural_residual, axis=0),
        -contribution.assembly.contact_force_on_structure,
        atol=1e-12,
    )


def test_implicit_facet_tracks_prescribed_pose_and_work_conjugate_dual():
    displacement, operator = _model()
    motion = boundary_models.prescribed_rigid_motion(translation=(0.01, 0, 0))
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        motion, start_time=0, end_time=1,
    )
    pair = boundary_models.rigid_contact_pair(
        operator.pair.slave_boundary,
        boundary_models.rigid_body(operator.surface, motion_schedule=schedule),
        penalty=200,
    )
    moving = dolfinx_implicit_facet_contact(pair, displacement)
    assert moving.has_prescribed_motion
    moving.update_motion(1.0)
    displacement.x.array.reshape((-1, 3))[:, 0] = 0.1
    contribution = moving.evaluate()
    assert contribution.active_points > 0
    assert np.all(np.isfinite(contribution.assembly.nodal_structural_residual))
    np.testing.assert_allclose(
        moving.projector.project(moving.adapter.evaluate(displacement).query_points,
                                 motion=motion, factor=1.0).signed_gaps,
        0.06 - moving.adapter.evaluate(displacement).query_points[:, 0],
    )
    moving.update_motion(0.0)
    assert float(moving.motion_factor.value) == 0.0


def test_implicit_facet_rejects_independent_physical_time_schedule():
    displacement, operator = _model()
    motion = boundary_models.prescribed_rigid_motion(translation=(0.01, 0, 0))
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        motion, start_time=0, end_time=2,
    )
    pair = boundary_models.rigid_contact_pair(
        operator.pair.slave_boundary,
        boundary_models.rigid_body(operator.surface, motion_schedule=schedule),
        penalty=200,
    )
    with pytest.raises(NotImplementedError, match="AFM-IMPLICIT-CONTACT-MOTION"):
        dolfinx_implicit_facet_contact(pair, displacement)


def test_implicit_facet_reports_and_bounds_initial_reference_penetration():
    displacement, original = _model()
    assert original.current_point_diagnostics()["initial_contact"]["policy"] == "report"
    assert original.current_point_diagnostics()["initial_contact"][
        "maximum_penetration"
    ] == 0.0
    tool = boundary_models.rigid_plane(
        point=(-0.05, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
    )

    def make_pair(limit):
        return boundary_models.rigid_contact_pair(
            original.pair.slave_boundary,
            boundary_models.rigid_body(tool),
            penalty=200,
            initial_penetration_limit=limit,
        )

    with pytest.raises(ValueError, match="AFM-CONTACT-INITIAL-OVERLAP"):
        dolfinx_implicit_rigid_contact(make_pair(0.01), displacement)
    accepted = dolfinx_implicit_rigid_contact(make_pair(0.06), displacement)
    evidence = accepted.current_point_diagnostics()["initial_contact"]
    assert evidence["maximum_penetration"] == pytest.approx(0.05)
    assert evidence["penetrating_point_count"] == accepted.adapter.trace.point_count
    assert evidence["admission_limit"] == 0.06
    assert evidence["policy"] == "reject_certified_above_limit"
    assert evidence["face_wise_bound_certified"] is True
    assert evidence["maximum_penetration_bound"] == pytest.approx(0.05)
    np.testing.assert_allclose(displacement.x.array, 0.0)
    assert make_pair(0.01).scientific_identity != make_pair(0.06).scientific_identity
    assert "initial_penetration_limit" not in original.pair.summary()
    assert make_pair(0.06).summary()["initial_penetration_limit"] == 0.06


def test_faceted_initial_overlap_limit_does_not_pretend_samples_bound_a_face():
    displacement, operator = _model()
    pair = boundary_models.rigid_contact_pair(
        operator.pair.slave_boundary, operator.pair.rigid_body,
        penalty=200, initial_penetration_limit=0.01,
    )
    with pytest.raises(NotImplementedError, match="AFM-CONTACT-INITIAL-BOUND"):
        dolfinx_implicit_facet_contact(pair, displacement)


@pytest.mark.parametrize("cell_type", [mesh.CellType.tetrahedron, mesh.CellType.hexahedron])
def test_quadratic_field_initial_plane_bound_uses_face_geometry_not_cell_dofs(cell_type):
    original, operator = _model(cell_type=cell_type)
    domain = original.function_space.mesh
    space = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 2, shape=(3,)))
    u = fem.Function(space)
    left = operator.pair.slave_boundary

    def create(point, normal):
        pair = boundary_models.rigid_contact_pair(
            left, boundary_models.rigid_body(boundary_models.rigid_plane(point=point, normal=normal)),
            penalty=200., initial_penetration_limit=0.)
        return dolfinx_implicit_rigid_contact(pair, u)

    # Interior cell DOFs are behind this plane but the entire selected left
    # boundary is separated. Taking all high-order trace cell nodes is wrong.
    provider = create((.05, 0., 0.), (-1., 0., 0.))
    evidence = provider.current_point_diagnostics()["initial_contact"]
    assert evidence["bound_method"] == "exact_plane_geometry_vertices"
    assert evidence["maximum_penetration_bound"] == 0.
    np.testing.assert_array_equal(provider.adapter.reference_facet_vertices[:, 0], 0.)
    assert not provider.adapter.reference_facet_vertices.flags.writeable
    from dataclasses import replace
    vertices = provider.adapter.reference_facet_vertices.copy()
    adapter = replace(provider.adapter, reference_facet_vertices=vertices)
    vertices[:] = 100.
    np.testing.assert_array_equal(adapter.reference_facet_vertices[:, 0], 0.)
    for invalid in (np.empty((0, 3)), np.ones((1, 3), dtype=complex),
                    np.full((1, 3), np.nan), np.zeros((3,))):
        with pytest.raises(ValueError, match="Reference facet vertices"):
            replace(provider.adapter, reference_facet_vertices=invalid)
    # Exact touching is admitted, while representable tiny reference overlap
    # is rejected without replacing the user's zero limit by a tolerance.
    assert create((0., 0., 0.), (1., 0., 0.))._initial_contact_diagnostics["maximum_penetration_bound"] == 0.
    with pytest.raises(ValueError, match="AFM-CONTACT-INITIAL-OVERLAP"):
        create((2**-45, 0., 0.), (1., 0., 0.))


@pytest.mark.parametrize("cell_type", [
    mesh.CellType.tetrahedron, mesh.CellType.hexahedron,
])
def test_analytical_plane_initial_bound_catches_unsampled_corner_penetration(cell_type):
    displacement, operator = _model(cell_type=cell_type)
    normal = np.asarray((-1.0, -0.1, 0.0), dtype=float)
    normal /= np.linalg.norm(normal)
    plane = boundary_models.rigid_plane(point=(0.09, 0.0, 0.0), normal=normal)

    def pair(limit):
        return boundary_models.rigid_contact_pair(
            operator.pair.slave_boundary,
            boundary_models.rigid_body(plane),
            penalty=200, initial_penetration_limit=limit,
        )

    accepted = dolfinx_implicit_rigid_contact(pair(0.02), displacement)
    evidence = accepted.current_point_diagnostics()["initial_contact"]
    assert evidence["maximum_penetration"] == 0.0
    assert evidence["maximum_penetration_bound"] == pytest.approx(
        0.01 / np.sqrt(1.01),
    )
    with pytest.raises(ValueError, match="AFM-CONTACT-INITIAL-OVERLAP"):
        dolfinx_implicit_rigid_contact(pair(0.001), displacement)


@pytest.mark.parametrize("limit", [-1.0, float("nan"), float("inf")])
def test_contact_pair_rejects_invalid_initial_penetration_limit(limit):
    _, operator = _model()
    with pytest.raises(ValueError, match="Initial contact penetration limit"):
        boundary_models.rigid_contact_pair(
            operator.pair.slave_boundary, operator.pair.rigid_body,
            penalty=200, initial_penetration_limit=limit,
        )


def test_closed_convex_sharp_tool_lowers_to_current_trial_newton_operator():
    displacement, original = _model()
    xmin, xmax, ymin, ymax, zmin, zmax = -1.0, 0.05, -1.0, 2.0, -1.0, 2.0
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
    tool = boundary_models.convex_polyhedral_rigid_surface(
        vertices=vertices, triangles=triangles,
    )
    pair = boundary_models.rigid_contact_pair(
        original.pair.slave_boundary,
        boundary_models.rigid_body(tool), penalty=200,
    )
    operator = dolfinx_implicit_rigid_contact(pair, displacement)
    contribution = operator.evaluate()
    assert contribution.active_points > 0
    assert operator.summary()["linearization"] == (
        "closed_convex_polyhedron_piecewise_affine_semismooth"
    )
    base = displacement.x.array.copy()
    direction = np.random.default_rng(22).normal(size=base.shape)
    step = 1e-7
    displacement.x.array[:] = base + step * direction
    plus = operator.evaluate()
    displacement.x.array[:] = base - step * direction
    minus = operator.evaluate()
    displacement.x.array[:] = base
    np.testing.assert_allclose(
        contribution.tangent_action(direction.reshape((-1, 3))),
        (plus.nodal_structural_residual - minus.nodal_structural_residual) / (2 * step),
        rtol=1e-7, atol=1e-8,
    )
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.01, 0, 0), rotation=(0, 0, 0.01),
        reference_point=(0.025, 0.5, 0.5),
    )
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        motion, start_time=0, end_time=1,
    )
    moving_pair = boundary_models.rigid_contact_pair(
        original.pair.slave_boundary,
        boundary_models.rigid_body(tool, motion_schedule=schedule), penalty=200,
    )
    moving = dolfinx_implicit_rigid_contact(moving_pair, displacement)
    moving.update_motion(1.0)
    moved = moving.evaluate()
    assert moved.active_points > 0
    assert moved.surface_generalized_moment is not None
    displacement.x.array[:] = base + step * direction
    plus = moving.evaluate()
    displacement.x.array[:] = base - step * direction
    minus = moving.evaluate()
    displacement.x.array[:] = base
    np.testing.assert_allclose(
        moved.tangent_action(direction.reshape((-1, 3))),
        (plus.nodal_structural_residual - minus.nodal_structural_residual) / (2 * step),
        rtol=1e-7, atol=1e-8,
    )
