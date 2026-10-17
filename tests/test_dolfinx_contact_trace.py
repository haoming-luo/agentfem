from __future__ import annotations

import basix.ufl
from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np
import pytest
import ufl

from agentfem import boundary_models
from agentfem import mesh as agent_mesh


@pytest.mark.parametrize('topology', ['triangle', 'quadrilateral'])
@pytest.mark.parametrize('integration', ['gauss', 'nodal_lumped'])
def test_batched_linear_trace_matches_scalar_rule_and_keeps_coincident_owners(topology, integration):
    from types import SimpleNamespace
    from agentfem.boundary_models.dolfinx_contact_trace import (
        _linear_trace_arrays, _TRIANGLE_DEGREE_TWO, _QUADRILATERAL_DEGREE_THREE,
        _quadrilateral_shape_values, _quadrilateral_shape_derivatives,
    )
    width = 3 if topology == 'triangle' else 4
    points = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [1.,1.,.2]])[:width]
    # More than one internal chunk; coincident bodies have separate node IDs.
    count = 4101
    coordinates = np.tile(points, (count, 1))
    blocks = np.arange(count*width).reshape(count, width)
    domain = SimpleNamespace(topology=SimpleNamespace(dim=3,
        connectivity=lambda *_: SimpleNamespace(links=lambda facet: [facet])))
    space = SimpleNamespace(dofmap=SimpleNamespace(list=blocks))
    ids, nodes, shapes, weights = _linear_trace_arrays(domain, space,
        np.arange(count), blocks, np.arange(count), count, topology=topology,
        coordinates=coordinates, dof_coordinates=coordinates,
        global_vertices=np.arange(count*width), tolerance=1.e-12, integration=integration)
    if width == 3:
        basis = _TRIANGLE_DEGREE_TWO
        reference = np.full(3, np.linalg.norm(np.cross(points[1]-points[0], points[2]-points[0]))/6.)
    else:
        basis = np.array([_quadrilateral_shape_values(r,s) for r,s in _QUADRILATERAL_DEGREE_THREE])
        reference = []
        for r,s in _QUADRILATERAL_DEGREE_THREE:
            dr,ds = _quadrilateral_shape_derivatives(r,s)
            reference.append(.25*np.linalg.norm(np.cross(dr@points, ds@points)))
        reference = np.array(reference)
    if integration == 'nodal_lumped':
        reference = basis.T@reference
        basis = np.eye(width)
    np.testing.assert_array_equal(nodes, np.repeat(blocks,width,axis=0))
    np.testing.assert_array_equal(shapes,np.tile(basis,(count,1)))
    np.testing.assert_allclose(weights,np.tile(reference,count),rtol=8*np.finfo(float).eps,atol=0.)
    offset = count*width if integration == 'nodal_lumped' else 0
    np.testing.assert_array_equal(ids,np.arange(count*width)+offset)
    # A duplicate within the *same* owning cell must still fail.
    coordinates[1] = coordinates[0]
    with pytest.raises(ValueError, match='exactly one'):
        _linear_trace_arrays(domain, space, np.arange(count), blocks, np.arange(count), count,
            topology=topology, coordinates=coordinates, dof_coordinates=coordinates,
            global_vertices=np.arange(count*width), tolerance=1.e-12, integration=integration)


def _cube(comm, *, cells_x: int = 1):
    return mesh.create_unit_cube(
        comm,
        cells_x,
        1,
        1,
        cell_type=mesh.CellType.tetrahedron,
    )


def _hex_cube(comm, *, cells_x: int = 1):
    return mesh.create_unit_cube(
        comm,
        cells_x,
        1,
        1,
        cell_type=mesh.CellType.hexahedron,
    )


def _quadratic_tetrahedron(comm):
    geometry = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.tetrahedron,
        2,
        lagrange_variant=basix.LagrangeVariant.gll_warped,
    )
    coordinates = np.asarray(geometry.points, dtype=float)
    coordinate_element = ufl.Mesh(
        basix.ufl.element(
            "Lagrange",
            "tetrahedron",
            2,
            shape=(3,),
        )
    )
    return agent_mesh.from_arrays(
        cells=[list(range(coordinates.shape[0]))],
        coordinates=coordinates,
        coordinate_element=coordinate_element,
        comm=comm,
    )


def _quadratic_hexahedron(comm):
    geometry = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.hexahedron,
        2,
        lagrange_variant=basix.LagrangeVariant.gll_warped,
    )
    coordinates = np.asarray(geometry.points, dtype=float)
    coordinate_element = ufl.Mesh(
        basix.ufl.element(
            "Lagrange",
            "hexahedron",
            2,
            shape=(3,),
        )
    )
    return agent_mesh.from_arrays(
        cells=[list(range(coordinates.shape[0]))],
        coordinates=coordinates,
        coordinate_element=coordinate_element,
        comm=comm,
    )


def _left_region(domain):
    facet_dimension = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain,
        facet_dimension,
        lambda x: np.isclose(x[0], 0.0),
    )
    tags = mesh.meshtags(
        domain,
        facet_dimension,
        facets,
        np.full(facets.size, 17, dtype=np.int32),
    )
    return agent_mesh.tagged_boundary_region(
        domain,
        tags,
        tag=17,
        name="left_slave",
    )


def _vector_space(domain, degree: int = 1):
    return fem.functionspace(
        domain,
        basix.ufl.element(
            "Lagrange",
            domain.basix_cell(),
            degree,
            shape=(3,),
        ),
    )


def test_dolfinx_contact_trace_builds_reference_triangle_quadrature():
    domain = _cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)

    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        function_space,
    )
    displacement = fem.Function(function_space)
    evaluation = adapter.evaluate(displacement)

    assert adapter.trace.point_count == 6
    assert np.unique(adapter.trace.point_ids).size == 6
    np.testing.assert_allclose(np.sum(adapter.trace.weights), 1.0)
    np.testing.assert_allclose(evaluation.query_points[:, 0], 0.0, atol=1.0e-14)
    assert adapter.summary()["function_space"] == "continuous_blocked_vector_cg1"
    assert adapter.summary()["linearization"] == "not_provided"


def test_dolfinx_contact_trace_drives_projection_response_and_assembly():
    domain = _cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        function_space,
    )
    displacement = fem.Function(function_space)
    displacement.x.array.reshape((-1, 3))[:, 0] = 0.1
    evaluation = adapter.evaluate(displacement)
    surface = boundary_models.rigid_plane(
        point=(0.05, 0.0, 0.0),
        normal=(-1.0, 0.0, 0.0),
    )
    projection = surface.project(evaluation.query_points)
    record = boundary_models.ContactProjectionRecord(
        adapter.trace.point_ids,
        projection,
    )
    response = boundary_models.frictionless_penalty_contact_law(200.0).evaluate(
        record.projection
    )

    assembly = evaluation.assemble(
        record,
        response,
        surface_reference_point=(0.0, 0.0, 0.0),
    )

    np.testing.assert_allclose(evaluation.query_points[:, 0], 0.1)
    np.testing.assert_allclose(response.penetration, 0.05)
    np.testing.assert_allclose(
        assembly.contact_force_on_structure,
        (-10.0, 0.0, 0.0),
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        assembly.surface_generalized_moment,
        (0.0, -5.0, 5.0),
        atol=1.0e-13,
    )
    np.testing.assert_allclose(assembly.potential_energy, 0.25, atol=1.0e-14)


def test_hexahedral_contact_trace_builds_bilinear_quadrature_and_assembly():
    domain = _hex_cube(MPI.COMM_SELF)
    function_space = _vector_space(domain)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        function_space,
    )
    displacement = fem.Function(function_space)
    displacement.x.array.reshape((-1, 3))[:, 0] = 0.1
    evaluation = adapter.evaluate(displacement)
    surface = boundary_models.rigid_plane(
        point=(0.05, 0.0, 0.0),
        normal=(-1.0, 0.0, 0.0),
    )
    projection = surface.project(evaluation.query_points)
    record = boundary_models.ContactProjectionRecord(
        adapter.trace.point_ids,
        projection,
    )
    response = boundary_models.frictionless_penalty_contact_law(200.0).evaluate(
        record.projection
    )
    assembly = evaluation.assemble(
        record,
        response,
        surface_reference_point=(0.0, 0.0, 0.0),
    )

    assert adapter.trace.point_count == 4
    assert adapter.trace.nodes_per_point == 4
    assert adapter.summary()["facet_topology"] == "quadrilateral"
    assert (
        adapter.summary()["quadrature_rule"]
        == "quadrilateral_gauss_degree_three_four_point"
    )
    np.testing.assert_allclose(np.sum(adapter.trace.weights), 1.0)
    np.testing.assert_allclose(evaluation.query_points[:, 0], 0.1)
    np.testing.assert_allclose(
        assembly.contact_force_on_structure,
        (-10.0, 0.0, 0.0),
        atol=1.0e-13,
    )
    np.testing.assert_allclose(assembly.potential_energy, 0.25, atol=1.0e-14)


@pytest.mark.parametrize("degree, point_count", [(5, 9), (7, 16)])
def test_hex_q1_contact_trace_refined_gauss_keeps_reference_force(
    degree, point_count,
):
    domain = _hex_cube(MPI.COMM_SELF)
    space = _vector_space(domain)
    region = _left_region(domain)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        region, space, contact_quadrature_degree=degree,
    )
    assert adapter.trace.point_count == point_count
    assert adapter.trace.nodes_per_point == 4
    assert f"degree_{degree}_{point_count}_point" in adapter.quadrature_rule
    np.testing.assert_allclose(np.sum(adapter.trace.weights), 1.0)
    assert np.unique(adapter.trace.point_ids).size == point_count
    displacement = fem.Function(space)
    displacement.x.array.reshape((-1, 3))[:, 0] = 0.1
    evaluation = adapter.evaluate(displacement)
    surface = boundary_models.rigid_plane(
        point=(0.05, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
    )
    record = boundary_models.ContactProjectionRecord(
        adapter.trace.point_ids, surface.project(evaluation.query_points),
    )
    response = boundary_models.frictionless_penalty_contact_law(200.0).evaluate(
        record.projection,
    )
    assembly = evaluation.assemble(record, response)
    np.testing.assert_allclose(
        assembly.contact_force_on_structure, (-10.0, 0.0, 0.0), atol=1.0e-12,
    )

    with pytest.raises(NotImplementedError, match="Gaussian CG1/Q1"):
        boundary_models.dolfinx_boundary_region_contact_trace(
            region, space, integration="nodal_lumped",
            contact_quadrature_degree=degree,
        )


def test_warped_hexahedral_face_uses_bilinear_map_and_pointwise_jacobian():
    domain = _hex_cube(MPI.COMM_SELF)
    region = _left_region(domain)
    moved = np.flatnonzero(domain.geometry.input_global_indices == 6)
    assert moved.size == 1
    domain.geometry.x[int(moved[0]), 0] = 0.2
    function_space = _vector_space(domain)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        region,
        function_space,
    )
    evaluation = adapter.evaluate(fem.Function(function_space))
    abscissae = np.asarray(
        (0.5 - 1.0 / (2.0 * np.sqrt(3.0)), 0.5 + 1.0 / (2.0 * np.sqrt(3.0)))
    )
    expected_x = np.asarray([0.2 * r * s for s in abscissae for r in abscissae])
    expected_weights = np.asarray(
        [
            0.25 * np.sqrt(1.0 + 0.2**2 * (r**2 + s**2))
            for s in abscissae
            for r in abscissae
        ]
    )

    np.testing.assert_allclose(
        np.sort(evaluation.query_points[:, 0]),
        np.sort(expected_x),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.sort(adapter.trace.weights),
        np.sort(expected_weights),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_warped_hex_contact_area_converges_under_explicit_gauss_refinement():
    domain = _hex_cube(MPI.COMM_SELF)
    region = _left_region(domain)
    moved = np.flatnonzero(domain.geometry.input_global_indices == 6)
    assert moved.size == 1
    domain.geometry.x[int(moved[0]), 0] = 0.2
    space = _vector_space(domain)
    points, weights = np.polynomial.legendre.leggauss(32)
    points = 0.5 * (points + 1.0)
    weights = 0.5 * weights
    exact = sum(
        wr * ws * np.sqrt(1.0 + 0.2**2 * (r**2 + s**2))
        for r, wr in zip(points, weights, strict=True)
        for s, ws in zip(points, weights, strict=True)
    )
    errors = []
    for degree in (None, 5, 7):
        adapter = boundary_models.dolfinx_boundary_region_contact_trace(
            region, space, contact_quadrature_degree=degree,
        )
        errors.append(abs(np.sum(adapter.trace.weights) - exact))
    assert errors[2] < errors[1] < errors[0]


def test_quadratic_displacement_trace_on_linear_tetrahedron():
    domain = _cube(MPI.COMM_SELF)
    region = _left_region(domain)
    quadratic = _vector_space(domain, degree=2)

    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        region,
        quadratic,
    )
    displacement = fem.Function(quadratic)
    displacement.x.array.reshape((-1, 3))[:, 0] = 0.1
    evaluation = adapter.evaluate(displacement)

    assert adapter.trace.point_count == 12
    assert adapter.trace.nodes_per_point == 10
    assert adapter.summary()["interpolation_degree"] == 2
    assert adapter.summary()["geometry_degree"] == 1
    assert adapter.summary()["function_space"] == "continuous_blocked_vector_cg2"
    np.testing.assert_allclose(np.sum(adapter.trace.weights), 1.0)
    np.testing.assert_allclose(evaluation.query_points[:, 0], 0.1, atol=1.0e-14)


def test_quadratic_curved_tetrahedral_trace_uses_coordinate_jacobian():
    domain = _quadratic_tetrahedron(MPI.COMM_SELF)
    region = _left_region(domain)
    candidate = np.flatnonzero(
        np.isclose(domain.geometry.x[:, 0], 0.0)
        & np.isclose(domain.geometry.x[:, 1], 0.5)
        & np.isclose(domain.geometry.x[:, 2], 0.0)
    )
    assert candidate.size == 1
    domain.geometry.x[int(candidate[0]), 0] = 0.15
    quadratic = _vector_space(domain, degree=2)

    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        region,
        quadratic,
    )
    evaluation = adapter.evaluate(fem.Function(quadratic))

    assert adapter.trace.point_count == 6
    assert adapter.summary()["interpolation_degree"] == 2
    assert adapter.summary()["geometry_degree"] == 2
    assert np.sum(adapter.trace.weights) > 0.5
    assert np.max(evaluation.query_points[:, 0]) > 0.05


def test_quadratic_curved_hexahedral_trace_uses_coordinate_jacobian():
    domain = _quadratic_hexahedron(MPI.COMM_SELF)
    region = _left_region(domain)
    candidate = np.flatnonzero(
        np.isclose(domain.geometry.x[:, 0], 0.0)
        & np.isclose(domain.geometry.x[:, 1], 0.5)
        & np.isclose(domain.geometry.x[:, 2], 0.5)
    )
    assert candidate.size == 1
    domain.geometry.x[int(candidate[0]), 0] = 0.15
    quadratic = _vector_space(domain, degree=2)

    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        region,
        quadratic,
    )
    evaluation = adapter.evaluate(fem.Function(quadratic))

    assert adapter.trace.point_count == 9
    assert adapter.summary()["facet_topology"] == "quadrilateral"
    assert adapter.summary()["interpolation_degree"] == 2
    assert adapter.summary()["geometry_degree"] == 2
    assert np.sum(adapter.trace.weights) > 1.0
    assert np.max(evaluation.query_points[:, 0]) > 0.05


def test_dolfinx_contact_trace_rejects_foreign_field():
    domain = _cube(MPI.COMM_SELF)
    region = _left_region(domain)

    first = _vector_space(domain)
    second = _vector_space(domain)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(region, first)
    with pytest.raises(ValueError, match="function-space instance"):
        adapter.evaluate(fem.Function(second))


def test_dolfinx_contact_trace_has_global_identity_and_empty_shards_under_mpi():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("DOLFINx contact trace is reviewed on two ranks.")
    comm = MPI.COMM_WORLD
    domain = _cube(comm, cells_x=2)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        _vector_space(domain),
    )
    local_ids = tuple(int(value) for value in adapter.trace.point_ids)
    gathered = tuple(comm.allgather(local_ids))
    flattened = tuple(value for values in gathered for value in values)
    global_measure = comm.allreduce(float(np.sum(adapter.trace.weights)), op=MPI.SUM)

    assert len(flattened) == 6
    assert len(set(flattened)) == 6
    np.testing.assert_allclose(global_measure, 1.0)
    evaluation = adapter.evaluate(fem.Function(adapter.function_space))
    assert evaluation.query_points.shape == (adapter.trace.point_count, 3)


@pytest.mark.parametrize(
    "contact_quadrature_degree, points_per_face",
    [(None, 4), (5, 9), (7, 16)],
)
def test_hexahedral_contact_trace_identity_is_partition_independent_under_mpi(
    contact_quadrature_degree, points_per_face,
):
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("Hexahedral DOLFINx contact trace is reviewed on two ranks.")
    comm = MPI.COMM_WORLD
    domain = _hex_cube(comm, cells_x=2)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        _vector_space(domain),
        contact_quadrature_degree=contact_quadrature_degree,
    )
    local_ids = tuple(int(value) for value in adapter.trace.point_ids)
    gathered = tuple(comm.allgather(local_ids))
    flattened = tuple(value for values in gathered for value in values)
    global_measure = comm.allreduce(float(np.sum(adapter.trace.weights)), op=MPI.SUM)

    assert len(flattened) == points_per_face
    assert len(set(flattened)) == points_per_face
    np.testing.assert_allclose(global_measure, 1.0)


def test_quadratic_hexahedral_trace_identity_is_partition_independent_under_mpi():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("Quadratic hexahedral contact trace is reviewed on two ranks.")
    comm = MPI.COMM_WORLD
    domain = _hex_cube(comm, cells_x=2)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        _vector_space(domain, degree=2),
    )
    local_ids = tuple(int(value) for value in adapter.trace.point_ids)
    gathered = tuple(comm.allgather(local_ids))
    flattened = tuple(value for values in gathered for value in values)
    global_measure = comm.allreduce(float(np.sum(adapter.trace.weights)), op=MPI.SUM)

    assert len(flattened) == 9
    assert len(set(flattened)) == 9
    np.testing.assert_allclose(global_measure, 1.0)
