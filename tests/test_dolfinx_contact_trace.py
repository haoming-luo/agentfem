from __future__ import annotations

import basix.ufl
from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np
import pytest
import ufl

from agentfem import boundary_models
from agentfem import mesh as agent_mesh


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


def test_hexahedral_contact_trace_identity_is_partition_independent_under_mpi():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("Hexahedral DOLFINx contact trace is reviewed on two ranks.")
    comm = MPI.COMM_WORLD
    domain = _hex_cube(comm, cells_x=2)
    adapter = boundary_models.dolfinx_boundary_region_contact_trace(
        _left_region(domain),
        _vector_space(domain),
    )
    local_ids = tuple(int(value) for value in adapter.trace.point_ids)
    gathered = tuple(comm.allgather(local_ids))
    flattened = tuple(value for values in gathered for value in values)
    global_measure = comm.allreduce(float(np.sum(adapter.trace.weights)), op=MPI.SUM)

    assert len(flattened) == 4
    assert len(set(flattened)) == 4
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
