# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""A cell average integrates physical measure, not the parameter square."""
import numpy as np
import pytest
import ufl
import basix
import basix.ufl
from dolfinx import fem, mesh
from mpi4py import MPI

from agentfem import constitutive, results
from agentfem import mesh as agent_mesh


@pytest.mark.parametrize("shape", [(), (2,), (2, 2)])
@pytest.mark.parametrize("dimension", [2, 3])
def test_distorted_q1_cell_recovery_matches_physical_centroid(shape, dimension):
    domain = (mesh.create_unit_square(MPI.COMM_SELF, 1, 1, cell_type=mesh.CellType.quadrilateral)
              if dimension == 2 else
              mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1, cell_type=mesh.CellType.hexahedron))
    x = domain.geometry.x
    axis = dimension - 1
    x[:, 0] *= 1. + x[:, axis]
    # X(r,s)=(r*(1+s),s): det(J)=1+s. The centroid ordinate is
    # integral(s*(1+s))/integral(1+s) = 5/9, not the reference mean 1/2.
    source = constitutive.QuadratureField.create(domain, name="IP", degree=4, value_shape=shape)
    ordinate = np.asarray(fem.Expression(ufl.SpatialCoordinate(domain)[axis], source.points)
                          .eval(domain, np.array([0], dtype=np.int32))).reshape(-1)
    components = np.arange(1, source.component_count + 1)
    values = (ordinate[:, None] * components).reshape((-1, *shape))
    source.assign(values)
    result = results.recover_integration_point_field(source)
    np.testing.assert_allclose(result.field.x.array, (5./9.) * components, rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(source.values, values)
    assert result.processing["averaging_measure"] == "mesh_geometry"
    assert result.processing["geometric_weighting"] == "absolute_jacobian_determinant"
    assert result.processing["deformation_gradient_weighting"] is False


def test_curved_triangle_recovery_uses_geometry_not_field_order():
    element = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 2,
                                   lagrange_variant=basix.LagrangeVariant.gll_warped)
    x = element.points.copy()
    x[:, 0] *= 1. + x[:, 1]
    domain = agent_mesh.from_arrays(
        cells=[list(range(len(x)))], coordinates=x,
        coordinate_element=ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 2, shape=(2,))),
        comm=MPI.COMM_SELF)
    source = constitutive.QuadratureField.create(domain, name="IP", degree=4)
    ordinate = np.asarray(fem.Expression(ufl.SpatialCoordinate(domain)[1], source.points)
                          .eval(domain, np.array([0], dtype=np.int32))).reshape(-1)
    source.assign(ordinate)
    # On r+s<=1, integral(s*(1+s))=1/4 and integral(1+s)=2/3.
    np.testing.assert_allclose(source.cell_average().x.array, 3./8., atol=2e-14)


def test_embedded_surface_recovery_uses_area_measure():
    # The physical surface lies on z=y; the surface Jacobian includes sqrt(2),
    # which cancels in the mean but must not require a square coordinate map.
    coordinates = np.asarray([[0., 0., 0.], [1., 0., 0.],
                              [0., 1., 1.], [2., 1., 1.]])
    domain = agent_mesh.from_arrays(
        cells=[[0, 1, 2, 3]], coordinates=coordinates,
        coordinate_element=ufl.Mesh(basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(3,))),
        comm=MPI.COMM_SELF)
    source = constitutive.QuadratureField.create(domain, name="IP", degree=4)
    source.assign(source.points[:, 1])
    np.testing.assert_allclose(source.cell_average().x.array, 5./9., atol=2e-14)


def test_recovery_does_not_reuse_measure_after_mesh_geometry_changes():
    domain = mesh.create_unit_square(MPI.COMM_SELF, 1, 1, cell_type=mesh.CellType.quadrilateral)
    source = constitutive.QuadratureField.create(domain, name="IP", degree=4)
    # Backend cell orientation need not align reference s with physical y.
    ordinate = np.asarray(fem.Expression(ufl.SpatialCoordinate(domain)[1], source.points)
                          .eval(domain, np.array([0], dtype=np.int32))).reshape(-1)
    source.assign(ordinate)
    np.testing.assert_allclose(source.cell_average().x.array, .5, atol=2e-14)
    domain.geometry.x[:, 0] *= 1. + domain.geometry.x[:, 1]
    np.testing.assert_allclose(source.cell_average().x.array, 5./9., atol=2e-14)
    np.testing.assert_array_equal(source.values, ordinate)


def test_recovery_reuses_only_expression_not_geometry_or_output(monkeypatch):
    domain = mesh.create_unit_square(MPI.COMM_SELF, 1, 1, cell_type=mesh.CellType.quadrilateral)
    source = constitutive.QuadratureField.create(domain, name="IP", degree=4)
    ordinate = np.asarray(fem.Expression(ufl.SpatialCoordinate(domain)[1], source.points)
                          .eval(domain, np.array([0], dtype=np.int32))).reshape(-1)
    source.assign(ordinate)
    original = fem.Expression
    preparations = []

    def prepare(*args, **kwargs):
        preparations.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(fem, "Expression", prepare)
    first = source.cell_average(name="FIRST")
    first_values = first.x.array.copy()
    domain.geometry.x[:, 0] *= 1. + domain.geometry.x[:, 1]
    second = source.cell_average(name="SECOND")
    assert len(preparations) == 1
    np.testing.assert_allclose(second.x.array, 5./9., atol=2e-14)
    np.testing.assert_array_equal(first.x.array, first_values)
    assert first.name == "FIRST" and second.name == "SECOND"
    source.assign(2 * ordinate)
    third = source.cell_average()
    assert len(preparations) == 1
    np.testing.assert_allclose(third.x.array, 10./9., atol=4e-14)
    np.testing.assert_allclose(second.x.array, 5./9., atol=2e-14)


def test_measure_preparation_invalidates_on_points_or_mesh(monkeypatch):
    domain = mesh.create_unit_square(MPI.COMM_SELF, 1, 1, cell_type=mesh.CellType.quadrilateral)
    domain.geometry.x[:, 0] *= 1. + domain.geometry.x[:, 1]
    source = constitutive.QuadratureField.create(domain, name="IP", degree=4)
    original = fem.Expression
    preparations = []

    def prepare(*args, **kwargs):
        preparations.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(fem, "Expression", prepare)
    source.owned_physical_weights()
    source.points[:] *= 0.8
    actual = source.owned_physical_weights()
    assert len(preparations) == 2
    expected = np.abs(np.asarray(original(ufl.JacobianDeterminant(domain), source.points)
                                .eval(domain, np.array([0], dtype=np.int32))).reshape(-1)) * source.weights
    np.testing.assert_array_equal(actual, expected)
    other = mesh.create_unit_square(MPI.COMM_SELF, 1, 1, cell_type=mesh.CellType.quadrilateral)
    other.geometry.x[:, 0] *= 3
    source.function = constitutive.QuadratureField.create(other, name="OTHER", degree=4).function
    changed = source.owned_physical_weights()
    assert len(preparations) == 3
    np.testing.assert_allclose(changed, 3 * source.weights, rtol=2e-14)
    source.weights[:] *= 2
    np.testing.assert_allclose(source.owned_physical_weights(), 2 * changed, rtol=2e-14)
    assert len(preparations) == 3


def test_measure_preparation_returns_detached_weights():
    domain = mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    source = constitutive.QuadratureField.create(domain, name="IP", degree=2)
    initial = source.owned_physical_weights()
    damaged = source.owned_physical_weights()
    damaged[:] = 0
    np.testing.assert_array_equal(source.owned_physical_weights(), initial)
    np.testing.assert_array_equal(source.owned_physical_weights(multipliers=np.full(initial.size, 2)), 2 * initial)
    source._measure_expression_cache = None
    np.testing.assert_array_equal(source.owned_physical_weights(), initial)


@pytest.mark.parametrize("cells", [1, 4])
def test_distributed_physical_average_preserves_integral_and_empty_rank(cells):
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        pytest.skip("requires two ranks")
    from dolfinx import graph
    def owner_zero(comm, parts, types, connectivity):
        count = (connectivity.num_nodes if hasattr(connectivity, "num_nodes")
                 else sum(array.size // 4 for array in connectivity))
        result = graph.adjacencylist(np.zeros((count, 1), dtype=np.int32))
        return getattr(result, "_cpp_object", result)
    # A one-cell graph is not an admissible ParMETIS partitioning problem.
    # Use an explicit owner to test the deliberately empty-rank contract.
    domain = mesh.create_unit_square(comm, cells, 1, cell_type=mesh.CellType.quadrilateral,
                                     partitioner=owner_zero if cells == 1 else None)
    domain.geometry.x[:, 0] *= 1. + domain.geometry.x[:, 1]
    source = constitutive.QuadratureField.create(domain, name="IP", degree=4)
    cell_map = domain.topology.index_map(domain.topology.dim)
    visible = np.arange(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32)
    ordinate = np.asarray(fem.Expression(ufl.SpatialCoordinate(domain)[1], source.points)
                          .eval(domain, visible)).reshape(-1)
    local_error = None
    try:
        source._array_indices()
    except Exception as exc:
        local_error = str(exc)
    errors = comm.allgather(local_error)
    assert not any(errors), errors
    source.assign(ordinate)
    recovered = source.cell_average()
    assert source._measure_expression_cache is None  # no rank-local JIT decision
    integral = comm.allreduce(fem.assemble_scalar(fem.form(recovered * ufl.dx)), op=MPI.SUM)
    assert integral == pytest.approx(5./6., rel=2e-14)
    if cells == 1:
        assert min(comm.allgather(cell_map.size_local)) == 0


def test_distributed_bad_local_measure_rejects_before_output_scatter(monkeypatch):
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        pytest.skip("requires two ranks")
    domain = mesh.create_unit_square(comm, 4, 1, cell_type=mesh.CellType.quadrilateral)
    source = constitutive.QuadratureField.create(domain, name="IP", degree=4)
    weights = source.owned_physical_weights()
    assert len(weights) > 0
    if comm.rank == 0:
        weights[:] = 0.
    monkeypatch.setattr(source, "owned_physical_weights", lambda: weights)
    with pytest.raises(ValueError, match="AFM-RECOVERY-MEASURE: rank 0"):
        source.cell_average()


@pytest.mark.parametrize("fault", ["measure", "value"])
def test_nonfinite_recovery_is_not_published(monkeypatch, fault):
    domain = mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    source = constitutive.QuadratureField.create(domain, name="IP", degree=2)
    if fault == "measure":
        weights = source.owned_physical_weights()
        weights[:] = np.nan
        monkeypatch.setattr(source, "owned_physical_weights", lambda: weights)
    else:
        source.function.x.array[:] = np.nan
    with pytest.raises(ValueError, match="AFM-RECOVERY-MEASURE"):
        source.cell_average()
