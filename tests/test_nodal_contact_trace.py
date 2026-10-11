# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from dolfinx import fem
from mpi4py import MPI
from agentfem import boundary_models as bm
from test_dolfinx_contact_trace import _cube, _hex_cube, _left_region, _vector_space


def assemble(adapter, u, surface):
    evaluation = adapter.evaluate(u)
    record = bm.ContactProjectionRecord(
        adapter.trace.point_ids, surface.project(evaluation.query_points)
    )
    response = bm.frictionless_penalty_contact_law(200.0).evaluate(record.projection)
    return evaluation.assemble(
        record, response, surface_reference_point=(0.0, 0.0, 0.0)
    )


@pytest.mark.parametrize("factory", [_cube, _hex_cube])
def test_nodal_area_force_moment_and_virtual_work(factory):
    domain = factory(MPI.COMM_SELF, cells_x=2)
    space = _vector_space(domain)
    region = _left_region(domain)
    nodal = bm.dolfinx_boundary_region_contact_trace(
        region, space, integration="nodal_lumped"
    )
    gauss = bm.dolfinx_boundary_region_contact_trace(region, space)
    assert np.all(nodal.trace.weights > 0)
    assert nodal.trace.weights.sum() == pytest.approx(gauss.trace.weights.sum())
    assert set(nodal.trace.point_ids).isdisjoint(gauss.trace.point_ids)
    assert np.all((nodal.trace.shape_values == 0) | (nodal.trace.shape_values == 1))
    u = fem.Function(space)
    u.x.array.reshape(-1, 3)[:, 0] = 0.1
    surface = bm.rigid_plane(point=(0.05, 0.0, 0.0), normal=(-1.0, 0.0, 0.0))
    result = assemble(nodal, u, surface)
    reference = assemble(gauss, u, surface)
    np.testing.assert_allclose(
        result.nodal_structural_residual,
        reference.nodal_structural_residual,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        result.surface_generalized_moment,
        reference.surface_generalized_moment,
        atol=1e-13,
    )
    assert result.potential_energy == pytest.approx(0.25)
    direction = np.random.default_rng(42).normal(size=u.x.array.shape)
    old = u.x.array.copy()
    h = 1e-7
    u.x.array[:] = old + h * direction
    plus = assemble(nodal, u, surface).potential_energy
    u.x.array[:] = old - h * direction
    minus = assemble(nodal, u, surface).potential_energy
    expected = result.nodal_structural_residual.ravel() @ direction
    assert (plus - minus) / (2 * h) == pytest.approx(expected, rel=2e-8, abs=1e-9)


def test_cylinder_node_contact_is_not_gauss_point_contact():
    domain = _hex_cube(MPI.COMM_SELF)
    space = _vector_space(domain)
    region = _left_region(domain)
    u = fem.Function(space)
    surface = bm.rigid_cylinder(
        axis_point=(-0.999, 0.0, -0.1),
        axis_direction=(0.0, 0.0, 1.0),
        radius=1.0,
        axial_interval=(0.0, 1.2),
    )
    gauss = bm.dolfinx_boundary_region_contact_trace(region, space)
    nodal = bm.dolfinx_boundary_region_contact_trace(
        region, space, integration="nodal_lumped"
    )
    assert assemble(gauss, u, surface).potential_energy == 0.0
    assert assemble(nodal, u, surface).potential_energy > 0.0


def test_high_order_and_unknown_lumping_rejected():
    domain = _hex_cube(MPI.COMM_SELF)
    region = _left_region(domain)
    with pytest.raises(NotImplementedError, match="first-order"):
        bm.dolfinx_boundary_region_contact_trace(
            region, _vector_space(domain, 2), integration="nodal_lumped"
        )
    with pytest.raises(ValueError, match="integration"):
        bm.dolfinx_boundary_region_contact_trace(
            region, _vector_space(domain), integration="auto"
        )


@pytest.mark.skipif(MPI.COMM_WORLD.size != 2, reason="requires exactly two ranks")
def test_nodal_trace_two_ranks_matches_serial_identity_and_area():
    domain = _hex_cube(MPI.COMM_WORLD, cells_x=2)
    distributed = bm.dolfinx_boundary_region_contact_trace(
        _left_region(domain), _vector_space(domain), integration="nodal_lumped"
    )
    serial_domain = _hex_cube(MPI.COMM_SELF, cells_x=2)
    serial = bm.dolfinx_boundary_region_contact_trace(
        _left_region(serial_domain),
        _vector_space(serial_domain),
        integration="nodal_lumped",
    )
    ids = np.concatenate(MPI.COMM_WORLD.allgather(distributed.trace.point_ids))
    np.testing.assert_array_equal(np.sort(ids), serial.trace.point_ids)
    assert MPI.COMM_WORLD.allreduce(distributed.trace.weights.sum()) == pytest.approx(
        serial.trace.weights.sum()
    )
