# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Actual affine offsets, not tangent rank alone, decide plane feasibility."""

import numpy as np
import pytest

from agentfem import boundary_models
from agentfem.boundary_models.contact_affine_gap_research import (
    AffinePlaneGapSystem, affine_plane_gap_system,
)
from agentfem.boundary_models.dolfinx_contact_trace import dolfinx_boundary_region_contact_trace
from test_finite_strain_j2_periodic import _periodic_j2_contact_step


@pytest.mark.parametrize("plane_x", [1.01, 1.03])
def test_periodic_plane_gap_uses_actual_offset_and_matches_direct_projection(plane_x):
    fixture, periodicity, step = _periodic_j2_contact_step()
    provider = step.contact_provider
    adapter = dolfinx_boundary_region_contact_trace(
        provider.pair.slave_boundary, step.solution.function_space, integration="nodal_lumped",
    )
    adapter = boundary_models.RigidContactTracePolicy(
        "nodal_lumped", coalesce_nodes=True,
    ).apply(adapter, provider.pair)
    reduction = periodicity.reduction(1.)
    matrix = reduction.matrix(fixture.domain.comm)
    dense = matrix.convert("dense")
    try:
        transformation = dense.getDenseArray().copy()
    finally:
        dense.destroy()
        matrix.destroy()
    plane = boundary_models.rigid_plane(point=(plane_x, 0., 0.), normal=(-1., 0., 0.))
    system = affine_plane_gap_system(
        adapter.trace, adapter.reference_nodal_positions, transformation, reduction.offset, plane,
    )
    # Nonzero macroscopic displacement is in the offset, not an arbitrary u=0 probe.
    rng = np.random.default_rng(8431)
    for _ in range(4):
        q = rng.normal(scale=.03, size=reduction.reduced_size)
        u = reduction.reconstruct(q)
        points = adapter.trace.evaluate(
            adapter.reference_nodal_positions + u.reshape(-1, 3),
        ).query_points
        np.testing.assert_allclose(system.gaps(q), plane.project(points).signed_gaps,
                                   atol=3e-16, rtol=1e-12)
    report = system.stationary_penetration(gap_tolerance=1e-10)
    assert len(report["stationary_point_ids"]) == 4
    assert report["minimum_stationary_gap"] == pytest.approx(plane_x-1.02)
    if plane_x < 1.02:
        assert len(report["incompatible_point_ids"]) == 4
        assert system.find_feasible_values(gap_tolerance=1e-10)["status"] == "stationary_incompatibility"
    else:
        assert not report["incompatible_point_ids"]
        result = system.find_feasible_values(gap_tolerance=1e-10)
        assert result["status"] == "feasible_witness"
        assert np.all(system.gaps(result["values"]) >= -1e-10)
    assert not report["certifies_spatial_stability"]
    assert not report["certifies_material_admissibility"]


def test_feasible_search_allows_negative_displacements():
    system = AffinePlaneGapSystem(np.array([[-1.]]), np.array([-2.]),
                                  np.array([7]), np.array([1.]))
    result = system.find_feasible_values(gap_tolerance=1e-10)
    assert result["status"] == "feasible_witness"
    assert result["values"][0] <= -2. + 1e-10


def test_conflicting_nonstationary_constraints_are_not_proclaimed_certified():
    # q >= 1 and q <= -1. HiGHS diagnoses it but this module has no dual certificate.
    system = AffinePlaneGapSystem(np.array([[1.], [-1.]]), np.array([-1., -1.]),
                                  np.array([7, 9]), np.array([1., 2.]))
    result = system.find_feasible_values(gap_tolerance=1e-10)
    assert result["status"] == "unresolved"
    assert result["optimizer_status"] == 2
    assert result["values"] is None
    assert not result["incompatible_point_ids"]


def test_tiny_nonzero_row_is_not_declared_stationary():
    system = AffinePlaneGapSystem(np.array([[1e-20]]), np.array([-1.]),
                                  np.array([0]), np.array([1.]))
    assert not system.stationary_penetration(gap_tolerance=1e-10)["incompatible_point_ids"]
    assert system.gaps([1e20])[0] == pytest.approx(0.)


def test_failed_numerical_witness_is_not_returned(monkeypatch):
    from types import SimpleNamespace
    from scipy import optimize
    system = AffinePlaneGapSystem(np.array([[1.]]), np.array([-1.]),
                                  np.array([0]), np.array([1.]))
    monkeypatch.setattr(optimize, "linprog", lambda *a, **k:
                        SimpleNamespace(x=np.array([0.]), success=True, status=0))
    result = system.find_feasible_values(gap_tolerance=1e-10)
    assert result["status"] == "unresolved" and result["values"] is None


def test_plane_only_budget_and_immutable_arrays():
    trace = boundary_models.ContactTrace(
        point_ids=np.array([0]), node_ids=np.array([[0]]),
        shape_values=np.ones((1, 1)), weights=np.ones(1),
        measure_configuration="reference", source="affine_gap_test",
    )
    reference = np.array([[1., 0., 0.]])
    plane = boundary_models.rigid_plane(point=(0., 0., 0.), normal=(1., 0., 0.))
    system = affine_plane_gap_system(trace, reference, np.eye(3), np.zeros(3), plane)
    with pytest.raises(ValueError):
        system.matrix[0, 0] = 2.
    with pytest.raises(TypeError, match="analytical plane"):
        affine_plane_gap_system(trace, reference, np.eye(3), np.zeros(3),
                                boundary_models.rigid_sphere(center=(0., 0., 0.), radius=1.))
    with pytest.raises(NotImplementedError, match="budget"):
        affine_plane_gap_system(trace, reference, np.eye(3), np.zeros(3), plane, maximum_dofs=2)
    with pytest.raises(ValueError, match="layouts"):
        affine_plane_gap_system(trace, reference, np.eye(3), np.zeros(2), plane)
    with pytest.raises(ValueError, match="tolerance"):
        system.stationary_penetration(gap_tolerance=-1.)
