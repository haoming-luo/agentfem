"""Gmsh lowering contracts for one fixed periodic multi-void realization."""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from agentfem import fields

from periodic_void_fixture import (
    SphericalVoidRealization,
    periodic_multi_spherical_void_cell,
    sample_hard_core_spherical_voids,
)


@pytest.fixture(scope="module")
def periodic_multi_void_fixture():
    pytest.importorskip("gmsh")
    realization = sample_hard_core_spherical_voids(
        side_length=1.0,
        count=3,
        radius=0.10,
        seed=1729,
        minimum_inter_void_clearance=0.06,
        minimum_boundary_clearance=0.05,
        maximum_attempts=500,
    )
    return periodic_multi_spherical_void_cell(
        MPI.COMM_SELF,
        realization=realization,
        mesh_size=0.16,
        stretch=1.01,
    )


def _equation_signature(equations):
    return tuple(
        tuple((term.node, term.dof, term.coefficient) for term in equation.terms)
        for equation in equations.equations
    )


def test_multi_void_mesh_has_one_matrix_and_unified_void_surface_semantics(
    periodic_multi_void_fixture,
):
    fixture = periodic_multi_void_fixture
    solid_volume = fem.assemble_scalar(
        fem.form(ufl.as_ufl(1.0) * ufl.dx(domain=fixture.domain))
    )

    assert fixture.domain.topology.index_map(3).size_global > 0
    assert fixture.void_surface_count == len(fixture.realization.spheres) == 3
    assert set(np.unique(fixture.cell_tags.values)) == {1}
    assert set(np.unique(fixture.facet_tags.values)) == {10, 20}
    assert np.count_nonzero(fixture.facet_tags.values == 20) > 0
    assert fixture.actual_void_fraction == pytest.approx(
        fixture.realization.actual_void_fraction
    )
    assert (
        fixture.realization_fingerprint
        == (fixture.realization.scientific_identity()["fingerprint"])
    )
    assert solid_volume == pytest.approx(fixture.exact_solid_volume, rel=8.0e-3)


def test_multi_void_mesh_preserves_exact_affine_periodic_source_equations(
    periodic_multi_void_fixture,
):
    fixture = periodic_multi_void_fixture
    displacement = fields.displacement(fixture.domain)
    periodicity = fixture.constraint(displacement)
    reduction = periodicity.reduction()
    global_vertices = fixture.domain.topology.index_map(0).size_global

    assert reduction.full_size == 3 * global_vertices
    assert 0 < reduction.reduced_size < reduction.full_size
    assert fixture.periodic_pairing_error < 1.0e-13
    assert len(fixture.equations.equations) > 0
    assert periodicity.reference_cell_volume == pytest.approx(1.0)

    periodicity.apply_affine_increment(0.0, 1.0)
    assert periodicity.mismatch() < 1.0e-12
    np.testing.assert_allclose(
        periodicity.measured_deformation_gradient(displacement),
        fixture.deformation_gradient,
        rtol=0.0,
        atol=2.0e-12,
    )


def test_multi_void_gmsh_source_semantics_are_stable_under_input_order(
    periodic_multi_void_fixture,
):
    first = periodic_multi_void_fixture
    source = first.realization
    reversed_realization = SphericalVoidRealization(
        side_length=source.side_length,
        spheres=tuple(reversed(source.spheres)),
        seed=source.seed,
        minimum_inter_void_clearance=source.minimum_inter_void_clearance,
        minimum_boundary_clearance=source.minimum_boundary_clearance,
        attempts=source.attempts,
        sampler=source.sampler,
    )
    repeated = periodic_multi_spherical_void_cell(
        MPI.COMM_SELF,
        realization=reversed_realization,
        mesh_size=0.16,
        stretch=1.01,
    )

    assert repeated.realization_fingerprint == first.realization_fingerprint
    assert repeated.anchor_node == first.anchor_node
    assert repeated.reference_nodes == first.reference_nodes
    np.testing.assert_array_equal(repeated.nodes.labels, first.nodes.labels)
    np.testing.assert_allclose(
        repeated.nodes.coordinates,
        first.nodes.coordinates,
        rtol=0.0,
        atol=0.0,
    )
    assert _equation_signature(repeated.equations) == _equation_signature(
        first.equations
    )
