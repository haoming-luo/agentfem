"""Contracts for the Zhang--Feng--Khandelwal 2021 external benchmark."""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from agentfem import fields

from zhang_2021_periodic_composite_fixture import (
    TABLE5,
    assess_table5,
    column_major_plane_components,
    young_poisson_from_bulk_shear,
    zhang_2021_periodic_composite,
)


@pytest.mark.skip(
    reason=(
        "Zhang 2021 Table 5 promotion requires content-bound evidence: the "
        "thin-3D P2/DG0 diagnostic is not the published 2D Q2/DPC1 "
        "formulation and has not passed energy/effective-tangent, load-path/"
        "mesh/cell-size convergence, or serial-MPI and restart equivalence."
    )
)
def test_zhang_table5_external_numerical_verification_gate():
    """Replace this skip only with converged external numerical evidence."""

    pytest.fail("External Table 5 verification has not been promoted.")


def test_table5_reference_preserves_published_component_order_and_evidence_gate():
    tensor = np.asarray(
        (
            (TABLE5.first_piola[0], TABLE5.first_piola[2], 0.0),
            (TABLE5.first_piola[1], TABLE5.first_piola[3], 0.0),
            (0.0, 0.0, 0.0),
        )
    )
    np.testing.assert_array_equal(
        column_major_plane_components(tensor),
        TABLE5.first_piola,
    )
    young, poisson = young_poisson_from_bulk_shear(17.5, 8.0)
    assert young == pytest.approx(20.826446280991735)
    assert poisson == pytest.approx(0.30165289256198347)

    incomplete = assess_table5(first_piola=tensor)
    assert incomplete["status"] == "incomplete"
    assert not incomplete["accepted"]
    assert not incomplete["comparison_accepted"]
    assert incomplete["evidence_authority"] == (
        "caller_supplied_assertions_not_content_bound"
    )
    assert not incomplete["content_bound"]
    assert not incomplete["benchmark_promotion_authorized"]
    assert incomplete["decision_scope"] == (
        "numeric_comparison_and_caller_asserted_completeness"
    )
    assert set(incomplete["missing_evidence"]) == {
        "published_elastic_energy",
        "effective_tangent",
        "load_increment_path_converged",
        "mesh_converged",
        "plane_strain_formulation_converged",
        "periodic_cell_size_invariant",
        "serial_mpi_equivalent",
        "restart_equivalent",
    }
    failed = assess_table5(first_piola=2.0 * tensor)
    assert failed["status"] == "failed"
    assert not failed["accepted"]
    assert "effective_tangent" in failed["missing_evidence"]
    accepted = assess_table5(
        first_piola=tensor,
        elastic_energy_density=TABLE5.elastic_energy_density,
        effective_tangent=TABLE5.effective_tangent,
        convergence_evidence={
            "load_increment_path_converged": True,
            "mesh_converged": True,
            "plane_strain_formulation_converged": True,
            "periodic_cell_size_invariant": True,
            "serial_mpi_equivalent": True,
            "restart_equivalent": True,
        },
    )
    assert accepted["status"] == "accepted"
    assert accepted["accepted"]
    assert accepted["comparison_accepted"]
    # Even a numerically exact candidate with every caller assertion set true
    # remains a comparison result.  This helper neither verifies an evidence
    # archive nor authorizes promotion of the benchmark capability.
    assert accepted["evidence_authority"] == (
        "caller_supplied_assertions_not_content_bound"
    )
    assert not accepted["content_bound"]
    assert not accepted["benchmark_promotion_authorized"]
    assert accepted["first_piola_componentwise_passed"]
    assert accepted["tolerance_authority"] == (
        "AgentFEM comparison contract; not published"
    )

    missing_path_convergence = assess_table5(
        first_piola=tensor,
        elastic_energy_density=TABLE5.elastic_energy_density,
        effective_tangent=TABLE5.effective_tangent,
        convergence_evidence={
            "mesh_converged": True,
            "plane_strain_formulation_converged": True,
            "periodic_cell_size_invariant": True,
            "serial_mpi_equivalent": True,
            "restart_equivalent": True,
        },
    )
    assert missing_path_convergence["status"] == "incomplete"
    assert not missing_path_convergence["accepted"]
    assert missing_path_convergence["missing_evidence"] == (
        "load_increment_path_converged",
    )

    # A vector norm can conceal a poor small component.  Perturb only P11 far
    # enough to violate the absolute-plus-relative component contract while
    # keeping the global relative L2 error below three percent.
    hidden_p11_error = tensor.copy()
    hidden_p11_error[0, 0] += 6.0e-3
    component_failure = assess_table5(
        first_piola=hidden_p11_error,
        elastic_energy_density=TABLE5.elastic_energy_density,
        effective_tangent=TABLE5.effective_tangent,
        convergence_evidence={
            "load_increment_path_converged": True,
            "mesh_converged": True,
            "plane_strain_formulation_converged": True,
            "periodic_cell_size_invariant": True,
            "serial_mpi_equivalent": True,
            "restart_equivalent": True,
        },
    )
    assert component_failure["first_piola_relative_l2_error"] < 0.03
    assert not component_failure["first_piola_componentwise_passed"]
    assert component_failure["first_piola_component_error_ratio"][0] > 1.0
    assert component_failure["status"] == "failed"
    assert not component_failure["accepted"]

    with pytest.raises(ValueError, match="no greater than"):
        assess_table5(first_piola=tensor, relative_tolerance=0.031)
    with pytest.raises(TypeError, match="must be bool"):
        assess_table5(
            first_piola=tensor,
            convergence_evidence={"mesh_converged": "yes"},
        )


def test_zhang_cell_geometry_materials_and_affine_periodicity_are_explicit():
    pytest.importorskip("gmsh")
    fixture = zhang_2021_periodic_composite(
        MPI.COMM_SELF,
        mesh_size=0.12,
        element_order=2,
    )
    matrix_region, inclusion_region = fixture.regions()
    matrix, inclusion = fixture.materials()
    unknown = fields.displacement_pressure(fixture.domain)
    periodicity = fixture.constraint(unknown)

    matrix_volume = fem.assemble_scalar(
        fem.form(ufl.as_ufl(1.0) * matrix_region.measure)
    )
    inclusion_volume = fem.assemble_scalar(
        fem.form(ufl.as_ufl(1.0) * inclusion_region.measure)
    )
    radius = 0.15
    expected_matrix = fixture.thickness * (1.0 - 3.0 * np.pi * radius**2)
    expected_inclusions = fixture.thickness * 2.0 * np.pi * radius**2

    assert set(np.unique(fixture.cell_tags.values)) == {1, 2}
    assert set(np.unique(fixture.facet_tags.values)) == {10, 20}
    assert fixture.element_order == 2
    assert matrix_volume == pytest.approx(expected_matrix, rel=1.0e-2)
    assert inclusion_volume == pytest.approx(expected_inclusions, rel=1.0e-2)
    assert fixture.periodic_pairing_error < 1.0e-13
    assert periodicity.reference_cell_volume == pytest.approx(fixture.thickness)
    assert matrix.bulk_modulus == pytest.approx(17.5)
    assert matrix.shear_modulus == pytest.approx(8.0)
    assert inclusion.bulk_modulus == pytest.approx(1750.0)
    assert inclusion.shear_modulus == pytest.approx(800.0)

    periodicity.apply_affine_increment(0.0, 1.0)
    assert periodicity.mismatch() < 1.0e-12
    np.testing.assert_allclose(
        periodicity.measured_deformation_gradient(unknown.displacement),
        fixture.deformation_gradient,
        rtol=0.0,
        atol=2.0e-12,
    )
