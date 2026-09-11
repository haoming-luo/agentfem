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
    zhang_2021_plane_strain_composite,
    zhang_2021_periodic_composite,
)


@pytest.mark.skip(
    reason=(
        "Zhang 2021 Table 5 promotion requires content-bound evidence: the "
        "formulation-correspondent 2D Q2/DPC1 diagnostic has not passed the "
        "stress, energy or "
        "effective-tangent comparison and still lacks load-path/mesh/cell-size "
        "convergence plus serial-MPI and restart equivalence. The thin-3D "
        "P2/DG0 diagnostic is not the published formulation."
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
    assert TABLE5.published_q9_element_count == 2823

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
        elastic_energy_semantics="primal_hencky_elastic_energy",
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
        elastic_energy_semantics="primal_hencky_elastic_energy",
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
        elastic_energy_semantics="primal_hencky_elastic_energy",
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
    with pytest.raises(ValueError, match="primal Hencky elastic energy"):
        assess_table5(
            first_piola=tensor,
            elastic_energy_density=TABLE5.elastic_energy_density,
            elastic_energy_semantics="mixed_condensed_elastic_energy",
        )
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


@pytest.mark.parametrize("mesh_size", (0.20, 0.12, 0.08))
def test_exact_plane_strain_geometry_is_q9_only_and_periodic(mesh_size):
    pytest.importorskip("gmsh")
    fixture = zhang_2021_plane_strain_composite(
        MPI.COMM_SELF,
        mesh_size=mesh_size,
    )
    matrix_region, inclusion_region = fixture.regions()
    matrix_area = fem.assemble_scalar(
        fem.form(ufl.as_ufl(1.0) * matrix_region.measure)
    )
    inclusion_area = fem.assemble_scalar(
        fem.form(ufl.as_ufl(1.0) * inclusion_region.measure)
    )
    boundary_measure = ufl.Measure(
        "ds",
        domain=fixture.domain,
        subdomain_data=fixture.facet_tags,
    )
    outer_length = fem.assemble_scalar(
        fem.form(ufl.as_ufl(1.0) * boundary_measure(fixture.periodic_boundary_tag))
    )
    void_length = fem.assemble_scalar(
        fem.form(ufl.as_ufl(1.0) * boundary_measure(fixture.void_boundary_tag))
    )

    radius = 0.15
    assert fixture.domain.topology.cell_type.name == "quadrilateral"
    assert fixture.domain.geometry.cmaps[0].degree == 2
    assert fixture.element_order == 2
    assert fixture.gmsh_element_name == "Quadrilateral 9"
    assert fixture.nodes_per_element == 9
    assert fixture.element_count > 0
    assert fixture.minimum_scaled_jacobian > 0.0
    assert fixture.region_tags == {"matrix": 1, "stiff_inclusions": 2}
    assert fixture.boundary_tags == {
        "periodic_boundary": 10,
        "void_boundary": 20,
    }
    assert set(np.unique(fixture.cell_tags.values)) == {1, 2}
    assert set(np.unique(fixture.facet_tags.values)) == {10, 20}
    assert fixture.inclusion_surface_count == 2
    assert fixture.void_curve_count >= 1
    assert matrix_area == pytest.approx(1.0 - 3.0 * np.pi * radius**2, rel=5.0e-3)
    assert inclusion_area == pytest.approx(2.0 * np.pi * radius**2, rel=5.0e-3)
    assert outer_length == pytest.approx(4.0, rel=1.0e-10)
    assert void_length == pytest.approx(2.0 * np.pi * radius, rel=5.0e-3)
    assert len(fixture.reference_nodes) == 2
    assert fixture.nodes.coordinates.shape[1] == 2
    assert fixture.deformation_gradient.shape == (2, 2)
    assert all(count >= 3 for count in fixture.periodic_pair_counts)
    assert fixture.periodic_pair_counts == fixture.periodic_expected_pair_counts
    assert fixture.periodic_pair_counts[0] == fixture.periodic_pair_counts[1]
    assert fixture.periodic_pairing_error < 1.0e-13
    equation_summary = fixture.equations.summary()
    assert equation_summary["equation_count"] > 0
    assert set(equation_summary["slave_dofs_by_component"]) == {1, 2}


def test_exact_plane_strain_fixture_prepares_three_dpc_pressure_modes():
    pytest.importorskip("gmsh")
    fixture = zhang_2021_plane_strain_composite(
        MPI.COMM_SELF,
        mesh_size=0.20,
    )
    unknown = fixture.mixed_field()
    displacement_element, pressure_element = (
        unknown.space.ufl_element().sub_elements
    )
    periodicity = fixture.constraint(unknown)

    assert fixture.pressure_modes_per_cell == 3
    assert displacement_element.dim == 18
    assert pressure_element.dim == 3
    assert unknown.summary()["pressure_unknowns_per_cell"] == 3
    assert periodicity.reference_cell_volume == pytest.approx(
        fixture.reference_cell_area
    )
    periodicity.apply_affine_increment(0.0, 1.0)
    assert periodicity.mismatch() < 1.0e-12
    np.testing.assert_allclose(
        periodicity.measured_deformation_gradient(unknown.displacement),
        fixture.deformation_gradient,
        rtol=0.0,
        atol=2.0e-12,
    )


def test_exact_plane_strain_fixture_restores_gmsh_global_options():
    gmsh = pytest.importorskip("gmsh")
    initialized_here = not gmsh.isInitialized()
    if initialized_here:
        gmsh.initialize()
    names = (
        "General.Verbosity",
        "Mesh.MeshSizeMin",
        "Mesh.MeshSizeMax",
        "Mesh.Algorithm",
        "Mesh.RecombineAll",
        "Mesh.SubdivisionAlgorithm",
        "Mesh.SecondOrderIncomplete",
        "Mesh.SecondOrderLinear",
    )
    original = {name: float(gmsh.option.getNumber(name)) for name in names}
    selected = {
        "General.Verbosity": 2.0,
        "Mesh.MeshSizeMin": 0.031,
        "Mesh.MeshSizeMax": 0.37,
        "Mesh.Algorithm": 5.0,
        "Mesh.RecombineAll": 1.0,
        "Mesh.SubdivisionAlgorithm": 0.0,
        "Mesh.SecondOrderIncomplete": 1.0,
        "Mesh.SecondOrderLinear": 1.0,
    }
    try:
        for name, value in selected.items():
            gmsh.option.setNumber(name, value)
        zhang_2021_plane_strain_composite(
            MPI.COMM_SELF,
            mesh_size=0.20,
        )
        for name, value in selected.items():
            assert gmsh.option.getNumber(name) == pytest.approx(value)
    finally:
        for name, value in original.items():
            gmsh.option.setNumber(name, value)
        gmsh.clear()
        if initialized_here:
            gmsh.finalize()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"mesh_size": 0.0}, "mesh_size must be finite and positive"),
        ({"mesh_size": np.nan}, "mesh_size must be finite and positive"),
        ({"element_order": 1}, "requires element_order=2"),
        ({"element_order": True}, "requires element_order=2"),
    ),
)
def test_exact_plane_strain_fixture_fails_closed_for_unsupported_mesh(
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        zhang_2021_plane_strain_composite(MPI.COMM_SELF, **kwargs)
