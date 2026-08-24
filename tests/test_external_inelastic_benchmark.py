"""External structure-level acceptance for distributed J2 equilibrium."""

from __future__ import annotations

import pytest
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import benchmarks


def test_nafems_creep_cylinder_reference_stresses_match_published_table():
    radius = (100.0, 125.0, 150.0, 175.0, 200.0)
    radial, hoop, axial = benchmarks.power_law_creep_cylinder_stress(
        radius,
        inner_radius=100.0,
        outer_radius=200.0,
        pressure=200.0,
        stress_exponent=5.0,
    )

    assert radial == pytest.approx(
        (-200.0, -129.47114, -76.33936, -34.34328, 0.0), abs=1.0e-5
    )
    assert hoop == pytest.approx(
        (130.38504, 172.70235, 204.58142, 229.77907, 250.38504), abs=1.0e-5
    )
    assert axial == pytest.approx(
        (-34.80748, 21.61561, 64.12103, 97.71789, 125.19252), abs=1.0e-5
    )


def test_nafems_creep_benchmark_rejects_nonpositive_increment_count():
    with pytest.raises(ValueError, match="positive integer"):
        benchmarks.creep_thick_cylinder_benchmark(
            comm=MPI.COMM_SELF,
            increments=0,
        )


def test_nafems_creep_benchmark_rejects_nonpositive_time_tolerance():
    with pytest.raises(
        ValueError,
        match="creep_strain_error_tolerance must be finite and positive",
    ):
        benchmarks.creep_thick_cylinder_benchmark(
            comm=MPI.COMM_SELF,
            radial_cells=1,
            angular_cells=2,
            increments=1,
            creep_strain_error_tolerance=0.0,
            progress=False,
        )


def test_thick_cylinder_sector_supports_structured_hexahedra():
    domain = benchmarks.thick_cylinder_sector_mesh(
        inner_radius=100.0,
        outer_radius=200.0,
        thickness=10.0,
        radial_cells=2,
        angular_cells=4,
        cell_type="hexahedron",
        comm=MPI.COMM_SELF,
    )

    assert domain.topology.cell_type == dolfinx_mesh.CellType.hexahedron
    assert domain.topology.index_map(domain.topology.dim).size_local == 8


def test_thick_cylinder_sector_rejects_unknown_cell_type():
    with pytest.raises(ValueError, match="tetrahedron.*hexahedron"):
        benchmarks.thick_cylinder_sector_mesh(
            inner_radius=100.0,
            outer_radius=200.0,
            thickness=10.0,
            radial_cells=2,
            angular_cells=4,
            cell_type="wedge",
            comm=MPI.COMM_SELF,
        )


def test_axisymmetric_j2_thick_cylinder_brackets_first_yield():
    assessment = benchmarks.j2_thick_cylinder_benchmark(
        comm=MPI.COMM_SELF,
        radial_cells=4,
        axial_cells=1,
        increments=24,
        formulation="axisymmetric",
    )

    assert assessment.acceptable
    assert assessment.quantities["maximum_equivalent_plastic_strain"] > 0.0
    assert assessment.quantities["yield_bracket_error"] < 0.01


def test_axisymmetric_nafems_creep_reaches_subpercent_stress_error():
    assessment = benchmarks.creep_thick_cylinder_benchmark(
        comm=MPI.COMM_SELF,
        radial_cells=4,
        axial_cells=1,
        increments=300,
        progress=False,
        formulation="axisymmetric",
    )

    assert assessment.acceptable
    assert assessment.quantities["formulation_axisymmetric"] == 1.0
    for name in (
        "radial_stress_relative_l2",
        "hoop_stress_relative_l2",
        "axial_stress_relative_l2",
    ):
        assert assessment.quantities[name] < 5.0e-3


def test_public_thick_cylinder_brackets_first_yield_in_serial_and_mpi():
    if MPI.COMM_WORLD.size not in {1, 2}:
        pytest.skip("the versioned external J2 contract covers one and two ranks")

    assessment = benchmarks.j2_thick_cylinder_benchmark()

    assert assessment.acceptable
    assert assessment.quantities["maximum_equivalent_plastic_strain"] > 0.0
    assert assessment.quantities["maximum_displacement"] == pytest.approx(
        0.003455482362744615,
        rel=2.0e-10,
        abs=2.0e-13,
    )
    assert assessment.quantities["maximum_equivalent_plastic_strain"] == pytest.approx(
        0.00024084814820469344,
        rel=2.0e-9,
        abs=2.0e-13,
    )
