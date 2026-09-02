"""External three-dimensional linear-elasticity verification."""

from __future__ import annotations

from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
import pytest

from agentfem import benchmarks


def test_nafems_le10_mesh_requires_an_exact_mid_thickness_layer():
    with pytest.raises(ValueError, match="must be even"):
        benchmarks.nafems_le10_mesh(
            radial_cells=2,
            angular_cells=4,
            thickness_cells=3,
            comm=MPI.COMM_SELF,
        )


def test_nafems_le10_mesh_is_curved_quadratic_hexahedral_geometry():
    domain = benchmarks.nafems_le10_mesh(
        radial_cells=2,
        angular_cells=4,
        thickness_cells=2,
        comm=MPI.COMM_SELF,
    )

    assert domain.topology.cell_type == dolfinx_mesh.CellType.hexahedron
    assert domain.geometry.cmaps[0].degree == 2
    assert domain.topology.index_map(domain.topology.dim).size_local == 16


def test_nafems_le10_public_3d_structural_benchmark():
    benchmark, simulation = benchmarks.nafems_le10_3d_benchmark(
        radial_cells=4,
        angular_cells=12,
        thickness_cells=2,
        comm=MPI.COMM_SELF,
    )

    assert benchmark.acceptable
    assert benchmark.quantities["sigma_yy_D_pa"] == pytest.approx(
        -5.398769829501652e6,
        rel=2.0e-7,
    )
    assert benchmark.quantities["relative_sigma_yy_D_error"] < 0.01
    assert benchmark.quantities["relative_point_D_displacement_error"] < 0.01
    assert benchmark.quantities["relative_force_balance_error"] < 1.0e-10
    assert benchmark.quantities["relative_energy_balance_error"] < 1.0e-10
    assert benchmark.quantities["relative_loaded_area_error"] < 5.0e-5
    assert benchmark.quantities["u_z_D_m"] == pytest.approx(
        -9.89161961e-5,
        rel=2.0e-7,
    )
    assert benchmark.extraction["nodal_recovery_is_default"] is False
    assert simulation.fields["S"].processing["interelement_smoothing"] is False
    assert simulation.metadata["external_benchmark"]["acceptable"] is True
