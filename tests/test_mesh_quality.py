from __future__ import annotations

import numpy as np
import pytest
import ufl
import basix
from basix.ufl import element
from mpi4py import MPI

from agentfem import mesh


def test_triangle_mesh_quality_is_collective_and_normalized():
    domain = mesh.rectangle(
        (0.0, 0.0), (1.0, 1.0), (1, 1),
        comm=MPI.COMM_SELF, cell_type="triangle",
    )

    values = mesh.cell_quality(domain)
    report = mesh.audit_quality(domain, threshold=0.8, strict=True)

    np.testing.assert_allclose(values, np.sqrt(3.0) / 2.0)
    assert report.global_cells == 2
    assert report.acceptable
    assert report.summary()["interpretation"] == "1 is equilateral; 0 is degenerate"


def test_mesh_quality_strict_mode_rejects_a_declared_threshold():
    domain = mesh.rectangle(
        (0.0, 0.0), (1.0, 1.0), (1, 1),
        comm=MPI.COMM_SELF, cell_type="triangle",
    )

    with pytest.raises(ValueError, match="poor_cells=2"):
        mesh.audit_quality(domain, threshold=0.9, strict=True)


def test_quadrilateral_quality_samples_the_coordinate_jacobian():
    domain = mesh.rectangle(
        (0.0, 0.0), (1.0, 1.0), (1, 1),
        comm=MPI.COMM_SELF, cell_type="quadrilateral",
    )

    regular = mesh.audit_quality(domain, threshold=0.9, strict=True)
    domain.geometry.x[domain.geometry.x[:, 1] > 0.5, 0] += 0.8
    distorted = mesh.audit_quality(domain)

    assert regular.metric == "sampled_scaled_jacobian"
    assert regular.minimum == pytest.approx(1.0)
    assert regular.samples_per_cell > 1
    assert distorted.minimum == pytest.approx(1.0 / np.sqrt(1.0 + 0.8**2))
    assert distorted.minimum < regular.minimum


def test_hexahedron_quality_samples_the_coordinate_jacobian():
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1),
        comm=MPI.COMM_SELF, cell_type="hexahedron",
    )

    report = mesh.audit_quality(domain, threshold=0.9, strict=True)

    assert report.cell_type.endswith("hexahedron")
    assert report.metric == "sampled_scaled_jacobian"
    assert report.minimum == pytest.approx(1.0)
    assert report.geometry_degree == 1


@pytest.mark.parametrize("cell_name", ("prism", "pyramid"))
def test_mixed_facet_cell_quality_uses_the_real_coordinate_map(cell_name):
    cell_type = getattr(basix.CellType, cell_name)
    coordinates = np.asarray(basix.cell.geometry(cell_type), dtype=float)
    coordinate_element = ufl.Mesh(
        element("Lagrange", cell_name, 1, shape=(3,))
    )
    domain = mesh.from_arrays(
        cells=[list(range(coordinates.shape[0]))],
        coordinates=coordinates,
        coordinate_element=coordinate_element,
        comm=MPI.COMM_SELF,
    )

    report = mesh.audit_quality(domain, threshold=0.9, strict=True)

    assert report.metric == "sampled_scaled_jacobian"
    assert report.minimum == pytest.approx(1.0)
    assert report.samples_per_cell > coordinates.shape[0]


def test_mesh_quality_rejects_a_topology_without_a_declared_metric():
    from dolfinx import mesh as dolfinx_mesh

    domain = dolfinx_mesh.create_unit_interval(MPI.COMM_SELF, 2)

    with pytest.raises(NotImplementedError, match="supports triangle"):
        mesh.audit_quality(domain)
