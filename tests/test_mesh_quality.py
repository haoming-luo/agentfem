from __future__ import annotations

import numpy as np
import pytest
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


def test_mesh_quality_does_not_assign_one_metric_to_unsupported_topologies():
    domain = mesh.rectangle(
        (0.0, 0.0), (1.0, 1.0), (1, 1),
        comm=MPI.COMM_SELF, cell_type="quadrilateral",
    )

    with pytest.raises(NotImplementedError, match="triangle and tetrahedron"):
        mesh.audit_quality(domain)
