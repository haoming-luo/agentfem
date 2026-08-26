from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from agentfem import fracture, mesh
from agentfem.constitutive.quadrature import QuadratureField


def test_distributed_interaction_integral_uses_owned_cells_once():
    domain = mesh.rectangle(
        (-0.6, -0.6),
        (0.6, 0.6),
        (72, 72),
        cell_type="triangle",
        comm=MPI.COMM_WORLD,
    )
    cracks = fracture.crack_set(
        fracture.segment("main", start=(-1.0, 0.0), end=(0.0, 0.0))
    )
    material = fracture.linear_elastic_fracture_material(
        young_modulus=1200.0,
        poisson_ratio=0.23,
        assumption="plane_strain",
    )
    exact = fracture.WilliamsField2D(
        cracks.tip("main:end"), material, k_i=8.0, k_ii=-3.0
    )
    stress = QuadratureField.create(
        domain, name="S_EXACT", degree=6, value_shape=(2, 2)
    )
    gradient = QuadratureField.create(
        domain, name="GRAD_U_EXACT", degree=6, value_shape=(2, 2)
    )
    cell_map = domain.topology.index_map(domain.topology.dim)
    cells = np.arange(
        int(cell_map.size_local + cell_map.num_ghosts), dtype=np.int32
    )
    coordinates = np.asarray(
        fem.Expression(ufl.SpatialCoordinate(domain), stress.points).eval(
            domain, cells
        ),
        dtype=float,
    ).reshape((-1, 2))
    stress.assign(exact.stress(coordinates))
    gradient.assign(exact.displacement_gradient(coordinates))

    report = fracture.dolfinx_interaction_integral_report(
        domain,
        stress.function,
        gradient.function,
        crack=cracks,
        tip_id="main:end",
        material=material,
        integration_radii=(0.18, 0.26, 0.34),
        quadrature_degree=6,
        relative_path_tolerance=0.025,
    )

    assert report.k_i == pytest.approx(8.0, rel=0.012)
    assert report.k_ii == pytest.approx(-3.0, rel=0.012)
    assert report.status == "accepted"
