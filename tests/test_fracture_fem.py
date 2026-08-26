from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from agentfem import benchmarks, fracture, mesh
from agentfem import fracture_fem
from agentfem.constitutive.quadrature import QuadratureField


def _exact_williams_quadrature_fields(domain, field, *, degree):
    stress = QuadratureField.create(
        domain, name="S_EXACT", degree=degree, value_shape=(2, 2)
    )
    gradient = QuadratureField.create(
        domain, name="GRAD_U_EXACT", degree=degree, value_shape=(2, 2)
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
    stress.assign(field.stress(coordinates))
    gradient.assign(field.displacement_gradient(coordinates))
    return stress.function, gradient.function


def test_dolfinx_adapter_recovers_exact_quadrature_williams_field():
    domain = mesh.rectangle(
        (-0.6, -0.6),
        (0.6, 0.6),
        (72, 72),
        cell_type="triangle",
        comm=MPI.COMM_SELF,
    )
    cracks = fracture.crack_set(
        fracture.segment("main", start=(-1.0, 0.0), end=(0.0, 0.0))
    )
    material = fracture.linear_elastic_fracture_material(
        young_modulus=1200.0, poisson_ratio=0.23, assumption="plane_strain"
    )
    exact = fracture.WilliamsField2D(
        cracks.tip("main:end"), material, k_i=8.0, k_ii=-3.0
    )
    stress, gradient = _exact_williams_quadrature_fields(domain, exact, degree=6)

    report = fracture.dolfinx_interaction_integral_report(
        domain,
        stress,
        gradient,
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
    assert report.metadata["provider"] == "dolfinx"


def test_dolfinx_adapter_rejects_non_tensor_fields_before_compilation():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (2, 2),
        cell_type="triangle",
        comm=MPI.COMM_SELF,
    )
    scalar = fem.Function(fem.functionspace(domain, ("DG", 0)))
    crack = fracture.segment("main", start=(-1.0, 0.0), end=(0.0, 0.0))
    cracks = fracture.crack_set(crack)
    material = fracture.linear_elastic_fracture_material(
        young_modulus=1000.0, poisson_ratio=0.2
    )

    with pytest.raises(ValueError, match="actual_stress"):
        fracture.dolfinx_interaction_integral_samples(
            domain,
            scalar,
            ufl.Identity(2),
            tip=cracks.tip("main:end"),
            auxiliary=fracture.WilliamsField2D(
                cracks.tip("main:end"), material, k_i=1.0
            ),
            inner_radius=0.1,
            outer_radius=0.2,
        )


def test_standard_fem_center_crack_recovers_mode_i_reference():
    evidence = benchmarks.center_crack_mode_i_benchmark()

    assert evidence.result.status == "completed"
    assert evidence.verification.relative_k_error < 0.03
    assert evidence.verification.relative_j_error < 0.05
    assert evidence.stress_intensity.path_variation < 0.02
    assert evidence.stress_intensity.k_ii == pytest.approx(
        0.0, abs=0.03 * evidence.reference.k_i
    )
    assert evidence.status == "accepted"


def test_center_crack_benchmark_rejects_non_tensile_loading_before_solve():
    with pytest.raises(ValueError, match="tensile"):
        benchmarks.center_crack_mode_i_benchmark(remote_strain=0.0)


def test_global_integral_accepts_a_rank_without_local_annulus(monkeypatch):
    class FakeComm:
        def __init__(self):
            self.calls = 0

        def allreduce(self, value, op=None):
            self.calls += 1
            return 1 if self.calls == 1 else 7.5

    def empty_local_samples(*args, **kwargs):
        raise ValueError(fracture_fem._EMPTY_ANNULUS_MESSAGE)

    monkeypatch.setattr(
        fracture_fem, "dolfinx_interaction_integral_samples", empty_local_samples
    )
    domain = SimpleNamespace(comm=FakeComm())

    value = fracture_fem.dolfinx_interaction_integral(domain, None, None)

    assert value == pytest.approx(7.5)


def test_global_integral_rejects_an_empty_global_annulus(monkeypatch):
    class EmptyComm:
        @staticmethod
        def allreduce(value, op=None):
            return 0

    def empty_local_samples(*args, **kwargs):
        raise ValueError(fracture_fem._EMPTY_ANNULUS_MESSAGE)

    monkeypatch.setattr(
        fracture_fem, "dolfinx_interaction_integral_samples", empty_local_samples
    )
    domain = SimpleNamespace(comm=EmptyComm())

    with pytest.raises(ValueError, match="global quadrature"):
        fracture_fem.dolfinx_interaction_integral(domain, None, None)
