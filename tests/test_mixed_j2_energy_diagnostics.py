"""Mixed J2 primal/condensed elastic-energy diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from agentfem import mesh, results
from agentfem.constitutive.quadrature import QuadratureField


def _domain():
    return mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (2, 1),
        cell_type="quadrilateral",
        comm=MPI.COMM_WORLD,
    )


def _fields(domain, *, logarithmic_volume, pressure, inverse_bulk, deviatoric):
    common = {"degree": 2}
    deformation_gradient = QuadratureField.create(
        domain,
        name="F",
        value_shape=(3, 3),
        **common,
    )
    pressure_field = QuadratureField.create(domain, name="PRESSURE_QP", **common)
    inverse_bulk_field = QuadratureField.create(
        domain,
        name="INV_BULK_MODULUS",
        **common,
    )
    condensed_field = QuadratureField.create(domain, name="ELENER", **common)
    count = len(deformation_gradient.values)

    def selected(values):
        return np.broadcast_to(np.asarray(values, dtype=float), (count,)).copy()

    logj = selected(logarithmic_volume)
    pressure_values = selected(pressure)
    inverse_values = selected(inverse_bulk)
    deviatoric_values = selected(deviatoric)
    gradients = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    gradients[:, 0, 0] = np.exp(logj)
    deformation_gradient.assign(gradients)
    pressure_field.assign(pressure_values)
    inverse_bulk_field.assign(inverse_values)
    condensed_field.assign(
        deviatoric_values + 0.5 * pressure_values**2 * inverse_values
    )
    return (
        deformation_gradient,
        pressure_field,
        inverse_bulk_field,
        condensed_field,
    )


def _quadrature_coordinates(field):
    domain = field.function.function_space.mesh
    cell_map = domain.topology.index_map(domain.topology.dim)
    cells = np.arange(
        int(cell_map.size_local + cell_map.num_ghosts),
        dtype=np.int32,
    )
    return np.asarray(
        fem.Expression(ufl.SpatialCoordinate(domain), field.points).eval(
            domain,
            cells,
        ),
        dtype=float,
    ).reshape((-1, domain.geometry.dim))


def test_uniform_stationary_mixed_energy_recovers_primal_channel():
    domain = _domain()
    logarithmic_volume = 0.12
    bulk_modulus = 25.0
    pressure = bulk_modulus * logarithmic_volume
    deviatoric_energy = 0.7
    fields = _fields(
        domain,
        logarithmic_volume=logarithmic_volume,
        pressure=pressure,
        inverse_bulk=1.0 / bulk_modulus,
        deviatoric=deviatoric_energy,
    )

    diagnostic = results.mixed_j2_elastic_energy_diagnostics(
        deformation_gradient=fields[0],
        pressure=fields[1],
        inverse_bulk_modulus=fields[2],
        condensed_elastic_energy_density=fields[3],
        reference_volume=1.25,
    )

    material_measure = 1.0
    expected = (
        deviatoric_energy
        + 0.5 * bulk_modulus * logarithmic_volume**2
    ) * material_measure / 1.25
    assert diagnostic.primal_elastic_energy_density == pytest.approx(expected)
    assert diagnostic.condensed_elastic_energy_density == pytest.approx(expected)
    assert diagnostic.signed_energy_gap_density == pytest.approx(0.0, abs=1.0e-15)
    assert diagnostic.pressure_orthogonality_density == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert diagnostic.pressure_constraint_defect_energy_density == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert diagnostic.maximum_absolute_pressure_constraint_residual == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert diagnostic.rms_pressure_constraint_residual == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert diagnostic.integrated_reference_measure == pytest.approx(1.0)
    expected_points = int(
        domain.topology.index_map(domain.topology.dim).size_global
    ) * len(fields[0].points)
    assert diagnostic.integration_point_count == expected_points
    assert diagnostic.algebraic_identity_verified


def test_heterogeneous_mixed_energy_preserves_weighted_decomposition():
    domain = _domain()
    reference = QuadratureField.create(
        domain,
        name="REFERENCE",
        degree=2,
        value_shape=(3, 3),
    )
    coordinates = _quadrature_coordinates(reference)
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    logj = 0.03 + 0.07 * x - 0.02 * y
    inverse_bulk = 0.025 + 0.015 * (x > 0.5)
    residual = 0.01 + 0.03 * x - 0.02 * y
    pressure = (logj - residual) / inverse_bulk
    deviatoric = 0.4 + 0.2 * x + 0.1 * y
    fields = _fields(
        domain,
        logarithmic_volume=logj,
        pressure=pressure,
        inverse_bulk=inverse_bulk,
        deviatoric=deviatoric,
    )

    diagnostic = results.mixed_j2_elastic_energy_diagnostics(
        deformation_gradient=fields[0],
        pressure=fields[1],
        inverse_bulk_modulus=fields[2],
        condensed_elastic_energy_density=fields[3],
        reference_volume=2.0,
    )

    owned = len(fields[0].owned_values)
    weights = fields[0].owned_physical_weights()
    owned_logj = logj[:owned]
    owned_pressure = pressure[:owned]
    owned_inverse = inverse_bulk[:owned]
    owned_deviatoric = deviatoric[:owned]
    owned_residual = owned_logj - owned_pressure * owned_inverse
    owned_condensed = (
        owned_deviatoric + 0.5 * owned_pressure**2 * owned_inverse
    )
    owned_primal = owned_deviatoric + 0.5 * owned_logj**2 / owned_inverse

    def normalized_global(values):
        local = float(np.sum(weights * values))
        return float(domain.comm.allreduce(local)) / 2.0

    expected_primal = normalized_global(owned_primal)
    expected_condensed = normalized_global(owned_condensed)
    expected_orthogonality = normalized_global(
        owned_pressure * owned_residual
    )
    expected_defect = normalized_global(
        0.5 * owned_residual**2 / owned_inverse
    )
    assert diagnostic.primal_elastic_energy_density == pytest.approx(
        expected_primal
    )
    assert diagnostic.condensed_elastic_energy_density == pytest.approx(
        expected_condensed
    )
    assert diagnostic.signed_energy_gap_density == pytest.approx(
        expected_primal - expected_condensed
    )
    assert diagnostic.pressure_orthogonality_density == pytest.approx(
        expected_orthogonality
    )
    assert diagnostic.pressure_constraint_defect_energy_density == pytest.approx(
        expected_defect
    )
    assert diagnostic.signed_energy_gap_density == pytest.approx(
        diagnostic.pressure_orthogonality_density
        + diagnostic.pressure_constraint_defect_energy_density
    )
    assert diagnostic.algebraic_identity_residual_density == pytest.approx(
        0.0,
        abs=2.0e-16,
    )
    assert diagnostic.integrated_absolute_algebraic_identity_residual_density == (
        pytest.approx(0.0, abs=2.0e-16)
    )
    assert diagnostic.algebraic_identity_verified
    record = diagnostic.as_dict()
    assert record["schema"] == "agentfem.mixed-j2-elastic-energy-diagnostics"
    assert record["pressure_constraint_residual"] == "ln(J) - p/kappa"


def test_mixed_energy_diagnostic_rejects_unaligned_quadrature_rule():
    domain = _domain()
    fields = list(
        _fields(
            domain,
            logarithmic_volume=0.0,
            pressure=0.0,
            inverse_bulk=0.04,
            deviatoric=0.0,
        )
    )
    pressure = QuadratureField.create(domain, name="PRESSURE_QP", degree=4)
    pressure.assign(np.zeros(len(pressure.values)))
    fields[1] = pressure

    with pytest.raises(
        RuntimeError,
        match="does not share the deformation-gradient quadrature rule",
    ):
        results.mixed_j2_elastic_energy_diagnostics(
            deformation_gradient=fields[0],
            pressure=fields[1],
            inverse_bulk_modulus=fields[2],
            condensed_elastic_energy_density=fields[3],
            reference_volume=1.0,
        )
