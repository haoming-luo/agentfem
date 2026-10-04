# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem
from mpi4py import MPI

from agentfem import mesh, results
from agentfem.constitutive.quadrature import QuadratureField


def _raw_scalar_field(domain):
    field = QuadratureField.create(
        domain,
        name="PEEQ",
        degree=2,
    )
    coordinates = field.owned_physical_points.reshape((-1, domain.geometry.dim))
    values = coordinates[:, 0] + 2.0 * coordinates[:, 1]
    field.assign(values)
    return field, values


def test_portable_integration_point_output_preserves_raw_values_and_semantics(
    tmp_path,
):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (2.0, 1.0),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    field, _ = _raw_scalar_field(domain)
    simulation = results.SimulationResult("raw-quadrature")
    simulation.add_field(
        "PEEQ",
        field.function,
        unit="1",
        location="quadrature_points",
        description="Equivalent plastic strain.",
        processing={"representation": "quadrature_values", "committed": True},
        sampling=field,
    )

    artifacts = results.write_integration_point_fields(
        simulation,
        tmp_path / "integration-points.h5",
    )
    restored = results.read_integration_point_fields(artifacts.hdf5)

    assert artifacts.field_names == ("PEEQ",)
    assert artifacts.rule_count == 1
    assert restored["schema"] == "agentfem.integration-point-dataset.v1"
    assert restored["mesh_identity"]["schema"] == (
        "agentfem.mesh-portable-identity.v2"
    )
    rule = restored["rules"][0]
    assert np.unique(rule["cell_id"]).size == len(rule["cell_id"])
    assert np.all(np.diff(rule["cell_id"]) > 0)
    assert rule["coordinates"].shape[:2] == rule["fields"]["PEEQ"].shape
    np.testing.assert_allclose(
        rule["fields"]["PEEQ"].reshape(-1),
        (
            rule["coordinates"][..., 0]
            + 2.0 * rule["coordinates"][..., 1]
        ).reshape(-1),
    )
    assert np.sum(rule["physical_weights"]) == pytest.approx(2.0)
    assert rule["field_metadata"]["PEEQ"]["unit"] == "1"
    assert rule["field_metadata"]["PEEQ"]["processing"]["committed"] is True


def test_completed_result_output_writes_xdmf_view_and_raw_quadrature_source(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    displacement = fem.Function(
        fem.functionspace(domain, ("Lagrange", 1, (2,))),
        name="Displacement",
    )
    raw, _ = _raw_scalar_field(domain)
    simulation = results.SimulationResult("dual-output")
    simulation.add_field(
        "Displacement",
        displacement,
        unit="m",
        processing={"method": "primary_finite_element_solution"},
    )
    simulation.add_field(
        "PEEQ",
        raw.function,
        unit="1",
        location="quadrature_points",
        processing={"representation": "quadrature_values"},
        sampling=raw,
    )

    from agentfem.results.output import attach_result_field_output

    attach_result_field_output(simulation, tmp_path / "fields.xdmf", strict=True)

    raw_path = tmp_path / "fields.integration-points.h5"
    assert (tmp_path / "fields.xdmf").exists()
    assert raw_path.exists()
    assert simulation.artifacts["integration_points_hdf5"] == raw_path
    assert simulation.metadata["integration_point_output"][
        "partition_independent"
    ] is True
    assert "PEEQ" in simulation.metadata["field_output_fields"]["omitted"]


def test_raw_integration_point_output_fails_closed_without_sampling(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    raw, _ = _raw_scalar_field(domain)
    simulation = results.SimulationResult("missing-sampling")
    simulation.add_field(
        "PEEQ",
        raw.function,
        location="quadrature_points",
    )

    with pytest.raises(ValueError, match="sampling contract"):
        results.write_integration_point_fields(
            simulation,
            tmp_path / "integration-points.h5",
        )
