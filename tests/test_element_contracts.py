from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import elements, fields, mesh, models, studies


def _solid_model(*, cell_type="triangle", dim=2, field_dim=None):
    if dim == 2:
        domain = mesh.rectangle(
            (0.0, 0.0),
            (1.0, 1.0),
            (1, 1),
            comm=MPI.COMM_SELF,
            cell_type=cell_type,
        )
    else:
        domain = mesh.cuboid(
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (1, 1, 1),
            comm=MPI.COMM_SELF,
            cell_type=cell_type,
        )
    study = studies.static_solid(
        dimension=dim,
        assumption="plane_strain" if dim == 2 else "three_dimensional",
    )
    model = models.create(study=study, mesh=domain, name="element_contract")
    model.field(fields.displacement(domain, dim=field_dim))
    return model


def test_runtime_topology_contract_separates_inspection_from_formulation():
    verified = mesh.describe_topology_compatibility("tetrahedron")
    verified_prism = mesh.describe_topology_compatibility("wedge")
    conditional = mesh.describe_topology_compatibility("pyramid")
    blocked = mesh.describe_topology_compatibility("polyhedron42")

    assert verified.release_ready
    assert verified.quality_metric == "simplex_mean_ratio"
    assert "tetra10" in verified.source_cell_types
    assert {
        item.name for item in verified.capabilities if item.verified
    } >= {
        "topology_inspection",
        "geometry_quality",
        "conforming_p1_patch",
        "quadratic_geometry_preflight",
    }
    assert verified_prism.release_ready
    assert verified_prism.topology == "prism"
    assert conditional.inspectable
    assert not conditional.release_ready
    assert conditional.topology == "pyramid"
    assert not blocked.inspectable


def test_element_identity_preserves_vector_and_mixed_subelements():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    displacement = fields.displacement(domain, degree=2)
    mixed = fields.displacement_pressure(
        domain,
        displacement_degree=2,
        pressure_family="DPC",
        pressure_degree=1,
    )

    displacement_identity = elements.describe_element(displacement.space)
    mixed_identity = elements.describe_element(mixed.space)

    assert displacement_identity.cell == "quadrilateral"
    assert displacement_identity.value_shape == (2,)
    assert displacement_identity.degree == 2
    assert not displacement_identity.mixed
    assert displacement_identity.embedded_subdegree == 2
    assert displacement_identity.embedded_superdegree == 2
    assert displacement_identity.mapping == "identity"
    assert mixed_identity.mixed
    assert mixed_identity.embedded_subdegree is None
    assert mixed_identity.embedded_superdegree == 2
    assert len(mixed_identity.subelements) == 2
    assert mixed_identity.subelements[0].value_shape == (2,)
    assert mixed_identity.subelements[1].discontinuous
    assert mixed_identity.subelements[1].polyset_type == "standard"


def test_discretization_audit_combines_topology_element_and_quality():
    model = _solid_model(cell_type="triangle")

    report = elements.audit(
        model,
        check_quality=True,
        quality_threshold=0.8,
    )

    assert report.acceptable
    assert report.topology is not None
    assert report.topology.release_ready
    assert report.geometry.cell_type == "triangle"
    assert report.geometry.degree == 1
    assert report.quality.acceptable
    assert report.fields[0].element.value_shape == (2,)
    assert report.summary()["quality"]["metric"] == "simplex_mean_ratio"


def test_discretization_quality_audit_is_mpi_collective():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (2.0, 1.0),
        (2, 1),
        comm=MPI.COMM_WORLD,
        cell_type="triangle",
    )
    study = studies.static_solid(dimension=2, assumption="plane_strain")
    model = models.create(study=study, mesh=domain, name="distributed_preflight")
    model.field(fields.displacement(domain))

    report = elements.audit(model, check_quality=True, quality_threshold=0.8)

    assert report.acceptable
    assert report.quality.global_cells == 4
    assert report.quality.minimum == pytest.approx(np.sqrt(3.0) / 2.0)


def test_discretization_audit_rejects_wrong_displacement_value_shape():
    model = _solid_model(cell_type="triangle", field_dim=3)

    report = elements.audit(model)

    assert not report.validation.is_valid
    mismatch = next(
        item
        for item in report.validation.errors
        if item.code == "AFM-DISCRETIZATION-006"
    )
    assert mismatch.path.endswith("value_shape")
    assert "requires value shape (2,)" in mismatch.message

    model_report = model.validate(target=model.fields[0])
    assert "AFM-DISCRETIZATION-006" in {
        item.code for item in model_report.errors
    }


def test_mesh_quality_policy_can_warn_or_fail_without_changing_metric():
    model = _solid_model(cell_type="quadrilateral")
    domain = model.domain
    domain.geometry.x[domain.geometry.x[:, 1] > 0.5, 0] += 0.8

    warning = elements.audit(
        model,
        check_quality=True,
        quality_threshold=0.9,
    )
    rejected = elements.audit(
        model,
        check_quality=True,
        quality_threshold=0.9,
        reject_poor_quality=True,
    )

    assert warning.validation.is_valid
    assert [item.code for item in warning.validation.warnings] == [
        "AFM-MESH-QUALITY-003"
    ]
    assert not rejected.validation.is_valid
    assert rejected.quality.minimum == pytest.approx(
        1.0 / np.sqrt(1.0 + 0.8**2)
    )


def test_mesh_summary_exposes_runtime_cell_and_coordinate_degree():
    model = _solid_model(cell_type="triangle")

    summary = mesh.summarize_mesh(model.domain).as_dict()

    assert summary["cell_type"] == "triangle"
    assert summary["geometry_degree"] == 1
    assert summary["geometry"]["basis_family"] == "P"
    assert summary["geometry"]["nodes_per_cell"] == 3
