from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentfem import models, studies
from agentfem.validation import ModelValidationError


class _IndexMap:
    size_global = 8
    size_local = 8


class _Topology:
    dim = 2

    def index_map(self, _dim):
        return _IndexMap()


class _Mesh:
    topology = _Topology()
    geometry = SimpleNamespace(dim=2)


class _Field:
    def __init__(self, mesh, name="Temperature", kind="temperature"):
        self.name = name
        self.kind = kind
        self.space = SimpleNamespace(mesh=mesh)

    def summary(self):
        return {"name": self.name, "kind": self.kind, "element": "Lagrange P1"}


def test_model_validation_returns_addressable_issues():
    model = models.Model(study=None, mesh=None, name="incomplete")

    report = model.validate()

    assert not report.is_valid
    assert {item.code for item in report.errors} == {
        "AFM-MODEL-001",
        "AFM-MODEL-002",
        "AFM-MODEL-004",
    }
    assert all(item.path.startswith("model.") for item in report.errors)
    with pytest.raises(ModelValidationError) as error:
        model.check()
    assert error.value.report is report or error.value.report.as_dict() == report.as_dict()


def test_valid_heat_model_has_no_structural_errors():
    mesh = _Mesh()
    study = studies.linear_static(
        physics="heat_transfer",
        dimension=2,
        name="steady_heat",
    )
    model = models.create(study=study, mesh=mesh, name="heat")
    model.field(_Field(mesh))
    model.material(SimpleNamespace(conductivity=1.0))

    report = model.validate()

    assert report.is_valid
    model.check()


def test_duplicate_names_are_warnings_not_execution_errors():
    mesh = _Mesh()
    study = studies.linear_static(physics="heat_transfer", dimension=2)
    model = models.create(study=study, mesh=mesh)
    model.field(_Field(mesh, name="T"))
    model.field(_Field(mesh, name="T"))
    model.material(SimpleNamespace(conductivity=1.0))

    report = model.validate()

    assert report.is_valid
    assert [warning.code for warning in report.warnings] == ["AFM-NAME-001"]


def test_heat_validation_rejects_missing_material_before_solve():
    mesh = _Mesh()
    model = models.create(
        study=studies.steady_heat_transfer(dimension=2),
        mesh=mesh,
        name="missing_heat_material",
    )
    model.field(_Field(mesh))

    report = model.validate()

    assert not report.is_valid
    assert {item.code for item in report.errors} == {
        "AFM-STUDY-002",
        "AFM-MATERIAL-001",
    }


def test_heat_validation_rejects_a_displacement_target():
    mesh = _Mesh()
    model = models.create(
        study=studies.steady_heat_transfer(dimension=2),
        mesh=mesh,
        name="wrong_heat_field",
    )
    model.field(_Field(mesh, name="Displacement", kind="displacement"))
    model.material(SimpleNamespace(conductivity=1.0))

    report = model.validate()

    assert not report.is_valid
    capability = next(
        item.context["capability"]
        for item in report.errors
        if item.code == "AFM-STUDY-002"
    )
    assert capability["target"]["kind"] == "displacement"


def test_capability_selects_the_compatible_field_in_a_multifield_model():
    mesh = _Mesh()
    model = models.create(
        study=studies.steady_heat_transfer(dimension=2),
        mesh=mesh,
        name="sequential_fields",
    )
    model.field(_Field(mesh, name="Displacement", kind="displacement"))
    model.field(_Field(mesh, name="Temperature", kind="temperature"))
    model.material(SimpleNamespace(conductivity=1.0))

    report = model.validate()

    assert report.is_valid
    capability = models.step_capability(model)
    assert capability["target"]["kind"] == "temperature"


def test_explicit_linear_operators_are_a_valid_material_free_expert_path():
    mesh = _Mesh()
    model = models.create(
        study=studies.steady_heat_transfer(dimension=2),
        mesh=mesh,
        name="operator_defined_heat",
    )
    target = model.field(_Field(mesh))

    report = model.validate(
        target=target,
        step_options={"K": object(), "F": object()},
    )

    assert report.is_valid
