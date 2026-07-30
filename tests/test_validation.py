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
    def __init__(self, mesh, name="Temperature"):
        self.name = name
        self.kind = "temperature"
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

    report = model.validate()

    assert report.is_valid
    model.check()


def test_duplicate_names_are_warnings_not_execution_errors():
    mesh = _Mesh()
    study = studies.linear_static(physics="heat_transfer", dimension=2)
    model = models.create(study=study, mesh=mesh)
    model.field(_Field(mesh, name="T"))
    model.field(_Field(mesh, name="T"))

    report = model.validate()

    assert report.is_valid
    assert [warning.code for warning in report.warnings] == ["AFM-NAME-001"]
