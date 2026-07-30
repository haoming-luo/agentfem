from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agentfem import ir, models, public_api, studies


class _IndexMap:
    size_global = 4
    size_local = 4


class _Topology:
    dim = 2

    def index_map(self, _dim):
        return _IndexMap()


class _Mesh:
    topology = _Topology()
    geometry = SimpleNamespace(dim=2)


class _Field:
    name = "Temperature"
    kind = "temperature"

    def __init__(self, mesh):
        self.space = SimpleNamespace(mesh=mesh)

    def summary(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "element": "Lagrange P1",
        }


def test_ir_document_is_deterministic_and_json_safe():
    document = ir.IRDocument(
        document_type="test",
        root={"b": (2, 1), "a": 3.0},
        generator={"version": "test", "name": "AgentFEM"},
    )

    first = document.to_json()
    second = document.to_json()

    assert first == second
    assert json.loads(first)["schema"] == ir.AFIR_SCHEMA


def test_ir_rejects_non_finite_floats():
    with pytest.raises(ir.IRSerializationError):
        ir.to_json_safe({"value": float("nan")})


def test_model_ir_records_scope_backend_and_validation(tmp_path):
    mesh = _Mesh()
    model = models.create(
        study=studies.linear_static(physics="heat_transfer", dimension=2),
        mesh=mesh,
        name="heat",
    )
    model.field(_Field(mesh))

    record = model.to_ir(metadata={"case": "unit_test"})
    output = model.write_ir(tmp_path / "heat.afir.json")

    assert record["schema"] == ir.AFIR_SCHEMA
    assert record["schema_version"] == ir.AFIR_SCHEMA_VERSION
    assert record["status"] == "experimental"
    assert record["root"]["execution_backend"]["name"] == "fenicsx"
    assert record["root"]["validation"]["valid"] is True
    assert record["root"]["mesh"]["reconstructable"] is False
    assert "local_cells" not in record["root"]["mesh"]
    assert record["metadata"]["case"] == "unit_test"
    assert json.loads(output.read_text(encoding="utf-8"))["root"]["name"] == "heat"


def test_runtime_objects_use_typed_opaque_markers():
    class BackendOnlyObject:
        pass

    record = ir.to_json_safe(BackendOnlyObject())

    assert record["kind"] == "opaque_runtime_object"
    assert record["serializable"] is False
    assert "BackendOnlyObject" in record["python_type"]


def test_scientific_values_are_retained_from_backend_constant_wrappers():
    class Constant:
        value = [1.0, -2.5]

    assert ir.describe_value(Constant()) == (1.0, -2.5)


def test_public_inspection_modules_are_discoverable():
    assert {"diagnostics", "ir", "validation"}.issubset(public_api())
