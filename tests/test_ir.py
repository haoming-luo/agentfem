from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from agentfem import ir, models, public_api, studies
from agentfem import _model_inspection


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


def test_ir_preserves_declared_units_and_engineering_stages_without_solving():
    from agentfem import units

    model = models.create(study=studies.static_solid(dimension=3),
                          units=units.n_mm_mpa(), name="semantic-review")
    loading = model.stage("loading", inherit_model_loads=True)
    unloading = model.stage("unloading", previous=loading, inherit_model_loads=False)
    record = model.to_ir(include_validation=False)
    root = record["root"]
    assert root["unit_system"] == model.unit_system.summary()
    assert root["unit_system"]["base"]["length"] == "mm"
    assert root["unit_system"]["automatic_conversion"] is False
    assert root["engineering_steps"] == [ir.to_json_safe(loading.summary()),
                                         ir.to_json_safe(unloading.summary())]
    assert root["steps"] == []
    assert "validation" not in root
    assert list(ir.WORKFLOW_ORDER).index("engineering_steps") < list(ir.WORKFLOW_ORDER).index("steps")
    assert record == json.loads(json.dumps(record, allow_nan=False))


def test_ir_does_not_infer_missing_units_or_stages():
    model = models.create(study=studies.static_solid(dimension=3))
    root = model.to_ir(include_validation=False)["root"]
    assert root["unit_system"] is None
    assert root["engineering_steps"] == []


@pytest.mark.parametrize("mapping", [{1: "numeric", "1": "text"},
                                     {False: "boolean", "False": "text"}])
def test_ir_rejects_lossy_mapping_key_conversion(mapping):
    with pytest.raises(ir.IRSerializationError, match="distinct mapping keys"):
        ir.to_json_safe(mapping, path="metadata")


def test_ir_canonicalizes_unordered_sets_but_preserves_sequences():
    values = {"alpha", "zeta", 4, (1, 2)}
    converted = ir.to_json_safe(values)
    assert converted == sorted(converted, key=lambda x: json.dumps(
        x, sort_keys=True, ensure_ascii=False, allow_nan=False))
    assert ir.to_json_safe(frozenset(values)) == converted
    assert ir.to_json_safe(["zeta", "alpha"]) == ["zeta", "alpha"]


def test_model_ir_records_scope_backend_and_validation(tmp_path):
    mesh = _Mesh()
    model = models.create(
        study=studies.linear_static(physics="heat_transfer", dimension=2),
        mesh=mesh,
        name="heat",
    )
    model.field(_Field(mesh))
    model.material(SimpleNamespace(conductivity=1.0))

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


def test_distributed_ir_builds_collective_document_on_nonroot(monkeypatch, tmp_path):
    calls = []

    class Comm:
        rank = 1

        def bcast(self, value, root):
            calls.append("bcast")
            return value

    fake_model = SimpleNamespace(mesh=SimpleNamespace(comm=Comm()))

    def build(*_args, **_kwargs):
        calls.append("build")
        return {"root": {"name": "distributed"}}

    monkeypatch.setattr(_model_inspection, "model_to_ir", build)
    monkeypatch.setattr(
        ir,
        "write_document",
        lambda *_args, **_kwargs: calls.append("write"),
    )

    output = _model_inspection.write_model_ir(
        fake_model,
        tmp_path / "distributed.afir.json",
    )

    assert output == tmp_path / "distributed.afir.json"
    assert calls == ["build", "bcast"]


@pytest.mark.parametrize("rank", [0, 1])
def test_ir_write_failure_reaches_all_ranks(monkeypatch, tmp_path, rank):
    class Comm:
        def bcast(self, value, root):
            assert root == 0
            if rank == 0:
                assert value == "PermissionError: read-only output"
            return "PermissionError: read-only output"

    comm = Comm()
    comm.rank = rank
    model = SimpleNamespace(mesh=SimpleNamespace(comm=comm))
    monkeypatch.setattr(_model_inspection, "model_to_ir", lambda *a, **kw: {})

    def fail_write(*args):
        assert rank == 0
        raise PermissionError("read-only output")

    monkeypatch.setattr(ir, "write_document", fail_write)
    with pytest.raises(RuntimeError, match="AFM-IR-WRITE.*read-only output"):
        _model_inspection.write_model_ir(model, tmp_path / "model.json")


def test_inspection_views_include_declared_engineering_context():
    from agentfem import units

    model = models.create(study=studies.static_solid(dimension=3), units=units.si())
    model.stage("loading")
    for name in ("unit_system", "amplitudes", "engineering_steps"):
        assert name in model.manifest()["workflow_order"]
        assert f"  {name}:" in model.tree()


def test_ir_write_failure_real_mpi(monkeypatch, tmp_path):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires two MPI ranks")
    model = SimpleNamespace(mesh=SimpleNamespace(comm=comm))
    monkeypatch.setattr(_model_inspection, "model_to_ir", lambda *a, **kw: {})
    # An existing directory is an invalid file destination on every platform.
    destination = comm.bcast(str(tmp_path) if comm.rank == 0 else None, root=0)
    with pytest.raises(RuntimeError, match="AFM-IR-WRITE") as caught:
        _model_inspection.write_model_ir(model, destination)
    errors = comm.allgather(str(caught.value))
    assert len(set(errors)) == 1


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


def test_public_api_exposes_progressive_discovery_levels():
    core = set(public_api("core"))
    advanced = set(public_api("advanced"))
    expert = set(public_api("expert"))

    assert {"studies", "mesh", "models", "fields", "results"} <= core
    assert {"fracture", "campaigns", "learning", "surrogates"} <= advanced
    assert {"ir", "diagnostics", "validation"} <= expert
    assert core | advanced | expert == set(public_api())
    with pytest.raises(ValueError, match="core, advanced, expert, or all"):
        public_api("beginnerish")


def test_model_api_separates_daily_advanced_and_compatibility_vocabulary():
    core = set(models.model_api("core"))
    advanced = set(models.model_api("advanced"))
    compatibility = set(models.model_api("compatibility"))

    assert {"field", "material", "clamp", "traction", "step", "check"} <= core
    assert {"stiffness", "internal_force", "to_ir"} <= advanced
    assert {"add_material", "linear_static_step", "creep_step"} <= compatibility
    assert not core.intersection(advanced)
    assert not core.intersection(compatibility)
    assert not advanced.intersection(compatibility)
    assert core | advanced | compatibility == set(models.model_api("all"))
    public_methods = {
        name
        for name, value in vars(models.Model).items()
        if callable(value) and not name.startswith("_")
    }
    assert public_methods == set(models.model_api("all"))
    with pytest.raises(ValueError, match="core, advanced, compatibility, or all"):
        models.model_api("beginnerish")


def test_model_api_contract_declares_compatibility_without_silent_rewrite():
    records = {item["name"]: item for item in models.model_api_contract()}

    assert records["step"]["lifecycle"] == "recommended"
    legacy = records["hyperelastic_step"]
    assert legacy["lifecycle"] == "compatibility"
    assert legacy["replacement"] == "model.step(...)"
    assert legacy["semantic_review"] is True
    assert legacy["automatic_rewrite"] is False


def test_package_api_tiers_are_disjoint_complete_and_importable():
    import agentfem

    tiers = tuple(set(agentfem.public_api(level)) for level in ("core", "advanced", "expert"))
    assert not tiers[0].intersection(tiers[1])
    assert not tiers[0].intersection(tiers[2])
    assert not tiers[1].intersection(tiers[2])
    assert set(agentfem.public_api()) == set().union(*tiers)
    assert set(agentfem.public_api()) <= set(agentfem.__all__)
    for name in agentfem.public_api():
        assert getattr(agentfem, name).__name__ == f"agentfem.{name}"


def test_model_step_signature_exposes_common_cross_physics_inputs():
    parameters = inspect.signature(models.Model.step).parameters

    assert {
        "target",
        "procedure",
        "material",
        "incrementation",
        "dt",
        "steps",
        "duration",
        "solver_options",
        "output",
        "progress",
        "checkpoint",
    } <= set(parameters)
    assert all(
        parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in parameters
        if name not in {"self", "kwargs"}
    )
