from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agentfem import cli, extensions, project


class _EntryPoints(tuple):
    def select(self, *, group):
        return self if group == extensions.ENTRY_POINT_GROUP else _EntryPoints()


class _EntryPoint:
    def __init__(self, name, value, payload, *, distribution="private-pack", version="1.2.3"):
        self.name = name
        self.value = value
        self._payload = payload
        self.loads = 0
        self.dist = SimpleNamespace(metadata={"Name": distribution}, version=version)

    def load(self):
        self.loads += 1
        return self._payload


@pytest.fixture(autouse=True)
def _isolated_extensions(monkeypatch):
    extensions._LOADED.clear()
    monkeypatch.setattr(extensions.metadata, "entry_points", lambda: _EntryPoints())
    yield
    extensions._LOADED.clear()


def test_discovery_is_lazy_and_activation_is_explicit(monkeypatch):
    calls = []
    extension = extensions.Extension(
        spec=extensions.ExtensionSpec(
            name="company-solids",
            version="1.2.3",
            capabilities=("company.material-model",),
        ),
        register=lambda context: calls.append(context.extension.name),
    )
    entry_point = _EntryPoint("company-solids", "company_agentfem:extension", extension)
    monkeypatch.setattr(
        extensions.metadata,
        "entry_points",
        lambda: _EntryPoints((entry_point,)),
    )

    descriptors = extensions.discover_extensions()
    assert descriptors[0].distribution == "private-pack"
    assert descriptors[0].loaded is False
    assert entry_point.loads == 0
    assert calls == []

    loaded = extensions.load_extension("company-solids")
    assert loaded.spec.capabilities == ("company.material-model",)
    assert loaded.registrations == {
        "step_providers": (),
        "backends": (),
        "materials": (),
    }
    assert entry_point.loads == 1
    assert calls == ["company-solids"]
    assert extensions.load_extension("company-solids") is loaded
    assert entry_point.loads == 1


def test_incompatible_extension_api_fails_before_registration(monkeypatch):
    calls = []
    extension = extensions.Extension(
        spec=extensions.ExtensionSpec(
            name="future-extension",
            version="1.0.0",
            api_version=99,
        ),
        register=lambda context: calls.append(context),
    )
    monkeypatch.setattr(
        extensions.metadata,
        "entry_points",
        lambda: _EntryPoints((_EntryPoint("future-extension", "future:extension", extension),)),
    )
    with pytest.raises(extensions.ExtensionError, match="supports 1"):
        extensions.load_extension("future-extension")
    assert calls == []


def test_registration_plan_rejects_duplicate_staged_names():
    context = extensions.ExtensionContext(
        extensions.ExtensionSpec(name="duplicate-test", version="1.0")
    )
    context.add_backend("duplicate", lambda: object())
    context.add_backend("duplicate", lambda: object())
    with pytest.raises(extensions.ExtensionError, match="duplicate backend"):
        context.commit()


def test_project_declares_required_extension_without_importing_it(tmp_path, monkeypatch):
    (tmp_path / "case.py").write_text("print('private workflow')\n", encoding="utf-8")
    (tmp_path / "agentfem.toml").write_text(
        """[project]
name = "private-workflow"
entrypoint = "case.py"

[extensions]
required = ["company-solids"]
""",
        encoding="utf-8",
    )
    config = project.ProjectConfig.load(tmp_path)
    assert config.extensions == ("company-solids",)
    assert "company-solids" in config.check()[0]

    extension = extensions.Extension(
        extensions.ExtensionSpec(name="company-solids", version="1.0"),
        lambda context: None,
    )
    entry_point = _EntryPoint("company-solids", "company_agentfem:extension", extension)
    monkeypatch.setattr(
        extensions.metadata,
        "entry_points",
        lambda: _EntryPoints((entry_point,)),
    )
    assert config.check() == ()
    assert entry_point.loads == 0


def test_execution_record_captures_activated_extension_identity(tmp_path, monkeypatch):
    extension = extensions.Extension(
        extensions.ExtensionSpec(
            name="traceable-extension",
            version="2.0",
            capabilities=("traceable.workflow",),
        ),
        lambda context: None,
    )
    monkeypatch.setattr(
        extensions.metadata,
        "entry_points",
        lambda: _EntryPoints((_EntryPoint("traceable-extension", "traceable:extension", extension),)),
    )
    extensions.load_extension("traceable-extension")
    config = project.ProjectConfig(
        root=tmp_path,
        name="traceable",
        entrypoint=tmp_path / "case.py",
        output_directory=tmp_path / "outputs",
    )
    run = project.RunContext.create(config, run_id="extension-run").prepare()
    run.write_execution("completed", structured_result=False)

    record = json.loads(run.execution_path.read_text(encoding="utf-8"))
    assert record["extensions"][0]["name"] == "traceable-extension"
    assert record["extensions"][0]["distribution"] == "private-pack"


def test_project_run_activates_declared_extension_and_records_it(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "case.py").write_text("print('extended case')\n", encoding="utf-8")
    (tmp_path / "agentfem.toml").write_text(
        """[project]
name = "extended-run"
entrypoint = "case.py"

[extensions]
required = ["company-solids"]
""",
        encoding="utf-8",
    )
    extension = extensions.Extension(
        extensions.ExtensionSpec(name="company-solids", version="3.0"),
        lambda context: None,
    )
    entry_point = _EntryPoint(
        "company-solids", "company_agentfem:extension", extension
    )
    monkeypatch.setattr(
        extensions.metadata,
        "entry_points",
        lambda: _EntryPoints((entry_point,)),
    )

    assert cli.main(
        [
            "run",
            "--project",
            str(tmp_path),
            "--run-id",
            "extension-cli",
            "--json",
        ]
    ) == 0
    record = json.loads(capsys.readouterr().out)
    execution = json.loads(
        (tmp_path / record["execution_record"]).read_text(encoding="utf-8")
        if not str(record["execution_record"]).startswith("/")
        else open(record["execution_record"], encoding="utf-8").read()
    )
    assert execution["extensions"][0]["name"] == "company-solids"
    assert entry_point.loads == 1


def test_extensions_cli_has_stable_empty_json(capsys):
    assert cli.main(["extensions", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["schema"] == "agentfem.extensions"
    assert record["installed"] == []
    assert record["loaded"] == []
