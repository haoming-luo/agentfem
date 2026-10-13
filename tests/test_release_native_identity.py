"""Release identity must include native artifacts absent from source checkouts."""
import importlib.util
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest


def test_native_wheel_identity_rejects_modified_artifact(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "release_native_gate", Path(__file__).parents[1] / "release_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    source = tmp_path / "source"
    installed = tmp_path / "installed" / "agentfem"
    source.mkdir()
    installed.mkdir(parents=True)
    (source / "__init__.py").write_text("# package\n")
    (installed / "__init__.py").write_text("# package\n")
    native = "_kernel" + EXTENSION_SUFFIXES[0]
    (installed / native).write_bytes(b"tested binary")
    wheel = tmp_path / "candidate.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path in installed.iterdir():
            archive.write(path, "agentfem/" + path.name)
    monkeypatch.setattr(gate, "SOURCE_PACKAGE", source)
    monkeypatch.setattr(gate, "check_versions", lambda: "1")
    monkeypatch.setattr(gate.metadata, "distribution", lambda name: SimpleNamespace(
        version="1", locate_file=lambda name: installed))
    monkeypatch.setattr(gate.importlib, "import_module", lambda name: SimpleNamespace(
        __file__=str(installed / "__init__.py")))
    gate._check_installed_identity(wheel=wheel)
    (installed / native).write_bytes(b"different binary")
    with pytest.raises(RuntimeError, match="different runtime files"):
        gate._check_installed_identity(wheel=wheel)
