# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import tomllib

import extension_gate


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "examples" / "extensions" / "reference_material"


def test_installed_core_hash_ignores_only_runtime_bytecode(tmp_path):
    package = tmp_path / "agentfem"
    package.mkdir()
    (package / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    before = extension_gate._tree_sha256(package)

    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "core.cpython-311.pyc").write_bytes(b"runtime cache")
    assert extension_gate._tree_sha256(package) == before

    (package / "core.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert extension_gate._tree_sha256(package) != before


def test_reference_extension_is_a_separate_distribution_and_entry_point():
    configuration = tomllib.loads((REFERENCE / "pyproject.toml").read_text())
    project = configuration["project"]

    assert project["name"] == "agentfem-reference-material"
    assert project["dependencies"] == ["agentfem>=0.3.8.dev0"]
    assert configuration["project"]["entry-points"]["agentfem.extensions"] == {
        "agentfem-reference-material": "agentfem_reference_material:extension"
    }
    assert not (REFERENCE / "src" / "agentfem").exists()


def test_reference_project_declares_the_extension_without_source_paths():
    configuration = tomllib.loads(
        (REFERENCE / "project" / "agentfem.toml").read_text()
    )
    source = (REFERENCE / "project" / "case.py").read_text(encoding="utf-8")

    assert configuration["extensions"]["required"] == [
        "agentfem-reference-material"
    ]
    assert "sys.path" not in source
    assert "agentfem_reference_material" not in source
