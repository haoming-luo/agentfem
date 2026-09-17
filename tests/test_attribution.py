import json
from pathlib import Path
import tomllib

from agentfem import provenance


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "agentfem"


def test_packaged_origin_is_the_single_runtime_origin_record():
    stored = json.loads((SOURCE / "origin.json").read_text(encoding="utf-8"))

    assert provenance.ORIGIN == stored
    assert stored["project"] == "AgentFEM"
    assert stored["initiated_by"] == "Haoming Luo"
    assert stored["repository"] == "https://github.com/haoming-luo/agentfem"
    assert stored["license"] == "Apache-2.0"
    assert stored["citation_file"] == "CITATION.cff"
    assert stored["release_verification"].startswith("gh attestation verify")


def test_every_packaged_python_source_carries_spdx_identity():
    copyright_tag = "SPDX-FileCopyright" + "Text: 2026 Haoming Luo"
    license_tag = "SPDX-License-" + "Identifier: Apache-2.0"
    missing = []
    for path in sorted(SOURCE.rglob("*.py")):
        header = "\n".join(path.read_text(encoding="utf-8").splitlines()[:6])
        if copyright_tag not in header:
            missing.append(str(path.relative_to(ROOT)))
        if license_tag not in header:
            missing.append(str(path.relative_to(ROOT)))

    assert not missing, "missing embedded SPDX identity: " + ", ".join(missing)


def test_repository_origin_contract_uses_open_standards_and_attestation():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    reuse = tomllib.loads((ROOT / "REUSE.toml").read_text(encoding="utf-8"))
    workflow = (ROOT / ".github/workflows/publish-pypi.yml").read_text(
        encoding="utf-8"
    )
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")

    assert project["project"]["license"] == "Apache-2.0"
    assert reuse["version"] == 1
    assert any(
        item.get("SPDX-License-Identifier") == "Apache-2.0"
        for item in reuse["annotations"]
    )
    assert "actions/attest-build-provenance" in workflow
    assert "attestations: write" in workflow
    assert "https://github.com/haoming-luo/agentfem" in notice
    assert (ROOT / "LICENSES/Apache-2.0.txt").read_bytes() == (
        ROOT / "LICENSE"
    ).read_bytes()
