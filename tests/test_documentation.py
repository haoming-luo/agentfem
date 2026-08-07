from __future__ import annotations

import json
from pathlib import Path

import build_docs


ROOT = Path(__file__).resolve().parents[1]


def test_documentation_machine_entrypoints_are_current():
    assert (ROOT / "docs" / "agentfem.json").read_text() == (
        build_docs.render_agent_manifest()
    )
    assert (ROOT / "docs" / "llms.txt").read_text() == build_docs.render_llms_entry()

    manifest = json.loads(build_docs.render_agent_manifest())
    assert manifest["schema"] == "agentfem.documentation-entry"
    assert manifest["version"] == build_docs.project_version()
    assert manifest["human_entrypoints"]["examples"] == "examples/"
    assert manifest["agent_entrypoints"]["guide"] == "agents/"
    assert "models" in manifest["public_workflow_modules"]


def test_generated_api_covers_public_workflow_objects():
    reference = build_docs.render_api_reference()
    assert "## `agentfem.studies`" in reference
    assert "linear_static" in reference
    assert "## `agentfem.models`" in reference
    assert "create" in reference
    assert "## `agentfem.mesh`" in reference
    assert "rectangle" in reference
    assert "## `agentfem.results`" in reference
    assert "SimulationResult" in reference


def test_site_navigation_uses_scientific_manual_structure():
    config = (ROOT / "mkdocs.yml").read_text()
    for section in (
        "Introduction",
        "Getting Started",
        "User Guide",
        "Examples",
        "Theory and Reference",
        "Extending AgentFEM",
        "Project",
    ):
        assert f"  - {section}:" in config
    assert "Engineering Notes:" in config
    assert "Theory and Conventions: reference/theory_and_conventions.md" in config
    assert "navigation.tabs" not in config
    assert "navigation.sections" in config
    assert "search.suggest" in config
    assert "pymdownx.arithmatex" in config


def test_theory_reference_states_equations_and_result_locations():
    theory = (ROOT / "docs" / "reference" / "theory_and_conventions.md").read_text()
    assert "## Static equilibrium" in theory
    assert "## Structural dynamics" in theory
    assert "## J2 plasticity and creep state" in theory
    assert "## Result locations and recovery" in theory
    assert r"\mathbf{M}\ddot{\mathbf{u}}" in theory
    assert "Integration points" in theory
