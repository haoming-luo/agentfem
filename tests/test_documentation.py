from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

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
    assert "models" in manifest["public_api"]["core"]
    assert "learning" in manifest["public_api"]["advanced"]
    assert "surrogates" in manifest["public_api"]["advanced"]
    assert "backends" in manifest["public_api"]["expert"]
    assert "step" in manifest["model_api"]["core"]
    assert "stiffness" in manifest["model_api"]["advanced"]
    assert "linear_static_step" in manifest["model_api"]["compatibility"]


def test_agentfem_skill_is_installable_and_routes_context_progressively():
    skill_dir = ROOT / "skills" / "agentfem"
    skill = (skill_dir / "SKILL.md").read_text()
    interface = (skill_dir / "agents" / "openai.yaml").read_text()

    assert skill.startswith("---\nname: agentfem\ndescription:")
    assert "\n---\n\n# AgentFEM\n" in skill
    assert "## Reference Routing" in skill
    for reference in (
        "workflow.md",
        "concepts.md",
        "module_map.md",
        "validation.md",
        "extension_rules.md",
    ):
        assert f"`references/{reference}`" in skill
        assert (skill_dir / "references" / reference).is_file()

    assert 'display_name: "AgentFEM"' in interface
    assert "$agentfem" in interface
    assert "allow_implicit_invocation: true" in interface

    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "recursive-include skills *.md *.yaml" in manifest


def test_knowledge_import_check_uses_the_current_checkout():
    completed = subprocess.run(
        [sys.executable, "build_knowledge.py", "--check", "--check-imports"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Validated 29 scientific function cards." in completed.stdout


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
    assert "## `agentfem.learning`" in reference
    assert "NeuralFieldSpec" in reference


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
    assert "Engineering Notes:" not in config
    assert "project/engineering_notes.md" in config
    assert "not_in_nav:" in config
    assert "Theory and Conventions: reference/theory_and_conventions.md" in config
    assert "navigation.tabs" not in config
    assert "navigation.instant" in config
    assert "navigation.instant.progress" in config
    assert "navigation.sections" not in config
    assert "navigation.indexes" in config
    assert "navigation.prune" in config
    assert "search.suggest" in config
    assert "pymdownx.arithmatex" in config
    assert "      - examples/index.md" in config
    assert config.index("Mesh Interoperability: mesh_interoperability.md") < config.index(
        "Results and Data:"
    )


def test_math_rendering_survives_instant_navigation_and_late_startup():
    config = (ROOT / "mkdocs.yml").read_text()
    script = (ROOT / "docs" / "javascripts" / "mathjax.js").read_text()
    stylesheet = (ROOT / "docs" / "stylesheets" / "extra.css").read_text()
    runtime = ROOT / "docs" / "vendor" / "mathjax" / "3.2.2"
    assert config.index("javascripts/mathjax.js") < config.index(
        "vendor/mathjax/3.2.2/tex-chtml-full.js"
    )
    assert "cdn.jsdelivr.net/npm/mathjax" not in config
    assert "unpkg.com/mathjax" not in config
    assert (runtime / "tex-chtml-full.js").stat().st_size > 1_000_000
    assert len(list((runtime / "output/chtml/fonts/woff-v2").glob("*.woff"))) == 23
    assert (runtime / "LICENSE").is_file()
    assert "document$.subscribe(requestTypeset)" in script
    assert "mathJax.startup.promise" in script
    assert "waitForMathJax" in script
    assert "performance.now() + 30000" in script
    assert "await delay(50)" in script
    assert "Promise.race([mathJax.startup.promise, delay(1000)])" in script
    assert 'root.classList.add("af-math-pending")' in script
    assert "requestAnimationFrame(resolve)" in script
    assert "await document.fonts.ready" in script
    assert "[data-md-component='content']" in script
    assert '!node.querySelector("mjx-container")' in script
    assert "mathJax.startup.output.clearCache()" in script
    assert "mathJax.typesetClear()" in script
    assert "mathJax.texReset()" in script
    assert "mathJax.typesetPromise(pending)" in script
    assert ".af-math-pending .arithmatex {" in stylesheet
    assert ".af-math-failed .arithmatex" in stylesheet


def test_theory_reference_states_equations_and_result_locations():
    theory = (ROOT / "docs" / "reference" / "theory_and_conventions.md").read_text()
    assert "## Static equilibrium" in theory
    assert "## Structural dynamics" in theory
    assert "## J2 plasticity and creep state" in theory
    assert "## Result locations and recovery" in theory
    assert r"\mathbf{M}\ddot{\mathbf{u}}" in theory
    assert "Integration points" in theory


def test_manual_layout_keeps_navigation_and_footer_visually_separate():
    stylesheet = (ROOT / "docs" / "stylesheets" / "extra.css").read_text()
    assert '.md-nav__link[tabindex="0"]:focus' in stylesheet
    assert ".md-sidebar__scrollwrap:hover" in stylesheet
    assert ".md-footer__inner," in stylesheet
    assert "pointer-events: none;" in stylesheet
    assert "pointer-events: auto;" in stylesheet
    assert "width: min(54rem, calc(100% - 25rem));" in stylesheet
    assert ".md-footer-meta.md-typeset .md-social__link" in stylesheet
    assert "font-weight: inherit;" in stylesheet
    assert ".af-home-lead" in stylesheet
    assert ".md-nav--primary > .md-nav__title" in stylesheet
    assert "Desktop navigation grammar" in stylesheet
    # Material couples its native scroll margin to the TOC observer. A custom
    # heading margin makes the highlighted TOC entry lag one section behind.
    assert "scroll-margin-top: 4rem;" not in stylesheet


def test_primary_navigation_preserves_its_scroll_position_between_pages():
    config = (ROOT / "mkdocs.yml").read_text()
    script = (
        ROOT / "docs" / "javascripts" / "navigation-state.js"
    ).read_text()
    assert "javascripts/navigation-state.js" in config
    assert ".md-sidebar--primary .md-sidebar__scrollwrap" in script
    assert "sessionStorage" in script
    assert "pagehide" in script
    assert ".md-sidebar--primary a.md-nav__link" in script


def test_table_of_contents_tracks_the_heading_below_the_fixed_header():
    config = (ROOT / "mkdocs.yml").read_text()
    script = (ROOT / "docs" / "javascripts" / "toc-current.js").read_text()
    stylesheet = (ROOT / "docs" / "stylesheets" / "extra.css").read_text()
    assert "javascripts/toc-current.js" in config
    assert ".md-sidebar--secondary" in script
    assert "headerBottom + 12" in script
    assert 'currentClass = "af-toc-current"' in script
    assert ".md-sidebar--secondary .md-nav__link.af-toc-current" in stylesheet


def test_homepage_starts_with_the_project_logo():
    homepage = (ROOT / "docs" / "index.md").read_text()
    assert homepage.index('class="af-home-logo"') < homepage.index("<h1>AgentFEM</h1>")
    assert 'class="af-home-lead"' in homepage
    assert "assets/images/AgentFEM_logo_transparent.png" in homepage
