from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT / "src" / "agentfem"
DOCS_DIR = ROOT / "docs"
SITE_DIR = ROOT / "site"
API_REFERENCE = DOCS_DIR / "reference" / "api.md"
AGENT_MANIFEST = DOCS_DIR / "agentfem.json"
LLMS_ENTRY = DOCS_DIR / "llms.txt"
LOGO_SOURCE = ROOT / "logo" / "AgentFEM_logo_transparent.png"
LOGO_TARGET = DOCS_DIR / "assets" / "images" / LOGO_SOURCE.name
PUBLIC_API_LEVELS = {
    "core": "CORE_WORKFLOW_MODULES",
    "advanced": "ADVANCED_WORKFLOW_MODULES",
    "expert": "EXPERT_WORKFLOW_MODULES",
}
PUBLIC_MODEL_API_LEVELS = {
    "core": "CORE_MODEL_API",
    "advanced": "ADVANCED_MODEL_API",
    "compatibility": "COMPATIBILITY_MODEL_API",
}


@dataclass(frozen=True)
class ApiObject:
    kind: str
    name: str
    signature: str
    summary: str


def project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise RuntimeError("Could not determine AgentFEM version from pyproject.toml")
    return match.group(1)


def public_api_levels() -> dict[str, tuple[str, ...]]:
    """Read progressive API declarations without importing the FEM runtime."""

    tree = ast.parse((PACKAGE_DIR / "__init__.py").read_text())
    assignments = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
        if isinstance(target, ast.Name)
    }
    levels = {}
    for level, name in PUBLIC_API_LEVELS.items():
        try:
            value = ast.literal_eval(assignments[name])
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"{name} must be a literal tuple in __init__.py"
            ) from exc
        if not isinstance(value, tuple) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise RuntimeError(f"{name} must contain non-empty module names.")
        levels[level] = value
    return levels


def public_modules() -> tuple[str, ...]:
    """Return the stable complete inventory in disclosure order."""

    return tuple(
        dict.fromkeys(
            module
            for modules in public_api_levels().values()
            for module in modules
        )
    )


def public_model_api_levels() -> dict[str, tuple[str, ...]]:
    """Read the Model facade vocabulary without importing FEniCSx."""

    tree = ast.parse((PACKAGE_DIR / "models.py").read_text())
    assignments = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
        if isinstance(target, ast.Name)
    }
    levels = {}
    for level, name in PUBLIC_MODEL_API_LEVELS.items():
        try:
            value = ast.literal_eval(assignments[name])
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError(f"{name} must be a literal tuple in models.py") from exc
        if not isinstance(value, tuple) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise RuntimeError(f"{name} must contain non-empty method names.")
        levels[level] = value
    return levels


def _annotation(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError):
        return ""


def _signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef, *, drop_first: bool = False
) -> str:
    args: list[str] = []
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(
        node.args.defaults
    )
    if drop_first and positional and positional[0].arg in {"self", "cls"}:
        positional = positional[1:]
        defaults = defaults[1:]
    for index, (argument, default) in enumerate(zip(positional, defaults)):
        if node.args.posonlyargs and index == len(node.args.posonlyargs):
            args.append("/")
        item = argument.arg
        annotation = _annotation(argument.annotation)
        if annotation:
            item += f": {annotation}"
        if default is not None:
            item += f" = {ast.unparse(default)}"
        args.append(item)
    if node.args.vararg is not None:
        item = f"*{node.args.vararg.arg}"
        annotation = _annotation(node.args.vararg.annotation)
        if annotation:
            item += f": {annotation}"
        args.append(item)
    elif node.args.kwonlyargs:
        args.append("*")
    for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        item = argument.arg
        annotation = _annotation(argument.annotation)
        if annotation:
            item += f": {annotation}"
        if default is not None:
            item += f" = {ast.unparse(default)}"
        args.append(item)
    if node.args.kwarg is not None:
        item = f"**{node.args.kwarg.arg}"
        annotation = _annotation(node.args.kwarg.annotation)
        if annotation:
            item += f": {annotation}"
        args.append(item)
    returns = _annotation(node.returns)
    suffix = f" -> {returns}" if returns else ""
    return f"{node.name}({', '.join(args)}){suffix}"


def _summary(node: ast.AST) -> str:
    doc = ast.get_docstring(node, clean=True) or ""
    return doc.split("\n\n", maxsplit=1)[0].replace("\n", " ").strip()


def _module_path(module: str) -> Path | None:
    file_path = PACKAGE_DIR / f"{module}.py"
    if file_path.exists():
        return file_path
    package_path = PACKAGE_DIR / module / "__init__.py"
    return package_path if package_path.exists() else None


def module_api(module: str) -> tuple[ApiObject, ...]:
    path = _module_path(module)
    if path is None:
        return ()
    tree = ast.parse(path.read_text(), filename=str(path))
    objects: list[ApiObject] = []
    seen: set[str] = set()

    def append_node(node: ast.AST, *, public_name: str | None = None) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                return
            name = public_name or node.name
            if name in seen:
                return
            signature = _signature(node)
            if name != node.name:
                signature = signature.replace(node.name, name, 1)
            objects.append(ApiObject("function", name, signature, _summary(node)))
            seen.add(name)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            name = public_name or node.name
            if name in seen:
                return
            init = next(
                (
                    item
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "__init__"
                ),
                None,
            )
            signature = name
            if init is not None:
                signature = _signature(init, drop_first=True).replace(
                    "__init__", name, 1
                )
            objects.append(ApiObject("class", name, signature, _summary(node)))
            seen.add(name)

    for node in tree.body:
        append_node(node)

    if path.name == "__init__.py":
        for statement in tree.body:
            if not isinstance(statement, ast.ImportFrom) or statement.level != 1:
                continue
            if statement.module is None:
                continue
            imported_path = path.parent.joinpath(*statement.module.split("."))
            if imported_path.with_suffix(".py").exists():
                imported_path = imported_path.with_suffix(".py")
            elif (imported_path / "__init__.py").exists():
                imported_path = imported_path / "__init__.py"
            else:
                continue
            imported_tree = ast.parse(
                imported_path.read_text(), filename=str(imported_path)
            )
            definitions = {
                node.name: node
                for node in imported_tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            for alias in statement.names:
                public_name = alias.asname or alias.name
                if public_name.startswith("_"):
                    continue
                selected = definitions.get(alias.name)
                if selected is not None:
                    append_node(selected, public_name=public_name)
    return tuple(objects)


def render_api_reference() -> str:
    modules = public_modules()
    lines = [
        "---",
        "title: Python API",
        "description: Automatically generated public AgentFEM Python API index.",
        "---",
        "",
        "# Python API",
        "",
        "This index is generated from the public workflow modules declared by",
        "AgentFEM. It is a discovery surface: detailed scientific meaning, maturity,",
        "and evidence remain in the linked guides and scientific function reference.",
        "",
        "!!! info \"Generated reference\"",
        "    Run `python build_docs.py` to refresh this page after public API changes.",
        "",
    ]
    for module in modules:
        objects = module_api(module)
        lines.extend((f"## `agentfem.{module}`", ""))
        if not objects:
            lines.extend(
                (
                    "This package exposes its public objects through focused submodules.",
                    "",
                )
            )
            continue
        lines.extend(("| Kind | Public object | Purpose |", "| --- | --- | --- |"))
        for item in objects:
            signature = item.signature.replace("|", "\\|")
            summary = item.summary.replace("|", "\\|") or "Public AgentFEM object."
            lines.append(f"| {item.kind} | `{signature}` | {summary} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_agent_manifest() -> str:
    api_levels = public_api_levels()
    model_api_levels = public_model_api_levels()
    manifest = {
        "schema": "agentfem.documentation-entry",
        "schema_version": "1.0",
        "project": "AgentFEM",
        "version": project_version(),
        "maturity": "early-release",
        "description": "AI-native finite-element computing for humans and agents.",
        "documentation": "https://haoming-luo.github.io/agentfem/",
        "repository": "https://github.com/haoming-luo/agentfem",
        "package": "https://pypi.org/project/agentfem/",
        "human_entrypoints": {
            "start": "get_started/",
            "guides": "guide/",
            "examples": "examples/",
            "reference": "reference/",
        },
        "agent_entrypoints": {
            "guide": "agents/",
            "llms_txt": "llms.txt",
            "knowledge_catalog": "https://raw.githubusercontent.com/haoming-luo/agentfem/main/knowledge/catalog.json",
            "skill": "agents/skill/",
        },
        "commands": {
            "environment_check": "agentfem doctor --json",
            "project_check": "agentfem check --json",
            "run": "agentfem run --json",
            "inspect": "agentfem inspect --json",
        },
        "public_workflow_modules": list(public_modules()),
        "public_api": {
            level: list(modules) for level, modules in api_levels.items()
        },
        "model_api": {
            level: list(methods) for level, methods in model_api_levels.items()
        },
        "workflow": [
            "study",
            "model",
            "mesh_and_regions",
            "fields",
            "materials",
            "loads_and_constraints",
            "step",
            "solve",
            "result_and_verification",
        ],
    }
    return json.dumps(manifest, indent=2) + "\n"


def render_llms_entry() -> str:
    lines = [
        "# AgentFEM",
        "",
        "> AI-native finite-element computing for humans and agents.",
        "",
        "AgentFEM keeps finite-element models readable, structured, inspectable, and",
        "operable through ordinary Python and a machine-readable CLI.",
        "",
        "## Start",
        "",
        "- [Human quick start](get_started/)",
        "- [AI agent quick start](agents/)",
        "- [Engineering guides](guide/)",
        "- [Examples](examples/)",
        "- [Python API](reference/api/)",
        "- [Machine manifest](agentfem.json)",
        "",
        "## Canonical sources",
        "",
        "- [Repository](https://github.com/haoming-luo/agentfem)",
        "- [Scientific knowledge catalog](https://raw.githubusercontent.com/haoming-luo/agentfem/main/knowledge/catalog.json)",
        "- [AgentFEM skill](agents/skill/)",
        "",
        "## Safety and scientific contract",
        "",
        "Do not treat a completed solver call as automatically verified. Inspect the",
        "structured result, convergence status, requested outputs, quality policy,",
        "and benchmark evidence before accepting or exporting simulation data.",
        "",
    ]
    return "\n".join(lines)


def write_generated_sources(*, check: bool) -> None:
    generated = {
        API_REFERENCE: render_api_reference(),
        AGENT_MANIFEST: render_agent_manifest(),
        LLMS_ENTRY: render_llms_entry(),
    }
    if check:
        stale = [path for path, expected in generated.items() if not path.exists() or path.read_text() != expected]
        if stale:
            joined = ", ".join(str(path.relative_to(ROOT)) for path in stale)
            raise SystemExit(f"Generated documentation is stale: {joined}. Run `python build_docs.py`.")
    else:
        for path, expected in generated.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected)
        LOGO_TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LOGO_SOURCE, LOGO_TARGET)


def build_site(*, strict: bool) -> None:
    command = [sys.executable, "-m", "mkdocs", "build", "--clean"]
    if strict:
        command.append("--strict")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AgentFEM documentation site")
    parser.add_argument(
        "--check",
        action="store_true",
        help="check generated documentation without rebuilding the site",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="allow MkDocs warnings during local exploration",
    )
    args = parser.parse_args()
    write_generated_sources(check=args.check)
    if args.check:
        print("AgentFEM generated documentation is current.")
        return
    build_site(strict=not args.no_strict)
    print(f"Built AgentFEM documentation site: {SITE_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
