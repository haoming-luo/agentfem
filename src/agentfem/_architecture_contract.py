# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free ownership contract for AgentFEM's stable middle layer.

The contract says who owns a scientific decision.  It is intentionally
smaller than the package inventory: modules may grow, but they must continue
to lower through these boundaries instead of turning ``Model`` or one solver
class into a numerical god object.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OwnershipBoundary:
    """One stable responsibility in the AgentFEM execution architecture."""

    name: str
    question: str
    owns: tuple[str, ...]
    excludes: tuple[str, ...]
    modules: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "question": self.question,
            "owns": self.owns,
            "excludes": self.excludes,
            "modules": self.modules,
        }


OWNERSHIP_BOUNDARIES = (
    OwnershipBoundary(
        name="model",
        question="What engineering problem is being solved?",
        owns=(
            "study and geometry",
            "regions and fields",
            "material assignments",
            "loads and constraints",
            "inspectable engineering steps",
        ),
        excludes=(
            "nonlinear iteration",
            "time integration",
            "constitutive history",
            "result acceptance",
        ),
        modules=(
            "studies",
            "models",
            "_model_support",
            "_model_validation",
            "_model_inspection",
            "mesh",
            "fields",
            "materials",
            "loads",
            "constraints",
            "boundary_models",
        ),
    ),
    OwnershipBoundary(
        name="constitutive",
        question="How does material response map kinematics and history to response?",
        owns=(
            "material-point update laws",
            "stress and consistent tangent",
            "declared internal-variable schema",
        ),
        excludes=(
            "global equilibrium",
            "mesh traversal policy",
            "accepted state lifetime",
        ),
        modules=("constitutive",),
    ),
    OwnershipBoundary(
        name="state",
        question="Which accepted and trial quantities must survive numerical evolution?",
        owns=(
            "accepted and trial history",
            "commit and rollback boundary",
            "restart snapshots",
            "time-level fields",
        ),
        excludes=(
            "material equations",
            "solver selection",
            "file-format policy",
        ),
        modules=("state", "checkpointing"),
    ),
    OwnershipBoundary(
        name="operator",
        question="What mathematical contribution is assembled or applied?",
        owns=(
            "residual, tangent, mass, damping, and source identity",
            "operator composition",
            "form provenance",
        ),
        excludes=(
            "analysis sequencing",
            "state acceptance",
            "verification claims",
        ),
        modules=("operators", "forms", "assembly"),
    ),
    OwnershipBoundary(
        name="procedure",
        question="How is the problem advanced and solved?",
        owns=(
            "solution family and algorithm",
            "increment and iteration policy",
            "provider dispatch and option contract",
        ),
        excludes=(
            "engineering model definition",
            "backend algebra implementation",
            "scientific acceptance",
        ),
        modules=(
            "procedures",
            "step_providers",
            "_step_provider_registry",
            "_step_provider_support",
            "_builtin_step_providers",
            "_step_builders",
            "_step_builders_thermal",
            "_step_builders_finite_strain",
            "_step_builders_inelastic",
            "_step_builders_frequency",
            "_step_builders_dynamics",
            "_nonlinear_problems",
            "_transient_problems",
            "_problem_fields",
            "_material_history",
            "_modal",
            "_modal_fem",
            "problems",
            "mechanics",
        ),
    ),
    OwnershipBoundary(
        name="backend",
        question="Which numerical runtime compiles, assembles, and executes the forms?",
        owns=(
            "form compilation",
            "finite-element assembly",
            "DOF and linear-algebra execution",
            "runtime capability identity",
        ),
        excludes=(
            "engineering semantics",
            "material selection",
            "verification policy",
        ),
        modules=("backends", "kernel", "solvers", "_solver_lifecycle"),
    ),
    OwnershipBoundary(
        name="result_verification",
        question="What was computed, and what evidence makes it usable?",
        owns=(
            "quantities, fields, histories, and artifacts",
            "provenance and failure semantics",
            "verification evidence and acceptance status",
        ),
        excludes=(
            "solver mutation",
            "constitutive state evolution",
            "model construction",
        ),
        modules=(
            "events",
            "results",
            "verification",
            "provenance",
            "convergence",
            "_work_energy",
        ),
    ),
)


# These are architectural impossibilities, not a complete import allow-list.
# The rules remain deliberately small so a useful implementation detail does
# not become an artificial abstraction layer.
_BUILDER_MODULES = (
    "_step_builders",
    "_step_builders_thermal",
    "_step_builders_finite_strain",
    "_step_builders_inelastic",
    "_step_builders_frequency",
    "_step_builders_dynamics",
)


FORBIDDEN_IMPORTS = {
    "models": ("problems", "results", "solvers", "time", "kernel"),
    "_model_support": (
        "models",
        "problems",
        "step_providers",
        *_BUILDER_MODULES,
        "results",
        "solvers",
    ),
    "_model_validation": ("models", "problems", "results", "solvers", "time"),
    "_model_inspection": ("models", "problems", "results", "solvers", "time"),
    "state": (
        "models",
        "problems",
        "step_providers",
        *_BUILDER_MODULES,
        "results",
    ),
    "operators": (
        "models",
        "problems",
        "step_providers",
        *_BUILDER_MODULES,
        "results",
    ),
    "procedures": ("models", "problems", "backends", "results"),
    "events": (
        "models",
        "problems",
        "solvers",
        "backends",
        "constitutive",
        "mechanics",
    ),
    "backends": (
        "models",
        "constitutive",
        "mechanics",
        "operators",
        "procedures",
        "problems",
        "results",
        "step_providers",
        *_BUILDER_MODULES,
    ),
}


def ownership_contract() -> tuple[dict[str, object], ...]:
    """Return the stable, machine-readable ownership inventory."""

    return tuple(item.as_dict() for item in OWNERSHIP_BOUNDARIES)


def ownership_of(module: str) -> str | None:
    """Return the declared owner of one package-relative module.

    The inventory intentionally covers stable architectural seams rather than
    every utility file.  ``None`` means the module has not yet earned a stable
    ownership declaration; it must not be guessed into a layer by callers.
    """

    selected = str(module).strip().removeprefix("agentfem.")
    root = selected.split(".", 1)[0]
    matches = tuple(
        boundary.name for boundary in OWNERSHIP_BOUNDARIES if root in boundary.modules
    )
    if len(matches) > 1:
        raise RuntimeError(
            f"Architecture module {root!r} has multiple owners: {matches}."
        )
    return None if not matches else matches[0]


def _module_name(path: Path, package_root: Path) -> str:
    relative = path.relative_to(package_root).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(("agentfem", *parts))


def _resolved_imports(path: Path, package_root: Path) -> set[str]:
    """Return eager AgentFEM imports for one implementation module."""

    module = _module_name(path, package_root)
    package = (
        module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
    )
    selected: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                if node.module:
                    names = [".".join((*base, node.module))]
                else:
                    names = [".".join((*base, item.name)) for item in node.names]
            elif node.module:
                names = [node.module]
        selected.update(name for name in names if name.startswith("agentfem."))
    return selected


def _dependency_cycles(graph: dict[str, set[str]]) -> tuple[tuple[str, ...], ...]:
    """Return strongly connected eager-import components."""

    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    active: set[str] = set()
    cycles: list[tuple[str, ...]] = []

    def visit(module: str) -> None:
        nonlocal index
        indices[module] = index
        lowlinks[module] = index
        index += 1
        stack.append(module)
        active.add(module)
        for dependency in graph[module]:
            if dependency not in indices:
                visit(dependency)
                lowlinks[module] = min(lowlinks[module], lowlinks[dependency])
            elif dependency in active:
                lowlinks[module] = min(lowlinks[module], indices[dependency])
        if lowlinks[module] != indices[module]:
            return
        component = []
        while True:
            selected = stack.pop()
            active.remove(selected)
            component.append(selected)
            if selected == module:
                break
        if len(component) > 1:
            cycles.append(tuple(sorted(component)))

    for module in graph:
        if module not in indices:
            visit(module)
    return tuple(sorted(cycles))


def audit_source_architecture(
    package_root: Path | None = None,
) -> dict[str, object]:
    """Audit the executable ownership boundary of a source or installed tree.

    The result is deliberately JSON-safe so release tooling can consume the
    same contract as CI.  Only eager top-level imports are considered; lazy
    provider imports remain an intentional extension seam.
    """

    root = (
        Path(__file__).resolve().parent if package_root is None else Path(package_root)
    )
    paths = tuple(sorted(root.rglob("*.py")))
    modules = {_module_name(path, root): path for path in paths}
    known = set(modules)
    graph = {
        module: {name for name in _resolved_imports(path, root) if name in known}
        for module, path in modules.items()
    }
    cycles = _dependency_cycles(graph)
    violations: list[str] = []
    for source, forbidden in FORBIDDEN_IMPORTS.items():
        candidates = [root / f"{source}.py"]
        package = root / source
        if package.is_dir():
            candidates.extend(sorted(package.rglob("*.py")))
        for path in candidates:
            if not path.is_file():
                continue
            imported_roots = {
                name.removeprefix("agentfem.").split(".", 1)[0]
                for name in _resolved_imports(path, root)
            }
            leaked = tuple(sorted(set(forbidden) & imported_roots))
            if leaked:
                violations.append(
                    f"{path.relative_to(root)} imports forbidden layer(s) {leaked}"
                )
    return {
        "schema": "agentfem.architecture-audit",
        "schema_version": "0.1.0",
        "status": "passed" if not cycles and not violations else "failed",
        "module_count": len(modules),
        "cycles": cycles,
        "violations": tuple(violations),
    }


__all__ = (
    "FORBIDDEN_IMPORTS",
    "OWNERSHIP_BOUNDARIES",
    "OwnershipBoundary",
    "audit_source_architecture",
    "ownership_contract",
    "ownership_of",
)
