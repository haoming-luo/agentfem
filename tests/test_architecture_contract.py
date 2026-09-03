from __future__ import annotations

import ast
from pathlib import Path

from agentfem import _architecture_contract


PACKAGE = Path(__file__).parents[1] / "src" / "agentfem"


def _module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(("agentfem", *parts))


def _agentfem_imports(path: Path) -> set[str]:
    module = _module_name(path)
    package = module.split(".")[:-1]
    if path.name == "__init__.py":
        package = module.split(".")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
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
        for name in names:
            if name == "agentfem":
                imported.add("")
            elif name.startswith("agentfem."):
                imported.add(name.removeprefix("agentfem.").split(".")[0])
    return imported


def _owned_files(prefix: str) -> tuple[Path, ...]:
    direct = PACKAGE / f"{prefix}.py"
    if direct.exists():
        return (direct,)
    return tuple(sorted((PACKAGE / prefix).rglob("*.py")))


def test_ownership_contract_is_small_stable_and_machine_readable():
    records = _architecture_contract.ownership_contract()

    assert tuple(item["name"] for item in records) == (
        "model",
        "constitutive",
        "state",
        "operator",
        "procedure",
        "backend",
        "result_verification",
    )
    assert all(item["question"] and item["owns"] and item["excludes"] for item in records)


def test_forbidden_cross_layer_imports_do_not_regrow():
    violations = []
    for source, forbidden in _architecture_contract.FORBIDDEN_IMPORTS.items():
        for path in _owned_files(source):
            selected = sorted(set(forbidden) & _agentfem_imports(path))
            if selected:
                violations.append(
                    f"{path.relative_to(PACKAGE)} imports forbidden layer(s) {selected}"
                )

    assert not violations, "\n".join(violations)


def test_model_does_not_construct_discrete_problem_objects_directly():
    source = (PACKAGE / "models.py").read_text(encoding="utf-8")

    assert "from . import problems" not in source
    assert "problems." not in source


def test_problem_compatibility_exports_point_to_new_owners():
    from agentfem import operators, problems, state

    assert problems.TransientState is state.TransientState
    assert problems.SecondOrderDynamicsState is state.SecondOrderDynamicsState
    assert problems.ExplicitDynamicsState is state.ExplicitDynamicsState
    assert problems.second_order_state is state.second_order_state
    assert problems.LumpedMassOperator is operators.LumpedMassOperator


def test_fatigue_work_contract_is_owned_by_shared_internal_layer():
    from agentfem import fatigue_fracture
    from agentfem import _work_energy

    assert fatigue_fracture.GeneralizedWorkSample is _work_energy.GeneralizedWorkSample
    assert fatigue_fracture.CyclicWorkEnergyLedger is _work_energy.CyclicWorkEnergyLedger
