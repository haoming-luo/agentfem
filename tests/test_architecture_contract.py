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
    assert all(
        item["question"] and item["owns"] and item["excludes"] for item in records
    )
    by_name = {item["name"]: item for item in records}
    assert by_name["constitutive"]["modules"] == ("constitutive",)
    assert "mechanics" in by_name["procedure"]["modules"]


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


def test_model_first_operator_facade_delegates_lowering_to_operator_owner():
    tree = ast.parse((PACKAGE / "models.py").read_text(encoding="utf-8"))
    model_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Model"
    )
    methods = {
        node.name: node
        for node in model_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    expected = {
        "stiffness": "_model_lowering.lower_stiffness",
        "mass": "_model_lowering.lower_mass",
        "conduction": "_model_lowering.lower_conduction",
        "heat_capacity": "_model_lowering.lower_heat_capacity",
    }
    for method_name, delegate in expected.items():
        method = methods[method_name]
        source = ast.unparse(method)
        assert delegate in source
        assert not any(
            isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension))
            for node in ast.walk(method)
        )

    lowering = PACKAGE / "operators" / "_model_lowering.py"
    assert lowering.exists()
    assert "models" not in _agentfem_imports(lowering)


def test_analysis_step_delegates_result_assembly_to_result_owner():
    from agentfem import results

    tree = ast.parse((PACKAGE / "problems.py").read_text(encoding="utf-8"))
    step_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AnalysisStep"
    )
    method = next(
        node
        for node in step_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve_result"
    )
    source = ast.unparse(method)

    assert "self.problem.solve()" in source
    assert "from_analysis_step(self, solution" in source
    assert not any(
        isinstance(node, (ast.For, ast.AsyncFor, ast.Try, ast.With))
        for node in ast.walk(method)
    )
    assert "add_field" not in source
    assert "add_quantities" not in source
    assert "static_force_balance" not in source
    assert "constraint_balance_contract" not in source

    result_factory = PACKAGE / "results" / "_analysis_step.py"
    assert result_factory.exists()
    assert "problems" not in _agentfem_imports(result_factory)
    assert "from_analysis_step" not in results.__all__
    assert not hasattr(results, "from_analysis_step")


def test_harmonic_step_delegates_petsc_problem_to_backend_owner():
    tree = ast.parse(
        (PACKAGE / "mechanics" / "viscoelasticity.py").read_text(encoding="utf-8")
    )
    step_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HarmonicViscoelasticStep"
    )
    source = ast.unparse(step_class)
    backend = (PACKAGE / "backends" / "_harmonic.py").read_text(encoding="utf-8")

    assert "fem_petsc.LinearProblem" not in source
    assert "PreparedHarmonicLinearProblem" in source
    assert "fem_petsc.LinearProblem" in backend
    assert "mechanics" not in _agentfem_imports(PACKAGE / "backends" / "_harmonic.py")


def test_modal_analysis_separates_procedure_backend_and_result_owners():
    problem_source = (PACKAGE / "problems.py").read_text(encoding="utf-8")
    mechanics_path = PACKAGE / "mechanics" / "modal.py"
    backend_path = PACKAGE / "backends" / "_modal.py"
    result_path = PACKAGE / "results" / "_modal.py"
    mechanics_source = mechanics_path.read_text(encoding="utf-8")
    backend_source = backend_path.read_text(encoding="utf-8")
    result_source = result_path.read_text(encoding="utf-8")

    assert "class ModalAnalysisStep" not in problem_source
    assert "from .mechanics.modal import ModalAnalysisStep" in problem_source
    assert "class ModalAnalysisStep" in mechanics_source
    assert "_select_modal_solution" in mechanics_source
    assert "SLEPc.EPS" not in mechanics_source
    assert "SimulationResult" not in mechanics_source
    assert "SLEPc.EPS" in backend_source
    assert "selected_clusters_are_complete" not in backend_source
    assert "mechanics" not in _agentfem_imports(backend_path)
    assert "from_modal_step" in result_source
    assert "problems" not in _agentfem_imports(result_path)


def test_problem_compatibility_exports_point_to_new_owners():
    from agentfem import dynamics, operators, problems, state

    assert problems.TransientState is state.TransientState
    assert problems.SecondOrderDynamicsState is state.SecondOrderDynamicsState
    assert problems.ExplicitDynamicsState is state.ExplicitDynamicsState
    assert problems.second_order_state is state.second_order_state
    assert problems.LumpedMassOperator is operators.LumpedMassOperator
    assert problems.ModalSolveInfo is dynamics.ModalSolveInfo


def test_material_history_orchestration_has_a_procedure_owner():
    from agentfem import constitutive

    material = constitutive.standard_linear_solid(
        equilibrium_modulus=2.0,
        relaxing_modulus=3.0,
        relaxation_time=1.0,
    )
    step = material.history([0.0, 1.0], [0.0, 0.1])

    assert step.__class__.__module__ == "agentfem._material_history"
    assert "class GeneralizedMaxwellHistoryStep" not in (
        PACKAGE / "constitutive" / "viscoelasticity.py"
    ).read_text(encoding="utf-8")


def test_fatigue_work_contract_is_owned_by_shared_internal_layer():
    from agentfem import fatigue_fracture
    from agentfem import _work_energy

    assert fatigue_fracture.GeneralizedWorkSample is _work_energy.GeneralizedWorkSample
    assert (
        fatigue_fracture.CyclicWorkEnergyLedger is _work_energy.CyclicWorkEnergyLedger
    )
