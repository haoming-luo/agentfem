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


def test_nonlinear_problems_delegate_result_assembly_to_result_owner():
    procedure_path = PACKAGE / "_nonlinear_problems.py"
    tree = ast.parse(procedure_path.read_text(encoding="utf-8"))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    expected = {
        "IncrementalNonlinearVariationalProblem": ("from_incremental_nonlinear_step"),
        "AffineNonlinearVariationalProblem": "from_affine_nonlinear_step",
    }
    for class_name, delegate in expected.items():
        method = next(
            node
            for node in classes[class_name].body
            if isinstance(node, ast.FunctionDef) and node.name == "solve_result"
        )
        source = ast.unparse(method)
        assert delegate in source
        assert "from_solution" not in source
        assert "add_field" not in source
        assert "add_history" not in source
        assert "constraint_balance_contract" not in source

    result_factory = PACKAGE / "results" / "_nonlinear_step.py"
    assert result_factory.exists()
    assert "problems" not in _agentfem_imports(result_factory)


def test_nonlinear_procedures_are_separate_from_discrete_problem_facade():
    problem_source = (PACKAGE / "problems.py").read_text(encoding="utf-8")
    procedure_path = PACKAGE / "_nonlinear_problems.py"
    procedure_source = procedure_path.read_text(encoding="utf-8")

    for name in (
        "AffineNonlinearVariationalProblem",
        "IncrementalNonlinearVariationalProblem",
        "NonlinearLoadIncrementInfo",
        "NonlinearLoadPathInfo",
    ):
        assert f"class {name}" not in problem_source
        assert f"class {name}" in procedure_source
    assert "from ._nonlinear_problems import" in problem_source
    assert "problems" not in _agentfem_imports(procedure_path)

    field_owner = PACKAGE / "_problem_fields.py"
    assert "def reaction_field" in field_owner.read_text(encoding="utf-8")
    assert "problems" not in _agentfem_imports(field_owner)


def test_transient_problems_delegate_result_assembly_to_result_owner():
    procedure_path = PACKAGE / "_transient_problems.py"
    tree = ast.parse(procedure_path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_solve_transient_result"
    )
    source = ast.unparse(method)

    assert "from_transient_step" in source
    assert "from_solution" not in source
    assert "add_field" not in source
    assert "add_history" not in source
    assert "add_artifact" not in source

    result_factory = PACKAGE / "results" / "_transient_step.py"
    assert result_factory.exists()
    assert "problems" not in _agentfem_imports(result_factory)


def test_transient_procedures_are_separate_from_discrete_problem_facade():
    problem_source = (PACKAGE / "problems.py").read_text(encoding="utf-8")
    procedure_path = PACKAGE / "_transient_problems.py"
    procedure_source = procedure_path.read_text(encoding="utf-8")

    for name in (
        "ExplicitDynamicsStep",
        "FirstOrderTransientStep",
        "ImplicitDynamicsStep",
    ):
        assert f"class {name}" not in problem_source
        assert f"class {name}" in procedure_source
    assert "from ._transient_problems import" in problem_source
    assert "problems" not in _agentfem_imports(procedure_path)


def test_direct_problems_delegate_result_assembly_to_result_owner():
    tree = ast.parse((PACKAGE / "problems.py").read_text(encoding="utf-8"))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    expected = {
        "LinearVariationalProblem": "from_linear_variational_problem",
        "LinearSystemProblem": "from_linear_system_problem",
        "NonlinearVariationalProblem": "from_nonlinear_variational_problem",
    }
    for class_name, delegate in expected.items():
        method = next(
            node
            for node in classes[class_name].body
            if isinstance(node, ast.FunctionDef) and node.name == "solve_result"
        )
        source = ast.unparse(method)
        assert delegate in source
        assert "from_solution" not in source
        assert "metadata=" not in source

    result_factory = PACKAGE / "results" / "_problem.py"
    assert result_factory.exists()
    assert "problems" not in _agentfem_imports(result_factory)


def test_model_delegates_validation_and_inspection_to_dedicated_owners():
    model_source = (PACKAGE / "models.py").read_text(encoding="utf-8")
    validation_path = PACKAGE / "_model_validation.py"
    inspection_path = PACKAGE / "_model_inspection.py"
    support_path = PACKAGE / "_model_support.py"

    assert "return validate_model(self" in model_source
    assert "return model_summary(self)" in model_source
    assert "return model_manifest(self)" in model_source
    assert "return model_tree(self)" in model_source
    assert "AFM-MODEL-001" not in model_source
    assert "agentfem_model_manifest" not in model_source

    for path in (validation_path, inspection_path, support_path):
        assert path.exists()
        assert "models" not in _agentfem_imports(path)

    validation_source = validation_path.read_text(encoding="utf-8")
    inspection_source = inspection_path.read_text(encoding="utf-8")
    assert "def validate_model" in validation_source
    assert "def model_summary" in inspection_source
    assert "def model_to_ir" in inspection_source


def test_model_first_operator_methods_delegate_numerical_lowering():
    model_source = (PACKAGE / "models.py").read_text(encoding="utf-8")
    lowering_path = PACKAGE / "operators" / "_model_lowering.py"
    lowering_source = lowering_path.read_text(encoding="utf-8")

    for name in (
        "lower_stiffness",
        "lower_mass",
        "lower_damping",
        "lower_conduction",
        "lower_heat_capacity",
        "lower_thermal_expansion",
        "lower_lumped_mass",
        "lower_load_vector",
        "lower_internal_force",
        "lower_boundary_force",
        "lower_force_balance",
    ):
        assert f"def {name}" in lowering_source
        assert name in model_source

    assert "assemble_lumped_mass" not in model_source
    assert "finite_strain_internal_force(" not in model_source
    assert "models" not in _agentfem_imports(lowering_path)


def test_step_provider_registry_owns_selection_not_scientific_lowering():
    provider_source = (PACKAGE / "step_providers.py").read_text(encoding="utf-8")
    registry_path = PACKAGE / "_step_provider_registry.py"
    registry_source = registry_path.read_text(encoding="utf-8")

    assert "class StepProviderRegistry(ProviderSelectionRegistry)" in provider_source
    assert "class ProviderSelectionRegistry" in registry_source
    assert "def candidates" in registry_source
    assert "def resolve" in registry_source
    assert "provider.lower" not in registry_source
    assert "_step_builders" not in registry_source
    assert "StepExecutionContext" not in registry_source
    assert "provider.lower" in provider_source


def test_public_step_provider_protocol_is_separate_from_builtin_catalog():
    public_path = PACKAGE / "step_providers.py"
    catalog_path = PACKAGE / "_builtin_step_providers.py"
    public_source = public_path.read_text(encoding="utf-8")
    catalog_source = catalog_path.read_text(encoding="utf-8")

    assert "class StepProvider" in public_source
    assert "class StepOptionContract" in public_source
    assert "def lower_step" in public_source
    assert "def _accept_linear_static" not in public_source
    assert "def _lower_linear_static" not in public_source
    assert "register_step_provider(" in catalog_source
    assert "def _accept_linear_static" in catalog_source
    assert "def _lower_linear_static" in catalog_source
    assert "models" not in _agentfem_imports(catalog_path)


def test_thermal_step_builders_have_a_separate_physics_owner():
    facade_source = (PACKAGE / "_step_builders.py").read_text(encoding="utf-8")
    family_path = PACKAGE / "_step_builders_thermal.py"
    family_source = family_path.read_text(encoding="utf-8")

    assert "from ._step_builders_thermal import" in facade_source
    assert "def heat_transfer" not in facade_source
    assert "def _nonlinear_heat_transfer" not in facade_source
    assert "def linear_static" not in facade_source
    assert "def heat_transfer" in family_source
    assert "def _nonlinear_heat_transfer" in family_source
    assert "def linear_static" in family_source
    assert "step_providers" not in _agentfem_imports(family_path)
    assert "models" not in _agentfem_imports(family_path)


def test_finite_strain_step_builders_have_a_separate_physics_owner():
    facade_source = (PACKAGE / "_step_builders.py").read_text(encoding="utf-8")
    family_path = PACKAGE / "_step_builders_finite_strain.py"
    family_source = family_path.read_text(encoding="utf-8")

    assert "from ._step_builders_finite_strain import" in facade_source
    for name in ("fabric_membrane", "hyperelastic", "mixed_hyperelastic"):
        assert f"def {name}" not in facade_source
        assert f"def {name}" in family_source
    assert "step_providers" not in _agentfem_imports(family_path)
    assert "models" not in _agentfem_imports(family_path)


def test_step_builder_facade_contains_no_scientific_construction():
    facade_path = PACKAGE / "_step_builders.py"
    tree = ast.parse(facade_path.read_text(encoding="utf-8"))
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    families = {
        "_step_builders_thermal.py": ("linear_static", "heat_transfer"),
        "_step_builders_inelastic.py": (
            "j2_plasticity",
            "finite_strain_j2",
            "creep",
            "viscoelastic",
        ),
        "_step_builders_frequency.py": (
            "direct_harmonic",
            "harmonic_viscoelastic",
        ),
        "_step_builders_dynamics.py": (
            "explicit_dynamics",
            "finite_strain_explicit_dynamics",
            "modal",
            "implicit_dynamics",
        ),
    }

    assert not functions
    for filename, names in families.items():
        path = PACKAGE / filename
        source = path.read_text(encoding="utf-8")
        for name in names:
            assert f"def {name}" in source
        assert "step_providers" not in _agentfem_imports(path)
        assert "models" not in _agentfem_imports(path)


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
