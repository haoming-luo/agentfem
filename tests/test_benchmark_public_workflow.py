"""Keep reusable runtime implementation out of ordinary benchmark providers."""
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "src/agentfem/benchmarks"


def test_backend_imports_are_confined_to_declared_formulation_references():
    # LEFM spells out grad(u) for its independent interaction-integral evidence.
    # Thermoelastic coupling deliberately compares against a raw mixed UFL solve.
    exceptions = {"lefm.py": {"ufl"},
                  "thermoelastic_coupling.py": {"ufl", "basix", "dolfinx", "petsc4py"}}
    failures = []
    for path in sorted(ROOT.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imported = [item.name.split('.')[0] for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported = [(node.module or '').split('.')[0]]
            else:
                continue
            for name in imported:
                if name in {'ufl', 'basix', 'dolfinx', 'petsc4py'} and name not in exceptions.get(path.name, set()):
                    failures.append(f"{path.name}:{node.lineno}: {name}")
    assert not failures, "Use platform APIs or document an independent reference: " + str(failures)


def test_structural_benchmarks_use_canonical_step_entry():
    failures = []
    legacy = {"finite_strain_explicit_dynamics_step", "linear_static_step",
              "nonlinear_static_step", "creep_step", "j2_step"}
    for path in sorted(ROOT.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in legacy:
                failures.append(f"{path.name}:{node.lineno}: {node.func.attr}")
    assert not failures, "Declare physics in Study and call model.step: " + str(failures)
