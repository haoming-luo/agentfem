"""Migration advice distinguishes application dispatch from provider lowering."""

from agentfem import upgrades
import pytest


def test_only_declared_lowerer_owns_add_step(tmp_path):
    source = """from agentfem.step_providers import StepProvider as Provider
from agentfem import io
def lower(model, request):
    with io.XDMFTimeSeries(path, domain) as writer:
        writer.write_fields(0, u)
    return model.add_step(problem)
def ordinary_case(model):
    return model.add_step(problem)
provider = Provider(name="external", lower=lower)
"""
    path = tmp_path / "provider.py"
    path.write_text(source)
    findings = upgrades._python_findings(path)
    compatibility = [item for item in findings if item.code == "AFM-UPG-104"]
    assert [item.line for item in compatibility] == [8]
    output = next(item for item in findings if item.code == "AFM-UPG-101")
    assert "io.ResultWriter" in output.replacement
    assert "solve_result" not in output.replacement
    assert "recursively" in output.message
    assert path.read_text() == source


def test_provider_directory_or_unrelated_constructor_is_not_an_exemption(tmp_path):
    path = tmp_path / "provider.py"
    path.write_text("""from unrelated import StepProvider
def lower(model, request):
    return model.add_step(problem)
provider = StepProvider(lower=lower)
""")
    assert [item.code for item in upgrades._python_findings(path)] == ["AFM-UPG-104"]


def test_nested_case_in_lowerer_still_gets_advice(tmp_path):
    path = tmp_path / "provider.py"
    path.write_text("""import agentfem.step_providers as sp
def lower(model, request):
    def application():
        return model.add_step(other)
    return model.add_step(problem)
provider = sp.StepProvider(lower=lower)
""")
    findings = upgrades._python_findings(path)
    assert [(item.code, item.line) for item in findings] == [("AFM-UPG-104", 4)]


def test_shadowed_constructor_does_not_exempt_lowerer(tmp_path):
    path = tmp_path / "provider.py"
    path.write_text("""from agentfem.step_providers import StepProvider
StepProvider = unrelated
def lower(model, request):
    return model.add_step(problem)
provider = StepProvider(lower=lower)
""")
    assert [item.code for item in upgrades._python_findings(path)] == ["AFM-UPG-104"]


@pytest.mark.parametrize(
    "shadow",
    [
        "class StepProvider: pass",
        "def StepProvider(**kwargs): pass",
        "del StepProvider",
    ],
)
def test_definition_or_deletion_shadowing_retains_advice(tmp_path, shadow):
    path = tmp_path / "provider.py"
    path.write_text(f"""from agentfem.step_providers import StepProvider
{shadow}
def lower(model, request):
    return model.add_step(problem)
provider = StepProvider(lower=lower)
""")
    assert [item.code for item in upgrades._python_findings(path)] == ["AFM-UPG-104"]
