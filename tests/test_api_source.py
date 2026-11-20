# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only API queries must not initialize a numerical runtime to exit."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import inspect
import pytest
from agentfem import _api_source


@pytest.mark.parametrize('name,code', [
    ('fields.displacement', 0), ('fields.prescribed', 0), ('fields.PrescribedField', 0),
    ('models.Model.step', 0), ('constitutive.StrainHistoryRuntime', 0),
    ('results.gauge', 0), ('fields.not_an_api', 2),
])
def test_fresh_api_query_completes_without_numerical_imports(tmp_path, name, code):
    script = '''
import importlib.abc, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mpi4py', 'petsc4py', 'dolfinx'}:
            raise AssertionError('Read-only query initialized ' + fullname)
sys.meta_path.insert(0, Deny())
from agentfem.cli import main
result = main(['capabilities', '--api', sys.argv[1], '--json'])
assert 'mpi4py.MPI' not in sys.modules
raise SystemExit(result)
'''
    done = subprocess.run([sys.executable, '-c', script, name], capture_output=True,
        text=True, timeout=15, env={**os.environ, 'AGENTFEM_TELEMETRY': 'off',
        'AGENTFEM_STATE_HOME': str(tmp_path/'state'),
        'PYTHONPATH': str(Path(_api_source.__file__).resolve().parents[1])})
    assert done.returncode == code, done.stdout+done.stderr
    record = json.loads(done.stdout)
    if code == 0:
        assert record['inspection'] == 'installed_source'
        assert record['case_executed'] is False


def test_static_signature_handles_python_parameter_kinds_and_literal_defaults():
    node = ast.parse('def example(a, /, b=2, *args, c, d=None, **kwargs) -> Answer: pass').body[0]
    signature = _api_source._signature(node)
    assert [p.kind for p in signature.parameters.values()] == [
        inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD]
    assert signature.parameters['c'].default is inspect.Parameter.empty
    assert signature.parameters['d'].default is None
    assert signature.return_annotation == 'Answer'


@pytest.mark.parametrize('source', [
    'def example(a=must_not_execute()): pass',
    '@must_not_execute\ndef example(a=1): pass',
])
def test_nonliteral_defaults_and_decorators_require_live_inspection(source):
    assert _api_source._signature(ast.parse(source).body[0]) is None


def test_simple_reexports_constructor_and_keyword_only_dataclass_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(_api_source, 'ROOT', tmp_path)
    (tmp_path/'public.py').write_text('from .implementation import Example\n__all__=["Example"]\n')
    source = tmp_path/'implementation.py'
    source.write_text('class Example:\n def __init__(self, value, *, trusted=False): pass\n')
    signature, kind, _ = _api_source.source_callable('public', 'Example')
    assert kind == 'constructor'
    assert list(signature.parameters) == ['value', 'trusted']
    source.write_text('@dataclass(kw_only=True)\nclass Example:\n value: int\n')
    assert _api_source.source_callable('public', 'Example') is None


@pytest.mark.parametrize('suffix', ['example = wrapped(example)', 'example.__signature__ = custom_signature'])
def test_rebound_or_custom_signature_remains_runtime_owned(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(_api_source, 'ROOT', tmp_path)
    (tmp_path/'public.py').write_text('def example(a): pass\n'+suffix+'\n')
    assert _api_source.source_callable('public', 'example') is None
