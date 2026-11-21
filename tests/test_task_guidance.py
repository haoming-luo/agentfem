# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Task discovery is installed, deterministic and independent of numerical imports."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from agentfem import cli
from agentfem._task_discovery import task_record


@pytest.mark.parametrize("task,words", [
    ("modal", ("固有频率", "modal")), ("point-history", ("端点位移随时间", "point")),
    ("load-path", ("载荷—位移曲线", "load displacement")),
    ("mesh-input", ("网格", "mesh import")), ("standard-output", ("云图", "output")),
    ("solver-options", ("求解器", "newton")),
])
def test_task_aliases_and_exact_ids(task, words):
    for word in words:
        assert task_record(word)["matches"][0]["id"] == task
        assert task_record(word) == task_record(word)
    exact = task_record("task:"+task)
    assert exact["matches"][0]["id"] == task
    assert exact["execution_checked"] is False
    assert "/Users/" not in json.dumps(exact)


def test_unknown_search_is_not_unsupported_capability(capsys):
    record = task_record("no-such-scientific-term-xyz")
    assert record['status'] == 'unmatched' and not record['matches']
    assert cli.main(['capabilities', '--task', 'task:no-such-id', '--json']) == 2
    assert json.loads(capsys.readouterr().out)['error']['message']


def test_fresh_task_query_does_not_import_numerics_or_execute_case(tmp_path):
    (tmp_path/'case.py').write_text('raise AssertionError("case must not execute")')
    source = '''
import importlib.abc, sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'dolfinx','mpi4py','petsc4py'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Deny())
from agentfem.cli import main
raise SystemExit(main(['capabilities','--task','载荷—位移曲线','--json']))
'''
    package = Path(cli.__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-c', source], cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': str(package), 'AGENTFEM_TELEMETRY': 'off'},
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout+result.stderr
    assert json.loads(result.stdout)['matches'][0]['id'] == 'load-path'
