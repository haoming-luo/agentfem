# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest
from agentfem import cli, feedback
from agentfem._project_runtime import local_runtime


def project(root, source="raise RuntimeError('must not execute')"):
    (root / 'agentfem.toml').write_text('[project]\nname="local-test"\n')
    (root / 'case.py').write_text(source)


def test_local_runtime_restores_environment_after_failure(tmp_path):
    before = dict(os.environ)
    original = tempfile.tempdir
    with pytest.raises(RuntimeError):
        with local_runtime(tmp_path):
            assert Path(tempfile.gettempdir()).is_relative_to(tmp_path)
            assert Path(os.environ['MPLCONFIGDIR']).is_dir()
            raise RuntimeError()
    assert dict(os.environ) == before
    assert tempfile.tempdir == original


def test_writable_check_does_not_execute_case(tmp_path, capsys):
    project(tmp_path)
    assert cli.main(['check', '--project', str(tmp_path), '--local-runtime', '--writable', '--json']) == 0
    record = json.loads(capsys.readouterr().out)
    assert record['model_executed'] is False
    assert len(record['write_access']) == 5
    assert all(item['writable'] for item in record['write_access'])
    assert all(Path(item['path']).is_relative_to(tmp_path) for item in record['write_access'])
    assert not list(tmp_path.rglob('execution.json'))


def test_writable_probe_reports_blocked_output(tmp_path, capsys):
    project(tmp_path)
    (tmp_path / 'outputs').write_text('preserve')
    assert cli.main(['check', '--project', str(tmp_path), '--local-runtime', '--writable', '--json']) == 2
    record = json.loads(capsys.readouterr().out)
    assert any('AFM-PREFLIGHT-WRITE' in item for item in record['errors'])
    assert (tmp_path / 'outputs').read_text() == 'preserve'


@pytest.mark.parametrize('exit_code', [0, 1])
def test_optional_record_failure_preserves_primary_exit(monkeypatch, capsys, exit_code):
    def broken(*args, **kwargs):
        raise PermissionError('private home path')
    monkeypatch.setattr(feedback, 'show_notice_once', broken)
    monkeypatch.setattr(feedback, 'observe_cli', broken)
    monkeypatch.setattr(cli, '_dispatch', lambda **kw: exit_code)
    assert cli.main(['templates', '--json']) == exit_code
    stderr = capsys.readouterr().err
    assert 'PermissionError' in stderr
    assert 'private home path' not in stderr


def test_shutdown_watchdog_captures_non_daemon_thread(tmp_path):
    # A bounded delay reproduces completed output followed by a thread join.
    project(tmp_path, '''import threading, time
threading.Thread(target=lambda: time.sleep(3), daemon=False).start()
''')
    run = subprocess.run([sys.executable, '-m', 'agentfem.cli', 'run', '--project', str(tmp_path),
                          '--local-runtime', '--trace-after', '1', '--json'],
                         capture_output=True, text=True, timeout=25)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)['status'] == 'completed'
    assert 'Timeout (' in run.stderr
    assert 'threading.py' in run.stderr


def test_local_runtime_setup_failure_is_structured(tmp_path, capsys):
    project(tmp_path)
    (tmp_path / '.agentfem-runtime').write_text('blocked')
    assert cli.main(['check', '--project', str(tmp_path), '--local-runtime', '--json']) == 2
    assert json.loads(capsys.readouterr().out)['error']['code'] == 'AFM-RUNTIME-SETUP'
