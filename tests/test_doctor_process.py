# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import json
import sys

from agentfem import _doctor_process as doctor


def test_output_is_not_exit():
    code, out, _, timeout = doctor.observe([sys.executable, '-c',
        'import time; print("passed",flush=True); time.sleep(30)'], timeout=.3)
    assert code == 124 and timeout and 'passed' in out


def test_normal_exit_and_nonzero():
    for expected in (0, 2):
        code, _, _, timeout = doctor.observe([sys.executable, '-c', f'raise SystemExit({expected})'])
        assert code == expected and not timeout


def test_timeout_does_not_publish_premature_readiness(monkeypatch, capsys):
    monkeypatch.setattr(doctor, 'observe', lambda *a, **k: (124, '{"solver_ready":true}', '', True))
    assert doctor.main(['doctor', '--json']) == 124
    record = json.loads(capsys.readouterr().out)
    assert not record['solver_ready']
    assert record['error']['code'] == 'AFM-RUNTIME-EXIT-TIMEOUT'


def test_success_records_observed_exit(monkeypatch, capsys):
    monkeypatch.setattr(doctor, 'observe', lambda *a, **k: (0, '{"solver_ready":true}', '', False))
    assert doctor.main(['doctor', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['process_exit_observed']
