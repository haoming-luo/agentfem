"""The product shell must preserve ordinary Python case invocation semantics."""

import json
import os
from pathlib import Path
import sys

import pytest

from agentfem import cli


def _case(root, source):
    (root / "agentfem.toml").write_text(
        "[project]\nname='invocation'\nentrypoint='case.py'\n"
    )
    (root / "case.py").write_text(source)
    return ["run", "--project", str(root), "--run-id", "test", "--json"]


def _record(root):
    return json.loads((root / "outputs/invocation/test/execution.json").read_text())


@pytest.mark.parametrize("forwarded", [[], ["--", "--value", "37", "--json"]])
def test_case_arguments_and_sibling_imports(tmp_path, monkeypatch, forwarded):
    args = _case(tmp_path, """
import argparse
import json
import sys
from invocation_sibling import ANSWER
from agentfem import project
p = argparse.ArgumentParser()
p.add_argument('--value', type=int, default=12)
p.add_argument('--json', action='store_true')
args = p.parse_args()
project.current_run().artifact('invocation.json').write_text(json.dumps({
    'argv': sys.argv, 'value': args.value, 'json': args.json, 'answer': ANSWER,
}))
""")
    (tmp_path / "invocation_sibling.py").write_text("ANSWER = 42\n")
    monkeypatch.setattr(sys, "argv", ["agentfem", *args, *forwarded])
    before_argv, before_path, before_cwd = sys.argv, sys.path, Path.cwd()
    try:
        assert cli.main() == 0
    finally:
        sys.modules.pop("invocation_sibling", None)
    result = json.loads((tmp_path / "outputs/invocation/test/invocation.json").read_text())
    assert result == {
        "argv": [str(tmp_path / "case.py"), *forwarded[1:]],
        "value": 37 if forwarded else 12, "json": bool(forwarded), "answer": 42,
    }
    assert sys.argv is before_argv and sys.path is before_path
    assert Path.cwd() == before_cwd
    assert _record(tmp_path)["status"] == "completed"


@pytest.mark.parametrize("code, expected", [(None, 0), (0, 0), (2, 1), ("bad input", 1)])
def test_case_system_exit_has_execution_evidence(tmp_path, monkeypatch, code, expected):
    args = _case(tmp_path, f"raise SystemExit({code!r})\n")
    monkeypatch.setenv("AGENTFEM_CLI_MANAGED", "previous")
    before_argv, before_path, before_cwd = sys.argv, sys.path, Path.cwd()
    assert cli.main(args) == expected
    assert sys.argv is before_argv and sys.path is before_path
    assert Path.cwd() == before_cwd
    assert os.environ["AGENTFEM_CLI_MANAGED"] == "previous"
    record = _record(tmp_path)
    assert record["status"] == ("failed" if expected else "completed")
    if expected:
        assert record["error"]["type"] == "builtins.SystemExit"
        assert record["error"]["exit_code"] == (code if isinstance(code, int) else 1)


def test_case_parser_failure_is_structured(tmp_path, capsys):
    args = _case(tmp_path, "import argparse\nargparse.ArgumentParser().parse_args()\n")
    assert cli.main([*args, "--", "--unknown"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    log = tmp_path / "outputs/invocation/test/logs/stderr.rank-0.log"
    assert "unrecognized arguments: --unknown" in log.read_text()
    assert _record(tmp_path)["error"]["exit_code"] == 2


def test_case_keyboard_interrupt_records_failure(tmp_path):
    args = _case(tmp_path, "raise KeyboardInterrupt()\n")
    assert cli.main(args) == 1
    assert _record(tmp_path)["error"]["type"] == "builtins.KeyboardInterrupt"


@pytest.mark.parametrize("status, expected", [("paused", 0), ("failed", 1)])
def test_case_explicit_status_is_not_reported_as_completed(tmp_path, capsys, status, expected):
    args = _case(tmp_path, f"from agentfem import project\nproject.current_run().write_execution({status!r})\n")
    assert cli.main(args[:-1]) == expected
    assert f"Run {status}" in capsys.readouterr().out
    assert _record(tmp_path)["status"] == status


def test_case_arguments_require_explicit_separator(tmp_path):
    args = _case(tmp_path, "raise AssertionError('must not run')\n")
    assert cli.main([*args, "unexpected-positional"]) == 2
    assert not (tmp_path / "outputs").exists()


def test_mpi_launcher_preserves_case_arguments(monkeypatch):
    args = cli.build_parser().parse_args([
        "run", "--project", "project with spaces", "--mpi", "2", "--json",
        "--", "--value", "37", "--name", "case value",
    ])
    commands = []
    monkeypatch.setattr(cli, "mpi_command", lambda count, child: ["launcher", str(count), *child])
    def execute(command, *, check):
        commands.append(command)
        return type("Result", (), {"returncode": 0})()
    monkeypatch.setattr(cli.subprocess, "run", execute)
    assert cli._launch_mpi(args) == 0
    child = commands[0]
    assert child[child.index("--inside-mpi") + 1:] == args.case_arguments
    assert "project with spaces" in child
