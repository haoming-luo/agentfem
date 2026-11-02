# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Source-usable onboarding must never infer success or repair user state."""
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "packaging/runtime/setup_environment.py"
spec = importlib.util.spec_from_file_location("environment_setup", SCRIPT)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


@pytest.fixture
def environment(tmp_path, monkeypatch):
    base = tmp_path / "Miniforge 中文 [test]"
    monkeypatch.setattr(setup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(setup, "manager_path", lambda requested: base / "bin/mamba")
    inventory = {"envs": []}
    monkeypatch.setattr(setup, "probe_json", lambda cmd: inventory if "list" in cmd else {"base environment": str(base)})
    args = setup.parser().parse_args(["--channel", "tuna", "--state-root", str(tmp_path / "state")])
    return args, base, inventory


def test_default_only_plans_one_channel_and_jit_toolchain(environment, tmp_path):
    args, base, _ = environment
    plan = setup.make_plan(args)
    assert plan["action"] == "create_then_verify"
    assert not plan["environment_changed"] and not plan["runtime_verified"]
    command = plan["package_command"]
    assert command.count("--channel") == 1
    assert "--no-rc" in command and "--override-channels" in command
    assert {"c-compiler", "cxx-compiler", "python=3.11", "fenics-dolfinx=0.11"} <= set(command)
    assert {"scipy>=1.11", "matplotlib-base", "gmsh>=4.11", "python-gmsh>=4.11"} <= set(command)
    assert command[command.index("--prefix") + 1] == str(base / "envs/agentfem-env")
    assert not base.exists() and not (tmp_path / "state").exists()


def test_existing_named_environment_is_reused_even_outside_default_prefix(environment):
    args, base, inventory = environment
    old = base.parent / "elsewhere/agentfem-env"
    (old / "conda-meta").mkdir(parents=True)
    (old / "conda-meta/history").touch()
    inventory["envs"] = [str(old)]
    plan = setup.make_plan(args)
    assert plan["action"] == "verify_existing" and plan["package_command"] is None
    assert plan["request"]["prefix"] == str(old)


def test_ambiguous_and_incomplete_existing_environment_are_not_repaired(environment):
    args, base, inventory = environment
    inventory["envs"] = [str(base / "envs/agentfem-env"), str(base / "other/agentfem-env")]
    with pytest.raises(setup.SetupError, match="Several"):
        setup.make_plan(args)
    inventory["envs"] = []
    prefix = base / "envs/agentfem-env"
    prefix.mkdir(parents=True)
    marker = prefix / "important.txt"
    marker.write_text("keep")
    with pytest.raises(setup.SetupError, match="not a complete"):
        setup.make_plan(args)
    assert marker.read_text() == "keep"


@pytest.mark.parametrize("option,value", [("--version", "0.4.4;echo bad"), ("--environment", "../base")])
def test_request_rejects_command_or_path_expressions(environment, option, value):
    args, _, _ = environment
    setattr(args, option[2:].replace("-", "_"), value)
    with pytest.raises(setup.SetupError):
        setup.make_plan(args)


def test_conda_route_does_not_receive_mamba_only_option(environment, monkeypatch):
    args, base, _ = environment
    monkeypatch.setattr(setup, "manager_path", lambda requested: base / "bin/conda")
    args.version = "0.4.3"
    args.prefix = str(base / "envs/exact")
    plan = setup.make_plan(args)
    assert "--no-rc" not in plan["package_command"]
    assert "agentfem=0.4.3" in plan["package_command"]
    args.prefix = str(base)
    with pytest.raises(setup.SetupError, match="base environment"):
        setup.make_plan(args)


class FakeRuntime:
    def __init__(self, prefix, fail=None, version="0.4.3"):
        self.prefix, self.fail, self.version = prefix, fail, version
        self.commands = []

    def __call__(self, cmd, **kwargs):
        self.commands.append(cmd)
        assert "PYTHONPATH" not in kwargs["env"]
        if self.fail and self.fail in cmd:
            kwargs["stdout"].write("simulated failure; retained for diagnosis")
            return subprocess.CompletedProcess(cmd, 1)
        data = {"status": "passed"}
        if "-c" in cmd:
            data = {"prefix": str(self.prefix), "version": self.version}
        if "workspace" in cmd:
            data = {"protected_from_distribution_removal": True}
        if "doctor" in cmd:
            data = {"solver_ready": True}
        if "agentfem" in cmd and cmd[cmd.index("agentfem") + 1] == "run":
            data = {"status": "completed"}
        if "verify" in cmd:
            data = {"verified": True}
        kwargs["stdout"].write(json.dumps(data))
        return subprocess.CompletedProcess(cmd, 0)


def make_run(environment, tmp_path, **fake_args):
    args, _, _ = environment
    plan = setup.make_plan(args)
    root = tmp_path / "attempt"
    root.mkdir()
    fake = FakeRuntime(plan["request"]["prefix"], **fake_args)
    return setup.SetupRun(plan, root, json_output=True, runner=fake), fake


def test_create_to_first_solve_records_each_stage_and_does_not_need_activation(environment, tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/wrong/source/checkout")
    run, fake = make_run(environment, tmp_path)
    run.execute()
    report = json.loads((run.root / "status.json").read_text())
    assert report["status"] == "completed" and report["first_solve_verified"]
    assert report["environment_changed"]
    assert "--dry-run" in fake.commands[0] and "--yes" in fake.commands[1]
    assert all(cmd[1:4] == ["run", "--prefix", run.plan["request"]["prefix"]] for cmd in fake.commands[2:])
    names = [r["name"] for r in report["stages"]]
    assert names[-4:] == ["project_create", "project_check", "first_solve", "result_verify"]
    assert all(Path(r["log"]).is_file() for r in report["stages"])


@pytest.mark.parametrize("failure", ["--dry-run", "--yes", "doctor", "workspace", "init", "check", "verify"])
def test_failed_stage_stops_without_fallback_or_false_success(environment, tmp_path, failure):
    run, fake = make_run(environment, tmp_path, fail=failure)
    with pytest.raises(setup.SetupError):
        run.execute()
    report = json.loads((run.root / "status.json").read_text())
    assert report["status"] == "failed" and not report["first_solve_verified"]
    assert failure in fake.commands[-1]
    assert report["stages"][-1]["status"] == "failed"
    assert "simulated failure" in Path(report["stages"][-1]["log"]).read_text()


def test_same_version_label_is_not_permission_to_ignore_requested_identity(environment, tmp_path):
    args, _, _ = environment
    args.version = "0.4.4"
    run, fake = make_run(environment, tmp_path, version="0.4.3")
    with pytest.raises(setup.SetupError, match="differs from the request"):
        run.execute()
    assert "doctor" not in fake.commands[-1]
    assert run.report["error_code"] == "AFM-SETUP-IDENTITY"


def test_existing_environment_has_no_package_transactions(environment, tmp_path):
    args, base, _ = environment
    prefix = base / "envs/agentfem-env"
    (prefix / "conda-meta").mkdir(parents=True)
    (prefix / "conda-meta/history").touch()
    run, fake = make_run(environment, tmp_path)
    run.execute()
    assert all(cmd[1] == "run" for cmd in fake.commands)
    assert not run.report["environment_changed"]


def test_lock_releases_after_failure_without_deleting_file(tmp_path):
    path = tmp_path / "setup.lock"
    with setup.setup_lock(path):
        with pytest.raises(setup.SetupError, match="Another"):
            with setup.setup_lock(path):
                pytest.fail("Duplicate setup admitted")
    with setup.setup_lock(path):
        assert path.exists()


def test_timeout_records_failed_stage_not_running(environment, tmp_path):
    run, _ = make_run(environment, tmp_path)
    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)
    run.runner = timeout
    with pytest.raises(setup.SetupError, match="time budget"):
        run.execute()
    assert run.report["stages"][-1]["status"] == "failed"
    assert run.report["error_code"] == "AFM-SETUP-TIMEOUT"


def test_real_owned_process_timeout(tmp_path):
    with (tmp_path / "process.log").open("w") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            setup.run_logged([sys.executable, "-c", "import time;time.sleep(30)"],
                             stdout=log, stderr=subprocess.STDOUT, text=True, timeout=.1, check=False)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_host_cancellation_records_interruption_and_restores_signal_handlers(environment, tmp_path, sig):
    run, _ = make_run(environment, tmp_path)
    previous = signal.getsignal(sig)
    def interrupted(cmd, **kwargs):
        signal.raise_signal(sig)
    run.runner = interrupted
    with pytest.raises(KeyboardInterrupt):
        with setup.interruptible_setup():
            run.execute()
    assert signal.getsignal(sig) == previous
    report = json.loads((run.root / "status.json").read_text())
    assert report["status"] == "interrupted"
    assert report["stages"][-1]["status"] == "failed"
    assert signal.Signals(sig).name in report["message"]


def test_source_cli_help_does_not_import_numerical_stack():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--apply" in result.stdout
    assert "import agentfem" not in SCRIPT.read_text()


@pytest.mark.parametrize("stage", ["doctor", "workspace", "project_check", "first_solve", "result_verify"])
def test_exit_zero_without_matching_evidence_is_not_acceptance(environment, tmp_path, monkeypatch, stage):
    monkeypatch.setattr(setup.platform, "release", lambda: "microsoft-standard-WSL2")
    run, _ = make_run(environment, tmp_path)
    original = run.stage
    def inconsistent(name, *a, **kw):
        value = original(name, *a, **kw)
        return {} if name == stage else value
    run.stage = inconsistent
    with pytest.raises(setup.SetupError):
        run.execute()
    assert not run.report["first_solve_verified"]
    assert run.report["current_stage"] == stage
    assert run.report["stages"][-1]["status"] == "failed"
    assert not run.report["stages"][-1]["evidence_validated"]


def test_json_evidence_is_separate_from_stderr(environment, tmp_path):
    run, _ = make_run(environment, tmp_path)
    # Use the real child-process runner: a warning must not corrupt stdout JSON.
    run.runner = setup.run_logged
    data = run.stage("fixture", [sys.executable, "-c",
        "import sys;print('{\"verified\":true}');print('diagnostic warning',file=sys.stderr)"],
        json_result=True)
    assert data == {"verified": True}
    record = run.report["stages"][-1]
    assert "diagnostic warning" in Path(record["stderr_log"]).read_text()
    assert "diagnostic warning" not in Path(record["log"]).read_text()


@pytest.mark.parametrize("value", ["[]", "null", "not-json"])
def test_invalid_json_evidence_stops_with_structured_failure(environment, tmp_path, value):
    run, _ = make_run(environment, tmp_path)
    def invalid(cmd, **kwargs):
        kwargs["stdout"].write(value)
        return subprocess.CompletedProcess(cmd, 0)
    run.runner = invalid
    with pytest.raises(setup.SetupError) as error:
        run.stage("invalid", ["fixture"], json_result=True)
    assert error.value.code == "AFM-SETUP-EVIDENCE"
    assert run.report["stages"][-1]["status"] == "failed"


def test_failed_solve_recheck_uses_existing_environment(environment, tmp_path):
    run, fake = make_run(environment, tmp_path, fail="verify")
    with pytest.raises(setup.SetupError):
        run.execute()
    prefix = Path(run.plan["request"]["prefix"])
    (prefix / "conda-meta").mkdir(parents=True)
    (prefix / "conda-meta/history").touch()
    attempt = tmp_path / "retry"
    attempt.mkdir()
    retry = setup.SetupRun(setup.make_plan(environment[0]), attempt,
                           json_output=True, runner=FakeRuntime(prefix))
    retry.execute()
    assert retry.report["status"] == "completed"
    assert retry.plan["package_command"] is None
    assert json.loads((run.root / "status.json").read_text())["status"] == "failed"


def test_error_record_storage_failure_does_not_hide_original_error(environment, tmp_path, monkeypatch):
    run, _ = make_run(environment, tmp_path, fail="doctor")
    original = run.save
    def save():
        if run.report["status"] == "failed":
            raise OSError("disk full")
        original()
    monkeypatch.setattr(run, "save", save)
    with pytest.raises(setup.SetupError, match="doctor failed"):
        run.execute()
    assert "STATE-WRITE" in run.report["recording_error"]
