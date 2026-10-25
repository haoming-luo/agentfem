#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Plan, create or verify a Linux/WSL AgentFEM environment using Miniforge.

This source-usable entry point needs only Python's standard library. It never
installs WSL, downloads an executable bootstrapper, changes shell configuration,
or repairs an existing environment implicitly. Package installation is opt-in;
completion requires an installed-template solve, not an import or a version.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import shutil
import subprocess
import sys
import time
import uuid


CHANNELS = {
    "conda-forge": "conda-forge",
    "tuna": "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge",
}
SCHEMA = "agentfem.environment-setup"


class SetupError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def atomic_json(path, record):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def setup_lock(path):
    # Kernel ownership is released on exit/crash; a persistent lock filename
    # is not mistaken for a running installation. Only Linux/WSL is admitted.
    import fcntl

    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SetupError("AFM-SETUP-BUSY", "Another environment setup is running. Do not start a duplicate.") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def manager_path(requested):
    if requested:
        candidate = shutil.which(requested) or str(Path(requested).expanduser())
    else:
        candidate = shutil.which("mamba") or shutil.which("conda")
        if not candidate:
            for name in ("mamba", "conda"):
                local = Path.home() / "miniforge3" / "bin" / name
                if local.is_file():
                    candidate = str(local)
                    break
    if not candidate or not Path(candidate).is_file():
        raise SetupError("AFM-SETUP-MANAGER-MISSING", "Use an existing Linux Miniforge installation, or install its verified official installer first; then rerun with --manager /absolute/path/to/mamba. Do not use a Windows .exe in WSL.")
    candidate = Path(candidate).absolute()
    if candidate.name not in ("mamba", "conda") or not os.access(candidate, os.X_OK):
        raise SetupError("AFM-SETUP-MANAGER", "--manager must name an executable Linux mamba or conda, not a shell command or Windows executable.")
    return candidate


def clean_environment():
    env = os.environ.copy()
    # Verify the installed package, never a checkout injected by an agent tool.
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    return env


def probe_json(command):
    try:
        response = subprocess.run(command, capture_output=True, text=True,
                                  env=clean_environment(), timeout=30, check=True)
        value = json.loads(response.stdout)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise SetupError("AFM-SETUP-PROBE", "Package-manager inventory failed; no environment was changed. Inspect the selected Miniforge installation.") from exc


def run_logged(command, *, timeout, **kwargs):
    """Stop only this invocation's process group on timeout/interruption.

    Killing just mamba/conda can leave its solver or first-solve child running.
    Never kill another installer, WSL distribution, service or simulation.
    """
    kwargs.pop("check", None)
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        return subprocess.CompletedProcess(command, process.wait(timeout=timeout))
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # The parent may exit before a child; terminate the owned group even
        # if wait() already reaped the package-manager parent.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def make_plan(args):
    if platform.system() != "Linux":
        raise SetupError("AFM-SETUP-HOST", "Run this environment step inside Linux/Ubuntu on WSL2, not in Windows Python or macOS.")
    if os.geteuid() == 0:
        raise SetupError("AFM-SETUP-ACCOUNT", "Use your ordinary Linux account. Finish Ubuntu's first account setup locally; never send passwords to an AI assistant.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.environment):
        raise SetupError("AFM-SETUP-NAME", "Use a simple environment name, not a path or package expression.")
    if args.version and not re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.dev\d+)?", args.version):
        raise SetupError("AFM-SETUP-VERSION", "--version needs one exact published version, not a range or shell expression.")
    manager = manager_path(args.manager)
    info = probe_json([str(manager), "info", "--json"])
    inventory = probe_json([str(manager), "env", "list", "--json"])
    base = info.get("root_prefix") or info.get("base environment")
    if not isinstance(base, str) or not Path(base).is_absolute():
        raise SetupError("AFM-SETUP-PROBE", "The package manager did not identify its absolute base environment.")
    base = Path(base).resolve()
    if args.prefix:
        prefix = Path(args.prefix).expanduser().resolve()
    else:
        matches = {Path(p).resolve() for p in inventory.get("envs", [])
                   if Path(p).name == args.environment}
        if len(matches) > 1:
            raise SetupError("AFM-SETUP-AMBIGUOUS", "Several environments share this name. Select one explicitly with --prefix; none was changed.")
        prefix = next(iter(matches)) if matches else base / "envs" / args.environment
    if prefix in (base, Path.home().resolve(), Path("/")):
        raise SetupError("AFM-SETUP-PREFIX", "Refusing to install into the base environment or a broad user/system directory. Choose a separate environment.")
    existing = prefix.exists()
    if existing and not (prefix / "conda-meta" / "history").is_file():
        raise SetupError("AFM-SETUP-INCOMPLETE", f"{prefix} exists but is not a complete conda environment. Preserve it for inspection or choose a new environment name; no files were removed.")
    command = [str(manager), "create", "--prefix", str(prefix)]
    if manager.name == "mamba":
        command.append("--no-rc")
    command += ["--override-channels", "--channel", CHANNELS[args.channel],
                "python=3.11", "fenics-dolfinx=0.11", "c-compiler", "cxx-compiler",
                "agentfem" + ("=" + args.version if args.version else "")]
    request = dict(manager=str(manager), prefix=str(prefix), channel=args.channel,
                   requested_version=args.version, protect_workspace=not args.no_protect_workspace)
    return dict(schema=SCHEMA, schema_version=1, status="planned", request=request,
                action="verify_existing" if existing else "create_then_verify",
                package_command=None if existing else command,
                environment_changed=False, runtime_verified=False,
                first_solve_verified=False, notes=[
                    "Existing environments are verified, never upgraded or repaired automatically.",
                    "One channel only; no global conda configuration or shell startup files are changed.",
                    "WSL consent, restart and private account setup remain separate platform steps."])


class SetupRun:
    def __init__(self, plan, root, *, json_output=False, runner=None):
        self.plan = plan
        self.root = root
        self.json_output = json_output
        self.runner = runner or run_logged
        self.report = dict(plan, status="running", stages=[],
                           started_at=datetime.now(timezone.utc).isoformat())
        self.env = clean_environment()
        self.env["XDG_CACHE_HOME"] = str(root / "cache")

    def save(self):
        atomic_json(self.root / "status.json", self.report)

    def save_finally(self):
        failing = sys.exc_info()[0] is not None
        try:
            self.save()
        except OSError as exc:
            if not failing:
                raise
            # Retain the original failure in the terminal result if recording
            # it also fails (e.g. full disk). A prior status.json may be stale.
            self.report["recording_error"] = f"AFM-SETUP-STATE-WRITE: {exc}"

    def stage(self, name, command, *, json_result=False, timeout=900):
        if not self.json_output:
            print(f"AgentFEM setup | {name.replace('_', ' ')}", flush=True)
        record = dict(name=name, status="running", command=command,
                      log=str(self.root / f"{len(self.report['stages']) + 1:02d}-{name}.log"))
        self.report["stages"].append(record)
        self.report["current_stage"] = name
        self.save()
        if not self.json_output and name in ("package_plan", "environment_create", "first_solve"):
            print(f"  Details: {record['log']} (leave this process running; do not start a second copy)", flush=True)
        started = time.monotonic()
        try:
            with ExitStack() as stack:
                log = stack.enter_context(Path(record["log"]).open("x", encoding="utf-8"))
                errors = subprocess.STDOUT
                if json_result:
                    record["stderr_log"] = record["log"] + ".stderr"
                    errors = stack.enter_context(Path(record["stderr_log"]).open("x", encoding="utf-8"))
                result = self.runner(command, cwd=self.root, env=self.env, stdout=log,
                                     stderr=errors, text=True, timeout=timeout, check=False)
            record["exit_code"] = result.returncode
            if result.returncode:
                logs = record["log"]
                if record.get("stderr_log"):
                    logs += " and " + record["stderr_log"]
                raise SetupError("AFM-SETUP-STAGE", f"{name.replace('_', ' ')} failed. See {logs}. No automatic source switch, deletion or repair was attempted.")
            value = None
            if json_result:
                try:
                    value = json.loads(Path(record["log"]).read_text(encoding="utf-8"))
                    if not isinstance(value, dict):
                        raise ValueError("Expected an object")
                except (ValueError, OSError) as exc:
                    raise SetupError("AFM-SETUP-EVIDENCE", f"{name} returned invalid evidence. See {record['log']}.") from exc
            record["status"] = "passed"
            return value
        except subprocess.TimeoutExpired as exc:
            record["status"] = "failed"
            raise SetupError("AFM-SETUP-TIMEOUT", f"{name} exceeded its time budget; inspect {record['log']} before resuming. Do not start a duplicate package transaction.") from exc
        except BaseException:
            record["status"] = "failed"
            raise
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - started, 2)
            self.save_finally()

    def command(self, *arguments):
        request = self.plan["request"]
        return [request["manager"], "run", "--prefix", request["prefix"], *arguments]

    def require_evidence(self, condition, code, message):
        """A zero process exit is not acceptance of its scientific/tool contract."""
        record = self.report["stages"][-1]
        record["evidence_validated"] = bool(condition)
        if not condition:
            record.update(status="failed", error_code=code)
            raise SetupError(code, message)

    def execute(self):
        try:
            self.save()
            if self.plan["package_command"]:
                # Exact requested package solve first; unavailable mirror versions
                # fail here rather than silently downgrading or mixing channels.
                self.stage("package_plan", self.plan["package_command"] + ["--dry-run", "--json"], timeout=1800)
                self.report["environment_change_attempted"] = True
                self.stage("environment_create", self.plan["package_command"] + ["--yes"], timeout=7200)
                self.report["environment_changed"] = True
            identity = self.stage("installed_identity", self.command("python", "-c",
                "import json,sys,importlib.metadata as m; print(json.dumps({'version':m.version('agentfem'),'prefix':sys.prefix}))"), json_result=True)
            requested = self.plan["request"]
            self.require_evidence(
                isinstance(identity.get("prefix"), str) and
                Path(identity["prefix"]).resolve() == Path(requested["prefix"]) and
                isinstance(identity.get("version"), str) and bool(identity["version"]) and
                (not requested["requested_version"] or identity["version"] == requested["requested_version"]),
                "AFM-SETUP-IDENTITY", "The installed environment/version differs from the request. It was not upgraded, downgraded or replaced.")
            self.report["installed"] = identity
            doctor = self.stage("doctor", self.command("agentfem", "doctor", "--json"), json_result=True)
            self.require_evidence(doctor.get("solver_ready") is True,
                "AFM-SETUP-DOCTOR", "The health report did not confirm solver readiness. Inspect its log; no repair was attempted.")
            self.report["runtime_verified"] = True
            workspace_args = ["agentfem", "workspace", "--json"]
            if requested["protect_workspace"]:
                workspace_args.append("--protect")
            workspace = self.stage("workspace", self.command(*workspace_args), json_result=True)
            self.require_evidence(
                "microsoft" not in platform.release().lower() or workspace.get("protected_from_distribution_removal") is True,
                "AFM-SETUP-WORKSPACE", "The WSL workspace is not protected from distribution removal. Inspect the workspace report before continuing.")
            self.report["workspace"] = workspace
            project = self.root / "first-solve"
            self.report["first_project"] = str(project)
            self.stage("project_create", self.command("agentfem", "init", "--template", "static-solid", str(project)))
            checked = self.stage("project_check", self.command("agentfem", "check", "--project", str(project), "--json"), json_result=True)
            self.require_evidence(checked.get("status") == "passed",
                "AFM-SETUP-CHECK", "The installed template did not pass its project check.")
            solved = self.stage("first_solve", self.command("agentfem", "run", "--project", str(project), "--name", "first-solve", "--json"), json_result=True)
            self.require_evidence(solved.get("status") == "completed",
                "AFM-SETUP-SOLVE", "The first solve did not complete; a paused run is not installation acceptance.")
            verification = self.stage("result_verify", self.command("agentfem", "verify", "--project", str(project), "--json"), json_result=True)
            self.require_evidence(verification.get("verified") is True,
                "AFM-SETUP-VERIFY", "Result provenance verification did not pass. Preserve the first-solve project for inspection.")
            self.report.update(first_solve_verified=True, verification=verification, status="completed")
        except BaseException as exc:
            self.report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                               error_code=getattr(exc, "code", "AFM-SETUP-FAILED"), message=str(exc),
                               next_action="inspect_stage_log_then_rerun_same_setup")
            raise
        finally:
            self.report["finished_at"] = datetime.now(timezone.utc).isoformat()
            self.save_finally()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manager", help="Existing Linux mamba/conda executable; never downloaded implicitly")
    result.add_argument("--environment", default="agentfem-env")
    result.add_argument("--prefix", help="Explicit existing or new environment directory")
    result.add_argument("--channel", choices=CHANNELS, default="conda-forge")
    result.add_argument("--version", default="", help="Exact published AgentFEM version, if required")
    result.add_argument("--apply", action="store_true", help="Execute the plan and first solve; default only plans")
    result.add_argument("--json", action="store_true", help="One structured result; detailed subprocess output stays in logs")
    result.add_argument("--no-protect-workspace", action="store_true", help="Inspect rather than migrate the default project workspace")
    result.add_argument("--state-root", type=Path, default=Path.home() / ".local/state/agentfem/setup")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    report = None
    code = 0
    try:
        report = make_plan(args)
        if args.apply:
            root = args.state_root.expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            # Per-user serialization, even when two requests select one target
            # through different manager paths. Never remove manager-owned locks.
            with setup_lock(root / "setup.lock"):
                # Re-probe after acquiring the lock; a previous run may have
                # created the environment while this request was being planned.
                report = make_plan(args)
                key = hashlib.sha256(json.dumps(report["request"], sort_keys=True).encode()).hexdigest()[:16]
                attempt = root / key / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
                attempt.mkdir(parents=True)
                run = SetupRun(report, attempt, json_output=args.json)
                report = run.report
                report["record"] = str(attempt / "status.json")
                run.execute()
    except (SetupError, OSError, KeyboardInterrupt) as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        report = dict(report or {}, schema=SCHEMA, schema_version=1,
                      status="interrupted" if code == 130 else "failed",
                      error_code=getattr(exc, "code", "AFM-SETUP-FAILED"), message=str(exc))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    elif report["status"] == "planned":
        print(f"AgentFEM setup | plan only\nEnvironment: {report['request']['prefix']}\nAction: {report['action']}\nChannel: {report['request']['channel']}\nNo packages changed. Review these settings, then rerun with --apply.")
    elif report["status"] == "completed":
        print(f"AgentFEM {report['installed']['version']} | first solve verified\nEnvironment: {report['request']['prefix']}\nOpen project: {report['first_project']}\nRecord: {report['record']}\nUse this environment through mamba/conda run --prefix; no AI is required.")
    else:
        print(f"AgentFEM setup | {report['status']}\n{report.get('error_code')}: {report.get('message')}\nRecord: {report.get('record', 'not created; no package operation started')}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
