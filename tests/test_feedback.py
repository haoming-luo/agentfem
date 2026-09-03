from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import pytest

from agentfem import __version__, cli, feedback


SAFE_RUNTIME = {
    "system": "Darwin",
    "route": "native macOS",
    "machine": "arm64",
    "python": "3.11.15",
    "dolfinx": "0.11.0",
    "petsc4py": "3.25.2",
    "mpi_vendor": "MPICH",
    "mpi_ranks": 1,
    "installation": "installed",
}


@pytest.fixture
def feedback_home(tmp_path, monkeypatch):
    home = tmp_path / "feedback"
    monkeypatch.setenv("AGENTFEM_FEEDBACK_HOME", str(home))
    monkeypatch.setenv("AGENTFEM_TELEMETRY_TESTING", "1")
    monkeypatch.delenv("AGENTFEM_TELEMETRY", raising=False)
    monkeypatch.delenv("AGENTFEM_FEEDBACK_ENDPOINT", raising=False)
    monkeypatch.delenv("AGENTFEM_FEEDBACK_ENDPOINTS", raising=False)
    monkeypatch.setattr(feedback, "_endpoints_from_package", lambda: ())
    monkeypatch.setattr(feedback, "_safe_runtime", lambda: dict(SAFE_RUNTIME))
    return home


def test_release_contains_the_owned_https_collector():
    endpoints = feedback._endpoints_from_package()
    assert endpoints == (
        "https://agentfem-reliability.horming-luo.workers.dev/v1/reliability",
    )


def _failed_execution(path: Path) -> Path:
    path.mkdir(parents=True)
    execution = path / "execution.json"
    execution.write_text(
        json.dumps(
            {
                "schema": "agentfem.run",
                "schema_version": "0.2.0",
                "status": "failed",
                "stage": "case_execution",
                "structured_result": False,
                "project_root": "/private/customer/project",
                "error": {
                    "type": "RuntimeError",
                    "code": "AFM-SOLVE-007",
                    "stage": "case_execution",
                    "message": "secret material=Inconel path=/private/customer/model.inp",
                    "traceback": (
                        'Traceback (most recent call last):\n'
                        '  File "/private/customer/project/case.py", line 7, in <module>\n'
                        '    solve(secret_material)\n'
                        'RuntimeError: secret material=Inconel\n'
                    ),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return execution


def test_basic_reliability_event_is_exactly_whitelisted(feedback_home):
    event = feedback.build_event(
        "run",
        "failed",
        duration_seconds=75,
        error={
            "type": "RuntimeError",
            "code": "AFM-SOLVE-007",
            "stage": "case_execution",
            "message": "secret material parameter and /customer/path",
            "traceback": "private source",
        },
        now=datetime(2026, 8, 31, 8, 42, 23, tzinfo=timezone.utc),
        event_id="00000000-0000-0000-0000-000000000001",
    )
    encoded = json.dumps(event)
    assert event["agentfem_version"] == __version__
    assert "occurred_at" not in event
    assert event["duration_bucket"] == "1-10m"
    assert event["failure"]["fingerprint"].startswith("AFM-FP-")
    assert "secret" not in encoded
    assert "customer" not in encoded
    assert "message" not in encoded
    assert "traceback" not in encoded
    feedback.validate_event(event)
    with pytest.raises(ValueError, match="whitelist"):
        feedback.validate_event({**event, "project": "confidential"})


def test_feedback_is_on_by_default_and_can_be_disabled(feedback_home):
    assert feedback.preferences().mode == "basic"
    event = feedback.build_event("run", "completed")
    assert feedback.enqueue(event).is_file()
    assert feedback.configure("off").mode == "off"
    assert feedback.queue_size() == 0
    assert feedback.enqueue(event) is None
    assert feedback.queue_size() == 0
    assert feedback.configure("basic").mode == "basic"
    assert feedback.enqueue(event).is_file()


def test_marking_notice_preserves_packaged_failover_routes(feedback_home, monkeypatch):
    routes = (
        "https://cn.example/v1/reliability",
        "https://global.example/v1/reliability",
    )
    monkeypatch.setattr(feedback, "_endpoints_from_package", lambda: routes)

    assert feedback.preferences().endpoints == routes
    configured = feedback.configure("basic", notice_shown=True)

    assert configured.notice_shown is True
    assert configured.endpoints == routes
    stored = json.loads((feedback_home / "preferences.json").read_text(encoding="utf-8"))
    assert stored["endpoints"] == list(routes)


def test_bounded_transport_sends_only_the_whitelisted_batch(
    feedback_home, monkeypatch
):
    feedback.configure("basic", endpoint="https://reliability.example/v1/reliability")
    event = feedback.build_event("run", "failed", error={"type": "RuntimeError"})
    feedback.enqueue(event)
    captured = {}

    class Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def open_request(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(feedback.urlrequest, "urlopen", open_request)
    delivered = feedback.flush()
    assert delivered == {"status": "sent", "sent": 1, "remaining": 0, "route": 0}
    payload = json.loads(captured["request"].data)
    assert payload["schema"] == "agentfem.reliability-batch"
    assert payload["events"] == [event]
    assert "occurred_at" not in payload["events"][0]
    assert captured["timeout"] == feedback.DEFAULT_TIMEOUT_SECONDS
    assert feedback.last_event() == event
    feedback.configure("off")
    assert feedback.last_event() is None


def test_transport_uses_first_successful_reviewed_route(feedback_home, monkeypatch):
    monkeypatch.setenv(
        "AGENTFEM_FEEDBACK_ENDPOINTS",
        "https://cn.example/v1/reliability,https://global.example/v1/reliability",
    )
    event = feedback.build_event("run", "failed", error={"type": "RuntimeError"})
    feedback.enqueue(event)
    attempted = []

    class Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def open_request(request, *, timeout):
        attempted.append((request.full_url, timeout))
        if "cn.example" in request.full_url:
            raise feedback.urlerror.URLError("unavailable")
        return Response()

    monkeypatch.setattr(feedback.urlrequest, "urlopen", open_request)
    delivered = feedback.flush()

    assert delivered == {"status": "sent", "sent": 1, "remaining": 0, "route": 1}
    assert tuple(item[0] for item in attempted) == (
        "https://cn.example/v1/reliability",
        "https://global.example/v1/reliability",
    )
    assert all(item[1] <= feedback.DEFAULT_TIMEOUT_SECONDS / 2 for item in attempted)
    assert feedback.preferences().summary()["delivery_route_count"] == 2


def test_repeated_failure_suggests_agent_only_once(feedback_home):
    error = {
        "type": "RuntimeError",
        "code": "AFM-SOLVE-007",
        "stage": "case_execution",
    }
    first = feedback.record_failure(error)
    second = feedback.record_failure(error)
    third = feedback.record_failure(error)
    fourth = feedback.record_failure(error)
    assert first["count"] == 1 and not first["suggest_agent_now"]
    assert second["count"] == 2 and not second["suggest_agent_now"]
    assert third["count"] == 3 and third["suggest_agent_now"]
    assert fourth["count"] == 4 and not fourth["suggest_agent_now"]


def test_notice_never_blocks_a_command_when_local_state_is_read_only(
    feedback_home, monkeypatch, capsys
):
    monkeypatch.setattr(
        feedback,
        "configure",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read only")),
    )
    assert feedback.show_notice_once() is True
    notice = capsys.readouterr().err
    assert "Models, meshes, parameters, code, paths, and results are never included" in notice
    assert "agentfem telemetry off" in notice


def test_support_bundle_is_local_sanitized_and_integrity_checked(
    tmp_path, feedback_home
):
    execution = _failed_execution(tmp_path / "run")
    target = tmp_path / "support"
    record = feedback.create_support_directory(execution, destination=target)
    assert record["privacy"]["contains_model"] is False
    assert record["privacy"]["contains_absolute_paths"] is False
    combined = "\n".join(
        item.read_text(encoding="utf-8")
        for item in target.iterdir()
        if item.suffix in {".md", ".json"}
    )
    assert "Inconel" not in combined
    assert "/private/customer" not in combined
    assert "solve(secret_material)" not in combined
    assert str(tmp_path) not in combined
    assert __version__ in (target / "runtime.json").read_text(encoding="utf-8")
    archive = feedback.create_feedback_archive(
        execution,
        destination=tmp_path / "feedback.zip",
    )
    assert archive["sent"] is False
    with zipfile.ZipFile(archive["archive"]) as bundle:
        assert "agentfem-support/manifest.json" in bundle.namelist()


def test_cli_telemetry_is_machine_readable(feedback_home, capsys):
    assert cli.main(["telemetry", "status", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["mode"] == "basic"
    assert status["delivery_available"] is False
    assert cli.main(["telemetry", "off", "--json"]) == 0
    disabled = json.loads(capsys.readouterr().out)
    assert disabled["mode"] == "off"


def test_cli_diagnose_assist_and_feedback_do_not_upload(
    tmp_path, feedback_home, capsys
):
    execution = _failed_execution(tmp_path / "run")
    assert cli.main(["diagnose", str(execution), "--json"]) == 0
    diagnosis = json.loads(capsys.readouterr().out)
    assert diagnosis["failure"]["code"] == "AFM-SOLVE-007"
    support = tmp_path / "agent-task"
    assert cli.main(
        ["assist", str(execution), "--output", str(support), "--json"]
    ) == 0
    created = json.loads(capsys.readouterr().out)
    assert Path(created["directory"]) == support
    archive = tmp_path / "feedback.zip"
    assert cli.main(
        ["feedback", str(execution), "--output", str(archive), "--json"]
    ) == 0
    packaged = json.loads(capsys.readouterr().out)
    assert packaged["sent"] is False
    assert archive.is_file()
