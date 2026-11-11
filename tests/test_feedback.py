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
    monkeypatch.delenv("AGENTFEM_FEEDBACK_ROUTE", raising=False)
    monkeypatch.setattr(feedback, "_endpoint_records_from_package", lambda: ())
    monkeypatch.setattr(feedback, "_safe_runtime", lambda: dict(SAFE_RUNTIME))
    return home


def test_release_contains_the_owned_https_collector():
    manifest = json.loads(
        (Path(feedback.__file__).parent / "feedback-endpoint.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["activation"] == "explicit-consent-only"
    endpoints = feedback._endpoints_from_package()
    assert endpoints == (
        "https://agentfem-reliability.horming-luo.workers.dev/v1/reliability",
        "https://1480699086-evo2jiworj.ap-guangzhou.tencentscf.com/v1/reliability",
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
        event_id="00000000-0000-4000-8000-000000000001",
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
    with pytest.raises(ValueError, match="UUID"):
        feedback.validate_event({**event, "event_id": "customer-or-project"})


def test_feedback_is_off_by_default_and_requires_explicit_opt_in(feedback_home):
    selected = feedback.preferences()
    assert selected.mode == "off"
    assert selected.consent_granted is False
    event = feedback.build_event("run", "completed")
    assert feedback.enqueue(event) is None
    assert feedback.queue_size() == 0
    enabled = feedback.configure("basic")
    assert enabled.mode == "basic"
    assert enabled.consent_granted is True
    assert enabled.consent_source == "api"
    assert feedback.enqueue(event).is_file()
    assert feedback.configure("off").mode == "off"
    assert feedback.queue_size() == 0


def test_marking_notice_preserves_packaged_failover_routes(feedback_home, monkeypatch):
    routes = (
        "https://cn.example/v1/reliability",
        "https://global.example/v1/reliability",
    )
    monkeypatch.setattr(
        feedback,
        "_endpoint_records_from_package",
        lambda: (
            feedback.FeedbackEndpoint("china", "china", routes[0]),
            feedback.FeedbackEndpoint("global", "global", routes[1]),
        ),
    )

    assert feedback.preferences().endpoints == ()
    configured = feedback.configure("basic", notice_shown=True)

    assert configured.notice_shown is True
    assert configured.endpoints == routes
    stored = json.loads((feedback_home / "preferences.json").read_text(encoding="utf-8"))
    assert stored["endpoints"] == []
    assert stored["route"] == "auto"
    assert stored["consent"] == {
        "granted": True,
        "source": "api",
        "version": feedback.CONSENT_VERSION,
    }


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
    assert delivered == {
        "status": "sent", "sent": 1, "remaining": 0,
        "route": 0, "route_name": "custom-0",
    }
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
    feedback.configure("basic")
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

    assert delivered == {
        "status": "sent", "sent": 1, "remaining": 0,
        "route": 1, "route_name": "custom-1",
    }
    assert tuple(item[0] for item in attempted) == (
        "https://cn.example/v1/reliability",
        "https://global.example/v1/reliability",
    )
    assert all(item[1] <= feedback.DEFAULT_TIMEOUT_SECONDS / 2 for item in attempted)
    assert feedback.preferences().summary()["delivery_route_count"] == 2


def test_route_preference_reorders_reviewed_collectors_without_geolocation(
    feedback_home, monkeypatch
):
    records = (
        feedback.FeedbackEndpoint("global", "global", "https://global.example/v1/reliability"),
        feedback.FeedbackEndpoint("china", "china", "https://cn.example/v1/reliability"),
    )
    monkeypatch.setattr(feedback, "_endpoint_records_from_package", lambda: records)

    assert feedback.preferences().route_names == ()
    feedback.configure("basic")
    assert feedback.preferences().route_names == ("global", "china")
    selected = feedback.configure("basic", route="china")
    assert selected.route == "china"
    assert selected.route_names == ("china", "global")
    assert selected.endpoints[0] == "https://cn.example/v1/reliability"

    feedback._delivery_success("global")
    automatic = feedback.configure("basic", route="auto")
    assert automatic.route_names == ("global", "china")


def test_fresh_install_ordinary_cli_never_collects_or_opens_network(
    feedback_home, monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr(
        feedback.urlrequest,
        "urlopen",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        feedback,
        "build_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("telemetry event constructed without consent")
        ),
    )

    assert cli.main(["templates", "--json"]) == 0
    capsys.readouterr()

    assert calls == []
    assert feedback.preferences().mode == "off"
    assert feedback.preferences().consent_granted is False
    assert feedback.queue_size() == 0
    assert feedback.last_event() is None


def test_legacy_basic_preferences_are_not_treated_as_consent(
    feedback_home, monkeypatch, capsys
):
    feedback_home.mkdir(parents=True)
    (feedback_home / "preferences.json").write_text(
        json.dumps(
            {
                "schema": feedback.PREFERENCES_SCHEMA,
                "schema_version": "0.1.0",
                "mode": "basic",
                "route": "global",
                "notice_shown": True,
                "endpoints": ["https://legacy.example/v1/reliability"],
            }
        ),
        encoding="utf-8",
    )
    queue = feedback_home / "queue"
    queue.mkdir()
    event = feedback.build_event("run", "completed")
    (queue / "legacy.json").write_text(json.dumps(event), encoding="utf-8")
    (feedback_home / "delivery.json").write_text(
        json.dumps({"attempts": 3, "retry_after": 0}), encoding="utf-8"
    )
    calls = []
    monkeypatch.setattr(
        feedback.urlrequest,
        "urlopen",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert cli.main(["templates", "--json"]) == 0
    capsys.readouterr()

    assert calls == []
    assert feedback.queue_size() == 0
    assert not (feedback_home / "delivery.json").exists()
    migrated = json.loads(
        (feedback_home / "preferences.json").read_text(encoding="utf-8")
    )
    assert migrated["schema_version"] == feedback.PREFERENCES_SCHEMA_VERSION
    assert migrated["mode"] == "off"
    assert migrated["consent"]["granted"] is False


def test_invalid_preferences_and_endpoint_override_fail_closed(
    feedback_home, monkeypatch
):
    feedback_home.mkdir(parents=True)
    (feedback_home / "preferences.json").write_text(
        '{"schema":"wrong","mode":"basic"}\n', encoding="utf-8"
    )
    monkeypatch.setenv(
        "AGENTFEM_FEEDBACK_ENDPOINT",
        "https://reliability.example/v1/reliability",
    )
    calls = []
    monkeypatch.setattr(
        feedback.urlrequest,
        "urlopen",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    selected = feedback._ensure_privacy_state()
    assert selected.mode == "off"
    assert selected.consent_granted is False
    assert selected.endpoints == ()
    assert feedback.flush()["status"] == "disabled"
    assert calls == []


def test_environment_basic_is_explicit_process_authorization(
    feedback_home, monkeypatch
):
    monkeypatch.setenv("AGENTFEM_TELEMETRY", "basic")
    selected = feedback._ensure_privacy_state()
    assert selected.mode == "basic"
    assert selected.consent_granted is True
    assert selected.consent_source == "environment"
    assert feedback.enqueue(feedback.build_event("run", "completed")).is_file()

    monkeypatch.delenv("AGENTFEM_TELEMETRY")
    assert feedback.preferences().mode == "off"


def test_explicit_opt_in_network_failure_is_bounded_and_retriable(
    feedback_home, monkeypatch
):
    feedback.configure("basic", endpoint="https://reliability.example/v1/reliability")
    feedback.enqueue(feedback.build_event("run", "completed"))
    monkeypatch.setattr(
        feedback.urlrequest,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            feedback.urlerror.URLError("offline")
        ),
    )

    delivered = feedback.flush(timeout=0.01)

    assert delivered["status"] == "unavailable"
    assert delivered["sent"] == 0
    assert feedback.queue_size() == 1


def test_revocation_removes_queue_retry_and_route_state(feedback_home):
    feedback.configure("basic", endpoint="https://reliability.example/v1/reliability")
    feedback.enqueue(feedback.build_event("run", "completed"))
    feedback._delivery_failure(0.0)
    feedback._delivery_success("custom-0")

    feedback.configure("off")

    assert feedback.queue_size() == 0
    assert feedback.last_event() is None
    assert not feedback._backoff_path().exists()
    assert not feedback._route_preference_path().exists()



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
    assert "off by default" in notice
    assert "agentfem telemetry basic" in notice


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
    assert status["mode"] == "off"
    assert status["consent"]["granted"] is False
    assert status["delivery_available"] is False
    assert cli.main(["telemetry", "route", "china", "--json"]) == 0
    routed_off = json.loads(capsys.readouterr().out)
    assert routed_off["mode"] == "off"
    assert routed_off["route"] == "china"
    assert cli.main(["telemetry", "basic", "--json"]) == 0
    enabled = json.loads(capsys.readouterr().out)
    assert enabled["mode"] == "basic"
    assert enabled["consent"] == {
        "required": True,
        "granted": True,
        "source": "cli",
    }
    assert cli.main(["telemetry", "off", "--json"]) == 0
    disabled = json.loads(capsys.readouterr().out)
    assert disabled["mode"] == "off"
    assert disabled["consent"]["granted"] is False


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
