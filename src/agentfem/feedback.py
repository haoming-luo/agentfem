"""Privacy-bounded reliability feedback and support escalation.

The reliability channel deliberately accepts only a small, schema-checked
event.  It never serializes a Model, mesh, material parameter, source file,
result field, project name, path, traceback, or exception message.  Richer
support evidence is generated locally and leaves the machine only through an
explicit user action such as ``agentfem feedback --github``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import platform as _platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Mapping, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest
import uuid
import zipfile


PREFERENCES_SCHEMA = "agentfem.feedback-preferences"
EVENT_SCHEMA = "agentfem.reliability-event"
BATCH_SCHEMA = "agentfem.reliability-batch"
DIAGNOSIS_SCHEMA = "agentfem.support-diagnosis"
SUPPORT_BUNDLE_SCHEMA = "agentfem.support-bundle"
SCHEMA_VERSION = "0.1.0"
DEFAULT_MODE = "basic"
REPOSITORY = "haoming-luo/agentfem"
ISSUES_URL = f"https://github.com/{REPOSITORY}/issues"
MAX_QUEUED_EVENTS = 64
MAX_EVENT_BYTES = 8 * 1024
MAX_BATCH_EVENTS = 8
# Preserve the original 0.75 s allowance per reviewed collector when two
# independent routes are packaged. The exponential backoff prevents this
# worst-case budget from recurring on every command while offline.
DEFAULT_TIMEOUT_SECONDS = 1.5
FAILURE_ESCALATION_COUNT = 3

_MODES = ("basic", "off")
_ROUTES = ("auto", "global", "china")
_OUTCOMES = ("completed", "failed", "cancelled")
_DURATION_BUCKETS = (
    "<1s",
    "1-10s",
    "10-60s",
    "1-10m",
    "10-60m",
    ">=1h",
    "unknown",
)
_EVENT_KEYS = {
    "schema",
    "schema_version",
    "event_id",
    "agentfem_version",
    "command",
    "outcome",
    "duration_bucket",
    "runtime",
    "failure",
}
_RUNTIME_KEYS = {
    "system",
    "route",
    "machine",
    "python",
    "dolfinx",
    "petsc4py",
    "mpi_vendor",
    "mpi_ranks",
    "installation",
}
_FAILURE_KEYS = {"code", "stage", "kind", "fingerprint"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    selected = (value or _utc_now()).astimezone(timezone.utc)
    # Minute precision is enough for product reliability and avoids recording
    # an unnecessarily precise user activity timestamp.
    return selected.replace(second=0, microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        temporary.chmod(0o600)
    except OSError:
        pass
    temporary.replace(path)


def _feedback_home() -> Path:
    override = os.environ.get("AGENTFEM_FEEDBACK_HOME")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if root:
            return Path(root) / "AgentFEM" / "feedback"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "AgentFEM" / "feedback"
    state = os.environ.get("XDG_STATE_HOME")
    return (
        Path(state).expanduser() / "agentfem" / "feedback"
        if state
        else Path.home() / ".local" / "state" / "agentfem" / "feedback"
    )


def _preferences_path() -> Path:
    return _feedback_home() / "preferences.json"


def _spool_directory() -> Path:
    return _feedback_home() / "queue"


def _failure_directory() -> Path:
    return _feedback_home() / "failures"


def _last_event_path() -> Path:
    return _feedback_home() / "last-event.json"


@dataclass(frozen=True)
class FeedbackEndpoint:
    """One reviewed collector route; never inferred from the user's address."""

    name: str
    region: str
    url: str


def _endpoint_records_from_package() -> tuple[FeedbackEndpoint, ...]:
    """Return reviewed delivery routes and their non-user geographic scope."""

    try:
        path = resources.files("agentfem") / "feedback-endpoint.json"
        record = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return ()
    declared = record.get("endpoints")
    if declared is None:
        value = record.get("endpoint")
        if not value:
            return ()
        try:
            return (FeedbackEndpoint("global", "global", _validated_endpoint(value)),)
        except (TypeError, ValueError):
            return ()
    if not isinstance(declared, list):
        return ()
    accepted: list[FeedbackEndpoint] = []
    for index, item in enumerate(declared):
        value = item.get("url") if isinstance(item, Mapping) else item
        try:
            endpoint = _validated_endpoint(value)
        except (TypeError, ValueError):
            continue
        if any(record.url == endpoint for record in accepted):
            continue
        name = str(item.get("name", f"route-{index}")) if isinstance(item, Mapping) else f"route-{index}"
        region = str(item.get("region", "global")) if isinstance(item, Mapping) else "global"
        accepted.append(FeedbackEndpoint(name=name[:32], region=region[:32], url=endpoint))
    return tuple(accepted)


def _endpoints_from_package() -> tuple[str, ...]:
    """Compatibility view of reviewed URLs in their declared order."""

    return tuple(record.url for record in _endpoint_records_from_package())


def _endpoint_from_package() -> str | None:
    """Compatibility view of the first reviewed delivery route."""

    endpoints = _endpoints_from_package()
    return endpoints[0] if endpoints else None


def _validated_endpoint(value: object) -> str:
    selected = str(value).strip()
    if not selected:
        raise ValueError("The reliability endpoint cannot be empty.")
    if not selected.startswith("https://") and not selected.startswith("http://127.0.0.1"):
        raise ValueError("The reliability endpoint must use HTTPS.")
    return selected.rstrip("/")


@dataclass(frozen=True)
class FeedbackPreferences:
    """Persistent, user-controllable reliability-report preferences."""

    mode: str = DEFAULT_MODE
    notice_shown: bool = False
    endpoint: str | None = None
    fallback_endpoints: tuple[str, ...] = ()
    route: str = "auto"
    route_names: tuple[str, ...] = ()

    @property
    def endpoints(self) -> tuple[str, ...]:
        """Return primary and fallback routes without duplicates."""

        result = []
        for value in (self.endpoint, *self.fallback_endpoints):
            if value and value not in result:
                result.append(value)
        return tuple(result)

    def summary(self) -> dict[str, object]:
        return {
            "schema": PREFERENCES_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode,
            "route": self.route,
            "notice_shown": self.notice_shown,
            "endpoint": self.endpoint,
            "delivery_available": bool(self.endpoints),
            "delivery_route_count": len(self.endpoints),
            "delivery_routes": list(self.route_names),
            "queue_size": queue_size(),
            "privacy": {
                "models": "never",
                "meshes": "never",
                "parameters": "never",
                "source_code": "never",
                "paths": "never",
                "results": "never",
                "tracebacks": "never_in_basic_reports",
            },
        }


def preferences() -> FeedbackPreferences:
    """Return the effective preferences without creating local state."""

    mode_override = os.environ.get("AGENTFEM_TELEMETRY")
    endpoint_override = os.environ.get("AGENTFEM_FEEDBACK_ENDPOINT")
    endpoints_override = os.environ.get("AGENTFEM_FEEDBACK_ENDPOINTS")
    route_override = os.environ.get("AGENTFEM_FEEDBACK_ROUTE")
    record: dict[str, object] = {}
    path = _preferences_path()
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    mode = str(mode_override or record.get("mode", DEFAULT_MODE)).strip().lower()
    if mode not in _MODES:
        mode = DEFAULT_MODE
    route = str(route_override or record.get("route", "auto")).strip().lower()
    if route not in _ROUTES:
        route = "auto"
    declared = ()
    recorded_endpoints = record.get("endpoints")
    if endpoints_override:
        declared = tuple(item.strip() for item in endpoints_override.split(",") if item.strip())
    elif endpoint_override:
        declared = (endpoint_override,)
    elif isinstance(recorded_endpoints, list) and recorded_endpoints:
        declared = tuple(recorded_endpoints)
    elif record.get("endpoint"):
        declared = (record.get("endpoint"),)
    else:
        packaged = list(_endpoint_records_from_package())
        preferred = _last_successful_route()
        if route != "auto":
            packaged.sort(key=lambda item: (item.region != route and item.name != route))
        elif preferred:
            packaged.sort(key=lambda item: item.name != preferred)
        declared = tuple(item.url for item in packaged)
    accepted = []
    for endpoint in declared:
        try:
            selected = _validated_endpoint(endpoint)
        except (TypeError, ValueError):
            continue
        if selected not in accepted:
            accepted.append(selected)
    packaged_names = {item.url: item.name for item in _endpoint_records_from_package()}
    route_names = tuple(packaged_names.get(endpoint, f"custom-{index}") for index, endpoint in enumerate(accepted))
    return FeedbackPreferences(
        mode=mode,
        notice_shown=bool(record.get("notice_shown", False)),
        endpoint=accepted[0] if accepted else None,
        fallback_endpoints=tuple(accepted[1:]),
        route=route,
        route_names=route_names,
    )


def configure(
    mode: str,
    *,
    route: str | None = None,
    endpoint: str | None = None,
    notice_shown: bool | None = None,
) -> FeedbackPreferences:
    """Persist ``basic`` or ``off`` and return the effective preferences."""

    selected = str(mode).strip().lower()
    if selected not in _MODES:
        raise ValueError(f"Feedback mode must be one of {_MODES}; received {mode!r}.")
    previous = preferences()
    selected_route = previous.route if route is None else str(route).strip().lower()
    if selected_route not in _ROUTES:
        raise ValueError(f"Feedback route must be one of {_ROUTES}; received {route!r}.")
    selected_endpoints = (
        previous.endpoints
        if endpoint is None and any(name.startswith("custom-") for name in previous.route_names)
        else () if endpoint is None
        else (_validated_endpoint(endpoint),)
    )
    record = {
        "schema": PREFERENCES_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "mode": selected,
        "route": selected_route,
        "notice_shown": previous.notice_shown if notice_shown is None else bool(notice_shown),
        "endpoints": list(selected_endpoints),
    }
    _atomic_json(_preferences_path(), record)
    if selected == "off":
        clear_queue()
    return preferences()


def notice_text() -> str:
    opening = (
        "AgentFEM shares a minimal anonymous reliability signal to improve this "
        "free, open-source software."
        if preferences().endpoint
        else "AgentFEM records anonymous error and reliability reports locally; "
        "online delivery is not configured in this build."
    )
    return (
        f"{opening} Models, meshes, parameters, code, paths, and results are never included.\n"
        "Turn off anytime: agentfem telemetry off"
    )


def show_notice_once(*, as_json: bool = False, stream=None) -> bool:
    """Show the one-time reliability notice on an interactive human route."""

    testing = bool(os.environ.get("AGENTFEM_TELEMETRY_TESTING"))
    if (
        as_json
        or (os.environ.get("CI") and not testing)
        or os.environ.get("AGENTFEM_TELEMETRY")
        or (
            os.environ.get("PYTEST_CURRENT_TEST")
            and not testing
        )
    ):
        return False
    selected = preferences()
    if selected.notice_shown:
        return False
    target = stream or sys.stderr
    print(notice_text(), file=target, flush=True)
    try:
        configure(selected.mode, notice_shown=True)
    except OSError:
        # A read-only home directory must never block doctor/check/run.
        pass
    return True


def _duration_bucket(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    if seconds < 1:
        return "<1s"
    if seconds < 10:
        return "1-10s"
    if seconds < 60:
        return "10-60s"
    if seconds < 600:
        return "1-10m"
    if seconds < 3600:
        return "10-60m"
    return ">=1h"


def _safe_error_kind(value: object) -> str:
    selected = str(value or "unknown").strip()
    leaf = selected.rsplit(".", 1)[-1]
    allowed = {
        "AssertionError",
        "FileNotFoundError",
        "ImportError",
        "IndexError",
        "KeyError",
        "MemoryError",
        "ModuleNotFoundError",
        "OSError",
        "RuntimeError",
        "SyntaxError",
        "TimeoutError",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
    }
    if leaf in allowed or leaf.startswith("AgentFEM"):
        return leaf[:80]
    if selected.startswith("agentfem.") and re.fullmatch(r"[A-Za-z0-9_.]+", selected):
        return leaf[:80]
    return "ExternalError"


def failure_fingerprint(error: Mapping[str, object] | None, *, command: str = "run") -> str | None:
    """Return a message/path-free identity for one class of failure."""

    if not error:
        return None
    payload = {
        "command": str(command),
        "code": str(error.get("code") or "unclassified"),
        "stage": str(error.get("stage") or "unknown"),
        "kind": _safe_error_kind(error.get("type")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "AFM-FP-" + hashlib.sha256(encoded).hexdigest()[:12].upper()


def _safe_runtime() -> dict[str, object]:
    from . import platforms

    report = platforms.runtime_report().summary()
    packages = report.get("packages", {})
    mpi = report.get("mpi", {})
    execution = report.get("execution", {})
    platform = report.get("platform", {})
    return {
        "system": str(report.get("operating_system", {}).get("system", "unknown"))[:32],
        "route": str(platform.get("route", "unknown"))[:80],
        "machine": str(report.get("machine", "unknown"))[:32],
        "python": str(report.get("python", "unknown"))[:32],
        "dolfinx": packages.get("fenics-dolfinx"),
        "petsc4py": packages.get("petsc4py"),
        "mpi_vendor": str(mpi.get("vendor", "unknown"))[:80],
        "mpi_ranks": int(mpi.get("rank_count", 1)),
        "installation": str(execution.get("mode", "unknown"))[:40],
    }


def build_event(
    command: str,
    outcome: str,
    *,
    duration_seconds: float | None = None,
    error: Mapping[str, object] | None = None,
    now: datetime | None = None,
    event_id: str | None = None,
) -> dict[str, object]:
    """Build the exact data allowed on the automatic reliability channel."""

    from . import __version__

    selected_outcome = str(outcome).strip().lower()
    if selected_outcome not in _OUTCOMES:
        raise ValueError(f"Reliability outcome must be one of {_OUTCOMES}.")
    failure = None
    if selected_outcome == "failed":
        source = error or {}
        failure = {
            "code": str(source.get("code") or "unclassified")[:80],
            "stage": str(source.get("stage") or "unknown")[:80],
            "kind": _safe_error_kind(source.get("type")),
            "fingerprint": failure_fingerprint(source, command=command),
        }
    record = {
        "schema": EVENT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "event_id": str(event_id or uuid.uuid4()),
        "agentfem_version": __version__,
        "command": str(command).strip().lower()[:64],
        "outcome": selected_outcome,
        "duration_bucket": _duration_bucket(duration_seconds),
        "runtime": _safe_runtime(),
        "failure": failure,
    }
    validate_event(record)
    return record


def validate_event(record: Mapping[str, object]) -> None:
    """Fail closed when an event contains an undeclared field."""

    if set(record) != _EVENT_KEYS or record.get("schema") != EVENT_SCHEMA:
        raise ValueError("Reliability event does not match the public whitelist.")
    runtime = record.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != _RUNTIME_KEYS:
        raise ValueError("Reliability runtime does not match the public whitelist.")
    failure = record.get("failure")
    if failure is not None and (
        not isinstance(failure, Mapping) or set(failure) != _FAILURE_KEYS
    ):
        raise ValueError("Reliability failure does not match the public whitelist.")
    if record.get("outcome") not in _OUTCOMES:
        raise ValueError("Reliability event outcome is invalid.")
    if record.get("duration_bucket") not in _DURATION_BUCKETS:
        raise ValueError("Reliability duration bucket is invalid.")
    try:
        event_id = uuid.UUID(str(record.get("event_id")))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Reliability event ID must be a UUID.") from exc
    if event_id.version not in {1, 2, 3, 4, 5}:
        raise ValueError("Reliability event ID must be a versioned UUID.")
    encoded = json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")
    if len(encoded) > MAX_EVENT_BYTES:
        raise ValueError("Reliability event exceeds the maximum safe size.")


def queue_size() -> int:
    try:
        return len(tuple(_spool_directory().glob("*.json")))
    except OSError:
        return 0


def clear_queue() -> int:
    """Remove every unsent automatic event when reporting is disabled."""

    removed = 0
    for path in _queued_paths():
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    for path in (
        _backoff_path(),
        _feedback_home() / "success-sample.json",
        _last_event_path(),
    ):
        try:
            path.unlink()
        except OSError:
            pass
    return removed


def _queued_paths() -> tuple[Path, ...]:
    try:
        return tuple(sorted(_spool_directory().glob("*.json")))
    except OSError:
        return ()


def enqueue(record: Mapping[str, object]) -> Path | None:
    """Queue one event atomically; never write when reporting is disabled."""

    if preferences().mode != "basic":
        return None
    validate_event(record)
    queue = _spool_directory()
    queue.mkdir(parents=True, exist_ok=True)
    paths = _queued_paths()
    for obsolete in paths[: max(0, len(paths) - MAX_QUEUED_EVENTS + 1)]:
        try:
            obsolete.unlink()
        except OSError:
            pass
    event_id = str(record["event_id"])
    path = queue / f"{int(time.time() * 1_000_000):020d}-{event_id}.json"
    _atomic_json(path, record)
    _atomic_json(_last_event_path(), record)
    return path


def last_event() -> dict[str, object] | None:
    paths = _queued_paths()
    selected = paths[-1] if paths else _last_event_path()
    try:
        record = json.loads(selected.read_text(encoding="utf-8"))
        validate_event(record)
        return record
    except (OSError, json.JSONDecodeError):
        return None
    except ValueError:
        return None


def _backoff_path() -> Path:
    return _feedback_home() / "delivery.json"


def _route_preference_path() -> Path:
    return _feedback_home() / "last-route.json"


def _last_successful_route() -> str | None:
    try:
        record = json.loads(_route_preference_path().read_text(encoding="utf-8"))
        value = str(record.get("route", "")).strip()
        return value or None
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _delivery_allowed(now: float) -> bool:
    try:
        record = json.loads(_backoff_path().read_text(encoding="utf-8"))
        return now >= float(record.get("retry_after", 0.0))
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return True


def _delivery_failure(now: float) -> None:
    attempts = 0
    try:
        attempts = int(json.loads(_backoff_path().read_text(encoding="utf-8")).get("attempts", 0))
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    attempts = min(attempts + 1, 8)
    delay = min(3600.0, 30.0 * (2 ** (attempts - 1)))
    _atomic_json(
        _backoff_path(),
        {"schema": "agentfem.feedback-delivery", "attempts": attempts, "retry_after": now + delay},
    )


def _delivery_success(route: str) -> None:
    try:
        _backoff_path().unlink()
    except FileNotFoundError:
        pass
    _atomic_json(
        _route_preference_path(),
        {"schema": "agentfem.feedback-route", "route": str(route)[:32]},
    )


def flush(*, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, object]:
    """Send a bounded batch and return status without raising into a solve."""

    selected = preferences()
    paths = _queued_paths()[:MAX_BATCH_EVENTS]
    if selected.mode != "basic":
        return {"status": "disabled", "sent": 0, "remaining": len(paths)}
    if not selected.endpoints:
        return {"status": "endpoint_unavailable", "sent": 0, "remaining": queue_size()}
    if not paths:
        return {"status": "empty", "sent": 0, "remaining": 0}
    now = time.time()
    if not _delivery_allowed(now):
        return {"status": "backoff", "sent": 0, "remaining": queue_size()}
    events = []
    valid_paths = []
    for path in paths:
        try:
            event = json.loads(path.read_text(encoding="utf-8"))
            validate_event(event)
        except (OSError, json.JSONDecodeError, ValueError):
            try:
                path.unlink()
            except OSError:
                pass
            continue
        events.append(event)
        valid_paths.append(path)
    if not events:
        return {"status": "empty", "sent": 0, "remaining": queue_size()}
    payload = json.dumps(
        {"schema": BATCH_SCHEMA, "schema_version": SCHEMA_VERSION, "events": events},
        separators=(",", ":"),
    ).encode("utf-8")
    delivered_route = None
    total_timeout = max(0.0, float(timeout))
    route_timeout = total_timeout / len(selected.endpoints)
    for route_index, endpoint in enumerate(selected.endpoints):
        request = urlrequest.Request(
            endpoint,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"AgentFEM/{events[0]['agentfem_version']} reliability",
            },
        )
        try:
            with urlrequest.urlopen(request, timeout=route_timeout) as response:
                if not 200 <= int(response.status) < 300:
                    raise OSError(f"collector returned HTTP {response.status}")
            delivered_route = route_index
            break
        except (OSError, TimeoutError, urlerror.URLError):
            continue
    if delivered_route is None:
        _delivery_failure(now)
        return {"status": "unavailable", "sent": 0, "remaining": queue_size()}
    for path in valid_paths:
        try:
            path.unlink()
        except OSError:
            pass
    delivered_name = (
        selected.route_names[delivered_route]
        if delivered_route < len(selected.route_names)
        else f"route-{delivered_route}"
    )
    _delivery_success(delivered_name)
    return {
        "status": "sent",
        "sent": len(valid_paths),
        "remaining": queue_size(),
        "route": delivered_route,
        "route_name": delivered_name,
    }


def _success_sample_allowed(now: datetime | None = None) -> bool:
    selected = (now or _utc_now()).astimezone(timezone.utc).date().isoformat()
    path = _feedback_home() / "success-sample.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("date") == selected:
            return False
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    _atomic_json(path, {"schema": "agentfem.feedback-success-sample", "date": selected})
    return True


def _load_execution_for_project(project: str | Path | None) -> dict[str, object] | None:
    try:
        from .project import discover

        selected = discover(project)
        pointer = selected.output_directory / selected.name / "latest.json"
        latest = json.loads(pointer.read_text(encoding="utf-8"))
        execution = Path(str(latest["execution_record"])).expanduser()
        if not execution.is_absolute():
            execution = (pointer.parent / execution).resolve()
        return json.loads(execution.read_text(encoding="utf-8"))
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError):
        return None


def observe_cli(
    command: str | None,
    exit_code: int,
    *,
    duration_seconds: float | None = None,
    project: str | Path | None = None,
) -> dict[str, object] | None:
    """Record one CLI outcome; this function is deliberately fail-open."""

    if (
        preferences().mode != "basic"
        or not command
        or command in {"telemetry", "diagnose", "assist", "feedback"}
    ):
        return None
    if (
        (os.environ.get("CI") or os.environ.get("PYTEST_CURRENT_TEST"))
        and not os.environ.get("AGENTFEM_TELEMETRY_TESTING")
    ):
        return None
    try:
        from mpi4py import MPI

        if MPI.COMM_WORLD.rank != 0:
            return None
    except Exception:
        pass
    outcome = "completed" if int(exit_code) == 0 else "failed"
    execution = _load_execution_for_project(project) if command == "run" else None
    error = execution.get("error") if execution and outcome == "failed" else None
    if outcome == "completed" and not _success_sample_allowed():
        return None
    try:
        event = build_event(
            command,
            outcome,
            duration_seconds=duration_seconds,
            error=error if isinstance(error, Mapping) else None,
        )
        path = enqueue(event)
        if outcome == "failed" and isinstance(error, Mapping):
            escalation = record_failure(error, command=command)
        else:
            escalation = None
        delivery = flush()
        return {
            "event": event,
            "queued": None if path is None else str(path),
            "delivery": delivery,
            "escalation": escalation,
        }
    except Exception:
        return None


def record_failure(error: Mapping[str, object], *, command: str = "run") -> dict[str, object]:
    """Update a local, version-scoped repetition counter for one failure class."""

    from . import __version__

    fingerprint = failure_fingerprint(error, command=command)
    if fingerprint is None:
        raise ValueError("Cannot record a failure without error evidence.")
    path = _failure_directory() / f"{fingerprint}.json"
    previous: dict[str, object] = {}
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    previous_version = previous.get("agentfem_version")
    count = int(previous.get("count", 0)) + 1 if previous_version == __version__ else 1
    previously_suggested = bool(previous.get("agent_suggested", False)) if previous_version == __version__ else False
    should_suggest = count >= FAILURE_ESCALATION_COUNT and not previously_suggested
    record = {
        "schema": "agentfem.failure-repetition",
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "agentfem_version": __version__,
        "code": str(error.get("code") or "unclassified")[:80],
        "stage": str(error.get("stage") or "unknown")[:80],
        "kind": _safe_error_kind(error.get("type")),
        "count": count,
        "first_seen": previous.get("first_seen") if previous_version == __version__ else _timestamp(),
        "last_seen": _timestamp(),
        "agent_suggested": previously_suggested or should_suggest,
        "suggest_agent_now": should_suggest,
    }
    _atomic_json(path, record)
    return record


def _resolve_execution(path: str | Path | None = None, *, project: str | Path | None = None) -> Path:
    if path is None:
        from .project import discover

        selected = discover(project)
        candidate = selected.output_directory / selected.name / "latest.json"
    else:
        candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        direct = candidate / "execution.json"
        candidate = direct if direct.is_file() else candidate / "latest.json"
    for _ in range(4):
        if not candidate.is_file():
            raise FileNotFoundError(f"No AgentFEM execution evidence found at {candidate}.")
        record = json.loads(candidate.read_text(encoding="utf-8"))
        if record.get("schema") == "agentfem.run" and "status" in record:
            return candidate
        target = record.get("execution_record")
        if target is None:
            sibling = candidate.parent / "execution.json"
            if sibling.is_file():
                return sibling
            raise ValueError(f"{candidate} does not reference an AgentFEM execution record.")
        selected = Path(str(target)).expanduser()
        candidate = selected.resolve() if selected.is_absolute() else (candidate.parent / selected).resolve()
    raise ValueError("AgentFEM execution pointer chain is unexpectedly deep.")


def _support_suggestions(error: Mapping[str, object] | None) -> tuple[str, ...]:
    if not error:
        return ()
    code = str(error.get("code") or "")
    stage = str(error.get("stage") or "")
    kind = _safe_error_kind(error.get("type"))
    suggestions = []
    if stage == "model_preflight" or code.startswith("AFM-STEP") or code.startswith("AFM-CONSTRAINT"):
        suggestions.append("Run `agentfem check --json` and resolve every addressable validation issue before solving.")
    if "MPI" in code or "MPI" in kind:
        suggestions.append("Run `agentfem doctor` and use `agentfem run --mpi N` so the launcher matches mpi4py.")
    if kind in {"ImportError", "ModuleNotFoundError"}:
        suggestions.append("Run `agentfem doctor` and install only the optional capability reported as missing.")
    if not suggestions:
        suggestions.append("Inspect the execution stage and the smallest reproducible case; do not change scientific assumptions blindly.")
    suggestions.append("If the same failure persists, give `agentfem assist` to Codex or another AI agent.")
    return tuple(suggestions)


def diagnose(path: str | Path | None = None, *, project: str | Path | None = None) -> dict[str, object]:
    """Explain one execution locally without uploading or modifying the case."""

    selected = _resolve_execution(path, project=project)
    record = json.loads(selected.read_text(encoding="utf-8"))
    error = record.get("error") if isinstance(record.get("error"), Mapping) else None
    fingerprint = failure_fingerprint(error, command="run")
    repetition = None
    if fingerprint:
        try:
            repetition = json.loads((_failure_directory() / f"{fingerprint}.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass
    return {
        "schema": DIAGNOSIS_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": record.get("status", "unknown"),
        "stage": record.get("stage"),
        "failure": None
        if error is None
        else {
            "code": str(error.get("code") or "unclassified")[:80],
            "kind": _safe_error_kind(error.get("type")),
            "fingerprint": fingerprint,
            "repetition_count": None if repetition is None else repetition.get("count"),
        },
        "suggestions": _support_suggestions(error),
        "execution_record": str(selected),
        "network_used": False,
    }


def format_diagnosis(record: Mapping[str, object]) -> str:
    lines = [f"AgentFEM diagnosis: {record.get('status', 'unknown')}"]
    failure = record.get("failure")
    if isinstance(failure, Mapping):
        lines.append(f"  code: {failure.get('code')}")
        lines.append(f"  kind: {failure.get('kind')}")
        lines.append(f"  fingerprint: {failure.get('fingerprint')}")
        if failure.get("repetition_count"):
            lines.append(f"  repeated: {failure.get('repetition_count')} run(s)")
    for item in record.get("suggestions", ()):
        lines.append(f"  next: {item}")
    return "\n".join(lines)


def _sanitized_runtime() -> dict[str, object]:
    from . import __version__

    return {
        "schema": "agentfem.support-runtime",
        "schema_version": SCHEMA_VERSION,
        "agentfem_version": __version__,
        **_safe_runtime(),
    }


def _sanitized_traceback(value: object, *, project_root: Path | None = None) -> str:
    text = str(value or "")
    home = str(Path.home())
    if home:
        text = text.replace(home, "<HOME>")
    if project_root is not None:
        text = text.replace(str(project_root), "<PROJECT>")
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("File "):
            match = re.match(r'^(\s*)File "([^"]+)", line (\d+), in (.+)$', line)
            if match:
                indent, raw_path, line_number, function = match.groups()
                path = raw_path
                if "agentfem" in Path(raw_path).parts:
                    parts = Path(raw_path).parts
                    index = parts.index("agentfem")
                    path = "/".join(parts[index:])
                elif project_root is not None:
                    try:
                        path = "<PROJECT>/" + str(Path(raw_path).resolve().relative_to(project_root))
                    except (OSError, ValueError):
                        path = f"<EXTERNAL>/{Path(raw_path).name}"
                else:
                    path = f"<EXTERNAL>/{Path(raw_path).name}"
                lines.append(f'{indent}File "{path}", line {line_number}, in {function}')
            continue
        # Source lines and exception messages can contain user parameters,
        # regions, paths, or code.  Preserve only structural traceback lines.
        if stripped.startswith("Traceback") or stripped.startswith("During handling"):
            lines.append(line)
        elif re.match(r"^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception):", stripped):
            kind = stripped.partition(":")[0]
            lines.append(f"{kind}: <message removed>")
    return "\n".join(lines).strip() + "\n"


def _sanitized_execution(path: Path) -> dict[str, object]:
    record = json.loads(path.read_text(encoding="utf-8"))
    error = record.get("error") if isinstance(record.get("error"), Mapping) else None
    root = None
    try:
        root = Path(str(record.get("project_root"))).resolve()
    except (TypeError, OSError):
        pass
    selected_error = None
    if error is not None:
        selected_error = {
            "type": _safe_error_kind(error.get("type")),
            "code": str(error.get("code") or "unclassified")[:80],
            "stage": str(error.get("stage") or "unknown")[:80],
            "fingerprint": failure_fingerprint(error),
            "traceback": _sanitized_traceback(error.get("traceback"), project_root=root),
        }
    return {
        "schema": "agentfem.sanitized-execution",
        "schema_version": SCHEMA_VERSION,
        "status": record.get("status"),
        "stage": record.get("stage"),
        "structured_result": bool(record.get("structured_result", False)),
        "error": selected_error,
        "privacy": "paths, names, messages, code, model data, and results removed",
    }


def _support_task(diagnosis_record: Mapping[str, object]) -> str:
    fingerprint = (diagnosis_record.get("failure") or {}).get("fingerprint") if isinstance(diagnosis_record.get("failure"), Mapping) else None
    return f"""# AgentFEM support task

Diagnose and repair the AgentFEM execution represented by the files in this
directory.  The bundle is deliberately sanitized and contains no model source,
mesh, material parameters, or result fields.

Failure fingerprint: `{fingerprint or 'none'}`

1. Read `diagnosis.json`, `execution.json`, and `runtime.json`.
2. Use the public AgentFEM workflow and stable AFM issue codes.
3. Ask the user for the project directory only if local inspection is needed.
4. Do not change physics, units, loads, materials, or verification tolerances
   merely to make the run complete.
5. Run `agentfem doctor`, `agentfem check`, the repaired case, and
   `agentfem verify` where a structured result exists.
6. If the failure remains, prepare `issue.md`; do not publish it without the
   user's authorization.
"""


def _issue_markdown(diagnosis_record: Mapping[str, object], runtime: Mapping[str, object]) -> str:
    failure = diagnosis_record.get("failure") if isinstance(diagnosis_record.get("failure"), Mapping) else {}
    suggestions = "\n".join(f"- {item}" for item in diagnosis_record.get("suggestions", ()))
    return f"""## AgentFEM diagnostic report

This report was generated by `agentfem feedback`. It contains no model,
mesh, source code, material parameters, paths, or result fields.

- Failure code: `{failure.get('code', 'unclassified')}`
- Failure kind: `{failure.get('kind', 'unknown')}`
- Failure fingerprint: `{failure.get('fingerprint', 'none')}`
- AgentFEM version: `{runtime.get('agentfem_version', runtime.get('agentfem', 'unknown'))}`
- Platform route: `{runtime.get('route', 'unknown')}`
- Python: `{runtime.get('python', 'unknown')}`
- DOLFINx: `{runtime.get('dolfinx', 'unknown')}`
- PETSc/petsc4py: `{runtime.get('petsc4py', 'unknown')}`
- MPI vendor/ranks: `{runtime.get('mpi_vendor', 'unknown')}` / `{runtime.get('mpi_ranks', 'unknown')}`

### Local guidance already attempted

{suggestions or '- No local suggestion was available.'}

### Minimal reproducer

Please add a synthetic reproducer if one can be shared safely. Do not attach
confidential geometry, material data, customer files, or simulation results.
"""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_support_directory(
    path: str | Path | None = None,
    *,
    project: str | Path | None = None,
    destination: str | Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Create a local, sanitized task that any AI agent can inspect."""

    execution_path = _resolve_execution(path, project=project)
    diagnosis_record = diagnose(execution_path)
    failure = diagnosis_record.get("failure") if isinstance(diagnosis_record.get("failure"), Mapping) else {}
    fingerprint = str(failure.get("fingerprint") or "completed")
    target = Path(destination or f"agentfem-support-{fingerprint.lower()}").expanduser().resolve()
    if target.exists() and any(target.iterdir()) and not force:
        raise FileExistsError(f"Support directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    runtime = _sanitized_runtime()
    portable_diagnosis = {
        key: value
        for key, value in diagnosis_record.items()
        if key != "execution_record"
    }
    files = {
        "TASK.md": _support_task(portable_diagnosis),
        "diagnosis.json": json.dumps(portable_diagnosis, indent=2, sort_keys=True) + "\n",
        "execution.json": json.dumps(_sanitized_execution(execution_path), indent=2, sort_keys=True) + "\n",
        "runtime.json": json.dumps(runtime, indent=2, sort_keys=True) + "\n",
        "issue.md": _issue_markdown(portable_diagnosis, runtime),
    }
    for name, content in files.items():
        (target / name).write_text(content, encoding="utf-8")
    manifest = {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "created_at": _timestamp(),
        "privacy": {
            "contains_model": False,
            "contains_mesh": False,
            "contains_parameters": False,
            "contains_source_code": False,
            "contains_result_fields": False,
            "contains_absolute_paths": False,
        },
        "files": {
            name: {"sha256": _file_sha256(target / name), "bytes": (target / name).stat().st_size}
            for name in sorted(files)
        },
    }
    _atomic_json(target / "manifest.json", manifest)
    return {**manifest, "directory": str(target)}


def create_feedback_archive(
    path: str | Path | None = None,
    *,
    project: str | Path | None = None,
    destination: str | Path | None = None,
) -> dict[str, object]:
    """Create a portable local support archive without sending it."""

    with tempfile.TemporaryDirectory(prefix="agentfem-support-") as raw:
        support = Path(raw) / "support"
        record = create_support_directory(path, project=project, destination=support)
        fingerprint = str(record["fingerprint"]).lower()
        archive = Path(destination or f"agentfem-feedback-{fingerprint}.zip").expanduser().resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(archive.suffix + ".tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for item in sorted(support.iterdir()):
                bundle.write(item, arcname=f"agentfem-support/{item.name}")
        temporary.replace(archive)
    return {
        "schema": "agentfem.feedback-archive",
        "schema_version": SCHEMA_VERSION,
        "archive": str(archive),
        "sha256": _file_sha256(archive),
        "bytes": archive.stat().st_size,
        "fingerprint": record["fingerprint"],
        "sent": False,
    }


def _github_available() -> bool:
    if shutil.which("gh") is None:
        return False
    return subprocess.run(
        ("gh", "auth", "status"),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def submit_github_issue(
    path: str | Path | None = None,
    *,
    project: str | Path | None = None,
) -> dict[str, object]:
    """Submit a sanitized issue after an explicit ``--github`` user action."""

    if not _github_available():
        raise RuntimeError(
            "GitHub CLI is not installed or authenticated. Run `gh auth login`, "
            "or create a local archive with `agentfem feedback`."
        )
    with tempfile.TemporaryDirectory(prefix="agentfem-github-") as raw:
        target = Path(raw) / "support"
        record = create_support_directory(path, project=project, destination=target)
        fingerprint = str(record["fingerprint"])
        query = subprocess.run(
            (
                "gh",
                "issue",
                "list",
                "--repo",
                REPOSITORY,
                "--state",
                "open",
                "--search",
                f'"{fingerprint}" in:body',
                "--json",
                "number,url,title",
                "--limit",
                "5",
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if query.returncode == 0:
            try:
                matches = json.loads(query.stdout)
            except json.JSONDecodeError:
                matches = []
            if matches:
                return {
                    "schema": "agentfem.github-feedback",
                    "schema_version": SCHEMA_VERSION,
                    "status": "existing_issue",
                    "fingerprint": fingerprint,
                    "issue": matches[0],
                }
        failure = diagnose(path, project=project).get("failure") or {}
        title = f"[Feedback] {failure.get('code', 'unclassified')} {fingerprint}"
        created = subprocess.run(
            (
                "gh",
                "issue",
                "create",
                "--repo",
                REPOSITORY,
                "--title",
                title,
                "--body-file",
                str(target / "issue.md"),
                "--label",
                "bug",
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if created.returncode != 0:
            raise RuntimeError(f"GitHub issue creation failed: {created.stderr.strip()}")
        return {
            "schema": "agentfem.github-feedback",
            "schema_version": SCHEMA_VERSION,
            "status": "created",
            "fingerprint": fingerprint,
            "url": created.stdout.strip(),
        }


__all__ = (
    "FeedbackPreferences",
    "build_event",
    "clear_queue",
    "configure",
    "create_feedback_archive",
    "create_support_directory",
    "diagnose",
    "failure_fingerprint",
    "flush",
    "format_diagnosis",
    "last_event",
    "notice_text",
    "observe_cli",
    "preferences",
    "queue_size",
    "record_failure",
    "show_notice_once",
    "submit_github_issue",
    "validate_event",
)
