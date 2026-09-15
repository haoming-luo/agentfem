"""Consent-first community support for people and installation agents.

The default path is local and read-only with respect to external services.
Checking GitHub is explicit; starring always remains an action performed by the
user or their agent after the user has approved it.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPOSITORY = "haoming-luo/agentfem"
REPOSITORY_URL = f"https://github.com/{REPOSITORY}"
DISCUSSIONS_URL = f"{REPOSITORY_URL}/discussions"
CITATION_URL = f"{REPOSITORY_URL}/blob/main/CITATION.cff"
_SCHEMA = "agentfem.community-support"
_SCHEMA_VERSION = "0.1.0"


def state_path() -> Path:
    """Return the local, account-free acknowledgement record."""

    override = os.environ.get("AGENTFEM_STATE_HOME")
    if override:
        root = Path(override).expanduser()
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "AgentFEM"
    else:
        configured = os.environ.get("XDG_STATE_HOME")
        root = (
            Path(configured).expanduser() / "agentfem"
            if configured
            else Path.home() / ".local" / "state" / "agentfem"
        )
    return root / "community-support.json"


def _read_state(path: Path | None = None) -> dict[str, Any]:
    selected = path or state_path()
    try:
        record = json.loads(selected.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return record if isinstance(record, dict) else {}


def _write_state(record: dict[str, Any], path: Path | None = None) -> None:
    selected = path or state_path()
    selected.parent.mkdir(parents=True, exist_ok=True)
    temporary = selected.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(selected)


def _try_write_state(record: dict[str, Any], path: Path) -> bool:
    """Persist optional invitation state without breaking an install workflow."""

    try:
        _write_state(record, path)
    except OSError:
        return False
    return True


def _series(version: str) -> str:
    parts = str(version).split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else str(version)


def acknowledge(*, kind: str = "supported", path: Path | None = None) -> dict[str, Any]:
    """Remember an explicit acknowledgement without storing account identity."""

    if kind not in {"github_star", "github_star_verified", "supported_elsewhere"}:
        raise ValueError(f"Unsupported community acknowledgement {kind!r}.")
    record = {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "acknowledged": True,
        "kind": kind,
        "acknowledged_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_state(record, path)
    return record


def _github_status() -> dict[str, Any]:
    """Explicitly inspect GitHub CLI state without exposing credentials."""

    executable = shutil.which("gh")
    if not executable:
        return {
            "cli_available": False,
            "authenticated": False,
            "starred": None,
            "reason": "github_cli_not_found",
        }
    try:
        authentication = subprocess.run(
            [executable, "auth", "status", "--hostname", "github.com"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {
            "cli_available": True,
            "authenticated": None,
            "starred": None,
            "reason": "github_auth_check_failed",
        }
    if authentication.returncode != 0:
        return {
            "cli_available": True,
            "authenticated": False,
            "starred": None,
            "reason": "github_cli_not_authenticated",
        }
    try:
        query = subprocess.run(
            [executable, "api", f"/user/starred/{REPOSITORY}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {
            "cli_available": True,
            "authenticated": True,
            "starred": None,
            "reason": "github_check_failed",
        }
    if query.returncode == 0:
        return {
            "cli_available": True,
            "authenticated": True,
            "starred": True,
            "reason": "verified_by_github_api",
        }
    combined = f"{query.stdout}\n{query.stderr}".lower()
    if "404" in combined or "not found" in combined:
        return {
            "cli_available": True,
            "authenticated": True,
            "starred": False,
            "reason": "not_starred",
        }
    return {
        "cli_available": True,
        "authenticated": True,
        "starred": None,
        "reason": "github_check_failed",
    }


def star(*, path: Path | None = None) -> dict[str, Any]:
    """Star AgentFEM through an existing GitHub CLI login.

    Calling this function is the explicit account-changing action.  It never
    starts from :func:`status`, ``doctor``, installation, or upgrade flows.
    """

    executable = shutil.which("gh")
    if not executable:
        raise RuntimeError(
            f"GitHub CLI is not available. Open {REPOSITORY_URL} and choose Star."
        )
    try:
        authentication = subprocess.run(
            [executable, "auth", "status", "--hostname", "github.com"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "Could not check the existing GitHub CLI login. No account action was taken."
        ) from exc
    if authentication.returncode != 0:
        raise RuntimeError(
            f"GitHub CLI is not signed in. Open {REPOSITORY_URL} and choose Star."
        )
    try:
        action = subprocess.run(
            [
                executable,
                "api",
                "--method",
                "PUT",
                f"/user/starred/{REPOSITORY}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "GitHub did not confirm the Star. No local acknowledgement was saved."
        ) from exc
    if action.returncode != 0:
        raise RuntimeError(
            f"GitHub did not confirm the Star. Open {REPOSITORY_URL} and choose Star."
        )
    acknowledgement = acknowledge(kind="github_star_verified", path=path)
    return {
        "requested_explicitly": True,
        "performed": True,
        "repository": REPOSITORY_URL,
        "acknowledgement": acknowledgement,
    }


def status(
    *,
    version: str,
    after_upgrade: bool = False,
    check_github: bool = False,
    path: Path | None = None,
) -> dict[str, Any]:
    """Return a machine-readable, consent-first support invitation."""

    selected = path or state_path()
    local = _read_state(selected)
    github = (
        _github_status()
        if check_github
        else {
            "cli_available": shutil.which("gh") is not None,
            "authenticated": None,
            "starred": None,
            "reason": "not_checked_without_explicit_request",
        }
    )
    state_persisted = True
    if github["starred"] is True and not local.get("acknowledged"):
        local = {
            "schema": _SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "acknowledged": True,
            "kind": "github_star_verified",
            "acknowledged_at": datetime.now(timezone.utc).isoformat(),
        }
        state_persisted = _try_write_state(local, selected)

    series = _series(version)
    acknowledged = bool(local.get("acknowledged"))
    already_invited = local.get("last_invited_series") == series
    invitation_due = bool(after_upgrade and not acknowledged and not already_invited)
    if invitation_due:
        local = {
            **local,
            "schema": _SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "last_invited_series": series,
            "last_invited_version": version,
        }
        state_persisted = _try_write_state(local, selected) and state_persisted

    return {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "agentfem_version": version,
        "repository": REPOSITORY_URL,
        "discussions": DISCUSSIONS_URL,
        "citation": CITATION_URL,
        "requires_explicit_user_confirmation": True,
        "automatic_account_action": False,
        "network_checked": check_github,
        "acknowledged": bool(local.get("acknowledged")),
        "invitation_due": invitation_due,
        "invitation_scope": "once_per_major_minor_series",
        "github": github,
        "local_acknowledgement_only": True,
        "state_persisted": state_persisted,
    }


def format_status(record: dict[str, Any]) -> str:
    """Render the compact human counterpart of :func:`status`."""

    if record["acknowledged"]:
        return "Thank you for supporting AgentFEM."
    if record["invitation_due"]:
        return (
            "AgentFEM is ready. If it helped, you may star "
            f"{record['repository']}. An AI agent must ask before acting for you."
        )
    if record["github"]["starred"] is False:
        return (
            "AgentFEM is ready. If it helped, you may star it at "
            f"{record['repository']}."
        )
    return (
        f"Support AgentFEM: {record['repository']}\n"
        "GitHub status is checked only with --check-github."
    )


__all__ = (
    "CITATION_URL",
    "DISCUSSIONS_URL",
    "REPOSITORY",
    "REPOSITORY_URL",
    "acknowledge",
    "format_status",
    "star",
    "state_path",
    "status",
)
