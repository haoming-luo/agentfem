from __future__ import annotations

import json
from types import SimpleNamespace

from agentfem import cli, community


def test_after_upgrade_invites_once_per_release_series(tmp_path):
    state = tmp_path / "community.json"

    first = community.status(version="0.3.6", after_upgrade=True, path=state)
    patch = community.status(version="0.3.7", after_upgrade=True, path=state)
    next_series = community.status(version="0.4.0", after_upgrade=True, path=state)

    assert first["invitation_due"] is True
    assert patch["invitation_due"] is False
    assert next_series["invitation_due"] is True
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["last_invited_series"] == "0.4"
    assert "username" not in saved


def test_acknowledgement_suppresses_future_invitations(tmp_path):
    state = tmp_path / "community.json"
    community.acknowledge(kind="github_star", path=state)

    record = community.status(version="1.0.0", after_upgrade=True, path=state)

    assert record["acknowledged"] is True
    assert record["invitation_due"] is False
    assert record["automatic_account_action"] is False
    assert record["requires_explicit_user_confirmation"] is True


def test_default_status_never_checks_github(monkeypatch, tmp_path):
    def forbidden():
        raise AssertionError("GitHub must not be contacted by default")

    monkeypatch.setattr(community, "_github_status", forbidden)

    record = community.status(version="0.3.6", path=tmp_path / "state.json")

    assert record["network_checked"] is False
    assert record["github"]["reason"] == "not_checked_without_explicit_request"
    assert "state_path" not in record


def test_explicit_github_check_remembers_verified_star(monkeypatch, tmp_path):
    state = tmp_path / "community.json"
    monkeypatch.setattr(
        community,
        "_github_status",
        lambda: {
            "cli_available": True,
            "authenticated": True,
            "starred": True,
            "reason": "verified_by_github_api",
        },
    )

    record = community.status(
        version="0.3.6", after_upgrade=True, check_github=True, path=state
    )

    assert record["acknowledged"] is True
    assert record["invitation_due"] is False
    assert record["state_persisted"] is True
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["kind"] == "github_star_verified"
    assert "username" not in saved


def test_invitation_survives_unwritable_optional_state(monkeypatch, tmp_path):
    monkeypatch.setattr(community, "_try_write_state", lambda record, path: False)

    record = community.status(
        version="0.3.6", after_upgrade=True, path=tmp_path / "community.json"
    )

    assert record["invitation_due"] is True
    assert record["state_persisted"] is False


def test_support_cli_is_machine_readable_and_account_free(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("AGENTFEM_STATE_HOME", str(tmp_path))

    assert cli._dispatch(["support", "--after-upgrade", "--json"]) == 0

    record = json.loads(capsys.readouterr().out)
    assert record["schema"] == "agentfem.community-support"
    assert record["invitation_due"] is True
    assert record["automatic_account_action"] is False
    assert record["network_checked"] is False


def test_explicit_star_uses_existing_github_login_and_acknowledges(
    monkeypatch, tmp_path
):
    state = tmp_path / "community.json"
    calls = []

    monkeypatch.setattr(community.shutil, "which", lambda name: "/usr/bin/gh")

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(community.subprocess, "run", run)

    record = community.star(path=state)

    assert record["requested_explicitly"] is True
    assert record["performed"] is True
    assert calls == [
        ["/usr/bin/gh", "auth", "status", "--hostname", "github.com"],
        [
            "/usr/bin/gh",
            "api",
            "--method",
            "PUT",
            "/user/starred/haoming-luo/agentfem",
        ],
    ]
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["kind"] == "github_star_verified"
    assert "username" not in saved


def test_support_star_is_never_implicit(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTFEM_STATE_HOME", str(tmp_path))
    monkeypatch.setattr(
        community,
        "star",
        lambda: (_ for _ in ()).throw(AssertionError("must remain explicit")),
    )

    assert cli._dispatch(["support", "--after-upgrade", "--json"]) == 0


def test_support_cli_explicit_star_is_one_command(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("AGENTFEM_STATE_HOME", str(tmp_path))

    def explicit_star():
        acknowledgement = community.acknowledge(kind="github_star_verified")
        return {
            "requested_explicitly": True,
            "performed": True,
            "repository": community.REPOSITORY_URL,
            "acknowledgement": acknowledgement,
        }

    monkeypatch.setattr(community, "star", explicit_star)

    assert cli._dispatch(["support", "--star", "--json"]) == 0

    record = json.loads(capsys.readouterr().out)
    assert record["action"]["performed"] is True
    assert record["acknowledged"] is True
    assert record["automatic_account_action"] is False
