# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real OOBE shell flow with OS operations replaced by stubs."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_existing_locked_account_resumes_password_setup(tmp_path):
    events = tmp_path / "events"
    script = r"""
set -e
getent() { printf 'researcher:x:1000:1000::/home/researcher:/bin/bash\n'; }
cat() { echo test; }
passwd() {
  if [[ "$1" == -S ]]; then echo 'researcher L';
  else echo password >> "$EVENTS"; fi
}
adduser() { echo unexpected-adduser >> "$EVENTS"; }
usermod() { :; }
mkdir() { :; }
chown() { :; }
su() { :; }
export AGENTFEM_UPGRADE_USER=researcher
source "$OOBE"
"""
    import os

    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "EVENTS": str(events),
            "OOBE": str(ROOT / "packaging/runtime/wsl/oobe.sh"),
        },
    )
    assert result.returncode == 0, result.stderr
    assert events.read_text().splitlines() == ["password"]
    assert "AgentFEM is ready" not in result.stdout


def test_existing_configured_account_keeps_password(tmp_path):
    events = tmp_path / "events"
    import os

    script = r"""
getent() { printf 'researcher:x:1000:1000::/home/researcher:/bin/bash\n'; }
cat() { echo test; }
passwd() {
  if [[ "$1" == -S ]]; then echo 'researcher P';
  else echo reset >> "$EVENTS"; fi
}
adduser() { echo recreated >> "$EVENTS"; }
usermod() { :; }; mkdir() { :; }; chown() { :; }; su() { :; }
export AGENTFEM_UPGRADE_USER=researcher
source "$OOBE"
"""
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "EVENTS": str(events),
            "OOBE": str(ROOT / "packaging/runtime/wsl/oobe.sh"),
        },
    )
    assert result.returncode == 0, result.stderr
    assert not events.exists()


def test_resume_does_not_traverse_user_data_or_repeat_doctor(tmp_path):
    import os

    events = tmp_path / "events"
    script = r"""
getent() { printf 'researcher:x:1000:1000::/home/researcher:/bin/bash\n'; }
cat() { echo test; }
passwd() { echo 'researcher P'; }
adduser() { echo recreated >> "$EVENTS"; }
usermod() { :; }; mkdir() { :; }
chown() { printf 'chown'; printf ' <%s>' "$@"; printf '\n'; }
su() { echo duplicate-doctor >> "$EVENTS"; }
export AGENTFEM_UPGRADE_USER=researcher
source "$OOBE"
"""
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True,
        env={**os.environ, "EVENTS": str(events),
             "OOBE": str(ROOT / "packaging/runtime/wsl/oobe.sh")},
    )
    assert result.returncode == 0, result.stderr
    assert not events.exists()
    assert "<-R>" not in result.stdout
    assert "</home/researcher/.local/state/agentfem>" in result.stdout
    assert "</home/researcher/AgentFEMProjects>" in result.stdout
