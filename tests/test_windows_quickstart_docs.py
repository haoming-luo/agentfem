# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Check executable documentation, without downloads or host mutations.

These are documentation contracts, not a Windows installation acceptance.
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
GUIDE = (ROOT / "docs/windows_agent_installation.md").read_text()


def blocks(language):
    return re.findall(rf"```{language}\n(.*?)\n```", GUIDE, re.S)


def test_front_door_does_not_require_repository_or_windows_features():
    intro = GUIDE.split("## 1.", 1)[0]
    assert "No GitHub clone" in intro
    install = (ROOT / "INSTALL.md").read_text().split("## Windows\n", 1)[1]
    assert install.index("WSL → Ubuntu → AgentFEM") < install.index("Optional tools")
    assert "wsl --install --no-distribution --web-download" in install


def test_powershell_examples_are_copy_safe_and_fail_closed():
    for block in blocks("powershell"):
        assert not re.search(r"<[a-zA-Z][^>]*>", block)
        assert "Invoke-Expression" not in block
    recipe = next(block for block in blocks("powershell") if "--import" in block)
    assert recipe.index("Get-FileHash") < recipe.index("wsl.exe --import")
    assert "-Algorithm SHA256" in recipe
    assert "$afmEntries.Count -ne 1" in recipe
    assert "$afmExisting -contains $afmDistro" in recipe
    assert "Test-Path -LiteralPath $afmTarget" in recipe
    assert "--version 2" in recipe
    assert "--unregister" not in recipe
    for line in recipe.splitlines():
        if line.startswith(("curl.exe ", "wsl.exe ")):
            following = recipe.split(line, 1)[1].lstrip()
            assert following.startswith("if ($LASTEXITCODE -ne 0)"), line


def test_account_creation_does_not_give_passwordless_administration():
    account = next(block for block in blocks("powershell") if "useradd" in block)
    assert "--create-home --shell /bin/bash afm" in account
    assert "-u afm" in account
    assert "NOPASSWD" not in account and "chpasswd" not in account
    assert "ubuntu.exe config --default-user" not in account


def test_miniforge_example_has_valid_bash_and_checks_before_execution():
    block, = blocks("bash")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash syntax check requires Bash")
    subprocess.run([bash, "-n"], input=block, text=True, check=True, timeout=10)
    assert block.index("sha256sum --check") < block.index('bash "$afm_file"')
    assert 'test ! -e "$HOME/miniforge3"' in block
    assert 'test "$(id -u)" -ne 0' in block
    assert "set -eu" in block


def test_compiler_contract_in_every_documented_create_command():
    for name in ("INSTALL.md", "docs/getting_started.md"):
        text = (ROOT / name).read_text()
        for command in re.findall(r"(?:mamba|conda) create .*?(?=\nmamba|\n```)", text, re.S):
            if "agentfem\n" in command or command.endswith("agentfem"):
                assert "c-compiler cxx-compiler" in command


def test_selected_channel_and_short_handoffs_survive_agent_entry():
    agent = (ROOT / "AGENT_GUIDE.md").read_text()
    assert "Retain the selected mirror" in agent
    assert "Do not require a GitHub clone" in agent
    assert "On first-solve completion, promptly report" in agent
    assert (ROOT / "docs/windows_installer_reference.md").is_file()
