# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Create an opt-in desktop entry bound to the installed interpreter."""
from pathlib import Path
import os
import shlex
import sys


def create_shortcut(destination, location, *, executable=None, platform=None):
    """Create, never overwrite, a macOS command or Linux desktop entry.

    The process owns its log rather than a chat/terminal pipe. WSL users use
    the existing WSL entry route; a Linux desktop file is not a Windows shortcut.
    """
    platform = platform or sys.platform
    executable = str(Path(executable or sys.executable).absolute())
    destination = Path(destination).expanduser().absolute()
    location = Path(location).expanduser().resolve(strict=True)
    if platform not in {"darwin", "linux"}:
        raise ValueError("Desktop shortcuts are supported on macOS and native Linux.")
    suffix = ".command" if platform == "darwin" else ".desktop"
    if destination.suffix != suffix:
        raise ValueError(f"Use a {suffix} filename for this platform.")
    command = [executable, "-m", "agentfem.viewer", str(location)]
    if platform == "darwin":
        command[2] = 'agentfem._viewer_launch'
        content = ("#!/bin/sh\n"
                   f'{shlex.join(command)}\n'
                   'status=$?\nif [ "$status" -ne 0 ]; then\n'
                   '  printf "Press Enter to close. "\n  read answer\nfi\nexit "$status"\n')
    else:
        # Desktop Exec has its own quoting grammar, not POSIX shell quoting.
        def quote(value):
            value = value.replace("%", "%%")
            for char in ("\\", '"', "`", "$"):
                value = value.replace(char, "\\" + char)
            return '"' + value + '"'
        content = ("[Desktop Entry]\nType=Application\nName=AgentFEM Viewer\n"
                   "Comment=Open local simulation results\nTerminal=false\n"
                   "Exec=" + " ".join(map(quote, command)) + "\n")
    if any("\n" in part or "\r" in part for part in command):
        raise ValueError("Shortcut paths must not contain line breaks.")
    # Exclusive creation protects an existing user-managed shortcut.
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(content)
    os.chmod(destination, 0o700)
    return destination
