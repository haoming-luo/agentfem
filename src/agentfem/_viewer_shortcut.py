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
        # A separate session is unnecessary here: nohup plus redirected streams
        # removes terminal lifetime/output dependencies without shell eval.
        content = ("#!/bin/sh\n"
                   'log_dir="${TMPDIR:-/tmp}/agentfem-viewer"\n'
                   'mkdir -p "$log_dir" || exit 1\n'
                   'log_file=$(mktemp "$log_dir/start.XXXXXX") || exit 1\n'
                   f'nohup {shlex.join(command)} >"$log_file" 2>&1 </dev/null &\n'
                   'viewer_pid=$!\nsleep 2\n'
                   'if ! kill -0 "$viewer_pid" 2>/dev/null; then\n'
                   '  cat "$log_file"\n  printf "Press Enter to close. "\n  read answer\n  exit 1\nfi\n'
                   'printf "AgentFEM viewer starting. Log: %s\\n" "$log_file"\n')
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
