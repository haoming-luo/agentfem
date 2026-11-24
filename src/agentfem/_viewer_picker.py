# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional desktop folder chooser, isolated from the render/event loop."""
import asyncio
import json
import os
from pathlib import Path
import signal
import sys


async def choose_folder(initial):
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "agentfem._viewer_picker", str(initial),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=os.name == "posix")
    try:
        out, _ = await asyncio.wait_for(process.communicate(), timeout=180)
        if process.returncode:
            raise RuntimeError("System folder chooser is unavailable here. Browse the listed folders or paste a path.")
        result = json.loads(out)
        return Path(result).resolve(strict=True) if result else None
    finally:
        if process.returncode is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            await process.wait()


async def launch_folder(folder):
    """An independent loopback session keeps existing views/exports scoped."""
    import re
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-m", "agentfem.viewer", str(folder),
        "--no-browser", "--port", "0", "--idle-timeout", "120",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    ready = False
    try:
        async with asyncio.timeout(30):
            while line := await process.stdout.readline():
                match = re.search(rb"Local:\s+(http://127\.0\.0\.1:\d+/)", line)
                if match:
                    ready = True
                    # Drain startup/runtime messages without retaining them.
                    async def drain():
                        while await process.stdout.readline():
                            pass
                        await process.wait()
                    task = asyncio.create_task(drain())
                    return match[1].decode(), task
            raise RuntimeError("Could not open the selected project.")
    finally:
        if not ready and process.returncode is None:
            process.terminate()
            await process.wait()


def main():
    if sys.platform == "darwin":
        # Use the OS panel on macOS. A non-bundled Python/Tk process can leave
        # its chooser behind the browser with no usable application focus.
        import subprocess
        script = '''on run argv
            activate
            try
                return POSIX path of (choose folder with prompt "Choose an AgentFEM project or results folder" default location (POSIX file (item 1 of argv)))
            on error number -128
                return ""
            end try
        end run'''
        result = subprocess.run(["/usr/bin/osascript", "-e", script, sys.argv[1]],
                                capture_output=True, text=True, check=True, timeout=170)
        print(json.dumps(result.stdout.strip()))
        return
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(parent=root, title="Choose an AgentFEM project or results folder",
                                           initialdir=sys.argv[1], mustexist=True)
        print(json.dumps(selected))
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
