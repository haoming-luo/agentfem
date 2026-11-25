import asyncio
import json
import subprocess
import os
import sys
import signal

import pytest

from agentfem._viewer_picker import choose_folder, launch_folder


@pytest.mark.parametrize("cancelled", [False, True])
def test_native_selection_does_not_upload_or_shell_interpolate(tmp_path, monkeypatch, cancelled):
    monkeypatch.setattr('sys.platform', 'linux')
    calls = []
    class Process:
        returncode = 0
        async def communicate(self):
            return json.dumps("" if cancelled else str(tmp_path)).encode(), b""
    async def spawn(*args, **kwargs):
        calls.append(args)
        return Process()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    assert asyncio.run(choose_folder(tmp_path)) == (None if cancelled else tmp_path)
    assert calls[0][-1] == str(tmp_path)
    assert "agentfem._viewer_picker" in calls[0]


def test_mac_chooser_avoids_nested_python_launch(tmp_path, monkeypatch):
    monkeypatch.setattr('sys.platform', 'darwin')
    calls = []
    class Process:
        returncode = 0
        async def communicate(self):
            return (str(tmp_path) + '\n').encode(), b''
    async def spawn(*args, **kwargs):
        calls.append(args)
        return Process()
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    assert asyncio.run(choose_folder(tmp_path)) == tmp_path
    assert calls[0][0] == '/usr/bin/osascript'
    assert calls[0][-1] == str(tmp_path)
    script = calls[0][-2]
    assert script.index('makeKeyAndOrderFront') < script.index('activateIgnoringOtherApps')
    assert 'NSFloatingWindowLevel' in script


def test_native_selection_unavailable_has_browse_fallback(tmp_path, monkeypatch):
    class Process:
        returncode = 1
        async def communicate(self):
            return b"", b"no desktop"
    async def spawn(*args, **kwargs):
        return Process()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(RuntimeError, match="Browse"):
        asyncio.run(choose_folder(tmp_path))


def test_selected_folder_launch_is_loopback_and_idle_bounded(tmp_path, monkeypatch):
    calls = []
    class Process:
        returncode = None
        def poll(self):
            self.returncode = 0
            return 0
    def spawn(args, **kwargs):
        calls.append(args)
        assert kwargs['stdout'] != asyncio.subprocess.PIPE
        assert kwargs['stdin'] == asyncio.subprocess.DEVNULL
        kwargs['stdout'].write(b" - Local: http://127.0.0.1:12345/\n")
        kwargs['stdout'].flush()
        return Process()
    monkeypatch.setattr(subprocess, "Popen", spawn)
    async def run():
        url, task = await launch_folder(tmp_path)
        assert url == "http://127.0.0.1:12345/"
        await task
    asyncio.run(run())
    assert calls[0][-2:] == ["--idle-timeout", "120"]


def test_selected_folder_startup_failure_is_reported(tmp_path, monkeypatch):
    class Process:
        returncode = 1
        def poll(self):
            return 1
    def spawn(*args, **kwargs):
        kwargs['stdout'].write(b"Missing viewer dependency")
        kwargs['stdout'].flush()
        return Process()
    monkeypatch.setattr(subprocess, 'Popen', spawn)
    with pytest.raises(RuntimeError, match="Missing viewer dependency"):
        asyncio.run(launch_folder(tmp_path))


@pytest.mark.skipif(os.name != 'posix', reason='POSIX detached-process lifecycle')
def test_child_survives_parent_event_loop_shutdown(tmp_path, monkeypatch):
    helper = tmp_path / 'viewer-helper'
    pid_file = tmp_path / 'pid'
    helper.write_text(f'#!{sys.executable}\n'
                      'import os, time\n'
                      f'open({str(pid_file)!r}, "w").write(str(os.getpid()))\n'
                      'print("Local: http://127.0.0.1:12345/", flush=True)\n'
                      'time.sleep(30)\n')
    helper.chmod(0o700)
    monkeypatch.setattr(sys, 'executable', str(helper))
    async def launch():
        url, watcher = await launch_folder(tmp_path)
        assert url.endswith(':12345/')
        assert not watcher.done()
        # asyncio.run cancels the old viewer's outstanding tasks on exit.
    try:
        asyncio.run(launch())
        pid = int(pid_file.read_text())
        os.kill(pid, 0)
    finally:
        if pid_file.exists():
            pid = int(pid_file.read_text())
            os.kill(pid, signal.SIGTERM)
            os.waitpid(pid, 0)
