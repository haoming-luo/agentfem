import asyncio
import json

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
    class Output:
        lines = [b"App running\n", b" - Local: http://127.0.0.1:12345/\n", b""]
        async def readline(self):
            return self.lines.pop(0) if self.lines else b""
    class Process:
        returncode = None
        stdout = Output()
        async def wait(self):
            self.returncode = 0
    async def spawn(*args, **kwargs):
        calls.append(args)
        return Process()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    async def run():
        url, task = await launch_folder(tmp_path)
        assert url == "http://127.0.0.1:12345/"
        await task
    asyncio.run(run())
    assert calls[0][-2:] == ("--idle-timeout", "120")
