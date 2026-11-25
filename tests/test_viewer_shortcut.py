import shlex
import subprocess
import os

import pytest

from agentfem._viewer_shortcut import create_shortcut


def test_mac_shortcut_is_bound_detached_and_quoted(tmp_path):
    location = tmp_path / "results 'with spaces'"
    location.mkdir()
    path = create_shortcut(tmp_path / "Viewer.command", location,
                           executable="/opt/my env/bin/python", platform="darwin")
    text = path.read_text()
    assert shlex.quote(str(location)) in text
    assert "'/opt/my env/bin/python'" in text
    assert 'agentfem._viewer_launch' in text
    assert 'status=$?' in text
    assert path.stat().st_mode & 0o777 == 0o700
    with pytest.raises(FileExistsError):
        create_shortcut(path, location, platform="darwin")


def test_linux_shortcut_escapes_desktop_field_codes(tmp_path):
    location = tmp_path / "test%f"
    location.mkdir()
    path = create_shortcut(tmp_path / "Viewer.desktop", location,
                           executable="/opt/python", platform="linux")
    assert "test%%f" in path.read_text()
    assert 'Exec="/opt/python" "-m" "agentfem.viewer"' in path.read_text()


@pytest.mark.parametrize("platform,name", [("win32", "Viewer.cmd"), ("darwin", "Viewer.txt")])
def test_shortcut_rejects_unsupported_target(tmp_path, platform, name):
    with pytest.raises(ValueError):
        create_shortcut(tmp_path / name, tmp_path, platform=platform)
    assert not (tmp_path / name).exists()


def test_shortcut_rejects_newline_in_path(tmp_path):
    location = tmp_path / "line\nbreak"
    location.mkdir()
    with pytest.raises(ValueError, match="line breaks"):
        create_shortcut(tmp_path / "Viewer.desktop", location, platform="linux")


@pytest.mark.skipif(os.name != 'posix', reason='Shell launcher execution')
@pytest.mark.parametrize('success', [True, False])
def test_launcher_waits_for_readiness_or_late_failure(tmp_path, success):
    executable = tmp_path / 'python stub'
    executable.write_text('#!/bin/sh\nsleep 3\n' + (
        'echo "Local: http://127.0.0.1:12345/"\nsleep 2\n' if success else
        'echo "Viewer dependency missing"\nexit 1\n'))
    executable.chmod(0o700)
    shortcut = create_shortcut(tmp_path / 'Viewer.command', tmp_path,
                               executable=executable, platform='darwin')
    result = subprocess.run([str(shortcut)], input='\n', capture_output=True,
                            text=True, timeout=12, env={**os.environ, 'TMPDIR': str(tmp_path)})
    assert result.returncode == (0 if success else 1)
    assert ('Local:' if success else 'Viewer dependency missing') in result.stdout


@pytest.mark.skipif(os.name != 'posix', reason='Detached POSIX process')
def test_startup_helper_survives_launcher_process_group(tmp_path):
    import sys
    import signal
    from pathlib import Path
    from agentfem import _viewer_launch
    worker = tmp_path / 'worker'
    pid = tmp_path / 'pid'
    worker.write_text(f'#!{sys.executable}\nimport os,time\n'
                      f'open({str(pid)!r},"w").write(str(os.getpid()))\n'
                      'print("Local: http://127.0.0.1:12345/",flush=True)\ntime.sleep(30)\n')
    worker.chmod(0o700)
    code = ('from agentfem._viewer_launch import launch; '
            f'launch({str(tmp_path)!r}, executable={str(worker)!r}); '
            'import time; time.sleep(30)')
    # Match the package under test, even when pytest's source path differs
    # from the interpreter's installed AgentFEM version.
    environment = {**os.environ, 'PYTHONPATH': str(Path(_viewer_launch.__file__).parents[1])}
    parent = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
                              env=environment)
    import time
    try:
        limit = time.monotonic()+5
        while not pid.exists() and time.monotonic() < limit:
            time.sleep(.05)
        child = int(pid.read_text())
        os.killpg(parent.pid, signal.SIGTERM)
        parent.wait(timeout=5)
        os.kill(child, 0)  # Child must not belong to its launcher's process group.
    finally:
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=5)
        if pid.exists():
            os.kill(int(pid.read_text()), signal.SIGTERM)
