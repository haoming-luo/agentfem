# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Detached viewer startup with a bounded readiness check and retained log."""
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def launch(location, *, executable=None, timeout=60):
    directory = Path(tempfile.gettempdir()) / 'agentfem-viewer'
    directory.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='start.', dir=directory, delete=False) as log:
        path = Path(log.name)
        process = subprocess.Popen(
            [executable or sys.executable, '-u', '-m', 'agentfem.viewer', str(location)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        output = path.read_text(errors='replace')
        if process.poll() is not None:
            raise RuntimeError(f'Viewer could not start. Log: {path}\n{output[-2000:]}')
        if 'Local:' in output and 'http://127.0.0.1:' in output:
            return path
        time.sleep(.1)
    raise RuntimeError(f'Viewer is still starting. Check the log before trying again: {path}')


def main():
    try:
        path = launch(sys.argv[1])
        print(f'AgentFEM viewer ready. Log: {path}')
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
