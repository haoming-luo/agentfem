# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit project-local runtime storage; no solver or telemetry policy changes."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


@contextmanager
def local_runtime(root):
    if root is None:
        yield
        return
    base = Path(root).resolve() / ".agentfem-runtime"
    locations = {"XDG_CACHE_HOME": base / "cache",
                 "MPLCONFIGDIR": base / "matplotlib",
                 "AGENTFEM_STATE_HOME": base / "state",
                 "TMPDIR": base / "tmp"}
    previous = {key: os.environ.get(key) for key in locations}
    previous_tempdir = tempfile.tempdir
    try:
        for key, path in locations.items():
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
        tempfile.tempdir = None
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        tempfile.tempdir = previous_tempdir


def check_write_access(output):
    """Explicit write probe. Preserve files; remove only the created probe."""
    from .community import state_path

    locations = {"output": Path(output), "runtime_records": state_path().parent,
                 "cache": Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")),
                 "matplotlib": Path(os.environ.get("MPLCONFIGDIR",
                     Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "matplotlib")),
                 "temporary": Path(tempfile.gettempdir())}
    records = []
    for name, path in locations.items():
        record = {"name": name, "path": str(path.resolve()), "writable": False, "error": None}
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=path) as probe:
                probe.write(b"agentfem-write-probe")
                probe.flush()
            record["writable"] = True
        except OSError as error:
            record["error"] = str(error)
        records.append(record)
    return records
