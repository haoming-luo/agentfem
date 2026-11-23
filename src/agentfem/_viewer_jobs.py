# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""One cancellable renderer per viewer, with completed-only publication."""

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from uuid import uuid4

from agentfem._atomic_text import atomic_write_text


class ExportJob:
    def __init__(self, destination):
        self.destination = Path(destination)
        self.process = None
        self.cancelled = False

    def cancel(self, **_):
        self.cancelled = True
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

    def close(self, **_):
        self.cancel()
        if self.process is not None:
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    async def run(self, request, progress):
        if self.process is not None:
            raise RuntimeError("An export is already running.")
        self.cancelled = False
        self.destination.mkdir(parents=True, exist_ok=True)
        # Staging is outside the HTTP-served directory. No partial MP4 is offered.
        stage = Path(tempfile.mkdtemp(prefix="agentfem-render-"))
        try:
            atomic_write_text(stage / "request.json", json.dumps(request, allow_nan=False))
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            with (stage / "worker.log").open("w") as log:
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "agentfem._viewer_export", str(stage / "request.json")],
                    stdout=log, stderr=subprocess.STDOUT, env=env,
                )
                while self.process.poll() is None:
                    await asyncio.sleep(.15)
                    if self.cancelled:
                        self.close()
                    record = stage / "progress.json"
                    if record.is_file():
                        item = json.loads(record.read_text())
                        progress(item["done"], item["total"])
                if self.cancelled:
                    return None
                if self.process.returncode:
                    detail = (stage / "worker.log").read_text(errors="replace")[-2000:]
                    raise RuntimeError("Export failed; source results are unchanged. " + detail)
            target = self.destination / ("export-" + uuid4().hex[:12])
            # Copy onto the destination filesystem before its atomic rename.
            # This also works when the temporary directory is on another drive.
            pending = self.destination / ("." + target.name)
            try:
                shutil.copytree(stage / "result", pending)
                pending.rename(target)
            finally:
                if pending.exists():
                    shutil.rmtree(pending)
            return target
        finally:
            self.close()
            self.process = None
            shutil.rmtree(stage)
