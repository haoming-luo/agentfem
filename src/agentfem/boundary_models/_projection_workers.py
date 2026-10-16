# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Instance-owned geometry services; no case reimport or MPI side effects."""

from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from ._cell_batches import join_batches
import os
import struct
import subprocess
import socket
import sys
from pathlib import Path
from threading import Lock
import numpy as np
from .._projection_service import MAGIC, MAX_BATCH, read_exact


def _service_environment():
    env = dict(os.environ)
    # A source checkout, editable install, or side-by-side wheel must launch
    # its own service, not an unrelated older site-packages installation.
    package_parent = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (package_parent, env.get("PYTHONPATH")))
    )
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    return env


class _Service:
    def __init__(self):
        self.process = None
        self.lock = Lock()
        self.lifecycle_lock = Lock()
        self.aborted = False
        self.channel = None
        self.reader = None

    def _send(self, data):
        # PETSc may install a fatal SIGPIPE handler. Suppress SIGPIPE on this
        # socket only; never change the solver's process-wide signal policy.
        self.channel.sendall(data, getattr(socket, "MSG_NOSIGNAL", 0))

    def evaluate(self, vertices, points):
        with self.lock:
            with self.lifecycle_lock:
                if self.aborted:
                    raise RuntimeError("Q1 geometry service was stopped")
                starting = self.process is None
                if starting:
                    self._start()
            if starting:
                if bytes(read_exact(self.reader, len(MAGIC))) != MAGIC:
                    raise RuntimeError("Q1 geometry worker protocol mismatch")
            values = np.concatenate((vertices.reshape(-1, 12), points), axis=1).astype(
                "<f8"
            )
            self._send(struct.pack("<I", len(points)))
            self._send(values.tobytes())
            rows = np.frombuffer(
                read_exact(self.reader, len(points) * 14 * 8), dtype="<f8"
            ).reshape(-1, 14)
            if not np.isfinite(rows).all() or not np.isin(rows[:, 0], (0, 1)).all():
                raise RuntimeError("Invalid Q1 geometry worker response")
            return [
                None
                if row[0] == 0
                else (
                    row[1:3].copy(),
                    row[3:7].copy(),
                    row[7:10].copy(),
                    row[10:13].copy(),
                    float(row[13]),
                )
                for row in rows
            ]

    def _start(self):
        env = _service_environment()
        parent, child = socket.socketpair()
        try:
            if hasattr(socket, "SO_NOSIGPIPE"):
                parent.setsockopt(socket.SOL_SOCKET, socket.SO_NOSIGPIPE, 1)
            self.process = subprocess.Popen(
                [sys.executable, "-m", "agentfem._projection_service"],
                stdin=child,
                stdout=child,
                env=env,
            )
            self.channel = parent
            self.reader = parent.makefile("rb")
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()

    def stop(self, *, abort=False):
        with self.lifecycle_lock:
            self.aborted = True
            child = self.process
        if child is None:
            return
        if child.poll() is None:
            if abort:
                child.terminate()
            else:
                try:
                    self._send(struct.pack("<I", 0))
                except (BrokenPipeError, OSError):
                    pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        self.reader.close()
        self.channel.close()


class ProjectionWorkers:
    def __init__(self, workers, *, timeout=300.0):
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("Positive integer worker count required")
        if timeout is not None and (
            isinstance(timeout, (bool, np.bool_))
            or not isinstance(timeout, (int, float, np.floating))
            or not np.isfinite(timeout) or timeout <= 0
        ):
            raise ValueError("Positive finite geometry worker timeout required")
        self.workers, self.timeout = workers, None if timeout is None else float(timeout)
        self.services = [_Service() for _ in range(workers)]
        self.pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="agentfem-projection-io"
        )
        self.closed = False

    def __call__(self, vertices, points):
        if self.closed:
            raise RuntimeError("Q1 geometry workers are closed")
        vertices, points = np.asarray(vertices), np.asarray(points)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or vertices.shape != (len(points), 4, 3)
            or np.iscomplexobj(vertices)
            or np.iscomplexobj(points)
            or not np.isfinite(vertices).all()
            or not np.isfinite(points).all()
        ):
            raise ValueError(
                "Geometry workers require finite matching Q1 vertices and points"
            )
        width = min(MAX_BATCH, max(1, (len(points) + self.workers - 1) // self.workers))
        futures = []
        try:
            for batch, start in enumerate(range(0, len(points), width)):
                service = self.services[batch % self.workers]
                futures.append(
                    self.pool.submit(
                        service.evaluate,
                        np.ascontiguousarray(vertices[start : start + width]),
                        np.ascontiguousarray(points[start : start + width]),
                    )
                )
            done, pending = wait(
                futures, timeout=self.timeout, return_when=FIRST_EXCEPTION
            )
            for future in done:
                future.result()
            if pending:
                raise TimeoutError("Q1 geometry worker batch timed out")
            return [answer for future in futures for answer in future.result()]
        except BaseException:
            for future in futures:
                future.cancel()
            for service in self.services:
                service.stop(abort=True)
            self.closed = True
            raise
        finally:
            join_batches(futures)

    def close(self):
        self.closed = True
        self.pool.shutdown(wait=True)
        for service in self.services:
            service.stop()
