# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import io
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

from agentfem._projection_service import MAGIC, MAX_BATCH, read_exact
from agentfem._q1_surface_geometry import q1_closest_point
from agentfem.boundary_models._projection_workers import (
    ProjectionWorkers,
    _service_environment,
)


def queries():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.2], [0.0, 1.0, 0.0]]
    )
    return np.tile(vertices, (4, 1, 1)), np.array(
        [[0.3, 0.4, 0.2], [2.0, 0.5, 0.0], [0.8, 0.7, -0.2], [0.0, 0.0, 0.0]]
    )


def test_geometry_services_preserve_answers_reuse_and_shutdown():
    vertices, points = queries()
    expected = [q1_closest_point(x, p) for x, p in zip(vertices, points)]
    env = {
        key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    workers = ProjectionWorkers(2)
    try:
        for _ in range(2):
            actual = workers(vertices, points)
            for a, b in zip(actual, expected):
                for value, reference in zip(a, b):
                    np.testing.assert_array_equal(value, reference)
        children = [service.process for service in workers.services]
        assert all(child is not None and child.poll() is None for child in children)
        bad = vertices.copy()
        bad[0] = 0.0
        assert workers(bad, points)[0] is None
        assert workers(vertices[:0], points[:0]) == []
        assert env == {key: os.environ.get(key) for key in env}
    finally:
        workers.close()
    assert all(child.poll() is not None for child in children)
    workers.close()
    with pytest.raises(RuntimeError, match="closed"):
        workers(vertices, points)


def test_pure_service_starts_without_mpi_and_rejects_bad_protocol():
    # The service refuses its handshake if MPI/PETSc/DOLFINx was imported.
    for payload in (struct.pack("<I", 0), struct.pack("<I", MAX_BATCH + 1), b"\x01"):
        result = subprocess.run(
            [sys.executable, "-m", "agentfem._projection_service"],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_service_environment(),
            timeout=20,
        )
        assert result.stdout == MAGIC
        assert (result.returncode == 0) == (payload == struct.pack("<I", 0))
    with pytest.raises(EOFError):
        read_exact(io.BytesIO(b"ab"), 3)


def test_geometry_worker_failure_closes_owned_children():
    vertices, points = queries()
    workers = ProjectionWorkers(2)
    try:
        workers(vertices, points)
        children = [service.process for service in workers.services]
        children[0].kill()
        children[0].wait()
        with pytest.raises((EOFError, BrokenPipeError, OSError)):
            workers(vertices, points)
        assert workers.closed
        assert all(child.poll() is not None for child in children)
    finally:
        workers.close()


def test_geometry_worker_timeout_reaps_children(monkeypatch):
    from agentfem.boundary_models import _projection_workers as implementation

    popen = subprocess.Popen

    def stalled_service(command, **kwargs):
        return popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)

    monkeypatch.setattr(implementation.subprocess, "Popen", stalled_service)
    workers = ProjectionWorkers(2, timeout=0.1)
    try:
        with pytest.raises(TimeoutError, match="timed out"):
            workers(*queries())
        assert workers.closed
        assert all(
            service.process is None or service.process.poll() is not None
            for service in workers.services
        )
    finally:
        workers.close()


def test_worker_input_rejection_does_not_spawn():
    workers = ProjectionWorkers(1)
    vertices, points = queries()
    try:
        bad = points.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            workers(vertices, bad)
        assert workers.services[0].process is None
    finally:
        workers.close()


def test_chunked_geometry_batches_keep_input_order(monkeypatch):
    from agentfem.boundary_models import _projection_workers as implementation

    monkeypatch.setattr(implementation, "MAX_BATCH", 2)
    vertices, points = queries()
    workers = ProjectionWorkers(1)
    try:
        actual = workers(vertices, points)
        for answer, vertex, point in zip(actual, vertices, points):
            for value, expected in zip(answer, q1_closest_point(vertex, point)):
                np.testing.assert_array_equal(value, expected)
    finally:
        workers.close()
