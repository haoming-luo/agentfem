# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Packaged Q1 geometry runtime; no material laws or case assets."""
import ctypes
from ._native_library import load_library
import numpy as np
from agentfem.boundary_models import Q1ContactExecution, exact_geometry_arithmetic
from ._native_bounds import make_bounds
from ._native_swept import make_swept
from ._native_batch import make_batch
from ._projection_workers import ProjectionWorkers
from ._cell_batches import CellBatches

class Q1ContactRuntime:

    def __init__(self, geometry_workers=0, projection_workers=1, backend='auto',
                 separation_reuse='incremental', geometry_timeout=300.0):
        if geometry_timeout is not None and (
            isinstance(geometry_timeout, (bool, np.bool_))
            or not isinstance(geometry_timeout, (int, float, np.floating))
            or not np.isfinite(geometry_timeout) or geometry_timeout <= 0
        ):
            raise ValueError('geometry_timeout must be positive finite seconds or None')
        if separation_reuse not in ('incremental', 'anchored'):
            raise ValueError('separation_reuse must be incremental or anchored')
        if isinstance(geometry_workers, bool) or not isinstance(geometry_workers, int) or geometry_workers < 0:
            raise ValueError('Nonnegative integer geometry_workers required')
        if isinstance(projection_workers, bool) or not isinstance(projection_workers, int) or projection_workers not in (1, 2, 4, 8):
            raise ValueError('projection_workers must be 1, 2, 4 or 8')
        self.workers = None
        self.batch = None
        self.winding_batches = None
        self.closed = False
        if backend not in ('auto', 'native', 'reference'):
            raise ValueError('Q1 backend must be auto, native or reference')
        self.statistics = dict(closest_native=0, closest_fallback=0, winding_native=0, winding_fallback=0)
        library = None if backend == 'reference' else load_library(required=backend == 'native')
        if library is None:
            self.execution = Q1ContactExecution({'provider': 'agentfem.q1_geometry',
                'revision': 1, 'arithmetic': 'reference'}, separation_reuse=separation_reuse)
            return
        self.convex_library = self.winding_library = library
        real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        integer = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        self.convex = self.convex_library.convex_q1
        self.convex.argtypes = [real, real, real]
        self.convex.restype = ctypes.c_int
        self.winding = self.winding_library.winding_batch_n
        self.winding.argtypes = [ctypes.c_int64] * 5 + [real, real, integer, integer, integer, real, real, real, real, ctypes.c_double, integer]
        self.winding.restype = ctypes.c_int
        # Queries are independent: preserve each query's ordered two-ray proof
        # while sharing immutable tree buffers. Do not re-run the whole batch
        # when only one native slice needs the reference fallback.
        self.winding_batches = CellBatches(
            self._winding_slice, workers=projection_workers, shared=(0,), minimum=64)
        self.directions = np.array([[1.0, 0.371, 0.193], [0.217, 1.0, 0.419], [0.613, 0.293, 1.0], [-1.0, 0.523, 0.317], [0.731, -1.0, 0.239], [0.419, 0.617, -1.0], [-0.317, -0.719, 1.0], [1.0, -0.233, -0.871]])
        self.directions /= np.linalg.norm(self.directions, axis=1)[:, None]
        self.statistics = dict(closest_native=0, closest_fallback=0, winding_native=0, winding_fallback=0)
        self.batch = make_batch(self.statistics, workers=projection_workers, rejected_evaluator=self._recover if geometry_workers else None)
        self.execution = Q1ContactExecution({
            'provider': 'agentfem.q1_geometry', 'revision': 1,
            'arithmetic': 'native_gmp', 'geometry_workers': geometry_workers,
            'projection_workers': projection_workers,
            'exact_arithmetic': exact_geometry_arithmetic(),
            'scope': 'projection-classification-conservative-bounds-swept-proof'},
            closest_point=self.closest, classify_inside=self.inside,
            distance_bounds=make_bounds(), separated_paths=make_swept(),
            project_batch=self.batch, separation_reuse=separation_reuse)
        if geometry_workers:
            self.workers = ProjectionWorkers(geometry_workers, timeout=geometry_timeout)

    def _recover(self, vertices, points):
        return self.workers(vertices, points)

    def _winding_slice(self, n, shared, query, lengths, answers):
        tree_args, vertices, padding = shared
        status = self.winding(n, *tree_args, vertices, query,
                              self.directions, lengths, padding, answers)
        if status:
            answers.fill(-1)

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Q1 geometry runtime is closed')
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.closed = True
        if self.batch is not None:
            self.batch.close()
            self.batch = None
        if self.winding_batches is not None:
            self.winding_batches.close()
            self.winding_batches = None
        if self.workers is not None:
            self.workers.close()
            self.workers = None

    def closest(self, vertices, point, reference):
        if self.closed:
            raise RuntimeError('Q1 geometry runtime is closed')
        vertices, point = np.asarray(vertices), np.asarray(point)
        if (vertices.shape != (4,3) or point.shape != (3,)
            or np.iscomplexobj(vertices) or np.iscomplexobj(point)
            or not np.isfinite(vertices).all() or not np.isfinite(point).all()):
            raise ValueError('Q1 projection requires four finite 3D vertices and one finite point')
        output = np.empty(13)
        status = self.convex(np.ascontiguousarray(vertices, dtype=float), np.ascontiguousarray(point, dtype=float), output)
        if not status:
            self.statistics['closest_fallback'] += 1
            return reference(vertices, point)
        self.statistics['closest_native'] += 1
        return (output[:2], output[2:6], output[6:9], output[9:12], float(output[12]))

    def inside(self, surface, points, reference):
        if self.closed:
            raise RuntimeError('Q1 geometry runtime is closed')
        points = np.asarray(points)
        if (points.ndim != 2 or points.shape[1] != 3 or np.iscomplexobj(points)
            or not np.isfinite(points).all()):
            raise ValueError('Q1 classification requires finite 3D query points')
        query = np.ascontiguousarray(points, dtype=float)
        tree = surface.search.tree
        lengths = np.ascontiguousarray(2 * (surface.scale + np.linalg.norm(query - (surface.lower + surface.upper) / 2, axis=1)))
        answers = np.full(len(query), -1, dtype=np.int64)
        shared = ((len(tree._facet), tree._root, len(surface.search.vertices),
                   len(self.directions), np.ascontiguousarray(tree._lower),
                   np.ascontiguousarray(tree._upper),
                   *[np.ascontiguousarray(a, dtype=np.int64)
                     for a in (tree._left, tree._right, tree._facet)]),
                  np.ascontiguousarray(surface.search.vertices),
                  surface.search.padding + 128 * np.finfo(float).eps * surface.scale)
        self.winding_batches(len(query), shared, query, lengths, answers)
        pending = np.flatnonzero(answers < 0)
        self.statistics['winding_native'] += len(query) - len(pending)
        self.statistics['winding_fallback'] += len(pending)
        if len(pending):
            answers[pending] = reference(query[pending])
        if not np.isin(answers, (0, 1)).all():
            raise ValueError('Native winding returned an invalid resolved classification')
        return answers.astype(bool)
