# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Instance-owned pure geometry workers; no global replacement or FEM state."""
from concurrent.futures import ProcessPoolExecutor
from ._cell_batches import join_batches
from multiprocessing import get_context
import numpy as np

def _portion(vertices, points):
    from agentfem._q1_surface_geometry import q1_closest_point
    answers = []
    for x, p in zip(vertices, points, strict=True):
        try:
            answers.append(q1_closest_point(x, p))
        except ValueError:
            answers.append(None)
    return answers

class ProjectionWorkers:

    def __init__(self, workers):
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError('Positive integer worker count required')
        self.workers = workers
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'))

    def __call__(self, vertices, points):
        width = max(1, (len(points) + self.workers - 1) // self.workers)
        futures = []
        try:
            for start in range(0, len(points), width):
                futures.append(self.pool.submit(_portion, np.ascontiguousarray(vertices[start:start + width]), np.ascontiguousarray(points[start:start + width])))
            return [answer for future in futures for answer in future.result()]
        finally:
            join_batches(futures)

    def close(self):
        self.pool.shutdown(wait=True)
