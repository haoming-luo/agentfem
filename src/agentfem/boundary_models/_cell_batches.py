# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Disjoint-cell arithmetic dispatch; no concurrent state commits."""
from concurrent.futures import ThreadPoolExecutor, wait


def join_batches(futures):
    """Do not let cancellation return while a native worker still writes."""
    pending = set(futures)
    interrupted = None
    while pending:
        try:
            _, pending = wait(pending)
        except BaseException as error:
            if interrupted is None:
                interrupted = error
            pending = {future for future in pending if not future.done()}
    if interrupted is not None:
        raise interrupted

class CellBatches:

    def __init__(self, function, *, workers, shared=(), minimum=8192):
        if workers not in (1, 2, 4, 8):
            raise ValueError('Unsupported cell worker count.')
        self.function = function
        self.workers = workers
        self.shared = frozenset(shared)
        self.minimum = minimum
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def __call__(self, n, *arrays):
        if self.workers == 1 or n < self.minimum:
            return self.function(n, *arrays)
        if any((i not in self.shared and len(a) != n for i, a in enumerate(arrays))):
            raise ValueError('Native cell argument does not span this batch.')
        width = (n + self.workers - 1) // self.workers
        pending = []
        try:
            for start in range(0, n, width):
                stop = min(n, start + width)
                selected = [a if i in self.shared else a[start:stop] for i, a in enumerate(arrays)]
                pending.append(self.pool.submit(self.function, stop - start, *selected))
            for future in pending:
                future.result()
        finally:
            join_batches(pending)

    def close(self):
        self.pool.shutdown(wait=True)
