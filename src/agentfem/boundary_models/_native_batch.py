# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Instance-local native batch search; preserves reference fallback and ties."""
import ctypes
from ._native_library import load_library
import numpy as np
from agentfem import _q1_surface_search as module
from ._cell_batches import CellBatches

def make_batch(statistics, variant='bernstein-convex', workers=1, rejected_evaluator=None):
    statistics.update(prepared_points=0, strictly_ordered_points=0, process_oracle_batches=0, process_oracle_points=0)
    if variant != 'bernstein-convex':
        raise ValueError('Only the certified Bernstein-convex filter is packaged')
    library = load_library()
    real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    integer = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    boxes = library.batch_boxes
    boxes.argtypes = [ctypes.c_int64] * 4 + [real, ctypes.c_double] + [real] * 2 + [integer] * 7
    boxes.restype = ctypes.c_int
    projection = library.batch_projection
    projection.argtypes = [ctypes.c_int64, real, real, integer, real]
    projection.restype = None
    projection = CellBatches(projection, workers=workers, minimum=1024)

    def project(self, points, reference, **options):

        def original_project(surface, query, **kw):
            return reference(query, **kw)
        raw = np.asarray(points)
        radius = options.get('maximum_distance')
        if raw.ndim != 2 or raw.shape[1:] != (3,) or np.iscomplexobj(raw) or (not np.isfinite(raw).all()) or (not len(raw)) or (not isinstance(radius, (int, float))) or (not np.isfinite(radius)) or (radius <= 0):
            return original_project(self, points, **options)
        points = np.ascontiguousarray(raw, dtype=float)
        tree = self.tree
        n = len(points)
        nn = len(tree._facet)
        capacity = 64
        ids = np.empty((n, capacity), dtype=np.int64)
        counts = np.empty(n, dtype=np.int64)
        integers = lambda *args: [np.ascontiguousarray(a, dtype=np.int64) for a in args]
        result = boxes(n, nn, tree._root, capacity, points, radius + self.padding, np.ascontiguousarray(tree._lower), np.ascontiguousarray(tree._upper), *integers(tree._left, tree._right, tree._facet), np.empty(nn, dtype=np.int64), np.empty(nn, dtype=np.int64), counts, ids)
        if result:
            return original_project(self, points, **options)
        rows = np.repeat(np.arange(n), counts)
        flat = ids[np.arange(capacity)[None, :] < counts[:, None]]
        vertices = np.ascontiguousarray(self.vertices[flat])
        queries = np.ascontiguousarray(points[rows])
        output = np.empty((len(flat), 13))
        status = np.empty(len(flat), dtype=np.int64)
        projection(len(flat), vertices, queries, status, output)
        maximum_candidates = options.get('maximum_candidates', 128)
        normal_policy = options.get('require_unique_normal', True)
        prefix = dict(getattr(self, '_retry_projection_prefix', {}))
        fast_rows = np.empty(0, dtype=np.int64)
        fast_winners = np.empty(0, dtype=np.int64)
        if len(flat) and isinstance(maximum_candidates, int) and (not isinstance(maximum_candidates, bool)) and (maximum_candidates > 0) and isinstance(normal_policy, bool) and (normal_policy or self.node_ids is not None):
            present = np.flatnonzero(counts)
            starts = np.r_[0, np.cumsum(counts)][present]
            bad = np.r_[0, np.cumsum(status == 0)]
            complete = bad[starts + counts[present]] == bad[starts]
            order = np.lexsort((flat, output[:, 12], rows))
            winners = order[starts]
            second = np.full(len(present), np.inf)
            multiple = counts[present] > 1
            second[multiple] = output[order[starts[multiple] + 1], 12]
            scales = np.maximum(radius, np.maximum.reduceat(self.facet_scales[flat], starts))
            eligible = complete & (counts[present] <= maximum_candidates) & (output[winners, 12] <= radius) & (second - output[winners, 12] > 128 * np.finfo(float).eps * scales)
            policy = (float(radius), maximum_candidates, normal_policy)
            fast_rows = present[eligible]
            fast_winners = winners[eligible]
            statistics['prepared_points'] += len(points)
            statistics['strictly_ordered_points'] += int(eligible.sum())
        pending = np.ones(n, dtype=bool)
        pending[fast_rows] = False
        slow = np.flatnonzero(pending)

        def saved(vertices, point):
            if self.execution is None:
                return module.q1_closest_point(vertices, point)
            return self.execution.closest(vertices, point, module.q1_closest_point)
        overlap = tree.overlapping_facets
        cache = {}
        for k in np.flatnonzero((status != 0) & pending[rows]):
            o = output[k]
            cache[vertices[k].tobytes(), queries[k].tobytes()] = (o[:2], o[2:6], o[6:9], o[9:12], float(o[12]))
        fallback = module.q1_closest_point if rejected_evaluator is not None else saved
        rejected = {(vertices[k].tobytes(), queries[k].tobytes()) for k in (np.flatnonzero((status == 0) & pending[rows]) if fallback is not saved else ())}
        if rejected_evaluator is not None and fallback is not saved:
            failed = np.flatnonzero((status == 0) & pending[rows])
            if len(failed) >= 64:
                recovered = rejected_evaluator(vertices[failed], queries[failed])
                statistics['process_oracle_batches'] += 1
                statistics['process_oracle_points'] += len(failed)
                if len(recovered) != len(failed):
                    raise RuntimeError('Projection worker returned an incomplete batch.')
                for k, answer in zip(failed, recovered, strict=True):
                    if answer is not None:
                        cache[vertices[k].tobytes(), queries[k].tobytes()] = answer
        box_radius = radius + self.padding
        box_cache = {((points[i] - box_radius).tobytes(), (points[i] + box_radius).tobytes()): 2 * ids[i, :counts[i]] for i in slow}

        def cached_closest(x, p):
            key = (np.asarray(x).tobytes(), np.asarray(p).tobytes())
            if key not in cache:
                cache[key] = (fallback if key in rejected else saved)(x, p)
            return cache[key]

        def cached_overlap(lower, upper):
            key = (np.asarray(lower).tobytes(), np.asarray(upper).tobytes())
            return box_cache[key] if key in box_cache else overlap(lower, upper)

        def original_project(surface, query, **kw):
            return reference(query, _closest=cached_closest, _overlap=cached_overlap, **kw)
        try:
            if not len(fast_rows):
                return original_project(self, points, **options)
            result = {'facet_indices': np.empty(n, dtype=np.int64), 'weights': np.empty((n, 4)), 'points': np.empty((n, 3)), 'normals': np.empty((n, 3)), 'distances': np.empty(n), 'candidate_counts': counts.copy(), 'unique_normal_required': normal_policy}
            result['facet_indices'][fast_rows] = flat[fast_winners]
            for name, columns in [('weights', slice(2, 6)), ('points', slice(6, 9)), ('normals', slice(9, 12)), ('distances', 12)]:
                result[name][fast_rows] = output[fast_winners, columns]
            if len(slow):
                try:
                    reference = original_project(self, points[slow], **options)
                except module.AmbiguousSurfaceProjection as error:
                    raise module.AmbiguousSurfaceProjection(int(slow[error.query_index]), facets=error.facets, distances=error.distances, point_separation=error.point_separation, distance_tolerance=error.distance_tolerance) from error
                except module.SurfaceOutsideSearchRadius as error:
                    raise module.SurfaceOutsideSearchRadius(int(slow[error.query_index])) from error
                except ValueError as error:
                    import re
                    match = re.match('Q1 query (\\d+):', str(error))
                    if match:
                        i = int(match.group(1))
                        raise ValueError(re.sub('^Q1 query \\d+:', f'Q1 query {slow[i]}:', str(error))) from error
                    raise
                for name in result:
                    if name != 'unique_normal_required':
                        result[name][slow] = reference[name]
            self._retry_projection_prefix = {}
            return result
        except BaseException:
            if len(fast_rows):
                prefix.update(getattr(self, '_retry_projection_prefix', {}))
                for i, k in zip(fast_rows, fast_winners):
                    o = output[k]
                    record = (o[:2], o[2:6], o[6:9], o[9:12], float(o[12]))
                    prefix[policy, points[i].tobytes()] = ((int(flat[k]), record), int(counts[i]))
                self._retry_projection_prefix = prefix
            raise
    project.close = projection.close
    return project
