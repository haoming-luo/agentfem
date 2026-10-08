# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Measure existing serial BVH against exhaustive projection, without solving FEM.

Tree construction, warm query batches and exact facet work are separate. Never
interpret this microbenchmark as an end-to-end solver speedup or MPI scaling.
"""

import argparse
import hashlib
import inspect
import json
import os
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import agentfem
from agentfem import boundary_models


def measure(cells=16, queries=16, repeats=3, seed=20261009):
    if any(type(value) is not int or value < 1 for value in (cells, queries, repeats)):
        raise ValueError("cells, queries and repeats must be positive integers")
    vertices = [
        (i / cells, j / cells, 0.0) for j in range(cells + 1) for i in range(cells + 1)
    ]
    triangles = []
    for j in range(cells):
        for i in range(cells):
            a = j * (cells + 1) + i
            triangles.extend(
                ((a, a + 1, a + cells + 2), (a, a + cells + 2, a + cells + 1))
            )
    surface = boundary_models.triangulated_rigid_surface(
        vertices=vertices, triangles=triangles, name="search_evidence_grid"
    )
    rng = np.random.default_rng(seed)
    points = np.column_stack(
        (rng.uniform(0, 1, (queries, 2)), rng.uniform(0.01, 0.1, queries))
    )
    started = perf_counter()
    search = boundary_models.triangle_surface_bvh(surface)
    build_seconds = perf_counter() - started
    # Untimed warmup and exact semantic comparison, including facet identity.
    accelerated = search.project_with_diagnostics(points)
    reference = surface.project(points)
    for key in ("valid", "status_codes", "entity_ids"):
        np.testing.assert_array_equal(
            getattr(accelerated.projection, key), getattr(reference, key)
        )
    for key in ("closest_points", "normals", "signed_gaps", "local_coordinates"):
        np.testing.assert_allclose(
            getattr(accelerated.projection, key),
            getattr(reference, key),
            atol=1e-13,
            rtol=1e-13,
        )
    elapsed = {"bvh": [], "exhaustive": []}
    callbacks = {
        "bvh": lambda: search.project_with_diagnostics(points),
        "exhaustive": lambda: surface.project(points),
    }
    for repetition in range(repeats):
        # Alternate order to reduce a fixed ordering bias; retain raw samples.
        order = ("bvh", "exhaustive") if repetition % 2 == 0 else ("exhaustive", "bvh")
        for name in order:
            started = perf_counter()
            callbacks[name]()
            elapsed[name].append(perf_counter() - started)
    bvh, exhaustive = (median(elapsed[key]) for key in ("bvh", "exhaustive"))
    evaluated = sum(accelerated.diagnostics.evaluated_facet_counts)
    exhaustive_work = queries * len(triangles)
    return {
        "cells_per_side": cells,
        "queries": queries,
        "repeats": repeats,
        "seed": seed,
        "geometry_fingerprint": accelerated.diagnostics.geometry_fingerprint,
        "search_implementation_sha256": hashlib.sha256(
            Path(inspect.getfile(type(search))).read_bytes()
        ).hexdigest(),
        "query_sha256": hashlib.sha256(points.astype("<f8").tobytes()).hexdigest(),
        "projection_equivalence": "passed",
        "tree_build_seconds": build_seconds,
        "query_seconds": elapsed,
        "median_query_seconds": {"bvh": bvh, "exhaustive": exhaustive},
        "query_speed_ratio": exhaustive / bvh,
        "exact_facet_evaluations": {"bvh": evaluated, "exhaustive": exhaustive_work},
        "exact_work_fraction": evaluated / exhaustive_work,
        "diagnostics": accelerated.diagnostics.summary(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": "agentfem.contact-search-microbenchmark.v1",
        "scope": "serial_planar_tool_warm_query_not_fem_or_mpi_speedup",
        "agentfem_version": agentfem.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "cases": [measure(n, args.queries, args.repeats) for n in args.cells],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
