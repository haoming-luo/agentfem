# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Paired local timing of one smooth-face geometry certificate.

This is deliberately a component screen, not a nonlinear Step benchmark.
The reference mode disables only the new squared-plane lower bound, leaving
the pre-existing conservative subdivision/fail-closed route intact. Run from
the source tree with ``PYTHONPATH=src python tools/benchmark_planar_branch_certificate.py``.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import platform
import sys
from time import perf_counter_ns
from unittest.mock import patch

import numpy as np

from agentfem import boundary_models
from agentfem.boundary_models.contact_planar_branch_bound import (
    certify_reviewed_planar_branch,
)


def _case(geometry):
    if geometry == "finite_vertex":
        extra_vertices = ((0.7, 1.6, 0.9), (0.9, 1.6, 0.9), (0.7, 2., 0.9))
        extra_faces = ((4, 5, 6),)
    else:
        extra_vertices = ((0.8, 2., 0.), (0.8, 2., 2.))
        extra_faces = ((3, 2, 5), (3, 5, 4))
    surface = boundary_models.triangulated_rigid_surface(
        vertices=(
            (1., 0., 0.), (1., 0., 2.), (1., 2., 2.), (1., 2., 0.),
            *extra_vertices,
        ),
        triangles=((0, 1, 2), (0, 2, 3), *extra_faces),
        name="planar_certificate_component_screen",
    )
    cells = np.asarray((((0.95, 0.4, 0.8), (0.95, 1.4, 0.8),
                         (0.95, 0.4, 1.1)),))
    projector = boundary_models.triangle_surface_bvh(surface)
    return cells, surface, projector


def _old_bound(*_args):
    # An inconclusive result forces the original four-child subdivision.
    return -float("inf"), np.zeros(3), np.zeros(3)


def _measure_once(cells, surface, projector, *, old):
    if old:
        with ExitStack() as disabled:
            disabled.enter_context(patch(
                "agentfem.boundary_models.contact_planar_branch_bound"
                "._quadratic_plane_distance_difference_lower_bound",
                _old_bound,
            ))
            disabled.enter_context(patch(
                "agentfem.boundary_models.contact_planar_branch_bound"
                "._quadratic_finite_feature_distance_difference_lower_bound",
                return_value=None,
            ))
            start = perf_counter_ns()
            result = certify_reviewed_planar_branch(
                cells, surface, projector=projector,
                maximum_refinement_levels=3,
            )
            elapsed = perf_counter_ns() - start
    else:
        start = perf_counter_ns()
        result = certify_reviewed_planar_branch(
            cells, surface, projector=projector,
            maximum_refinement_levels=3,
        )
        elapsed = perf_counter_ns() - start
    if not result.certified:
        raise RuntimeError(f"Certificate did not pass: {result.status}")
    return elapsed * 1e-6, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry", choices=("plane_competitor", "finite_vertex"),
        default="plane_competitor",
    )
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--warmup-pairs", type=int, default=20)
    args = parser.parse_args()
    if args.pairs < 3 or args.warmup_pairs < 0:
        parser.error("Require at least three measured pairs and nonnegative warmup.")
    cells, surface, projector = _case(args.geometry)
    timings = {"subdivision_reference_ms": [], "quadratic_bound_ms": []}
    reference = None
    for pair in range(args.warmup_pairs + args.pairs):
        # Reverse order on alternating pairs to reduce clock/thermal bias.
        modes = (True, False) if pair % 2 == 0 else (False, True)
        outcomes = {}
        for old in modes:
            duration, result = _measure_once(cells, surface, projector, old=old)
            outcomes[old] = result
            if pair >= args.warmup_pairs:
                timings[
                    "subdivision_reference_ms" if old else "quadratic_bound_ms"
                ].append(duration)
        old, new = outcomes[True], outcomes[False]
        np.testing.assert_allclose(new.plane_points, old.plane_points, atol=1e-12)
        np.testing.assert_allclose(new.plane_normals, old.plane_normals, atol=1e-12)
        reference = (old, new)

    def summary(values):
        return {"median": float(np.median(values)),
                "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75))}

    old, new = reference
    result = {
        "scope": "single_geometry_certificate_not_full_step",
        "geometry": args.geometry,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "machine": platform.machine(),
        "system": platform.system(),
        "warmup_pairs": args.warmup_pairs,
        "measured_pairs": args.pairs,
        "reference_examined_facets": old.examined_master_facet_count,
        "new_examined_facets": new.examined_master_facet_count,
        "subdivision_reference_ms": summary(timings["subdivision_reference_ms"]),
        "quadratic_bound_ms": summary(timings["quadratic_bound_ms"]),
    }
    result["median_reference_over_new"] = (
        result["subdivision_reference_ms"]["median"]
        / result["quadratic_bound_ms"]["median"]
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
