# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded serial Q1 interface capacity and independent polynomial checks.

Reference unit-square traces only, not a forming or finite-rotation benchmark.
Reported wall time is descriptive; do not infer speedups from a concurrent run.
"""

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np

from agentfem import interfaces
from agentfem._interface_pairing import FixedReferenceCohesiveAssembler
from benchmark_nonmatching_blocks import surface


def run(negative, positive, maximum_points):
    start = perf_counter()
    pair = interfaces.pair_reference_traces(
        surface(negative),
        surface(positive, True),
        tolerance=1e-10,
        maximum_points=maximum_points,
    )
    prepared = perf_counter() - start
    audit = pair.constant_traction_audit()
    nodal_errors = {
        side: record["relative_nodal_measure_error_l2"]
        for side, record in audit["sides"].items()
    }
    # Coverage belongs to the common-refinement constructor. The independent
    # nodal patch deliberately does not claim that it proves surface coverage.
    coverage = pair.summary()["coverage"]
    if coverage != "coplanar-per-facet-area-and-self-overlap-checked":
        raise AssertionError(f"Unexpected coverage contract: {coverage}")
    if max(nodal_errors.values()) > 1e-9:
        raise AssertionError(f"Nodal patch failed: {audit}")
    stiffness = 10.0
    assembler = FixedReferenceCohesiveAssembler(
        pair,
        interfaces.elastic_cohesive(
            normal_stiffness=stiffness, tangential_stiffness=stiffness
        ),
        tangential="mixed",
    )
    un = np.zeros_like(pair.negative.vertices)
    up = np.zeros_like(pair.positive.vertices)
    up[:, 2] = pair.positive.vertices[:, 0] * pair.positive.vertices[:, 1]
    response = assembler.evaluate(un, up)
    expected_energy = stiffness / 18
    positive_force = response.positive_residual.sum(axis=0)
    negative_force = response.negative_residual.sum(axis=0)
    moment = np.cross(pair.negative.vertices, response.negative_residual).sum(
        axis=0
    ) + np.cross(pair.positive.vertices, response.positive_residual).sum(axis=0)
    np.testing.assert_allclose(
        response.stored_energy, expected_energy, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        positive_force, [0, 0, stiffness / 4], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(negative_force + positive_force, 0, atol=1e-10)
    np.testing.assert_allclose(moment, 0, atol=1e-10)
    rng = np.random.default_rng(561)
    dn, dp = rng.normal(size=un.shape), rng.normal(size=up.shape)
    tn, tp = assembler.tangent_action(un, up, dn, dp)
    nodal_work = float(np.sum(dn * tn) + np.sum(dp * tp))
    jump = pair.jump(dn, dp)
    quadrature_work = float(stiffness * np.sum(pair.weights[:, None] * jump**2))
    np.testing.assert_allclose(nodal_work, quadrature_work, rtol=1e-11, atol=1e-11)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "schema": "agentfem.nonmatching-q1-capacity.v1",
        "scope": "serial_fixed_reference_coplanar_q1_polynomial_and_virtual_work",
        "negative_facets": negative**2,
        "positive_facets": positive**2,
        "quadrature_points": len(pair.weights),
        "maximum_points": maximum_points,
        "pairing_fingerprint": pair.fingerprint,
        "coverage_check": coverage,
        "nodal_measure_relative_errors": nodal_errors,
        "polynomial_energy_absolute_error": abs(
            response.stored_energy - expected_energy
        ),
        "resultant_force_norm": float(np.linalg.norm(negative_force + positive_force)),
        "reference_moment_norm": float(np.linalg.norm(moment)),
        "virtual_work_relative_error": abs(nodal_work - quadrature_work)
        / abs(quadrature_work),
        "seed": 561,
        "pairing_preparation_seconds": prepared,
        "total_seconds": perf_counter() - start,
        "peak_rss_bytes": int(rss if sys.platform == "darwin" else 1024 * rss),
        "numpy_version": np.__version__,
        "accepted": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative", type=int, default=94)
    parser.add_argument("--positive", type=int, default=70)
    parser.add_argument("--maximum-points", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        not 1
        <= min(args.negative, args.positive)
        <= max(args.negative, args.positive)
        <= 100
    ):
        parser.error("Use 1..100 facets per axis")
    if not 1 <= args.maximum_points <= 1_000_000:
        parser.error("Use a point budget of 1..1000000")
    if args.output is not None and args.output.exists():
        parser.error("Refusing to overwrite evidence")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    )
    record = run(args.negative, args.positive, args.maximum_points)
    record.update(revision=revision, tracked_source_dirty=dirty)
    print(json.dumps(record, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
