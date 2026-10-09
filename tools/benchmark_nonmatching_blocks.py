# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Local response/block microbenchmark; not a global solver speedup claim."""

import argparse
import json
from time import perf_counter

import numpy as np

from agentfem import interfaces
from agentfem._interface_pairing import FixedReferenceCohesiveAssembler


def surface(n, reverse=False):
    vertices = np.array(
        [[i / n, j / n, 0.0] for j in range(n + 1) for i in range(n + 1)]
    )
    cells = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            cells.append([a, a + 1, a + n + 2, a + n + 1])
    cells = np.array(cells)
    return interfaces.reference_trace(
        vertices,
        cells[:, ::-1] if reverse else cells,
        topology="quadrilateral",
        tolerance=1e-10,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-cells", type=int, default=10)
    parser.add_argument("--positive-cells", type=int, default=13)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.negative_cells, args.positive_cells, args.repeats) < 1:
        parser.error("counts must be positive")
    if max(args.negative_cells, args.positive_cells) > 100 or args.repeats > 20:
        parser.error("bounded benchmark: at most 100 cells per side axis and 20 repeats")
    start = perf_counter()
    pair = interfaces.pair_reference_traces(
        surface(args.negative_cells),
        surface(args.positive_cells, True),
        tolerance=1e-10,
    )
    preparation_seconds = perf_counter() - start
    assembler = FixedReferenceCohesiveAssembler(
        pair,
        interfaces.elastic_cohesive(normal_stiffness=10, tangential_stiffness=5),
        tangential="mixed",
    )
    un, up = (
        np.zeros_like(pair.negative.vertices),
        np.zeros_like(pair.positive.vertices),
    )
    records = {}
    for mode in ("point_blocks", "pair_blocks"):
        durations = []
        for _ in range(args.repeats):
            start = perf_counter()
            total = 0.0
            count = 0
            if mode == "point_blocks":
                tangent = assembler._point_response(un, up, begin=False)["tangent"][
                    :, 0
                ]
                for q, weight in enumerate(pair.weights):
                    shape = np.concatenate(
                        (-pair.negative_weights[q], pair.positive_weights[q])
                    )
                    block = weight * np.einsum("a,ij,b->aibj", shape, tangent[q], shape)
                    total += float(block.reshape(24, 24).trace())
                    count += 1
            else:
                for _, _, block in assembler.tangent_blocks(un, up):
                    total += float(block.trace())
                    count += 1
            durations.append(perf_counter() - start)
        records[mode] = dict(
            median_seconds=float(np.median(durations)), blocks=count, trace=total
        )
    if not np.isclose(
        records["point_blocks"]["trace"], records["pair_blocks"]["trace"], rtol=1e-12
    ):
        raise RuntimeError("Benchmark block traces disagree.")
    print(
        json.dumps(
            dict(
                scope="local_response_and_block_generation_not_global_solve",
                quadrature_points=len(pair.weights),
                pairing_preparation_seconds=preparation_seconds,
                repeats=args.repeats,
                measurements=records,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
