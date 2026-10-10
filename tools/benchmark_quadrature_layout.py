# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Cold quadrature layout construction only; not time-stepping speedup."""

import argparse
import json
from statistics import median
from time import perf_counter

import numpy as np
from dolfinx import mesh
from mpi4py import MPI

from agentfem.constitutive.quadrature import QuadratureField


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.size <= 64 or not 1 <= args.repeats <= 10:
        parser.error("Use size 1..64 and repeats 1..10.")
    if MPI.COMM_WORLD.size != 1:
        parser.error("Serial component benchmark only.")
    domain = mesh.create_unit_cube(
        MPI.COMM_SELF,
        args.size,
        args.size,
        args.size,
        cell_type=mesh.CellType.hexahedron,
    )
    field = QuadratureField.create(domain, name="tensor", degree=1, value_shape=(3, 3))
    count = domain.topology.index_map(3).size_local
    V = field.function.function_space
    slow, fast = [], []
    for repeat in range(args.repeats):
        # Alternate order to reduce systematic warm-cache ordering bias.
        outputs = {}
        for kind in ("cellwise", "bulk") if repeat % 2 == 0 else ("bulk", "cellwise"):
            start = perf_counter()
            if kind == "cellwise":
                dofs = np.empty((count, len(field.points)), dtype=np.int64)
                for c in range(count):
                    dofs[c] = V.dofmap.cell_dofs(c)
                output = (
                    dofs[..., None] * V.dofmap.bs + np.arange(V.dofmap.bs)
                ).ravel()
                slow.append(perf_counter() - start)
            else:
                field._array_indices_cache = None
                output = field._array_indices()
                fast.append(perf_counter() - start)
            outputs[kind] = output
        np.testing.assert_array_equal(outputs["cellwise"], outputs["bulk"])
    print(
        json.dumps(
            {
                "scope": "cold quadrature layout only; not full simulation",
                "cells": count,
                "components": 9,
                "repeats": args.repeats,
                "cellwise_seconds": slow,
                "bulk_seconds": fast,
                "component_speedup": median(slow) / median(fast),
                "indices_identical": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
