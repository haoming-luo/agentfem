# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare old/new boundary snapshots; this is not a FEM solve speed benchmark.

Run from an installed development environment or with PYTHONPATH=src.
No MPI, solver, network, or output files are created.
"""

import argparse
import json
import platform
import statistics
import timeit
import tracemalloc
from types import SimpleNamespace

import numpy as np

from agentfem._operator_lifecycle import boundary_dof_identity


def _previous_identity(bcs):
    records = []
    for bc in bcs:
        dofs, owned = bc.dof_indices()
        records.append((tuple(int(item) for item in dofs), int(owned)))
    return tuple(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dofs", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--number", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()
    if min(*args.dofs, args.number, args.repeat) <= 0:
        parser.error("sizes, number and repeat must be positive")
    measurements = []
    for size in args.dofs:
        indices = np.arange(size, dtype=np.int32)
        bcs = (
            SimpleNamespace(
                dof_indices=lambda indices=indices, size=size: (indices, size)
            ),
        )
        record = {"boundary_dofs": size}
        for name, implementation in (
            ("previous", _previous_identity),
            ("current", boundary_dof_identity),
        ):
            baseline = implementation(bcs)
            samples = timeit.repeat(
                lambda implementation=implementation, bcs=bcs, baseline=baseline: (
                    implementation(bcs) == baseline
                ),
                number=args.number,
                repeat=args.repeat,
            )
            tracemalloc.start()
            snapshot = implementation(bcs)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            assert snapshot == baseline
            record[name] = {
                "median_seconds": statistics.median(samples) / args.number,
                "snapshot_peak_python_bytes": peak,
            }
        measurements.append(record)
    print(
        json.dumps(
            {
                "scope": "boundary_snapshot_and_equality_only_not_end_to_end_solve",
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "number": args.number,
                "repeat": args.repeat,
                "measurements": measurements,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
