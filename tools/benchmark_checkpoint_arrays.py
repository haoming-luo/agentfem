# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare auxiliary serialization only; excludes mesh, solve and nodal archive."""

import argparse
import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import tracemalloc

import numpy as np

from agentfem import checkpointing
from agentfem._checkpoint_arrays import decode_tree, encode_tree
from benchmark_finite_hex_trial import make_case


def run(size):
    u, residual = make_case(size)
    x = u.function_space.tabulate_dof_coordinates()
    u.x.array[:] = (0.02 * x).ravel()
    residual.update_time(1e-6)
    vector = residual.assemble_vector()
    vector.destroy()
    residual.commit()
    expected = residual.transaction_snapshot()
    records = []
    with TemporaryDirectory(prefix="agentfem-auxiliary-benchmark-") as directory:
        root = Path(directory)
        for kind in ("json", "numeric_tree"):
            gc.collect()
            tracemalloc.start()
            start = perf_counter()
            metadata = root / f"{kind}.json"
            if kind == "json":
                record = residual.snapshot()
                checkpointing.atomic_write_text(
                    metadata, json.dumps(record, indent=2, allow_nan=False)
                )
                paths = [metadata]
            else:
                tree, arrays = encode_tree(residual.checkpoint_snapshot())
                payload = root / "arrays.npz"
                checkpointing.atomic_savez(payload, **arrays)
                checkpointing.atomic_write_text(
                    metadata, json.dumps(tree, indent=2, allow_nan=False)
                )
                paths = [metadata, payload]
            elapsed = perf_counter() - start
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            if kind == "json":
                restored = json.loads(metadata.read_text())
            else:
                with np.load(payload, allow_pickle=False) as loaded:
                    restored = decode_tree(json.loads(metadata.read_text()), loaded)
            np.testing.assert_array_equal(restored["gradient"], expected["gradient"])
            for name, values in expected["fields"].items():
                np.testing.assert_array_equal(restored["fields"][name], values)
            residual.restore(restored)
            records.append(
                dict(
                    encoding=kind,
                    seconds=elapsed,
                    python_traced_peak_bytes=peak,
                    file_bytes=sum(path.stat().st_size for path in paths),
                )
            )
            del restored
            if kind == "json":
                del record
    return dict(cells=size**3, scope="auxiliary_serialization_only", records=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.size <= 24:
        parser.error("Use size 1..24 for the JSON comparison memory budget")
    print(json.dumps(run(args.size), indent=2))
