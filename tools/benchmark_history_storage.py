# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded result-publication benchmark, not a finite-element solve benchmark."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter

import numpy as np
import agentfem
from agentfem import results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100001)
    parser.add_argument("--channels", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 2 or args.channels < 1 or args.output.exists():
        parser.error("Use at least two samples, one channel and a fresh output file.")
    simulation = results.SimulationResult("history-storage-benchmark")
    coordinate = np.linspace(0., .1, args.samples)
    for index in range(args.channels):
        simulation.add_history(f"quantity_{index}", coordinate,
                               np.sin((index + 1) * coordinate), unit="1")
    records = []
    # Temporary generated artifacts are removed; retain only the measured report.
    with tempfile.TemporaryDirectory(prefix="agentfem-history-storage-") as directory:
        # Runtime provenance and reader imports belong to neither encoding's
        # steady publication cost. Warm both routes before alternating trials.
        warmup = results.SimulationResult("history-storage-warmup")
        warmup.add_history("q", [0., 1.], [0., 1.])
        for storage in ("inline", "npz"):
            target = Path(directory) / f"warmup-{storage}" / "result.json"
            warmup.write_manifest(target, include_histories=True, history_storage=storage)
            results.read_histories(target)
        for index, storage in enumerate(("inline", "npz", "npz", "inline")):
            target = Path(directory) / f"trial-{index}" / "result.json"
            start = perf_counter()
            simulation.write_manifest(target, include_histories=True, history_storage=storage)
            write_seconds = perf_counter() - start
            start = perf_counter()
            restored = results.read_histories(target)
            read_seconds = perf_counter() - start
            for old, new in zip(simulation.histories.values(), restored, strict=True):
                assert old.as_dict(include_values=False) == new.as_dict(include_values=False)
                assert old.abscissa.tobytes() == new.abscissa.tobytes()
                assert old.values.tobytes() == new.values.tobytes()
            records.append(dict(storage=storage, write_seconds=write_seconds,
                                read_seconds=read_seconds,
                                bytes=sum(p.stat().st_size for p in target.parent.iterdir()),
                                complete_bitwise_roundtrip=True))
    report = dict(scope="synthetic complete-history publication and reading, not solver throughput",
                  environment={"platform": platform.platform(), "python": sys.version,
                               "numpy": np.__version__, "agentfem": agentfem.__version__},
                  benchmark_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  implementation_sha256={
                      name: sha256((Path(agentfem.__file__).parent / name).read_bytes()).hexdigest()
                      for name in ("results/history_archive.py", "results/core.py", "results/__init__.py", "provenance.py")
                  },
                  samples_per_channel=args.samples, channels=args.channels, records=records,
                  warmup="one two-sample publication/read per encoding, excluded",
                  execution_order=[item["storage"] for item in records],
                  compression="none", sample_decimation="none", data_precision="binary64")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
