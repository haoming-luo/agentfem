"""Run the scheduled NAFEMS R0027 Test 7 structural creep benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from agentfem import __version__, benchmarks, platforms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radial-cells", type=int, default=4)
    parser.add_argument("--angular-cells", type=int, default=8)
    parser.add_argument("--increments", type=int, default=40)
    parser.add_argument("--duration", type=float, default=1000.0)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-acceptable", action="store_true")
    options = parser.parse_args()

    started = perf_counter()
    assessment = benchmarks.creep_thick_cylinder_benchmark(
        radial_cells=options.radial_cells,
        angular_cells=options.angular_cells,
        increments=options.increments,
        duration=options.duration,
        progress=True,
    )
    record = {
        "schema": "agentfem.external-structural-benchmark",
        "schema_version": "0.1.0",
        "benchmark": "NAFEMS R0027 Test 7",
        "status": "passed" if assessment.acceptable else "not_promoted",
        "agentfem_version": __version__,
        "configuration": {
            "radial_cells": options.radial_cells,
            "angular_cells": options.angular_cells,
            "increments": options.increments,
            "duration_hour": options.duration,
        },
        "assessment": assessment.as_dict(),
        "runtime_seconds": perf_counter() - started,
        "runtime": platforms.runtime_report().summary(),
    }
    options.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = options.report.with_suffix(options.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(options.report)
    print(json.dumps(record, indent=2, sort_keys=True))
    if options.require_acceptable and not assessment.acceptable:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
