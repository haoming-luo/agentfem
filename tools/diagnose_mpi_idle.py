# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Measure POSIX MPI idle CPU in isolated children; never change runtime defaults.

Optional provider comparisons are diagnostics, not a recommendation to replace
an HPC fabric. Every provider must separately pass actual distributed tests.
The parent deliberately does not import mpi4py or initialize MPI.
"""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys


CHILD = """
import json, resource, sys, time
from mpi4py import MPI
MPI.COMM_WORLD.Barrier()
before = resource.getrusage(resource.RUSAGE_SELF)
started = time.perf_counter()
time.sleep(float(sys.argv[1]))
wall = time.perf_counter() - started
after = resource.getrusage(resource.RUSAGE_SELF)
user = after.ru_utime - before.ru_utime
system = after.ru_stime - before.ru_stime
print(json.dumps({
    "vendor": MPI.get_vendor(),
    "ranks": MPI.COMM_WORLD.size,
    "wall_seconds": wall,
    "user_cpu_seconds": user,
    "system_cpu_seconds": system,
    "idle_cpu_equivalent_cores": (user + system) / wall,
}))
"""


def probe(*, provider=None, seconds=2.0):
    if not math.isfinite(seconds) or not 0.1 <= seconds <= 10:
        raise ValueError("Idle observation must be 0.1..10 seconds.")
    environment = dict(os.environ)
    if provider is not None:
        if not provider or any(character.isspace() for character in provider):
            raise ValueError("A provider must be one nonempty libfabric expression.")
        environment["FI_PROVIDER"] = provider
    # Match serial library import, not a user-requested nested MPI job.
    if any(
        key in environment for key in ("PMI_SIZE", "PMIX_SIZE", "OMPI_COMM_WORLD_SIZE")
    ):
        raise ValueError("Run idle diagnostics outside an existing MPI job.")
    record = {
        "requested_provider": provider,
        "effective_provider_environment": environment.get("FI_PROVIDER"),
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-c", CHILD, str(seconds)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=seconds + 10,
            check=False,
        )
        if completed.returncode != 0:
            return {
                **record,
                "status": "failed",
                "returncode": completed.returncode,
                "diagnostic": completed.stderr[-4000:],
            }
        measured = json.loads(completed.stdout)
        if measured["ranks"] != 1 or not all(
            math.isfinite(measured[name]) and measured[name] >= 0
            for name in (
                "wall_seconds",
                "user_cpu_seconds",
                "system_cpu_seconds",
                "idle_cpu_equivalent_cores",
            )
        ):
            raise ValueError("Invalid idle observation.")
        return {**record, "status": "completed", **measured}
    except subprocess.TimeoutExpired:
        return {
            **record,
            "status": "timeout",
            "diagnostic": "MPI child exceeded its bounded initialization/observation window.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-provider", action="append", default=[])
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("POSIX resource accounting is required; use WSL on Windows.")
    if not 1 <= args.repetitions <= 5 or args.output.exists():
        parser.error("Use 1..5 repetitions and a new output path.")
    if not math.isfinite(args.seconds) or not 0.1 <= args.seconds <= 10:
        parser.error("Use an observation of 0.1..10 seconds.")
    records = []
    for repeat in range(args.repetitions):
        providers = [None, *args.compare_provider]
        for provider in providers if repeat % 2 == 0 else providers[::-1]:
            record = {
                "repeat": repeat + 1,
                **probe(provider=provider, seconds=args.seconds),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    report = {
        "schema": "agentfem.mpi-idle-diagnostic.v1",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "scope": "single_rank_idle_cpu_not_solver_speed_or_distributed_correctness",
        "environment_modified_in_parent": False,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
