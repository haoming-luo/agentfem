# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep the MPI launcher supervisor outside the MPI execution runtime."""

import argparse
import json
import os
import subprocess
import sys


def _probe_vendor():
    # Query the linked implementation in a short-lived process without MPI_Init.
    # Never change the environment inherited by the actual simulation ranks.
    environment = dict(os.environ, MPI4PY_RC_INITIALIZE="0", MPI4PY_RC_FINALIZE="0")
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "import json; from mpi4py import MPI; "
            "assert not MPI.Is_initialized(); print(json.dumps(MPI.get_vendor()))",
        ),
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    name, version = json.loads(completed.stdout)
    if not isinstance(name, str) or not isinstance(version, list):
        raise ValueError("Invalid MPI vendor probe response.")
    return name, tuple(int(item) for item in version)


def configure_mpi_parser(parser):
    """One parser contract shared with full CLI discovery."""
    parser.add_argument("-n", "--ranks", type=int, required=True)
    parser.add_argument(
        "--timeout",
        type=float,
        help="Stop the complete MPI process group after this many seconds.",
    )
    parser.add_argument("child_command", nargs=argparse.REMAINDER)


def _mpi_main(argv):
    from .mpi_runtime import audit_mpi_runtime, run_mpi_command

    parser = argparse.ArgumentParser(prog="agentfem mpi-run")
    configure_mpi_parser(parser)
    args = parser.parse_args(argv)
    child = args.child_command
    if child and child[0] == "--":
        child = child[1:]
    if args.ranks <= 0 or not child:
        parser.error("A positive rank count and child command are required.")
    try:
        # Nested launchers are not supported; do not start one supervisor per rank.
        for key in (
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
            "PMIX_SIZE",
            "MV2_COMM_WORLD_SIZE",
        ):
            if key in os.environ:
                raise ValueError("mpi-run must run outside an existing MPI job.")
        audit = audit_mpi_runtime(vendor=_probe_vendor(), rank_count=1)
        if not audit.compatible:
            from .mpi_runtime import MPILauncherError

            raise MPILauncherError(audit)
        print(
            f"AgentFEM MPI: {audit.family} launcher {audit.selected_launcher} "
            f"({args.ranks} ranks)",
            flush=True,
        )
        return run_mpi_command(args.ranks, child, timeout=args.timeout, audit=audit)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"AgentFEM MPI: {error}", file=sys.stderr, flush=True)
        return 1


def main(argv=None):
    selected = list(sys.argv[1:] if argv is None else argv)
    if selected and selected[0] == "mpi-run":
        return _mpi_main(selected[1:])
    from .cli import main as product_main

    return product_main(selected)


if __name__ == "__main__":
    raise SystemExit(main())
