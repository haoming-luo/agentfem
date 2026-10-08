# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Run the bounded 3D two-way thermoelastic verification prototype."""

import argparse
from pathlib import Path

from mpi4py import MPI

from agentfem.benchmarks.thermoelastic_coupling import _ThermoelasticPrototype
from agentfem.provenance import collective_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.002)
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--nonuniform", action="store_true")
    parser.add_argument("--inward-heat-flux", type=float, default=0.0)
    parser.add_argument("--dilation-rate", type=float)
    parser.add_argument("--temperature-rate", type=float)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        parser.error("--stop-after must lie within the declared path")
    comm = MPI.COMM_WORLD
    with _ThermoelasticPrototype(
        comm=comm,
        cells=args.cells,
        dt=args.dt,
        alpha=args.alpha,
        nonuniform=args.nonuniform,
        inward_heat_flux=args.inward_heat_flux,
        dilation_rate=args.dilation_rate,
        temperature_rate=args.temperature_rate,
    ) as case:
        if args.resume:
            case.load_checkpoint(args.resume, total_steps=args.steps)
        for _ in range(case.completed_steps, args.stop_after or args.steps):
            record = case.advance(relaxation=args.relaxation)
            if comm.rank == 0:
                print(
                    f"step={record['step']} t={record['time']:.4g} "
                    f"outer_iterations={record['outer_iterations']} "
                    f"T_error={record['temperature_reference_error']:.3e} "
                    f"u_error={record['displacement_reference_error']:.3e}"
                )
        if args.checkpoint:
            case.save_checkpoint(args.checkpoint, total_steps=args.steps)
        result = case.result()
        if args.output:
            collective_call(
                lambda: result.write_manifest(args.output) if comm.rank == 0 else None,
                comm=comm,
                label="write prototype result",
            )
        if comm.rank == 0:
            print(result.format())
            print(
                "Experimental benchmark only; no production coupled Step or general first-law claim."
            )


if __name__ == "__main__":
    main()
