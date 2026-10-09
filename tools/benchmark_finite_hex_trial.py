# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Serial private finite-Hex trial cost, not a full trajectory benchmark."""

import argparse
import cProfile
import json
import pstats
import resource
import sys
from statistics import median
from time import perf_counter

import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI

from agentfem import constitutive
from agentfem.constitutive.material_driver import MaterialQuadratureResponse
from agentfem.elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual
from agentfem.mechanics._finite_hex_explicit import FiniteHexExplicitResidual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.size <= 32 or not 1 <= args.repeats <= 20:
        parser.error("Use size 1..32 and repeats 1..20.")
    if MPI.COMM_WORLD.size != 1:
        parser.error("This private bridge is serial only.")
    domain = mesh.create_unit_cube(
        MPI.COMM_SELF,
        args.size,
        args.size,
        args.size,
        cell_type=mesh.CellType.hexahedron,
    )
    u = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
    law = constitutive.finite_strain_j2_logarithmic(
        young=100, poisson=0.3, yield_stress=1e9, hardening_modulus=0
    )
    response = MaterialQuadratureResponse.create(
        domain,
        law.state_schema,
        degree=1,
        stored_energy_component_names=law.stored_energy_component_names,
    )
    internal = FiniteUniformHexResidual(
        u, response, density=2, hourglass_modulus=40, hourglass_scale=0.1
    )
    residual = FiniteHexExplicitResidual(internal, law, omega_squared_bound=1e9)
    snapshot = residual.transaction_snapshot()
    x = u.function_space.tabulate_dof_coordinates()
    u.x.array[:] = (0.02 * x).ravel()
    timings = []
    norm = None
    profiler = cProfile.Profile()
    for index in range(args.repeats + 1):
        residual.update_time(1e-6)
        if args.profile and index:
            profiler.enable()
        start = perf_counter()
        vector = residual.assemble_vector()
        timings.append(perf_counter() - start)
        if args.profile and index:
            profiler.disable()
        norm = float(np.linalg.norm(vector.array))
        vector.destroy()
        residual.restore(snapshot)
    print(
        json.dumps(
            {
                "cells": args.size**3,
                "repeats": args.repeats,
                "scope": "private_serial_finite_hex_trial_no_commit_or_io",
                "median_seconds": median(timings[1:]),
                "force_norm": norm,
                "profiled": args.profile,
                "peak_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024)
                ),
            },
            indent=2,
        )
    )
    if args.profile:
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(25)


if __name__ == "__main__":
    main()
