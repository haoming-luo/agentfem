# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private serial finite-Hex affine-path endurance oracle, not product promotion.

The six faces prescribe their normal displacement, u_i = rate*t at X_i=1
and zero at X_i=0. With v(X,0)=rate*X, homogeneous dilation is exact: the
Piola stress is spatially constant and every unconstrained acceleration is zero.
This checks assembly/lifecycle across many increments, not spatial convergence.
"""

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np
from dolfinx import mesh
from mpi4py import MPI

from agentfem import (
    amplitudes,
    constitutive,
    fields,
    models,
    problems,
    state,
    studies,
    time,
)
from agentfem.constitutive.material_driver import MaterialQuadratureResponse
from agentfem.diagnostics import MechanicalEnergyMonitor
from agentfem.elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual
from agentfem.mechanics._finite_hex_explicit import FiniteHexExplicitResidual


def run(size, steps, max_rss_mb=4096):
    if MPI.COMM_WORLD.size != 1:
        raise ValueError("The private finite bridge is serial only.")
    domain = mesh.create_unit_cube(
        MPI.COMM_SELF, size, size, size, cell_type=mesh.CellType.hexahedron
    )
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    duration, rate = 0.1, 1.0
    amplitude = amplitudes.Amplitude(
        "affine_dilation", lambda t: rate * t, metadata={"rate": rate}
    )
    for component in range(3):
        model.fix(
            u, on=lambda x, i=component: np.isclose(x[i], 0), components=component
        )
        model.fix(
            u,
            on=lambda x, i=component: np.isclose(x[i], 1),
            components=component,
            value=amplitude,
        )
    law = constitutive.finite_strain_j2_logarithmic(
        young=100, poisson=0.3, yield_stress=1e9
    )
    response = MaterialQuadratureResponse.create(
        domain,
        law.state_schema,
        degree=1,
        stored_energy_component_names=law.stored_energy_component_names,
    )
    history = state.second_order_state(u)
    internal = FiniteUniformHexResidual(
        history.u, response, density=2, hourglass_modulus=40, hourglass_scale=0.1
    )
    residual = FiniteHexExplicitResidual(internal, law, omega_squared_bound=1e7)
    x = history.u.value.function_space.tabulate_dof_coordinates()
    history.v.value.x.array[:] = (rate * x).ravel()
    step = problems.explicit_dynamics(
        state=history,
        integrator=time.explicit.central_difference(state=history, mass=internal),
        residual=residual,
        prescribed=tuple(model.constraints),
        dt=duration / steps,
        steps=steps,
        history_every=max(1, steps // 10),
        progress=False,
        history_monitor=MechanicalEnergyMonitor(mass=internal.mass_diagonal),
    )
    start = perf_counter()
    samples = []
    for station in sorted({1, 5, 10} | {round(steps * k / 10) for k in range(1, 11)}):
        step.run(until_step=station)
        t = step.completed_steps * step.dt
        stretch = 1 + rate * t
        bulk = 100 / (3 * (1 - 2 * 0.3))
        expected_stress = np.eye(3) * 3 * bulk * np.log(stretch) / stretch**3
        displacement_error = float(
            np.max(np.abs(history.u.value.x.array.reshape(-1, 3) - rate * t * x))
        )
        stress_error = float(
            np.max(np.abs(response.cauchy_stress.values - expected_stress))
        )
        energy = float(
            internal.cells.volume @ response.strain_energy_density.values.reshape(-1)
        )
        expected_energy = 4.5 * bulk * np.log(stretch) ** 2
        if (
            displacement_error > 1e-9
            or stress_error > 1e-6
            or abs(energy - expected_energy) > 1e-7
        ):
            raise AssertionError(
                f"Affine oracle failed: {displacement_error=}, {stress_error=}, {energy=}, {expected_energy=}"
            )
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        sample = {
            "step": station,
            "time": t,
            "displacement_error": displacement_error,
            "stress_error": stress_error,
            "energy_error": abs(energy - expected_energy),
            "peak_rss_bytes": int(rss if sys.platform == "darwin" else 1024 * rss),
            "wall_seconds": perf_counter() - start,
        }
        samples.append(sample)
        print(json.dumps(sample), flush=True)
        if sample["peak_rss_bytes"] > max_rss_mb * 1024**2:
            raise MemoryError("Observed process RSS exceeded the verification budget.")
    return {
        "cells": size**3,
        "steps": steps,
        "scope": "private_affine_path_endurance_not_spatial_convergence",
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-rss-mb",
        type=int,
        default=4096,
        help="Stop at an observation station above this RSS budget; not an OS hard limit.",
    )
    args = parser.parse_args()
    if not 1 <= args.size <= 64 or not 200 <= args.steps <= 2000:
        parser.error("Use size 1..64 and steps 200..2000.")
    if not 512 <= args.max_rss_mb <= 8192:
        parser.error("Use an observed RSS budget between 512 and 8192 MiB.")
    if args.size > 32 and args.max_rss_mb < 8192:
        parser.error("Larger meshes require an explicit --max-rss-mb 8192 budget.")
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new evidence path.")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "HEAD",
                "--",
                "src/agentfem",
                "tools/verify_finite_hex_patch.py",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    result = run(args.size, args.steps, args.max_rss_mb)
    result.update(
        {
            "revision": revision,
            "tracked_source_dirty_at_start": dirty,
            "numpy_version": np.__version__,
            "observed_rss_budget_mb": args.max_rss_mb,
        }
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
