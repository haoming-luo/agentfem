# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent proportional finite-plastic bar oracle and bounded capacity run.

This verifies homogeneous uniaxial strain, not general forming or spatial
convergence. Preparation, run time and observed peak process RSS stay separate.
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
    fracture,
    models,
    problems,
    state,
    studies,
    time,
)
from agentfem.constitutive.material_driver import MaterialQuadratureResponse
from agentfem.elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual
from agentfem.mechanics._finite_hex_explicit import FiniteHexExplicitResidual
from agentfem.mechanics._finite_hex_energy import FiniteHexEnergyMonitor


def prepare(size, steps, *, public_step=False):
    if MPI.COMM_WORLD.size != 1:
        raise ValueError("Use the serial capacity oracle without an MPI launcher.")
    domain = mesh.create_unit_cube(
        MPI.COMM_SELF, size, size, size, cell_type=mesh.CellType.hexahedron
    )
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(1, 2))
    model.fix(u, on=lambda x: np.isclose(x[0], 0), components=0)
    model.fix(
        u,
        on=lambda x: np.isclose(x[0], 1),
        components=0,
        value=amplitudes.Amplitude(
            "constant_stretch_rate", lambda t: 2 * t, metadata={"rate": 2}
        ),
    )
    law = constitutive.finite_strain_j2_logarithmic(
        young=100, poisson=0.3, yield_stress=1, hardening_modulus=5,
        density=2 if public_step else None,
    )
    if public_step:
        from agentfem import elements

        model.material(law)
        step = model.step(
            target=u,
            element_policy=elements.uniform_strain_hex8(
                hourglass_modulus=40, hourglass_scale=0.1,
                kinematics="finite_strain",
            ),
            omega_squared_bound=1e8,
            maximum_negative_growth_per_increment=0.1,
            dt=0.1 / steps,
            steps=steps,
            history_every=max(1, steps // 20),
            progress=False,
        )
        xyz = u.value.function_space.tabulate_dof_coordinates()
        step.state.v.value.x.array[::3] = 2 * xyz[:, 0]
        return step, xyz
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
    residual = FiniteHexExplicitResidual(
        internal,
        law,
        omega_squared_bound=1e8,
        maximum_negative_growth_per_increment=0.1,
    )
    residual.enable_energy(
        law.initial_array_response(len(response.state.reference_field.values))
    )
    xyz = history.u.value.function_space.tabulate_dof_coordinates()
    history.v.value.x.array[::3] = 2 * xyz[:, 0]
    ledger = fracture.DynamicEnergyLedger(
        energy=FiniteHexEnergyMonitor(residual),
        state=history,
        mass=internal.mass_diagonal,
        residual=residual,
        prescribed=tuple(model.constraints),
    )
    step = problems.explicit_dynamics(
        state=history,
        integrator=time.explicit.central_difference(state=history, mass=internal),
        residual=residual,
        prescribed=tuple(model.constraints),
        dt=0.1 / steps,
        steps=steps,
        history_every=max(1, steps // 20),
        history_monitor=ledger,
        progress=False,
    )
    return step, xyz


def oracle(t):
    stretch = 1 + 2 * t
    log_stretch = np.log(stretch)
    mu, bulk, hardening, yield_stress = 100 / 2.6, 100 / 1.2, 5, 1
    peeq = max(0, (2 * mu * log_stretch - yield_stress) / (3 * mu + hardening))
    elastic_deviator = log_stretch - 1.5 * peeq
    tau = np.diag(
        [
            bulk * log_stretch + 4 * mu / 3 * elastic_deviator,
            bulk * log_stretch - 2 * mu / 3 * elastic_deviator,
            bulk * log_stretch - 2 * mu / 3 * elastic_deviator,
        ]
    )
    energy = (
        bulk / 2 * log_stretch**2
        + 2 * mu / 3 * elastic_deviator**2
        + hardening / 2 * peeq**2
    )
    return tau / stretch, energy, peeq, yield_stress * peeq


def run(size, steps, max_rss_mb, *, public_step=False):
    start = perf_counter()
    step, xyz = prepare(size, steps, public_step=public_step)
    preparation = perf_counter() - start
    start = perf_counter()
    samples = []
    # Check capacity frequently; no per-increment JSON or growing field copies.
    for station in range(5, steps + 5, 5):
        step.run(until_step=min(station, steps))
        t = step.completed_steps * step.dt
        stress, energy, peeq, dissipation = oracle(t)
        actual = step.residual.internal.response
        ledger = step.history_monitor.evaluate(
            displacement=step.state.u, velocity=step.state.v
        )
        sample = {
            "step": step.completed_steps,
            "time": t,
            "displacement_error": float(
                np.max(np.abs(step.state.u.value.x.array[::3] - 2 * t * xyz[:, 0]))
            ),
            "stress_error": float(np.max(np.abs(actual.cauchy_stress.values - stress))),
            "stored_energy_error": abs(ledger["bulk_stored_energy"] - energy),
            "peeq_error": float(
                np.max(np.abs(actual.state.committed_state_vectors()[:, 9] - peeq))
            ),
            "dissipation_error": abs(ledger["material_dissipation"] - dissipation),
            "energy_balance_error": ledger["energy_balance_error"],
            "relative_energy_balance_error": ledger["relative_energy_balance_error"],
            "hourglass_energy": ledger["hourglass_energy"],
            "wall_seconds": perf_counter() - start,
        }
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        sample["peak_rss_bytes"] = int(rss if sys.platform == "darwin" else 1024 * rss)
        if sample["peak_rss_bytes"] > max_rss_mb * 1024**2:
            raise MemoryError("Observed process RSS exceeded the capacity budget.")
        if (
            sample["displacement_error"] > 1e-8
            or sample["stress_error"] > 1e-5
            or sample["stored_energy_error"] > 1e-6
            or sample["peeq_error"] > 1e-7
            or sample["dissipation_error"] > 1e-7
        ):
            raise AssertionError(f"Independent plastic oracle mismatch: {sample}")
        samples.append(sample)
        print(json.dumps(sample), flush=True)
    if samples[-1]["relative_energy_balance_error"] > 5e-4:
        raise AssertionError("Finite-plastic path did not meet the energy gate.")
    return {
        "schema": "agentfem.private-finite-plastic-capacity.v1",
        "cells": size**3,
        "steps": steps,
        "dt": step.dt,
        "preparation_seconds": preparation,
        "scope": "homogeneous_uniaxial_strain_not_spatial_convergence_or_forming",
        "entrypoint": "ordinary_model_step" if public_step else "private_oracle_procedure",
        "material": step.residual.material.summary(),
        "samples": samples,
        "stability": step.residual.summary(),
        "observed_rss_budget_mb": max_rss_mb,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-rss-mb", type=int, default=4096)
    parser.add_argument("--public-step", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.size <= 64 or not 1000 <= args.steps <= 4000 or args.steps % 5:
        parser.error("Use size 1..64 and steps 1000..4000, divisible by 5.")
    if not 512 <= args.max_rss_mb <= 8192 or args.size > 32 and args.max_rss_mb < 8192:
        parser.error("RSS budget must be 512..8192 MiB; size >32 requires 8192.")
    if args.output.exists():
        parser.error("Refusing to overwrite evidence.")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    )
    result = run(args.size, args.steps, args.max_rss_mb, public_step=args.public_step)
    result.update(revision=revision, tracked_source_dirty=dirty)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
