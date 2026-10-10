# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent nonuniform finite-plastic manufactured trajectory.

The prescribed exact displacement is u_x=A(t) sin(pi X)/pi, u_y=u_z=0.
A(t)=0.2(t/T)^2 and T=0.1. Each point follows monotonic coaxial tension or
compression, so the finite Hencky J2 return has an independent scalar solution.
Reference body force is rho*u_tt - dP11/dX, evaluated with nodal quadrature.
This is a convergence check, not a forming benchmark or a public solve factory.
"""

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
from dolfinx import mesh
from mpi4py import MPI

from agentfem import (
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
from agentfem.mechanics._finite_hex_energy import FiniteHexEnergyMonitor
from agentfem.mechanics._finite_hex_explicit import FiniteHexExplicitResidual


MU, BULK, HARDENING, YIELD, RHO, DURATION = 100 / 2.6, 100 / 1.2, 5, 1, 2, 0.1


def reference(x, t):
    amplitude = 0.2 * (t / DURATION) ** 2
    stretch = 1 + amplitude * np.cos(np.pi * x)
    logarithm = np.log(stretch)
    peeq = np.maximum(0, (2 * MU * np.abs(logarithm) - YIELD) / (3 * MU + HARDENING))
    elastic = logarithm - 1.5 * peeq * np.sign(logarithm)
    tau = BULK * logarithm + 4 * MU / 3 * elastic
    slope = np.where(
        peeq > 0,
        BULK + 4 * MU * HARDENING / (3 * (3 * MU + HARDENING)),
        BULK + 4 * MU / 3,
    )
    acceleration = 0.4 / DURATION**2 * np.sin(np.pi * x) / np.pi
    force_density = (
        RHO * acceleration
        + amplitude * np.pi * np.sin(np.pi * x) * (slope - tau) / stretch**2
    )
    return {
        "u": amplitude * np.sin(np.pi * x) / np.pi,
        "a": acceleration,
        "force_density": force_density,
        "stress": tau / stretch,
        "peeq": peeq,
    }


class ManufacturedForce:
    """Verification load with no constitutive or state update implementation."""

    def __init__(self, residual, x):
        self.residual, self.x = residual, x

    def assemble_vector(self):
        internal = self.residual.internal
        force = internal.displacement.x.petsc_vec.duplicate()
        force.set(0)
        force.array[::3] = (
            internal.mass_diagonal[::3]
            / RHO
            * reference(self.x, self.residual.time)["force_density"]
        )
        return force


class LoadedVerificationResidual:
    """Test-only additive load composition; the core owns all history."""

    def __init__(self, internal, force):
        self.internal_residual, self.force = internal, force

    def __getattr__(self, name):
        return getattr(self.internal_residual, name)

    def _subtract(self, vector):
        force = self.force.assemble_vector()
        try:
            vector.axpy(-1, force)
            return vector
        finally:
            force.destroy()

    def assemble_vector(self):
        return self._subtract(self.internal_residual.assemble_vector())

    def assemble_accepted_vector(self):
        return self._subtract(self.internal_residual.assemble_accepted_vector())


def run(size, steps=2000):
    if MPI.COMM_WORLD.size != 1:
        raise ValueError("This independent manufactured driver is serial.")
    started = perf_counter()
    domain = mesh.create_unit_cube(
        MPI.COMM_SELF, size, 2, 2, cell_type=mesh.CellType.hexahedron
    )
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(1, 2))
    model.fix(u, on=lambda x: np.isclose(x[0], 0) | np.isclose(x[0], 1), components=0)
    law = constitutive.finite_strain_j2_logarithmic(
        young=100, poisson=0.3, yield_stress=YIELD, hardening_modulus=HARDENING
    )
    response = MaterialQuadratureResponse.create(
        domain,
        law.state_schema,
        degree=1,
        stored_energy_component_names=law.stored_energy_component_names,
    )
    history = state.second_order_state(u)
    internal = FiniteUniformHexResidual(
        history.u, response, density=RHO, hourglass_modulus=40, hourglass_scale=0.1
    )
    residual = FiniteHexExplicitResidual(
        internal,
        law,
        omega_squared_bound=1e8,
        maximum_negative_growth_per_increment=0.1,
    )
    residual.enable_energy(law.initial_array_response(len(internal.cell_nodes)))
    x = history.u.value.function_space.tabulate_dof_coordinates()[:, 0]
    history.a.value.x.array[::3] = reference(x, 0)["a"]
    force = ManufacturedForce(residual, x)
    total = LoadedVerificationResidual(residual, force)
    ledger = fracture.DynamicEnergyLedger(
        energy=FiniteHexEnergyMonitor(residual),
        state=history,
        mass=internal.mass_diagonal,
        residual=total,
        natural_force=force,
        prescribed=tuple(model.constraints),
    )
    step = problems.explicit_dynamics(
        state=history,
        integrator=time.explicit.central_difference(state=history, mass=internal),
        residual=total,
        prescribed=tuple(model.constraints),
        dt=DURATION / steps,
        steps=steps,
        history_every=steps,
        history_monitor=ledger,
        progress=False,
    )
    step.run()
    exact = reference(x, DURATION)
    cell_x = internal.cells.coordinates[:, :, 0].mean(1)
    exact_cell = reference(cell_x, DURATION)
    mass = internal.mass_diagonal[::3]
    numerical = history.u.value.x.array[::3]
    weighted = lambda value, weight: float(np.sqrt(np.dot(weight, value**2)))
    record = {
        "size": size,
        "cells": len(internal.cell_nodes),
        "steps": steps,
        "dt": step.dt,
        "displacement_relative_l2_nodal_quadrature": weighted(
            numerical - exact["u"], mass
        )
        / weighted(exact["u"], mass),
        "stress_relative_l2_cell": weighted(
            response.cauchy_stress.values[:, 0, 0] - exact_cell["stress"],
            internal.cells.volume,
        )
        / weighted(exact_cell["stress"], internal.cells.volume),
        "peeq_relative_l2_cell": weighted(
            response.state.committed_state_vectors()[:, 9] - exact_cell["peeq"],
            internal.cells.volume,
        )
        / weighted(exact_cell["peeq"], internal.cells.volume),
        "energy": {
            key: value
            for key, value in step.history_records[-1].items()
            if "energy" in key or "work" in key or "dissipation" in key
        },
        "wall_seconds": perf_counter() - started,
    }
    if not all(
        np.isfinite(record[key])
        for key in (
            "displacement_relative_l2_nodal_quadrature",
            "stress_relative_l2_cell",
            "peeq_relative_l2_cell",
        )
    ):
        raise AssertionError("Non-finite manufactured response.")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        any(not 4 <= value <= 128 for value in args.sizes)
        or not 1000 <= args.steps <= 8000
    ):
        parser.error("Use sizes 4..128 and steps 1000..8000.")
    if args.output.exists():
        parser.error("Refusing to overwrite evidence.")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    )
    records = []
    for size in args.sizes:
        records.append(run(size, args.steps))
        print(json.dumps(records[-1]), flush=True)
    result = {
        "schema": "agentfem.finite-plastic-manufactured.v1",
        "revision": revision,
        "tracked_source_dirty": dirty,
        "scope": "coaxial_nonuniform_tension_compression_with_yield_fronts",
        "body_force_quadrature": "nodal_lumped",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
