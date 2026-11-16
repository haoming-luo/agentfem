# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Plane longitudinal/shear half-sine waves in a constrained 3D prism.

Orthogonal displacement components are fixed throughout the prism, admitting
an analytical scalar bulk wave. Uniform end traction is required for that
reference; this is explicitly NOT the blade's point-force boundary condition.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from mpi4py import MPI
from agentfem import (
    mesh,
    models,
    studies,
    fields,
    elements,
    constitutive,
    constraints,
    amplitudes,
    results,
)

E, nu, rho = 210e9, 0.3, 7800.0
mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
tau = 25e-6
traction = 1230 / 0.01
out = Path(__file__).parent / "evidence" / "half_sine_waves"
out.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--meshes", nargs="+", type=int, default=[16, 32, 64])
args = parser.parse_args()
if any(n < 16 or n % 16 for n in args.meshes):
    parser.error("mesh counts must be positive multiples of 16")
records = (
    json.loads((out / "convergence.json").read_text())
    if (out / "convergence.json").exists()
    else []
)
for mode, component, speed in [
    ("longitudinal", 0, np.sqrt((lam + 2 * mu) / rho)),
    ("shear", 1, np.sqrt(mu / rho)),
]:
    arrival = 0.5 / speed
    duration = arrival + tau + 20e-6
    for nx in args.meshes:
        for fraction in (0.8, 0.4):
            domain = mesh.cuboid(
                (0, 0, 0),
                (1, 0.1, 0.1),
                (nx, nx // 16, nx // 16),
                comm=MPI.COMM_SELF,
                cell_type="tetrahedron",
            )
            model = models.create(
                study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=domain
            )
            u = model.field(
                fields.displacement(domain, element=elements.tet15_mass_lumped())
            )
            model.material(
                constitutive.isotropic_elastic(young=E, poisson=nu, density=rho)
            )
            for other in set(range(3)) - {component}:
                model.constraint(
                    constraints.DirichletConstraint.component(
                        u.space, other, lambda x: np.ones(x.shape[1], dtype=bool)
                    )
                )
            load = np.zeros(3)
            load[component] = traction
            model.traction(
                load,
                on=mesh.boundary(
                    domain, lambda x: np.isclose(x[0], 0), tag=1, name="loaded_end"
                ),
                amplitude=amplitudes.half_sine(tau),
            )
            step = model.step(
                target=u,
                mass=model.lumped_mass(u, method="tet15_nodal"),
                dt="auto",
                steps=1,
                progress=False,
            )
            step.steps = int(np.ceil(duration / (step.dt * fraction)))
            step.dt = duration / step.steps
            history = results.dynamic_history(
                "sensor",
                at=(0.5, 0.05, 0.05),
                variables=("V",),
                length_unit="m",
                time_unit="s",
            )
            step.solve_result(history=history)
            t = np.array([r["time"] for r in step.history_records])
            v = np.array([r[f"sensor.V{component + 1}"] for r in step.history_records])
            phase = (t - arrival) / tau
            exact = np.where(
                (phase > 0) & (phase < 1),
                traction / (rho * speed) * np.sin(np.pi * phase),
                0,
            )
            threshold = 0.01 * traction / (rho * speed)
            hits = np.flatnonzero(abs(v) >= threshold)
            measured = None if not len(hits) else float(t[hits[0]])
            expected = arrival + tau / np.pi * np.arcsin(0.01)
            row = {
                "mode": mode,
                "nx": nx,
                "dt_fraction": fraction,
                "dt": step.dt,
                "steps": step.steps,
                "analytic_speed": float(speed),
                "arrival_threshold": float(threshold),
                "first_threshold_time": measured,
                "expected_threshold_time": float(expected),
                "relative_arrival_error": None
                if measured is None
                else abs(measured / expected - 1),
                "relative_peak_error": float(abs(max(abs(v)) / max(exact) - 1)),
                "relative_waveform_L2": float(
                    np.sqrt(
                        np.trapezoid((v - exact) ** 2, t) / np.trapezoid(exact**2, t)
                    )
                ),
            }
            records = [
                r
                for r in records
                if (r["mode"], r["nx"], r["dt_fraction"]) != (mode, nx, fraction)
            ]
            records.append(row)
            print(json.dumps(row), flush=True)
            np.savetxt(
                out / f"{mode}_{nx}_{fraction}.csv",
                np.column_stack((t, v, exact)),
                delimiter=",",
                header="time,velocity,analytic_velocity",
                comments="",
            )
            (out / "convergence.json").write_text(json.dumps(records, indent=2) + "\n")
