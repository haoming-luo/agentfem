# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent Fourier-mode convergence oracle for the public coupled Step.

On a unit box with zero normal displacement and insulated faces, let
theta=cos(w*x)*exp(-gamma*t), u_x=beta/(H*w)*sin(w*x)*exp(-gamma*t),
u_y=u_z=0, w=2*pi, H=lambda+2*mu. Equilibrium gives H*u_x'=beta*theta;
the heat equation gives gamma=k*w*w/(C+beta*beta*T0/H).
There is no applied source. This is a 1D mode embedded in 3D, not a test of
arbitrary multidimensional singular geometries or a general first law.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from agentfem import constitutive, fields, mesh, models, studies, units


def run(cells, steps, *, final_time=1.0):
    if type(cells) is not int or cells < 2 or type(steps) is not int or steps < 1:
        raise ValueError(
            "Positive integer steps and at least two axial cells required."
        )
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (cells, 1, 1),
        comm=MPI.COMM_WORLD,
        cell_type="hexahedron",
    )
    model = models.create(
        study=studies.coupled_thermoelastic(), mesh=domain, units=units.si()
    )
    u = model.field(fields.displacement(domain))
    theta = model.field(fields.temperature(domain, value=0.0))
    E, nu, alpha, T0, capacity, conductivity = 1000.0, 0.25, 0.002, 300.0, 100.0, 1.0
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    H, beta, wave = lam + 2 * mu, (3 * lam + 2 * mu) * alpha, 2 * np.pi
    rate = conductivity * wave**2 / (capacity + beta**2 * T0 / H)
    model.material(
        constitutive.thermoelastic(
            young=E,
            poisson=nu,
            density=1.0,
            specific_heat=capacity,
            conductivity=conductivity,
            thermal_expansion=alpha,
            reference_temperature=T0,
        )
    )
    theta.value.interpolate(lambda x: np.cos(wave * x[0]))
    u.value.interpolate(
        lambda x: np.vstack(
            (
                beta / (H * wave) * np.sin(wave * x[0]),
                np.zeros_like(x[0]),
                np.zeros_like(x[0]),
            )
        )
    )
    u.value.x.scatter_forward()
    theta.value.x.scatter_forward()
    for axis in range(3):
        face = mesh.boundary(
            domain,
            lambda x, axis=axis: np.isclose(x[axis], 0.0) | np.isclose(x[axis], 1.0),
            name=f"normal_{axis}",
            tag=axis + 1,
        )
        model.fix(u, on=face, component=axis, value=0.0)
    with model.step(
        target=u,
        temperature_departure=theta,
        dt=final_time / steps,
        steps=steps,
        rtol=1e-11,
        displacement_atol=1e-15,
        temperature_atol=1e-13,
    ) as step:
        result = step.solve_result()
        x = ufl.SpatialCoordinate(domain)
        decay = float(np.exp(-rate * final_time))
        exact_theta = decay * ufl.cos(wave * x[0])
        exact_u = ufl.as_vector(
            (decay * beta / (H * wave) * ufl.sin(wave * x[0]), 0.0, 0.0)
        )
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 10})

        def norm(expr):
            return np.sqrt(
                domain.comm.allreduce(
                    fem.assemble_scalar(fem.form(ufl.inner(expr, expr) * dx)),
                    op=MPI.SUM,
                )
            )

        return {
            "cells": cells,
            "transverse_cells": [1, 1],
            "steps": steps,
            "dt": final_time / steps,
            "final_time": final_time,
            "temperature_relative_l2": float(
                norm(theta.value - exact_theta) / norm(exact_theta)
            ),
            "displacement_relative_l2": float(norm(u.value - exact_u) / norm(exact_u)),
            "matrix_assemblies": result.metadata["matrix_assemblies"],
            "performance": result.performance,
            "max_quadratic_residual": max(
                r["quadratic_balance_absolute"] for r in step.history
            ),
        }


def study():
    spatial = [run(n, 128) for n in (8, 16, 32)]
    temporal = [run(128, n) for n in (4, 8, 16)]
    orders = {}
    for axis, records in (("space", spatial), ("time", temporal)):
        for field in ("temperature", "displacement"):
            errors = [r[field + "_relative_l2"] for r in records]
            orders[axis + "_" + field] = [
                float(np.log2(a / b)) for a, b in zip(errors, errors[1:])
            ]
    accepted = all(
        min(v) > (1.7 if k.startswith("space") else 0.8) for k, v in orders.items()
    )
    return {
        "schema": "agentfem.coupled-decay-convergence.v1",
        "spatial": spatial,
        "temporal": temporal,
        "observed_orders": orders,
        "accepted": accepted,
        "scope": "analytical one-dimensional Fourier mode embedded in 3D; no general validation",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = study()
    if MPI.COMM_WORLD.rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {"accepted": report["accepted"], "orders": report["observed_orders"]}
            )
        )
    if not report["accepted"]:
        raise SystemExit(1)
