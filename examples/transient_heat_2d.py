"""Transient heat-conduction example using the AgentFEM workflow."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import ufl
from mpi4py import MPI

SOURCE_PARENT = Path(__file__).resolve().parents[2]
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from agentfem import constraints
from agentfem import fields
from agentfem import io as fem_io
from agentfem import mesh as fem_mesh
from agentfem import operators
from agentfem import time as fem_time
from agentfem.problems import LinearSystemProblem
from agentfem.solvers import LinearSolverOptions


def main() -> None:
    comm = MPI.COMM_WORLD
    domain = fem_mesh.rectangle(
        lower=(0.0, 0.0),
        upper=(1.0, 0.4),
        cells=(60, 24),
        comm=comm,
        cell_type="quadrilateral",
    )

    temperature = fields.temperature(domain, degree=1, value=300.0)
    previous_temperature = fields.scalar_unknown(
        domain,
        name="TemperaturePrevious",
        degree=1,
        value=300.0,
    )

    rho_cp = 3.9e6
    conductivity = 45.0
    heat_source = 0.0

    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], 1.0)

    left_boundary = fem_mesh.boundary(domain, left, name="left", tag=1)
    right_boundary = fem_mesh.boundary(domain, right, name="right", tag=2)
    hot_boundary = constraints.fixed(
        temperature,
        location=left_boundary,
        value=400.0,
    )
    cold_boundary = constraints.fixed(
        temperature,
        location=right_boundary,
        value=300.0,
    )
    bcs = hot_boundary.bcs + cold_boundary.bcs

    dt = 10
    total_steps = 1500
    dx = ufl.dx(domain=domain)
    stepper = fem_time.TimeStepper(
        total_steps=total_steps,
        dt=dt,
        save_every=10,
        print_every=50,
    )

    M_over_dt = operators.mass_operator(temperature, rho_cp / dt, measure=dx)
    K = operators.diffusion_operator(temperature, conductivity, measure=dx)
    F_source = operators.body_force_vector(heat_source, temperature, measure=dx)

    effective_K = M_over_dt.expression + K.expression
    effective_F = (
        operators.mass_operator(
            previous_temperature.value,
            temperature,
            rho_cp / dt,
            measure=dx,
        ).expression
        + F_source.expression
    )
    system = operators.LinearSystem(
        stiffness=effective_K,
        force=effective_F,
        name="implicit_euler_heat_Keff_T_eq_Feff",
    )

    problem = LinearSystemProblem(
        system=system,
        unknown=temperature,
        bcs=bcs,
        solver_options=LinearSolverOptions(ksp_type="preonly", pc_type="lu"),
    )

    out = Path(__file__).resolve().parents[1] / "examples_output" / "transient_heat_2d.xdmf"
    with fem_io.XDMFTimeSeries(out, domain) as xdmf:
        xdmf.write_fields(0.0, temperature.value)
        for info in stepper:
            problem.solve()
            previous_temperature.assign_from(temperature)
            if info.should_save:
                xdmf.write_fields(info.time, temperature.value)
            if info.should_print and comm.rank == 0:
                print(
                    f"step {info.index:4d}/{total_steps} "
                    f"t={info.time:.3e} maxT={temperature.max_value():.3f}",
                    flush=True,
                )

    if comm.rank == 0:
        print(f"Transient heat result: {out}")


if __name__ == "__main__":
    main()
