"""Transient heat-conduction example using the AgentFEM workflow."""

from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np
from mpi4py import MPI

SOURCE_PARENT = Path(__file__).resolve().parents[2]
if (
    os.environ.get("AGENTFEM_INSTALLED_SMOKE") != "1"
    and str(SOURCE_PARENT) not in sys.path
):
    sys.path.insert(0, str(SOURCE_PARENT))

from agentfem import constitutive
from agentfem import fields
from agentfem import mesh as fem_mesh
from agentfem import models
from agentfem import studies
from agentfem.diagnostics import print_on_root
from agentfem.solvers import LinearSolverOptions


def main() -> None:
    comm = MPI.COMM_WORLD
    smoke = os.environ.get("AGENTFEM_RELEASE_SMOKE") == "1"
    study = studies.first_order_transient(
        physics="heat_transfer",
        dimension=2,
        name="transient_heat_conduction",
    )
    domain = fem_mesh.rectangle(
        lower=(0.0, 0.0),
        upper=(1.0, 0.4),
        cells=(8, 4) if smoke else (60, 24),
        comm=comm,
        cell_type="quadrilateral",
    )
    model = models.create(study=study, mesh=domain, name="heat_transfer_model")
    temperature = model.field(fields.temperature(domain, degree=1, value=300.0))
    model.material(
        constitutive.thermoelastic(
            name="generic steel",
            young=200.0e9,
            poisson=0.3,
            density=7800.0,
            thermal_expansion=12.0e-6,
            conductivity=45.0,
            specific_heat=500.0,
            reference_temperature=300.0,
        )
    )

    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], 1.0)

    left_boundary = fem_mesh.boundary(domain, left, name="left", tag=1)
    right_boundary = fem_mesh.boundary(domain, right, name="right", tag=2)
    model.fix(temperature, on=left_boundary, value=400.0)
    model.fix(temperature, on=right_boundary, value=300.0)
    model.check()

    step = model.step(
        target=temperature,
        dt=10.0,
        steps=5 if smoke else 1500,
        save_every=1 if smoke else 10,
        print_every=1 if smoke else 50,
        solver_options=LinearSolverOptions(ksp_type="preonly", pc_type="lu"),
        name="heat_implicit_euler",
    )

    out = Path(__file__).resolve().parents[1] / "examples_output" / "transient_heat_2d.xdmf"
    step.run(output=out)

    print_on_root(
        comm,
        f"Transient heat result: {out}; maxT={temperature.max_value():.3f}",
    )


if __name__ == "__main__":
    main()
