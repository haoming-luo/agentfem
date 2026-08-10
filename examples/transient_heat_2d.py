"""Transient heat-conduction example using the AgentFEM workflow."""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import ufl
from mpi4py import MPI

from agentfem import benchmarks
from agentfem import constitutive
from agentfem import fields
from agentfem import mesh as fem_mesh
from agentfem import models
from agentfem import results
from agentfem import studies
from agentfem.diagnostics import print_on_root
from agentfem.solvers import LinearSolverOptions


def main() -> dict[str, float]:
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
    simulation = step.solve_result(
        output=out,
        history=(
            results.probe_history(
                "center_temperature",
                at=(0.5, 0.2),
                unit="K",
                description="Temperature at the plate center.",
            ),
        ),
    )
    observables = {
        "final_mean_temperature": results.average(
            temperature.value,
            measure=ufl.Measure("dx", domain=domain),
        ),
    }
    if smoke:
        golden = benchmarks.golden_benchmark(
            "agentfem.benchmark.transient_heat_release"
        )
    simulation.add_quantity(
        "final_mean_temperature",
        observables["final_mean_temperature"],
        unit="K",
    )
    quality = simulation.verify(
        "release" if smoke else "engineering",
        claims=golden.claims(observables) if smoke else (),
        required_quantities=("final_mean_temperature",),
        required_artifacts=("fields_xdmf",),
    )
    quality.require()
    if comm.rank == 0:
        simulation.write_manifest(
            out.with_suffix(".result.json"),
            include_histories=True,
        )

    print_on_root(
        comm,
        f"Transient heat result: {out}; maxT={temperature.max_value():.3f}",
    )
    print_on_root(
        comm,
        "Transient heat observable: "
        f"meanT={observables['final_mean_temperature']:.16e}",
    )
    print_on_root(comm, simulation.format())
    return observables


if __name__ == "__main__":
    main()
