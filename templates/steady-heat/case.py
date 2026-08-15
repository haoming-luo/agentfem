"""Steady heat conduction with convection to an ambient environment."""

from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import constitutive, fields, mesh, models, project, studies
from agentfem.diagnostics import print_on_root


def main():
    run = project.current_run(project_root=Path(__file__).resolve().parent)
    study = studies.steady_heat_transfer(dimension=2)
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (30, 6),
        comm=MPI.COMM_WORLD,
        cell_type="quadrilateral",
    )
    model = models.create(study=study, mesh=domain, name="heated_wall")
    temperature = model.field(fields.temperature(domain, value=300.0))
    model.material(
        constitutive.thermoelastic(
            name="steel",
            young=200.0e9,
            poisson=0.3,
            density=7800.0,
            thermal_expansion=12.0e-6,
            conductivity=45.0,
            specific_heat=500.0,
            reference_temperature=300.0,
        )
    )

    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="hot", tag=1)
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="cooled", tag=2)
    model.prescribed_temperature(temperature, 400.0, on=left)
    model.convection(on=right, coefficient=25.0, ambient_temperature=300.0)

    simulation = model.step(
        target=temperature,
        name="steady_heat",
        output=run.artifact("temperature.xdmf"),
    ).solve_result()
    simulation.add_dof_statistics(temperature, prefix="temperature", unit="K")
    if MPI.COMM_WORLD.rank == 0:
        run.publish(simulation)
        print_on_root(MPI.COMM_WORLD, simulation.format())
        print_on_root(MPI.COMM_WORLD, f"Result manifest: {run.manifest_path}")
    MPI.COMM_WORLD.barrier()
    return simulation


if __name__ == "__main__":
    main()
