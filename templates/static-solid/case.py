"""Minimal cantilever-like linear-static AgentFEM project."""

from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import fields, io, mesh, models, project, results, studies
from agentfem.constitutive import elasticity
from agentfem.diagnostics import print_on_root


def main():
    run = project.current_run(project_root=Path(__file__).resolve().parent)
    study = studies.static_solid(
        dimension=2,
        assumption="plane_strain",
        name="cantilever_static",
    )
    domain = mesh.rectangle(
        lower=(0.0, 0.0),
        upper=(1.0, 0.2),
        cells=(20, 4),
        comm=MPI.COMM_WORLD,
        cell_type="quadrilateral",
    )
    model = models.create(study=study, mesh=domain, name="cantilever")
    displacement = model.field(fields.displacement(domain, degree=1))
    model.material(
        elasticity.isotropic_elastic(
            young=210.0e9,
            poisson=0.30,
            density=7800.0,
            name="steel",
        )
    )

    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left", tag=1)
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right", tag=2)
    model.clamp(displacement, on=left)
    model.traction((0.0, -1.0e6), on=right)
    model.check()

    simulation = model.step(target=displacement, name="static_load").solve_result()
    simulation.add_dof_statistics(displacement, prefix="displacement", unit="m")
    field_path = run.artifact("fields.xdmf")
    output_displacement = io.interpolate_for_xdmf(displacement.value, degree=1)
    with io.XDMFTimeSeries(field_path, domain) as writer:
        writer.write_fields(0.0, output_displacement)
    simulation.add_artifact("fields", field_path)
    if MPI.COMM_WORLD.rank == 0:
        run.publish(simulation)
        print_on_root(MPI.COMM_WORLD, simulation.format())
        print_on_root(MPI.COMM_WORLD, f"Result manifest: {run.manifest_path}")
    MPI.COMM_WORLD.barrier()
    return simulation


if __name__ == "__main__":
    main()
