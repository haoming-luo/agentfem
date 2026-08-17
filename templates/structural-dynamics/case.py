"""Small implicit structural-dynamics AgentFEM project."""

from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import constitutive, fields, mesh, models, project, results, studies


def main():
    run = project.current_run(project_root=Path(__file__).resolve().parent)
    study = studies.dynamic_solid(
        dimension=2,
        assumption="plane_stress",
        method="newmark",
        name="beam_dynamics",
    )
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_WORLD,
        cell_type="triangle",
    )
    model = models.create(study=study, mesh=domain, name="dynamic_beam")
    displacement = model.field(fields.displacement(domain))
    model.material(
        constitutive.isotropic_elastic(
            young=2.0e9,
            poisson=0.3,
            density=1000.0,
        )
    )
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left", tag=1)
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right", tag=2)
    model.clamp(displacement, on=left)
    model.traction((0.0, -1.0e4), on=right)

    step = model.step(
        target=displacement,
        dt=1.0e-4,
        steps=3,
        print_every=1,
        name="newmark_response",
        output=run.artifact("dynamics.xdmf"),
    )
    simulation = step.solve_result(
        history=(
            results.probe_history(
                "tip_U2",
                at=(1.0, 0.1),
                component=1,
                unit="m",
            ),
        ),
    )
    simulation.add_dof_statistics(
        step.state.u,
        prefix="final_displacement",
        unit="m",
    )
    if MPI.COMM_WORLD.rank == 0:
        run.publish(simulation)
        print(simulation.format())
        print(f"Result manifest: {run.manifest_path}")
    MPI.COMM_WORLD.barrier()
    return simulation


if __name__ == "__main__":
    main()
