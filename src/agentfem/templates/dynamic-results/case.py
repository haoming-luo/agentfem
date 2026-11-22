# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Small implicit structural-dynamics AgentFEM project."""

import os
import json
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import constitutive, fields, mesh, models, project, provenance, results, studies, units


def main(*, review=False):
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
    model = models.create(study=study, mesh=domain, name="dynamic_beam", units=units.si())
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
    options = dict(dt=1.0e-4, steps=6, save_every=3, print_every=1, name="newmark_response")
    if review:
        card = model.review(target=displacement, step_options=options, output_fields=(displacement,),
            output_directory=Path(__file__).parent/"outputs")
        if domain.comm.rank == 0:
            print(json.dumps(card, ensure_ascii=False, allow_nan=False))
        return card
    run = project.current_run(project_root=Path(__file__).resolve().parent)
    settings = provenance.model_settings(model, target=displacement, step_options=options, source_files=(Path(__file__),))
    step = model.step(target=displacement, output=run.artifact("dynamics.xdmf"), **options)
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
    simulation.metadata["model_settings"] = settings
    simulation.add_dof_statistics(
        step.state.u,
        prefix="final_displacement",
        unit="m",
    )
    run.publish(simulation, comm=MPI.COMM_WORLD)
    if MPI.COMM_WORLD.rank == 0:
        if not os.environ.get("AGENTFEM_CLI_MANAGED"):
            print(simulation.format())
            print(f"Result manifest: {run.manifest_path}")
    return simulation


if __name__ == "__main__":
    main(review="--review" in sys.argv[1:])
