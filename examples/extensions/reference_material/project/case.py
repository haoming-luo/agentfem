# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Installed external material through the ordinary AgentFEM lifecycle."""

from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import benchmarks, fields, materials, mesh, models, project, studies
from agentfem.diagnostics import max_magnitude


def main():
    run = project.current_run(project_root=Path(__file__).resolve().parent)
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (40, 8),
        comm=MPI.COMM_WORLD,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain"),
        mesh=domain,
        name="external_material_cantilever",
    )
    displacement = model.field(fields.displacement(domain, degree=2))
    material = materials.load("agentfem_reference_alloy")
    model.material(material)
    model.clamp(
        displacement,
        on=mesh.boundary(
            domain, lambda x: np.isclose(x[0], 0.0), name="left", tag=1
        ),
    )
    model.traction(
        (1.0e8, -1.0e6),
        on=mesh.boundary(
            domain, lambda x: np.isclose(x[0], 1.0), name="right", tag=2
        ),
    )
    model.check()
    result = model.step(
        target=displacement,
        name="external_material_static",
        output=run.artifact("fields.xdmf"),
    ).solve_result()
    maximum = max_magnitude(displacement.value)
    result.add_quantity("maximum_displacement", maximum, unit="m")
    result.metadata["external_material"] = material.summary()
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.linear_static_cantilever"
    )
    result.verify(
        "release",
        claims=golden.claims({"maximum_displacement": maximum}),
        required_quantities=("maximum_displacement",),
        required_artifacts=("fields_xdmf",),
    ).require()
    if MPI.COMM_WORLD.rank == 0:
        run.publish(result)
    MPI.COMM_WORLD.barrier()
    return result


if __name__ == "__main__":
    main()
