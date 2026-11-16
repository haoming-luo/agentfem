# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Minimal standard project for the composite Tet10 explicit formulation.

An integration/teaching fixture, not an application prediction. Consistent SI
units. The formulation is explicitly named; it is not an Abaqus C3D10M alias.
"""

from pathlib import Path
import numpy as np
from mpi4py import MPI
from agentfem import (
    mesh,
    studies,
    models,
    fields,
    constitutive,
    elements,
    loads,
    results,
    project,
)


def main():
    run = project.current_run(project_root=Path(__file__).parent)
    domain = mesh.cuboid(
        (0, 0, 0), (1, 1, 1), (1, 1, 1), comm=MPI.COMM_WORLD, cell_type="tetrahedron"
    )
    model = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=domain
    )
    u = model.field(
        fields.displacement(
            domain, element_policy=elements.composite_tet10_policy(mean_dilatation=True)
        )
    )
    model.material(constitutive.isotropic_elastic(young=100, poisson=0.25, density=2))
    left = mesh.boundary_region(domain, lambda x: np.isclose(x[0], 0), name="fixed")
    right = mesh.boundary_region(domain, lambda x: np.isclose(x[0], 1), name="loaded")
    model.clamp(u, on=left)
    model.load(loads.traction((1, 0, 0), on=right))
    step = model.step(target=u, dt="auto", steps=5, save_every=1, progress=False)
    result = step.solve_result(
        output=run.output_directory / "fields.xdmf",
        field_variables=("S", "E", "MISES", "SENER"),
        history=results.dynamic_history(
            "station", at=(1, 0.5, 0.5), length_unit="m", time_unit="s"
        ),
    )
    if domain.comm.rank == 0:
        results.write_dynamic_history(result, run.output_directory / "history.csv")
    run.publish(result)
    return result


if __name__ == "__main__":
    main()
