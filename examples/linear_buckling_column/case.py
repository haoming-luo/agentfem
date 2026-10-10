# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Elastic column: static reference solve followed by initial-stress buckling."""

from pathlib import Path
import numpy as np
from mpi4py import MPI
from agentfem import fields, mesh, models, project, studies
from agentfem.constitutive import isotropic_elastic


def main():
    run = project.current_run(project_root=Path(__file__).resolve().parent)
    length, height, young = 20.0, 1.0, 1000.0
    domain = mesh.rectangle(
        (0.0, -0.5), (length, 0.5), (60, 4), cell_type="quadrilateral"
    )
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="clamp", tag=1)
    right = mesh.boundary(
        domain, lambda x: np.isclose(x[0], length), name="compression", tag=2
    )
    material = isotropic_elastic(young=young, poisson=0.0, density=1.0)
    reference = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_stress"), mesh=domain
    )
    u0 = reference.field(fields.displacement(domain, degree=2))
    reference.material(material)
    reference.clamp(u0, on=left)
    reference.traction((-1.0, 0.0), on=right)
    reference.step(target=u0).solve_result()
    model = models.create(
        study=studies.buckling_solid(dimension=2, assumption="plane_stress"),
        mesh=domain,
    )
    u = model.field(fields.displacement(domain, degree=2))
    model.material(material)
    model.clamp(u, on=left)
    result = model.step(
        target=u,
        reference_displacement=u0,
        reference_name="uniform compression: unit stress",
        modes=3,
    ).solve_result(output=run.artifact("buckling.xdmf"), strict_output=True)
    expected = np.pi**2 * young * height**2 / (48.0 * length**2)
    result.add_quantity("euler_reference_stress", expected)
    result.add_quantity(
        "euler_relative_error", abs(result.quantity("load_factors")[0] / expected - 1)
    )
    if MPI.COMM_WORLD.rank == 0:
        run.publish(result)
        print(result.format())
    return result


if __name__ == "__main__":
    main()
