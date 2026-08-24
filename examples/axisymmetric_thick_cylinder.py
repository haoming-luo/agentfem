"""Pressurized thick cylinder through the native axisymmetric workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import constitutive, constraints, fields, mesh, models, studies


def main() -> None:
    inner_radius, outer_radius = 1.0, 2.0
    domain = mesh.rectangle(
        (inner_radius, 0.0),
        (outer_radius, 0.2),
        (8, 1),
        comm=MPI.COMM_WORLD,
        cell_type="quadrilateral",
    )
    study = studies.static_solid(dimension=2, assumption="axisymmetric")
    model = models.create(study=study, mesh=domain, name="thick_cylinder")
    displacement = model.field(fields.displacement(domain, degree=2))
    model.material(
        constitutive.isotropic_elastic(
            young=1000.0,
            poisson=0.3,
            density=1.0,
        )
    )

    # Long-cylinder plane strain is a specialization of axisymmetry, not its
    # general definition. It fixes U_z throughout the short meridian strip.
    model.constraint(constraints.axisymmetric_plane_strain(displacement))
    inner_wall = mesh.face(
        domain,
        axis="x",
        value=inner_radius,
        name="inner_wall",
        tag=1,
    )
    model.pressure(10.0, on=inner_wall)

    output = (
        Path(__file__).resolve().parents[1]
        / "examples_output"
        / "axisymmetric_thick_cylinder.xdmf"
    )
    result = model.step(target=displacement, output=output).solve_result()
    result.add_quantity(
        "maximum_displacement",
        float(np.max(np.abs(displacement.value.x.array))),
    )
    if domain.comm.rank == 0:
        result.write_manifest(output.with_suffix(".result.json"))
        print(result.format())


if __name__ == "__main__":
    main()
