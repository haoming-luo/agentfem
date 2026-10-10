# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Periodic effective properties versus a finite honeycomb's apparent response."""

from pathlib import Path
import numpy as np
from mpi4py import MPI
from agentfem import (
    constraints,
    fields,
    mechanics,
    mesh,
    models,
    project,
    results,
    studies,
)
from agentfem.constitutive import isotropic_elastic


def main():
    run = project.current_run(project_root=Path(__file__).resolve().parent)
    data = mesh.reentrant_honeycomb(mesh_size=0.06)
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_stress"),
        mesh=data.domain,
    )
    w = model.field(fields.displacement(data.domain, degree=2))
    material = isotropic_elastic(young=1000.0, poisson=0.3, density=1.0)
    model.material(material)
    effective = mechanics.periodic_elasticity(model, w, anchor=(0.0, 0.0))
    # Same geometry generator, ordinary static solver; only boundary physics changes.
    data = mesh.reentrant_honeycomb(
        repeats=(3, 3), mesh_size=0.08, connected_specimen=True
    )
    domain = data.domain
    width = 3 * np.sqrt(3.0)
    height = 6.0
    finite = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_stress"), mesh=domain
    )
    u = finite.field(fields.displacement(domain, degree=2))
    finite.material(material)
    left = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="left_gauge", tag=1
    )
    right = mesh.boundary(
        domain, lambda x: np.isclose(x[0], width), name="right_gauge", tag=2
    )
    bottom = mesh.boundary(
        domain, lambda x: np.isclose(x[1], 0.0), name="bottom_gauge", tag=3
    )
    top = mesh.boundary(
        domain, lambda x: np.isclose(x[1], height), name="top_gauge", tag=4
    )
    finite.constraint(constraints.fixed_component(u, component=0, on=left, value=0.0))
    finite.constraint(
        constraints.fixed_component(u, component=0, on=right, value=0.001 * width)
    )
    finite.constraint(constraints.pin(u, at=(0.0, 0.0), components=(1,)))
    step = finite.step(target=u)
    result = step.solve_result(
        output=run.artifact("finite_array.xdmf"), strict_output=True
    )
    apparent = mechanics.apparent_poisson_ratio(
        u,
        axial_axis=0,
        transverse_axis=1,
        axial_faces=(left, right),
        transverse_faces=(bottom, top),
        axial_length=width,
        transverse_length=height,
    )
    result.add_quantity("apparent_poisson_ratio", apparent["apparent_poisson_ratio"])
    result.add_quantity(
        "right_reaction",
        results.reaction_resultant(step.problem, on=right, component=0),
    )
    result.metadata["gauge_measurement"] = apparent
    if MPI.COMM_WORLD.rank == 0:
        effective.write_manifest(run.artifact("periodic_properties.json"))
        run.publish(result)
        print("Periodic nu_xy:", effective.quantity("poisson_ratios")[0, 1])
        print("Finite-array apparent nu_xy:", apparent["apparent_poisson_ratio"])
    return result


if __name__ == "__main__":
    main()
