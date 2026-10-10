# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Clamped shallow solid arch traced through two load limit points."""

import numpy as np
from agentfem import (
    mesh,
    models,
    studies,
    fields,
    constitutive,
    procedures,
    solvers,
    results,
    project,
)


def solve_arch(*, nx=32, arc_size=0.04, increments=80, output=None):
    domain = mesh.rectangle(
        (-1.0, -0.02), (1.0, 0.02), (nx, 2), cell_type="quadrilateral"
    )
    # Geometry mapping only; all equilibrium and continuation live in the library.
    domain.geometry.x[:, 1] += 0.2 * (1 - domain.geometry.x[:, 0] ** 2)
    model = models.create(
        study=studies.static_solid(
            dimension=2, assumption="plane_strain", nonlinear=True
        ),
        mesh=domain,
    )
    u = model.field(fields.displacement(domain, degree=2))
    model.material(constitutive.neo_hookean(young=1000.0, poisson=0.0))
    ends = mesh.boundary(
        domain, lambda x: np.isclose(np.abs(x[0]), 1.0), name="clamped_ends"
    )
    # Fixed patch endpoints x=+-0.1875 align for nx=32 and nx=64.
    crown = mesh.boundary(
        domain,
        lambda x: (
            (np.abs(x[0]) <= 0.1875 + 1.0e-10)
            & np.isclose(x[1] - 0.2 * (1 - x[0] ** 2), 0.02)
        ),
        name="crown_load",
    )
    model.clamp(u, on=ends)
    model.traction((0.0, -1.0), on=crown)
    step = model.step(
        target=u,
        procedure=procedures.arc_length(),
        increments=increments,
        arc_options=solvers.ArcLengthOptions(
            initial=arc_size, maximum=arc_size, displacement_scale=0.2, load_scale=0.1
        ),
    )
    result = step.solve_result(output=output)
    displacement = [
        float(results.probe(step.displacement_at(i), at=(0.0, 0.2))[1])
        for i in range(len(step.path.history))
    ]
    factors = [record["load_factor"] for record in step.path.history]
    result.add_history(
        "load_displacement",
        -np.asarray(displacement),
        factors,
        abscissa_name="downward_crown_displacement",
    )
    return result, np.asarray(displacement), np.asarray(factors)


def main():
    run = project.current_run()
    result, displacement, factors = solve_arch(output=run.artifact("arch.xdmf"))
    np.savetxt(
        run.artifact("load_displacement.csv"),
        np.column_stack((-displacement, factors)),
        delimiter=",",
        header="downward_crown_displacement,load_factor",
        comments="",
    )
    run.publish(result)
    return result


if __name__ == "__main__":
    main()
