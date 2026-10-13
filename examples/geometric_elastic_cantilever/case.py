# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Finite-rotation thin cantilever under a fixed-direction transverse load."""

import numpy as np
from agentfem import (
    mesh,
    models,
    studies,
    fields,
    constitutive,
    mechanics,
    steps,
    project,
    solvers,
)


def solve_cantilever(*, nx=80, ny=2, alpha=2.0, increment=0.05, output=None):
    length, height, young = 100.0, 1.0, 1e6
    force_per_thickness = alpha * young * height**3 / (12 * length**2)
    d = mesh.rectangle((0.0, -0.5), (length, 0.5), (nx, ny), cell_type="quadrilateral")
    m = models.create(
        mesh=d,
        study=studies.static_solid(
            dimension=2, assumption="plane_stress", nonlinear=True
        ),
    )
    u = m.field(fields.displacement(d, degree=2))
    m.material(
        constitutive.saint_venant_kirchhoff(young=young, poisson=0.0, plane_stress=True)
    )
    left = mesh.boundary(d, lambda x: np.isclose(x[0], 0.0), name="clamp")
    right = mesh.boundary(d, lambda x: np.isclose(x[0], length), name="free_end")
    m.clamp(u, on=left)
    m.traction((0.0, force_per_thickness / height), on=right)
    step = m.step(
        target=u,
        solver_options=solvers.NewtonSolverOptions(
            line_search="basic", absolute_tolerance=1e-7
        ),
        incrementation=steps.automatic(
            initial=increment, maximum=increment, minimum=1e-6, max_increments=2000
        ),
        progress=False,
    )
    r = step.solve_result(output=output)
    transverse = mechanics.load_controlled_response(
        m, step, support=left, monitor=right, component=1, monitor_component=1
    )
    axial = mechanics.load_controlled_response(
        m, step, support=left, monitor=right, component=1, monitor_component=0
    )
    curve = dict(
        alpha=transverse["load_factor"] * alpha,
        tip_x_over_L=1 + axial["monitor_displacement"] / length,
        tip_y_over_L=transverse["monitor_displacement"] / length,
        support_reaction_per_thickness=transverse["reaction"],
    )
    r.add_history(
        "tip_y_over_L",
        curve["alpha"],
        curve["tip_y_over_L"],
        abscissa_name="dimensionless_load",
    )
    return r, curve


def main():
    run = project.current_run()
    r, c = solve_cantilever(output=run.artifact("displacement.xdmf"))
    np.savetxt(
        run.artifact("load_displacement.csv"),
        np.column_stack(list(c.values())),
        delimiter=",",
        header=",".join(c),
        comments="",
    )
    run.publish(r)
    return r


if __name__ == "__main__":
    main()
