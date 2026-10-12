# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Stress-free imperfect cantilever, N/mm/MPa, axial reference dead traction."""

import numpy as np
from dolfinx import fem
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


def solve_column(
    *,
    nx=80,
    ny=4,
    imperfection=0.1,
    increment=0.025,
    total_load=10700.0,
    output=None,
    solver_options=None,
):
    length, height, thickness, young = 200.0, 10.0, 10.0, 210000.0
    domain = mesh.rectangle(
        (0.0, -height / 2), (length, height / 2), (nx, ny), cell_type="quadrilateral"
    )
    # Material end faces are tagged BEFORE perturbing the stress-free mesh.
    ends = []
    for axis_position, tag, name in [(0.0, 1, "clamp"), (length, 2, "free_tip")]:
        boundary = mesh.boundary(
            domain, lambda x, a=axis_position: np.isclose(x[0], a), tag=tag, name=name
        )
        ends.append(
            mesh.tagged_boundary_region(domain, boundary.facet_tags, tag=tag, name=name)
        )
    left, right = ends
    mode = fem.Function(fields.displacement(domain, degree=2).space)
    # First Euler cantilever mode; transverse amplitude is exactly imperfection.
    mode.interpolate(
        lambda x: np.vstack(
            (np.zeros_like(x[0]), 1 - np.cos(np.pi * x[0] / (2 * length)))
        )
    )
    receipt = mesh.apply_mode_imperfection(domain, [mode], amplitudes=[imperfection])
    model = models.create(
        study=studies.static_solid(
            dimension=2, assumption="plane_stress", nonlinear=True
        ),
        mesh=domain,
    )
    u = model.field(fields.displacement(domain, degree=2))
    model.material(
        constitutive.saint_venant_kirchhoff(young=young, poisson=0.3, plane_stress=True)
    )
    model.clamp(u, on=left)
    model.traction((-total_load / (height * thickness), 0.0), on=right)
    # For this smooth elastic problem use full Newton, with load-step cutback.
    # A dimensional absolute tolerance of 1e-7 N/mm is below the reaction budget.
    if solver_options is None:
        solver_options = solvers.NewtonSolverOptions(
            line_search="basic", absolute_tolerance=1e-7
        )
    step = model.step(
        target=u,
        solver_options=solver_options,
        incrementation=steps.automatic(
            initial=increment,
            maximum=increment,
            minimum=1e-6,
            max_increments=2000,
            max_cutbacks=15,
        ),
        output_every=1,
        progress=False,
    )
    result = step.solve_result(output=output)
    response = mechanics.load_controlled_response(
        model, step, support=left, monitor=right
    )
    curve = dict(
        load_N=response["load_factor"] * total_load,
        added_tip_displacement_mm=response["monitor_displacement"],
        support_reaction_N=response["reaction"] * thickness,
    )
    for name in ("added_tip_displacement_mm", "support_reaction_N"):
        result.add_history(name, curve["load_N"], curve[name], abscissa_name="load_N")
    result.metadata["geometry"] = dict(
        length_mm=length,
        height_mm=height,
        thickness_mm=thickness,
        initial_tip_offset_mm=imperfection,
        minimum_mesh_quality=receipt.quality_report.minimum,
    )
    result.metadata["material"] = model.materials[0].item.as_dict()
    result.add_quantity(
        "Euler_load_N",
        np.pi**2 * young * (thickness * height**3 / 12) / (4 * length**2),
    )
    return result, curve, step


def main():
    run = project.current_run()
    result, curve, step = solve_column(output=run.artifact("displacement.xdmf"))
    if step.solution.function_space.mesh.comm.rank == 0:
        np.savetxt(
            run.artifact("load_displacement.csv"),
            np.column_stack(list(curve.values())),
            delimiter=",",
            header=",".join(curve),
            comments="",
        )
        run.publish(result)
    return result


if __name__ == "__main__":
    main()
