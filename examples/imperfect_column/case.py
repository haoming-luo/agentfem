# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Displacement-controlled imperfect clamped-guided column, using public platform functions."""

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
)


def solve_column(
    *, imperfection=0.02, nx=40, shortening=0.05, increment=0.05, output=None
):
    length = 20.0
    domain = mesh.rectangle(
        (0.0, -0.5), (length, 0.5), (nx, 4), cell_type="quadrilateral"
    )
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), tag=1, name="clamp")
    right = mesh.boundary(
        domain, lambda x: np.isclose(x[0], length), tag=2, name="loaded_tip"
    )
    left = mesh.tagged_boundary_region(domain, left.facet_tags, tag=1, name="clamp")
    right = mesh.tagged_boundary_region(
        domain, right.facet_tags, tag=2, name="loaded_tip"
    )
    linear = models.create(
        study=studies.buckling_solid(dimension=2, assumption="plane_strain"),
        mesh=domain,
    )
    mode_target = linear.field(fields.displacement(domain, degree=2))
    linear.material(
        constitutive.isotropic_elastic(young=1000.0, poisson=0.0, density=1.0)
    )
    linear.clamp(
        mode_target,
        on=mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="clamp"),
    )
    linear.fix(mode_target, on=right, component=0, value=0.0)
    reference = fem.Function(mode_target.space)
    reference.interpolate(lambda x: np.vstack((-x[0] / 1000.0, np.zeros_like(x[0]))))
    buckling = linear.step(
        target=mode_target,
        reference_displacement=reference,
        reference_name="unit compressive stress",
        modes=1,
    ).solve_result()
    receipt = mesh.apply_mode_imperfection(
        domain, [buckling.field("Buckling_mode_1")], amplitudes=[imperfection]
    )
    nonlinear = models.create(
        study=studies.static_solid(
            dimension=2, assumption="plane_strain", nonlinear=True
        ),
        mesh=domain,
    )
    u = nonlinear.field(fields.displacement(domain, degree=2))
    material = nonlinear.material(constitutive.neo_hookean(young=1000.0, poisson=0.0))
    nonlinear.clamp(u, on=left)
    nonlinear.prescribe(u, -shortening, on=right, component=0)
    step = nonlinear.step(
        target=u,
        material=material,
        incrementation=steps.automatic(
            initial=increment,
            maximum=increment,
            minimum=1.0e-5,
            max_increments=500,
            max_cutbacks=12,
        ),
        output_every=1,
        progress=False,
    )
    result = step.solve_result(output=output)
    curve = mechanics.displacement_controlled_response(nonlinear, step, on=right)
    result.add_history(
        "compression_force",
        -curve["displacement"],
        -curve["reaction"],
        abscissa_name="end_shortening",
    )
    result.add_history(
        "lateral_displacement",
        -curve["displacement"],
        curve["monitor_displacement"],
        abscissa_name="end_shortening",
    )
    result.metadata["imperfection"] = {
        "amplitude": imperfection,
        "quality": receipt.quality_report.minimum,
        "reference": "stress-free perturbed geometry",
    }
    result.add_quantity(
        "ideal_critical_stress", float(buckling.quantity("load_factors")[0])
    )
    return result, curve


def main():
    run = project.current_run()
    result, curve = solve_column(output=run.artifact("displacement.xdmf"))
    if result.field("Displacement").function_space.mesh.comm.rank == 0:
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
