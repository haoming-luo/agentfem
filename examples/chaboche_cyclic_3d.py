"""Small three-dimensional combined-hardening cycle through the public API."""

from __future__ import annotations

import numpy as np

from agentfem import (
    amplitudes,
    constitutive,
    fields,
    mesh,
    models,
    solvers,
    steps,
    studies,
)


domain = mesh.cuboid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1))
model = models.create(
    study=studies.static_solid(dimension=3, nonlinear=True),
    mesh=domain,
    name="chaboche_cyclic_bar",
)
u = model.field(fields.displacement(domain))
steel = model.material(
    constitutive.chaboche(
        young=200.0e3,
        poisson=0.3,
        yield_stress=200.0,
        backstresses=((22.22e3, 34.65), (8.0e3, 5.0)),
        isotropic_saturation=2000.0,
        isotropic_rate=0.25,
    )
)

model.fix(u, on=mesh.face(domain, axis="x", value=0.0), component=0, value=0.0)
model.fix(u, on=mesh.face(domain, axis="y", value=0.0), component=1, value=0.0)
model.fix(u, on=mesh.face(domain, axis="z", value=0.0), component=2, value=0.0)
model.fix(u, on=mesh.face(domain, axis="x", value=1.0), component=0, value=0.005)

cycle = amplitudes.tabular(
    (0.0, 0.25, 0.5, 0.75, 1.0),
    (0.0, 1.0, 0.0, -1.0, 0.0),
    name="fully_reversed_displacement",
)
step = model.step(
    target=u,
    material=steel,
    amplitude=cycle,
    incrementation=steps.fixed(4),
    solver_options=solvers.newton(
        relative_tolerance=1.0e-9,
        absolute_tolerance=1.0e-10,
        maximum_iterations=20,
        line_search="backtracking",
    ),
)
result = step.solve_result(output="results/chaboche_cyclic_3d.xdmf")

print(
    {
        "completed": step.last_solve_info.completed_step,
        "maximum_peeq": result.quantity("maximum_equivalent_plastic_strain"),
        "maximum_backstress": float(np.max(np.abs(step.state.backstresses.values))),
        "fields": tuple(result.fields),
    }
)
