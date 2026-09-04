"""First modes of a clamped plane-stress cantilever."""

from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import fields, mesh, models, studies
from agentfem.constitutive import isotropic_elastic


domain = mesh.rectangle(
    (0.0, 0.0),
    (1.0, 0.2),
    (24, 4),
    cell_type="quadrilateral",
)
model = models.create(
    study=studies.modal_solid(dimension=2, assumption="plane_stress"),
    mesh=domain,
    name="cantilever_modes",
)
u = model.field(fields.displacement(domain, degree=2))
model.material(
    isotropic_elastic(
        young=210.0e9,
        poisson=0.3,
        density=7800.0,
        name="steel",
    )
)
model.clamp(
    u,
    on=mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 0.0),
        name="fixed_end",
        tag=1,
    ),
)

output = Path("outputs/modal_cantilever_2d")
result = model.step(target=u, modes=6).solve_result(
    output=output / "modes.xdmf",
    strict_output=True,
)
if MPI.COMM_WORLD.rank == 0:
    output.mkdir(parents=True, exist_ok=True)
    result.write_manifest(output / "result.json")
    print("Natural frequencies:", result.quantity("frequencies"))
