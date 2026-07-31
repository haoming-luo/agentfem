"""Global 3D J2 plasticity with automatic incrementation."""

from pathlib import Path

import numpy as np
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import constitutive, fields, io, mesh, models, solvers, steps, studies


def main() -> None:
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_WORLD, 4, 2, 2)
    study = studies.nonlinear_static(
        physics="solid_mechanics",
        dimension=3,
        name="small_strain_j2_tension",
    )
    model = models.create(study=study, mesh=domain)
    displacement = model.field(fields.displacement(domain))
    steel = model.material(
        constitutive.J2LinearIsotropicHardening(
            young=200.0e3,
            poisson=0.3,
            yield_stress=200.0,
            hardening_modulus=2.0e3,
        )
    )

    left = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="left", tag=1
    )
    right = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 1.0), name="right", tag=2
    )
    model.fix(displacement, on=left, value=0.0)
    model.traction((250.0, 0.0, 0.0), on=right)

    step = model.step(
        target=displacement,
        material=steel,
        incrementation=steps.automatic(
            initial=0.1,
            minimum=1.0e-4,
            maximum=0.25,
            max_increments=100,
        ),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            maximum_iterations=20,
            linear_solver=solvers.direct_solver(package="mumps"),
        ),
    )
    step.solve()

    output = Path(__file__).resolve().parents[1] / "examples_output" / "j2_plasticity_3d.xdmf"
    with io.XDMFTimeSeries(output, domain) as writer:
        writer.write_fields(
            1.0,
            displacement.value,
            *step.state.output_fields(),
        )


if __name__ == "__main__":
    main()
