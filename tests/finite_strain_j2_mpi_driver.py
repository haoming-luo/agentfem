"""Two-rank global finite-strain J2 acceptance driver."""

from __future__ import annotations

import numpy as np
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import constitutive, fields, mesh, models, solvers, steps, studies


def main() -> None:
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        raise RuntimeError("Finite-strain J2 MPI acceptance requires two ranks.")
    domain = dolfinx_mesh.create_unit_cube(comm, 2, 1, 1)
    study = studies.nonlinear_static(physics="solid_mechanics", dimension=3)
    model = models.create(study=study, mesh=domain, name="finite_strain_j2_mpi")
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="left", tag=1
    )
    right = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 1.0), name="right", tag=2
    )
    y_symmetry = mesh.boundary(
        domain, lambda x: np.isclose(x[1], 0.0), name="y_symmetry", tag=3
    )
    z_symmetry = mesh.boundary(
        domain, lambda x: np.isclose(x[2], 0.0), name="z_symmetry", tag=4
    )
    model.fix(displacement, on=left, component=0, value=0.0)
    model.fix(displacement, on=y_symmetry, component=1, value=0.0)
    model.fix(displacement, on=z_symmetry, component=2, value=0.0)
    model.fix(displacement, on=right, component=0, value=0.02)
    material = constitutive.finite_strain_j2_logarithmic(
        young=200_000.0,
        poisson=0.3,
        yield_stress=200.0,
        hardening_modulus=2_000.0,
    )
    model.material(material)
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(4),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        progress=False,
    )
    solution = step.solve()

    state = step.response.state.committed_state_vectors()
    points_per_cell = len(step.response.state.reference_field.points)
    cell_map = domain.topology.index_map(domain.topology.dim)
    owned_points = int(cell_map.size_local) * points_per_cell
    owned = state[:owned_points]
    local_min = float(np.min(owned[:, -1], initial=np.inf))
    local_max = float(np.max(owned[:, -1], initial=-np.inf))
    global_min = float(comm.allreduce(local_min, op=MPI.MIN))
    global_max = float(comm.allreduce(local_max, op=MPI.MAX))
    local_displacement = float(np.max(solution.x.array, initial=-np.inf))
    maximum_displacement = float(comm.allreduce(local_displacement, op=MPI.MAX))
    local_fp_error = max(
        (
            abs(float(np.linalg.det(value)) - 1.0)
            for value in owned[:, :-1].reshape((-1, 3, 3))
        ),
        default=0.0,
    )
    fp_error = float(comm.allreduce(local_fp_error, op=MPI.MAX))

    if step.accepted_load_factor != 1.0:
        raise RuntimeError("Distributed finite-strain J2 path did not complete.")
    if global_min <= 0.0 or global_max - global_min > 2.0e-10:
        raise RuntimeError("Distributed finite-strain J2 state is nonuniform.")
    if abs(maximum_displacement - 0.02) > 2.0e-10:
        raise RuntimeError("Distributed prescribed displacement is incorrect.")
    if fp_error > 5.0e-10:
        raise RuntimeError("Distributed finite-strain J2 violated det(Fp)=1.")


if __name__ == "__main__":
    main()
