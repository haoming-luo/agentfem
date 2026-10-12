"""Run with mpiexec -n 2; explicit global point support and nonlinear solve."""

import numpy as np
from mpi4py import MPI
from agentfem import (
    mesh,
    models,
    studies,
    fields,
    constitutive,
    constraints,
    solvers,
    results,
)

comm = MPI.COMM_WORLD
for dim in (2, 3):
    d = (
        mesh.rectangle((0.0, 0.0), (1.0, 1.0), (6, 4), comm=comm)
        if dim == 2
        else mesh.cuboid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (4, 2, 2), comm=comm)
    )
    m = models.create(
        mesh=d,
        study=studies.static_solid(
            dimension=dim,
            assumption="plane_stress" if dim == 2 else None,
            nonlinear=True,
        ),
    )
    u = m.field(fields.displacement(d, degree=2))
    m.material(
        constitutive.saint_venant_kirchhoff(
            young=1000, poisson=0.0, plane_stress=dim == 2
        )
    )
    pin = constraints.pin(u, at=(0.0,) * dim, components=(1,))
    assert pin.dirichlet[0].match_evidence["owned_global_dofs"] == 1
    owned = pin.bcs[0].dof_indices()[1]
    assert sum(comm.allgather(int(owned) == 0)) >= comm.size - 1
    try:
        constraints.pin(u, at=(-10.0,) * dim, components=(1,))
    except ValueError:
        pass
    else:
        raise AssertionError("Empty point support not rejected")
    m.clamp(u, on=mesh.boundary(d, lambda x: np.isclose(x[0], 0.0)))
    m.traction(
        (1.0,) + (0.0,) * (dim - 1),
        on=mesh.boundary(d, lambda x: np.isclose(x[0], 1.0)),
    )
    step = m.step(
        target=u,
        solver_options=solvers.NewtonSolverOptions(
            linear_solver=solvers.direct_solver(package="mumps")
        ),
        progress=False,
    )
    step.solve()
    tip = float(results.probe(u.value, at=(1.0,) + (0.5,) * (dim - 1))[0])
    assert abs(500 * ((1 + tip) ** 3 - (1 + tip)) - 1.0) < 1e-7
    if comm.rank == 0:
        print(
            {"ranks": comm.size, "dimension": dim, "tip": tip, "global_pin_dofs": 1},
            flush=True,
        )
