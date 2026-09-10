from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import constitutive, fields, mechanics, mesh, models, studies


def test_mixed_hyperelastic_acceptance_is_mpi_safe():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed mixed solve requires at least two MPI ranks")
    domain = mesh.cuboid(
        (0, 0, 0), (1, 1, 1), (1, 1, 1),
        comm=MPI.COMM_WORLD, cell_type="tetrahedron",
    )
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain,
    )
    unknown = model.field(fields.displacement_pressure(domain))
    material = model.material(
        constitutive.mixed_neo_hookean(young=1.0e6, poisson=0.499),
    )
    exterior = mesh.boundary(domain, lambda x: np.full(x.shape[1], True), name="all")
    model.fix(unknown.displacement, on=exterior)

    problem = model.step(target=unknown, material=material, increments=1, progress=False)
    problem.solve()

    local = float(np.max(np.abs(unknown.value.x.array)))
    assert MPI.COMM_WORLD.allreduce(local, op=MPI.MAX) == pytest.approx(0.0)
    assert problem.last_solve_info.converged


def test_mixed_finite_strain_j2_rejects_unverified_distributed_reduction():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed mixed guard requires at least two MPI ranks")
    domain = mesh.cuboid(
        (0, 0, 0),
        (1, 1, 1),
        (1, 1, 1),
        comm=MPI.COMM_WORLD,
        cell_type="tetrahedron",
    )
    unknown = fields.displacement_pressure(domain)
    material = constitutive.finite_strain_j2_logarithmic(
        young=20_000.0,
        poisson=0.45,
        yield_stress=80.0,
    )
    with pytest.raises(
        NotImplementedError,
        match="Distributed mixed displacement-pressure MPC",
    ):
        mechanics.finite_strain_j2_mixed_affine_problem(
            target=unknown,
            material=material,
            constraint=SimpleNamespace(target=unknown),
        )
