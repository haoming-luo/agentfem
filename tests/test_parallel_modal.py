from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import fields, mesh, models, studies
from agentfem.constitutive import isotropic_elastic


def test_structural_modes_use_one_distributed_reduced_eigenproblem():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed modal verification requires at least two ranks")

    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (8, 2),
        comm=MPI.COMM_WORLD,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.modal_solid(dimension=2, assumption="plane_stress"),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain, degree=2))
    model.material(
        isotropic_elastic(
            young=210.0e9,
            poisson=0.3,
            density=7800.0,
        )
    )
    model.clamp(
        displacement,
        on=mesh.boundary(
            domain,
            lambda x: np.isclose(x[0], 0.0),
            name="fixed_end",
            tag=1,
        ),
    )

    result = model.step(target=displacement, modes=3).solve_result()
    frequency = np.asarray(result.quantity("frequencies"))
    gathered = MPI.COMM_WORLD.allgather(frequency)

    for rank_frequency in gathered:
        np.testing.assert_allclose(rank_frequency, frequency, rtol=1.0e-10)
    assert frequency[0] == pytest.approx(163.27832561, rel=1.0e-7)
    assert result.metadata["solve"]["free_dofs"] > 0
    assert result.metadata["solve"]["constrained_dofs"] > 0
