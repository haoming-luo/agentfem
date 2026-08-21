from __future__ import annotations

import numpy as np
from dolfinx import fem
from mpi4py import MPI

from agentfem import constraints, fields, mesh, operators, results, solvers


def test_prepared_linear_problem_reuses_matrix_for_updated_rhs():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (6, 6),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    unknown = fields.scalar_unknown(domain, degree=1, name="u")
    boundary = mesh.boundary(
        domain,
        lambda x: (
            np.isclose(x[0], 0.0)
            | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
        ),
        name="boundary",
        tag=1,
    )
    fixed = constraints.fixed(unknown, on=boundary, value=0.0)
    source = fem.Constant(domain, 1.0)
    stiffness = operators.diffusion_operator(unknown, 1.0)
    force = operators.source_vector(source, unknown)

    with solvers.prepare_linear_problem(
        stiffness.expression,
        force.expression,
        unknown.value,
        bcs=fixed.bcs,
        options=solvers.direct_solver(),
    ) as prepared:
        first = prepared.solve()
        first_center = results.probe(first, at=(0.5, 0.5))
        source.value = 2.0
        second = prepared.solve()
        second_center = results.probe(second, at=(0.5, 0.5))
        summary = prepared.summary()

    assert second_center > first_center
    np.testing.assert_allclose(second_center, 2.0 * first_center, rtol=1.0e-12)
    assert summary["matrix_reused"] is True
    assert summary["solve_count"] == 2
    assert summary["last_solve"]["converged"] is True
