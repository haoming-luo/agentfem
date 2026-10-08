# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from dolfinx import fem
from mpi4py import MPI

from agentfem import fields, mesh
from agentfem.time._staggered import StaggeredFieldIteration


def make_iteration():
    domain = mesh.rectangle((0.0, 0.0), (1.0, 1.0), (2, 2), comm=MPI.COMM_WORLD)
    value = fields.temperature(domain, value=0.0).value
    return value, StaggeredFieldIteration({"temperature": value})


def test_norm_forms_are_reused_and_unrelaxed_residual_is_reported(monkeypatch):
    value, iteration = make_iteration()

    def forbidden(*args, **kwargs):
        raise AssertionError("Residual form rebuilt during iteration")

    monkeypatch.setattr(fem, "form", forbidden)

    def solve():
        value.x.array[:] = 0.5 * (value.x.array + 1.0)
        value.x.scatter_forward()

    records = iteration.run(
        (solve,),
        absolute_tolerances={"temperature": 1e-10},
        rtol=1e-10,
        relaxation=1.0,
        max_iterations=100,
    )
    assert records[0]["temperature_absolute"] == pytest.approx(0.5)
    assert records[-1]["temperature_absolute"] < 2e-10
    np.testing.assert_allclose(value.x.array, 1.0, atol=4e-10)


@pytest.mark.parametrize("limit", [True, 1.5, 0])
def test_iteration_limit_is_an_actual_positive_integer(limit):
    _, iteration = make_iteration()
    with pytest.raises(ValueError, match="integer iteration"):
        iteration.run(
            (lambda: None,),
            absolute_tolerances={"temperature": 1e-10},
            rtol=1e-10,
            relaxation=1.0,
            max_iterations=limit,
        )


def test_near_zero_field_keeps_absolute_criterion():
    _, iteration = make_iteration()
    records = iteration.run(
        (lambda: None,),
        absolute_tolerances={"temperature": 1e-10},
        rtol=1e-10,
        relaxation=1.0,
        max_iterations=2,
    )
    assert records == [
        {"iteration": 1, "temperature_absolute": 0.0, "temperature_relative": None}
    ]
