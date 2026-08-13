"""External structure-level acceptance for distributed J2 equilibrium."""

from __future__ import annotations

import pytest
from mpi4py import MPI

from agentfem import benchmarks


def test_public_thick_cylinder_brackets_first_yield_in_serial_and_mpi():
    if MPI.COMM_WORLD.size not in {1, 2}:
        pytest.skip("the versioned external J2 contract covers one and two ranks")

    assessment = benchmarks.j2_thick_cylinder_benchmark()

    assert assessment.acceptable
    assert assessment.quantities["maximum_equivalent_plastic_strain"] > 0.0
    assert assessment.quantities["maximum_displacement"] == pytest.approx(
        0.003455482362744615,
        rel=2.0e-10,
        abs=2.0e-13,
    )
    assert assessment.quantities["maximum_equivalent_plastic_strain"] == pytest.approx(
        0.00024084814820469344,
        rel=2.0e-9,
        abs=2.0e-13,
    )
