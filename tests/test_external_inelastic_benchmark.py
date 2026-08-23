"""External structure-level acceptance for distributed J2 equilibrium."""

from __future__ import annotations

import pytest
from mpi4py import MPI

from agentfem import benchmarks


def test_nafems_creep_cylinder_reference_stresses_match_published_table():
    radius = (100.0, 125.0, 150.0, 175.0, 200.0)
    radial, hoop, axial = benchmarks.power_law_creep_cylinder_stress(
        radius,
        inner_radius=100.0,
        outer_radius=200.0,
        pressure=200.0,
        stress_exponent=5.0,
    )

    assert radial == pytest.approx(
        (-200.0, -129.47114, -76.33936, -34.34328, 0.0), abs=1.0e-5
    )
    assert hoop == pytest.approx(
        (130.38504, 172.70235, 204.58142, 229.77907, 250.38504), abs=1.0e-5
    )
    assert axial == pytest.approx(
        (-34.80748, 21.61561, 64.12103, 97.71789, 125.19252), abs=1.0e-5
    )


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
