# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Whole-mesh synthetic convergence gates, independent analytic solutions."""

import runpy
from pathlib import Path
import pytest

harness = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "validation/dynamic_parity/composite_accuracy/checks.py"
    )
)


@pytest.mark.parametrize("nu", [0.3, 0.4999])
@pytest.mark.parametrize("distorted", [False, True])
def test_manufactured_bending_refinement(nu, distorted):
    coarse = harness["bending"](2, nu, True, distorted)
    fine = harness["bending"](4, nu, True, distorted)
    assert fine["relative_nodal_error"] < 0.06
    assert fine["relative_nodal_error"] < 0.5 * coarse["relative_nodal_error"]
    assert fine["energy_relative_error"] < 0.08
    assert fine["energy_relative_error"] < 0.4 * coarse["energy_relative_error"]


def test_longitudinal_frequency_space_and_time():
    coarse = harness["longitudinal_mode"](4)
    fine = harness["longitudinal_mode"](16)
    assert fine["relative_error"] < 0.002
    assert fine["relative_error"] < 0.15 * coarse["relative_error"]
    errors = fine["temporal_frequency_errors"]
    assert errors[1] < 0.26 * errors[0]
    assert errors[2] < 0.26 * errors[1]


def test_p2_reproduces_quadratic_manufactured_field():
    r = harness["bending"](2, 0.3, False, False, reference=2)
    assert r["relative_nodal_error"] < 1e-11
    assert r["energy_relative_error"] < 1e-11
