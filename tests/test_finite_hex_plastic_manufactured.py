# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent plastic manufactured forcing and a bounded refinement gate."""

from pathlib import Path
import runpy

import numpy as np


def fixture():
    return runpy.run_path(
        str(
            Path(__file__).parents[1]
            / "tools/verify_finite_hex_plastic_manufactured.py"
        )
    )


def test_body_force_is_independent_divergence_of_reference_piola():
    module = fixture()
    reference = module["reference"]
    x = np.array([0.1, 0.3, 0.497, 0.7, 0.9])
    t, h = 0.1, 1e-6
    # Coaxial uniaxial strain has J=F11, hence P11=sigma11.
    derivative = (reference(x + h, t)["stress"] - reference(x - h, t)["stress"]) / (
        2 * h
    )
    exact = reference(x, t)
    np.testing.assert_allclose(
        exact["force_density"],
        module["RHO"] * exact["a"] - derivative,
        rtol=1e-8,
        atol=1e-7,
    )
    assert np.count_nonzero(exact["peeq"] > 0) == 4
    assert exact["stress"][0] > 0 > exact["stress"][-1]


def test_nonuniform_plastic_manufactured_trajectory_refines():
    run = fixture()["run"]
    coarse, fine = run(8, 1000), run(16, 1000)
    for name in (
        "displacement_relative_l2_nodal_quadrature",
        "stress_relative_l2_cell",
        "peeq_relative_l2_cell",
    ):
        assert fine[name] < 0.4 * coarse[name]
    assert fine["displacement_relative_l2_nodal_quadrature"] < 0.003
    assert fine["stress_relative_l2_cell"] < 0.015
    assert fine["peeq_relative_l2_cell"] < 0.02
    for result in (coarse, fine):
        assert result["energy"]["material_dissipation"] > 0
        assert result["energy"]["relative_energy_balance_error"] < 1e-6
