# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Structural regressions use public projects and independent reference mechanics."""

from pathlib import Path
import runpy
import numpy as np
from agentfem import constitutive

ROOT = Path(__file__).resolve().parents[1]


def test_imperfect_column_reaction_and_increment_independence():
    solve = runpy.run_path(str(ROOT / "examples/geometric_elastic_column/case.py"))[
        "solve_column"
    ]
    tips = []
    for increment in (0.05, 0.025):
        _, curve, _ = solve(nx=40, ny=2, increment=increment)
        load = np.asarray(curve["load_N"])
        reaction = np.asarray(curve["support_reaction_N"])
        np.testing.assert_allclose(reaction, load, atol=1e-4, rtol=1e-6)
        tips.append(float(curve["added_tip_displacement_mm"][-1]))
    assert 11 < tips[0] < 13
    np.testing.assert_allclose(tips[0], tips[1], rtol=1e-5)


def test_large_rotation_cantilever_against_independent_elastica():
    solve = runpy.run_path(str(ROOT / "examples/geometric_elastic_cantilever/case.py"))[
        "solve_cantilever"
    ]
    reference = runpy.run_path(
        str(ROOT / "validation/geometric_elasticity/run_elastica.py")
    )["reference"]
    _, curve = solve(nx=40, alpha=4.0)
    ref = reference(4.0)
    assert ref["angle"] > 1.0
    np.testing.assert_allclose(curve["tip_y_over_L"][-1], ref["y"], rtol=1e-3)
    np.testing.assert_allclose(curve["tip_x_over_L"][-1], ref["x"], rtol=1e-3)


def test_stvk_arch_crosses_limit_point():
    solve = runpy.run_path(str(ROOT / "examples/shallow_arch_arc_length/case.py"))[
        "solve_arch"
    ]
    result, displacement, load = solve(
        nx=32,
        arc_size=0.04,
        increments=80,
        material=constitutive.saint_venant_kirchhoff(young=1000, poisson=0.0),
    )
    assert np.max(load) > 0.19
    assert np.any(np.diff(load) < 0)
    assert -displacement[-1] > 0.3
    assert np.max(result.quantity("equilibrium_residuals")) < 1e-6
