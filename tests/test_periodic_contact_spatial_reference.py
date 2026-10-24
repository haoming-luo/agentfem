# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Check the independent spatial diagnostic, not a new product solver."""

from pathlib import Path
import runpy

import numpy as np
import pytest


def test_independent_linear_reference_scaling_and_constraint_residual():
    solve = runpy.run_path(str(Path(__file__).resolve().parents[1]
                               / "tools/diagnose_periodic_contact_spatial.py"))["solve_linear_reference"]
    a, b = solve(4), solve(4, amplitude=.5)
    assert b["reaction_magnitude"] == pytest.approx(.5 * a["reaction_magnitude"], rel=2e-12)
    assert b["elastic_energy"] == pytest.approx(.25 * a["elastic_energy"], rel=2e-12)
    for row in (a, b):
        assert row["free_residual_norm"] < 1e-10
        assert row["maximum_absolute_gap"] < 1e-12
        assert row["minimum_nodal_force"] > 0
        assert 0 < row["edge_nodal_force_fraction"] < 1


def test_graded_fixture_retains_patch_and_actual_affine_support():
    root = Path(__file__).resolve().parents[1]
    make = runpy.run_path(str(root / "examples/finite_j2_periodic_augmented_plane.py"))["make_step"]
    grading = runpy.run_path(str(root / "tools/diagnose_periodic_contact_spatial.py"))["graded_coordinate_lines"]
    lines = grading(4)
    step = make(cells=4, increments=4, affine_end_face_support=True, coordinate_lines=lines)
    step.solve(until=.25)
    assert step.constraint.mismatch(.25) < 1e-12
    summary = runpy.run_path(str(root / "tools/verify_affine_augmented_path.py"))["observe"](
        step, .25, volume_weighted=True)
    assert np.isfinite(summary["peeq_reference_volume_mean"])
    assert summary["maximum_penetration"] < 1e-8
    bad = lines.copy()
    bad[1, 1] += .001
    with pytest.raises(ValueError, match="patch edges"):
        make(cells=4, coordinate_lines=bad)
