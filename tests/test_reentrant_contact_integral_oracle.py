"""Independent L-boundary quadrature counterexample; not a solver claim."""

from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np


ORACLE = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "tools" / "verify_reentrant_contact_integral.py"
))


def test_reentrant_face_can_hide_positive_penalty_energy_from_three_point_rule():
    vertices = ORACLE["SLAVE"]
    barycentric = np.asarray(((2 / 3, 1 / 6, 1 / 6),
                              (1 / 6, 2 / 3, 1 / 6),
                              (1 / 6, 1 / 6, 2 / 3)))
    assert np.all(ORACLE["signed_gap"](barycentric @ vertices) > 0.0)
    assert ORACLE["signed_gap"](vertices[:1])[0] < 0.0
    assert ORACLE["penalty_potential"](subdivisions=None) == 0.0
    coarse = ORACLE["penalty_potential"](subdivisions=128)
    medium = ORACLE["penalty_potential"](subdivisions=256)
    refined = ORACLE["penalty_potential"](subdivisions=512)
    assert 0.0 < coarse < medium < refined
    assert (refined - medium) / refined < 1.0e-3
