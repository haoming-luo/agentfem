# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem._orientation import positive_finite_determinants


def test_admission_matches_lu_for_mixed_gradients():
    rng = np.random.default_rng(326)
    f = np.concatenate((np.eye(3) + rng.normal(size=(200, 3, 3)) * .01,
                        rng.normal(size=(200, 3, 3)),
                        np.array([np.diag([1e-100, 1e100, 1.]),
                                  np.diag([-1., 1., 1.]), np.zeros((3, 3))])))
    np.testing.assert_array_equal(positive_finite_determinants(f),
                                  np.linalg.det(f) > 0)


def test_near_identity_avoids_lu_and_boundary_falls_back(monkeypatch):
    original = np.linalg.det
    seen = []
    def det(f):
        seen.append(f.copy())
        return original(f)
    monkeypatch.setattr(np.linalg, "det", det)
    f = np.tile(np.eye(3), (3, 1, 1))
    f[:, 0, 1] = [.499999, .5, 10.]
    assert positive_finite_determinants(f).all()
    assert len(seen) == 1 and len(seen[0]) == 2


def test_invalid_and_empty():
    f = np.tile(np.eye(3), (4, 1, 1))
    f[0, 0, 0] = np.nan
    f[1, 0, 0] = np.inf
    f[2] *= 1e200
    f[3] *= 1e-200
    assert not positive_finite_determinants(f).any()
    assert positive_finite_determinants(np.empty((0, 3, 3))).size == 0
    with pytest.raises(ValueError):
        positive_finite_determinants(np.eye(3))
