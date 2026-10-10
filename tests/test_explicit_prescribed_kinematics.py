# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Compact prescribed kinematics retain overlap and owned/ghost semantics."""

from types import SimpleNamespace

import numpy as np
import pytest

from agentfem.time.explicit import (
    _assign_prescribed_component,
    _owned_dirichlet_dofs,
    _prescribed_kinematics,
)


def boundary(indices, *, owned=None, amplitude=None):
    selected = np.asarray(indices, dtype=np.int32)
    bc = SimpleNamespace(
        dof_indices=lambda: (selected, len(selected) if owned is None else owned)
    )
    return SimpleNamespace(bc=bc, amplitude=amplitude)


def test_compact_motion_stationary_fallback_and_ghost_exclusion():
    items = (
        boundary([0, 4, 8], owned=2, amplitude=lambda t: 3 * t**2),
        boundary([1, 6]),
    )
    resolved = _prescribed_kinematics(items, time=0.5, dt=0.1, stationary_dofs=[0, 2])
    actual = dict(zip(resolved.dofs, resolved.values))
    assert set(actual) == {0, 1, 2, 4, 6}
    np.testing.assert_allclose(actual[0], [2.7, 6, 3])
    np.testing.assert_allclose(actual[4], actual[0])
    for index in (1, 2, 6):
        np.testing.assert_array_equal(actual[index], 0)


def test_duplicate_histories_preserve_last_compatible_values():
    first = boundary([1, 3, 3], amplitude=lambda t: t)
    last = boundary([3, 5], amplitude=lambda t: (1 + 1e-6) * t)
    resolved = _prescribed_kinematics((first, last), time=0.5, dt=0.1)
    np.testing.assert_array_equal(resolved.dofs, [1, 3, 5])
    np.testing.assert_allclose(resolved.values[:, 0], [1, 1.000001, 1.000001])
    with pytest.raises(ValueError, match="Conflicting prescribed.*3"):
        _prescribed_kinematics((first, boundary([3])), time=0.5, dt=0.1)


@pytest.mark.parametrize("value", [np.inf, np.nan])
def test_nonfinite_motion_rejected_even_on_rank_without_owned_boundary(value):
    with pytest.raises(ValueError, match="must be finite"):
        _prescribed_kinematics(
            (boundary([], amplitude=lambda t: value),), time=0.5, dt=0.1
        )


@pytest.mark.parametrize("indices", [[], [1, 3]])
def test_assignment_is_vectorized_and_empty_partitions_still_scatter(monkeypatch, indices):
    calls = []
    field = SimpleNamespace(
        x=SimpleNamespace(array=np.full(5, -1.0), scatter_forward=lambda: calls.append(1))
    )
    monkeypatch.setattr("agentfem.time.explicit.fields.unwrap", lambda value: value)
    resolved = _prescribed_kinematics((boundary(indices),), time=0.5, dt=0.1)
    _assign_prescribed_component(field, resolved, component=0)
    assert calls == [1]
    np.testing.assert_array_equal(field.x.array[indices], 0)
    remaining = np.setdiff1d(np.arange(5), indices)
    np.testing.assert_array_equal(field.x.array[remaining], -1)


def test_backend_indices_are_unique_owned_scalar_dofs():
    np.testing.assert_array_equal(
        _owned_dirichlet_dofs([boundary([4, 2, 99], owned=2).bc, boundary([2, 3]).bc]),
        [2, 3, 4],
    )
