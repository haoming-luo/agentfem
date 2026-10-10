# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem._contact_path_guard import unresolved_outside_paths


def test_thin_solid_crossing_with_two_separated_endpoints_is_unresolved():
    assert unresolved_outside_paths([0.1], [0.2], [[0, 0, 1.3]], np.zeros((8, 3)))[0]
    assert not unresolved_outside_paths(
        [0.1], [0.09], [[0, 0, 0.01]], np.zeros((8, 3))
    )[0]


def test_common_translation_does_not_trigger_false_crossing():
    shift = np.array([200.0, -400.0, 50.0])
    assert not unresolved_outside_paths(
        [0.01], [0.01], shift[None], np.tile(shift, (8, 1))
    )[0]


def test_primary_motion_is_included_and_detected_contact_is_not_rejected():
    assert unresolved_outside_paths(
        [0.1], [0.2], [[0, 0, 0]], np.tile([0, 0, -1.3], (8, 1))
    )[0]
    assert not unresolved_outside_paths([0.1], [-0.2], [[0, 0, 0.3]], np.zeros((8, 3)))[
        0
    ]


@pytest.mark.parametrize("value", [np.nan, np.inf, 1j])
def test_nonfinite_or_complex_path_rejected(value):
    with pytest.raises(ValueError):
        unresolved_outside_paths([value], [0.1], [[0, 0, 1]], np.zeros((8, 3)))


def test_random_linear_paths_crossing_unit_solid_never_pass_guard():
    rng = np.random.default_rng(728)
    # Through opposite x faces, strictly interior in y/z. Distances to the
    # solid at endpoints are exact; arbitrary common motion cancels.
    for _ in range(100):
        left, right = rng.uniform(0.001, 2, 2)
        shift = rng.normal(size=3) * 10
        delta = np.array([1 + left + right, 0, 0]) + shift
        assert unresolved_outside_paths(
            [left], [right], [delta], np.tile(shift, (8, 1))
        )[0]
