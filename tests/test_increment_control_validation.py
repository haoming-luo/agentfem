# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Numerical controls reject invalid values before any solver owns a state."""

import json

import numpy as np
import pytest

from agentfem import steps


@pytest.mark.parametrize("value", [True, np.bool_(True), 2.5, 4.0, "4", np.nan, np.inf, 0, -1])
def test_fixed_count_is_a_positive_integer_without_truncation(value):
    with pytest.raises(ValueError, match="increments"):
        steps.fixed(value)


@pytest.mark.parametrize("name", ["max_increments", "max_cutbacks", "fast_iterations", "slow_iterations"])
@pytest.mark.parametrize("value", [True, np.bool_(True), 2.5, np.nan, np.inf, "4"])
def test_automatic_counts_are_integers(name, value):
    with pytest.raises(ValueError, match=name):
        steps.automatic(**{name: value})


@pytest.mark.parametrize("name", ["initial", "minimum", "maximum", "growth_factor",
                                 "cutback_factor", "maximum_inelastic_increment"])
@pytest.mark.parametrize("value", [True, np.bool_(True), np.nan, np.inf, -np.inf, "0.1", 1j])
def test_automatic_sizes_are_finite_real_controls(name, value):
    with pytest.raises(ValueError, match=name):
        steps.automatic(**{name: value})


@pytest.mark.parametrize("value", [True, np.bool_(True), "1", 1j, np.nan, np.inf])
def test_fixed_path_factors_are_finite_real_controls(value):
    with pytest.raises(ValueError, match="load factors"):
        steps.at(value)


def test_valid_numpy_controls_normalize_without_changing_default_identity():
    automatic = steps.automatic(
        initial=np.float64(.1), max_increments=np.int64(100), max_cutbacks=np.int32(0),
        growth_factor=np.float32(1.5), fast_iterations=np.int64(4))
    assert type(automatic.initial) is float
    assert type(automatic.max_increments) is int
    assert type(automatic.max_cutbacks) is int
    assert automatic.max_cutbacks == 0  # Explicitly disabling retries is valid.
    assert json.loads(json.dumps(automatic.summary()))["max_cutbacks"] == 0
    expected = steps.automatic().summary()
    actual = automatic.summary()
    actual["max_cutbacks"] = 5
    assert actual == expected
    assert steps.fixed(np.int64(4)).summary() == steps.fixed(4).summary()
    assert steps.at(np.float64(.5), np.float64(1.)).load_factors == (.5, 1.)


def test_compatibility_normalizer_cannot_bypass_count_validation():
    with pytest.raises(ValueError, match="increments"):
        steps.normalize(increments=2.5)


def test_existing_integer_spelling_of_real_controls_keeps_checkpoint_identity():
    policy = steps.automatic(initial=1, maximum=1, growth_factor=2,
                             maximum_inelastic_increment=1)
    summary = policy.summary()
    for name in ("initial", "maximum", "growth_factor", "maximum_inelastic_increment"):
        assert type(summary[name]) is int
    assert all(type(value) is float for value in steps.at(1).load_factors)
