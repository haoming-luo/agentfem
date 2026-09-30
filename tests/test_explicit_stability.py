# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from agentfem import time


def test_stability_contribution_round_trips_time_and_spectral_bound():
    contribution = time.ExplicitStabilityContribution.from_time_increment(
        "body",
        0.25,
        method="test_limit",
    )

    assert contribution.spectral_radius_upper_bound == pytest.approx(64.0)
    restored = time.ExplicitStabilityContribution.from_spectral_bound(
        "body",
        contribution.spectral_radius_upper_bound,
        method="test_bound",
    )
    assert restored.unsafed_time_increment == pytest.approx(0.25)
    assert restored.summary()["method"] == "test_bound"


def test_additive_spectral_composition_is_stricter_than_minimum_limit():
    body = time.ExplicitStabilityContribution.from_time_increment("body", 1.0)
    interface = time.ExplicitStabilityContribution.from_time_increment(
        "interface",
        1.0,
    )

    estimate = time.combine_explicit_stability(
        (body, interface),
        safety_factor=0.8,
    )

    assert estimate.unsafed_time_increment == pytest.approx(1.0 / np.sqrt(2.0))
    assert estimate.selected == pytest.approx(0.8 / np.sqrt(2.0))
    assert estimate.selected < 0.8 * min(
        body.unsafed_time_increment,
        interface.unsafed_time_increment,
    )
    assert estimate.summary()["composition"] == "additive_spectral_upper_bounds"


def test_composed_bound_dominates_exact_small_matrix_frequency():
    mass = np.diag([2.0, 1.0])
    body_stiffness = np.array([[3.0, -1.0], [-1.0, 2.0]])
    interface_stiffness = np.array([[2.0, 2.0], [2.0, 2.0]])
    inverse_sqrt_mass = np.diag(1.0 / np.sqrt(np.diag(mass)))

    def spectral_radius(stiffness):
        scaled = inverse_sqrt_mass @ stiffness @ inverse_sqrt_mass
        return float(np.linalg.eigvalsh(scaled)[-1])

    body = time.ExplicitStabilityContribution.from_spectral_bound(
        "body",
        spectral_radius(body_stiffness),
    )
    interface = time.ExplicitStabilityContribution.from_spectral_bound(
        "interface",
        spectral_radius(interface_stiffness),
    )
    estimate = time.combine_explicit_stability((body, interface), safety_factor=1.0)
    exact_total = spectral_radius(body_stiffness + interface_stiffness)

    assert estimate.spectral_radius_upper_bound >= exact_total
    assert estimate.selected <= 2.0 / np.sqrt(exact_total)


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_stability_contribution_rejects_invalid_limits(value):
    with pytest.raises(ValueError):
        time.ExplicitStabilityContribution.from_time_increment("body", value)


def test_stability_composition_rejects_duplicate_names():
    contribution = time.ExplicitStabilityContribution.from_time_increment(
        "body",
        1.0,
    )
    with pytest.raises(ValueError, match="unique"):
        time.combine_explicit_stability((contribution, contribution))
