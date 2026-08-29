"""Fast contracts for deterministic multi-spherical-void realizations."""

from __future__ import annotations

import json

import numpy as np
import pytest

from periodic_void_fixture import (
    SPHERICAL_VOID_REALIZATION_SCHEMA,
    SPHERICAL_VOID_SAMPLER,
    SphericalVoid,
    SphericalVoidRealization,
    sample_hard_core_spherical_voids,
)


def _sample(*, seed: int = 1729) -> SphericalVoidRealization:
    return sample_hard_core_spherical_voids(
        side_length=1.0,
        count=4,
        radius=0.09,
        seed=seed,
        minimum_inter_void_clearance=0.04,
        minimum_boundary_clearance=0.03,
        maximum_attempts=500,
    )


def test_hard_core_realization_is_reproducible_and_scientifically_identified():
    first = _sample()
    repeated = _sample()
    changed_seed = _sample(seed=1730)

    assert first == repeated
    assert first.canonical_json() == repeated.canonical_json()
    assert (
        first.scientific_identity()["fingerprint"]
        == repeated.scientific_identity()["fingerprint"]
    )
    assert (
        first.scientific_identity()["fingerprint"]
        != (changed_seed.scientific_identity()["fingerprint"])
    )
    assert first.sampler == SPHERICAL_VOID_SAMPLER
    assert first.attempts == 4
    assert first.scientific_identity()["fingerprint"] == (
        "d3f0879a06d1092926544b465109cdff4b979cf384b7db55bb0a67fd1291e49d"
    )
    assert first.observed_boundary_clearance >= 0.03
    assert first.observed_periodic_inter_void_clearance is not None
    assert first.observed_periodic_inter_void_clearance >= 0.04

    expected_fraction = 4.0 * (4.0 * np.pi * 0.09**3 / 3.0)
    assert first.actual_void_fraction == pytest.approx(expected_fraction)
    payload = json.loads(first.canonical_json())
    assert payload["schema"] == SPHERICAL_VOID_REALIZATION_SCHEMA
    assert payload["constraints"]["periodic_boundary_crossing"] is False
    assert payload["observed_periodic_inter_void_clearance"] >= 0.04
    assert [item["id"] for item in payload["voids"]] == [
        "void-0001",
        "void-0002",
        "void-0003",
        "void-0004",
    ]


def test_realization_identity_is_independent_of_input_sphere_order():
    realization = _sample()
    reversed_realization = SphericalVoidRealization(
        side_length=realization.side_length,
        spheres=tuple(reversed(realization.spheres)),
        seed=realization.seed,
        minimum_inter_void_clearance=realization.minimum_inter_void_clearance,
        minimum_boundary_clearance=realization.minimum_boundary_clearance,
        attempts=realization.attempts,
        sampler=realization.sampler,
    )

    assert reversed_realization.spheres == realization.spheres
    assert reversed_realization.canonical_json() == realization.canonical_json()
    assert reversed_realization.scientific_identity() == (
        realization.scientific_identity()
    )


@pytest.mark.parametrize(
    ("spheres", "message"),
    (
        (
            (SphericalVoid(center=(0.1, 0.5, 0.5), radius=0.09),),
            "boundary clearance",
        ),
        (
            (
                SphericalVoid(center=(0.35, 0.5, 0.5), radius=0.12),
                SphericalVoid(center=(0.55, 0.5, 0.5), radius=0.12),
            ),
            "inter-void clearance",
        ),
    ),
)
def test_explicit_realization_rejects_violated_clearances(spheres, message):
    with pytest.raises(ValueError, match=message):
        SphericalVoidRealization(
            side_length=1.0,
            spheres=spheres,
            seed=0,
            minimum_inter_void_clearance=0.02,
            minimum_boundary_clearance=0.02,
            attempts=len(spheres),
        )


def test_explicit_realization_checks_periodic_minimum_image_clearance():
    with pytest.raises(ValueError, match="periodic inter-void clearance"):
        SphericalVoidRealization(
            side_length=1.0,
            spheres=(
                SphericalVoid(center=(0.10, 0.5, 0.5), radius=0.04),
                SphericalVoid(center=(0.90, 0.5, 0.5), radius=0.04),
            ),
            seed=0,
            minimum_inter_void_clearance=0.13,
            minimum_boundary_clearance=0.02,
            attempts=2,
        )


def test_sampler_fails_closed_when_boundary_clearance_cannot_fit():
    with pytest.raises(ValueError, match="does not fit inside the cell"):
        sample_hard_core_spherical_voids(
            side_length=1.0,
            count=1,
            radius=0.49,
            seed=4,
            minimum_inter_void_clearance=0.01,
            minimum_boundary_clearance=0.02,
            maximum_attempts=10,
        )


def test_sampler_fails_closed_after_bounded_attempts_for_impossible_population():
    with pytest.raises(
        ValueError,
        match=r"accepted 1 of 2 voids after 20 attempts \(seed=7\)",
    ):
        sample_hard_core_spherical_voids(
            side_length=1.0,
            count=2,
            radius=0.24,
            seed=7,
            minimum_inter_void_clearance=0.60,
            minimum_boundary_clearance=0.01,
            maximum_attempts=20,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("count", 0, "count must be a positive integer"),
        ("seed", -1, "seed must be a non-negative integer"),
        (
            "minimum_inter_void_clearance",
            0.0,
            "minimum_inter_void_clearance must be finite and positive",
        ),
        (
            "minimum_boundary_clearance",
            0.0,
            "minimum_boundary_clearance must be finite and positive",
        ),
    ),
)
def test_sampler_rejects_ambiguous_or_unsafe_inputs(keyword, value, message):
    options = {
        "side_length": 1.0,
        "count": 2,
        "radius": 0.1,
        "seed": 0,
        "minimum_inter_void_clearance": 0.02,
        "minimum_boundary_clearance": 0.02,
        "maximum_attempts": 100,
    }
    options[keyword] = value
    with pytest.raises(ValueError, match=message):
        sample_hard_core_spherical_voids(**options)
