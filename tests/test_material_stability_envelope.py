# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Provider evidence is bounded, immutable, serializable and actively checked."""

from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

from agentfem.constitutive import FirstPiolaTangentEnvelope


def envelope(**overrides):
    return FirstPiolaTangentEnvelope(**{
        "positive_modulus": 100, "negative_modulus": 20,
        "minimum_stretch": 0.8, "maximum_stretch": 1.3,
        "source": "independent reference derivation v1", **overrides,
    })


def test_envelope_is_immutable_and_json_evidence_not_a_verification_claim():
    item = envelope()
    assert json.loads(json.dumps(item.summary())) == item.summary()
    assert item.summary()["configuration"] == "reference"
    assert "not_automatic_verification" in item.summary()["evidence"]
    with pytest.raises(FrozenInstanceError):
        item.positive_modulus = 300


@pytest.mark.parametrize("options", [
    {"positive_modulus": -1}, {"negative_modulus": np.nan},
    {"minimum_stretch": 0}, {"minimum_stretch": 1.01},
    {"maximum_stretch": 0.99}, {"maximum_stretch": np.inf},
    {"source": " "}, {"positive_modulus": True},
])
def test_invalid_declarations_are_rejected(options):
    with pytest.raises(ValueError):
        envelope(**options)


@pytest.mark.parametrize("invalid", ["domain", "reflection", "positive", "negative", "asymmetric", "nan"])
def test_observed_violation_is_rejected(invalid):
    f = np.eye(3)[None]
    a = np.eye(9)[None] * 30
    if invalid == "domain":
        f[0, 0, 0] = 1.4
    elif invalid == "reflection":
        f[0, 0, 0] = -1
    elif invalid == "positive":
        a[0, 0, 0] = 101
    elif invalid == "negative":
        a[0, 0, 0] = -21
    elif invalid == "asymmetric":
        a[0, 0, 1] = 1
    else:
        a[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        envelope().validate_response(f, a)


def test_empty_partition_and_equivalent_tensor_encoding():
    item = envelope()
    item.validate_response(np.empty((0, 3, 3)), np.empty((0, 9, 9)))
    item.validate_response(np.eye(3)[None], np.eye(9).reshape(1, 3, 3, 3, 3))
    with pytest.raises(ValueError, match="point-matched"):
        item.validate_response(np.eye(3)[None], np.empty((0, 9, 9)))
    assert replace(item, source="v2").summary() != item.summary()


def test_chunked_guard_checks_last_point_and_rejects_complex_inputs():
    f = np.tile(np.eye(3), (2051, 1, 1))
    a = np.tile(np.eye(9), (2051, 1, 1))
    envelope().validate_response(f, a)
    a[-1, 0, 0] = 200
    with pytest.raises(ValueError, match="violates"):
        envelope().validate_response(f, a)
    with pytest.raises(ValueError, match="real"):
        envelope().validate_response(f.astype(complex), a)
    with pytest.raises(ValueError, match="real"):
        envelope().validate_response(f, a.astype(complex))


def test_stretch_fast_admission_keeps_svd_for_inconclusive_points(monkeypatch):
    original = np.linalg.svd
    counts = []
    def tracked(value, *args, **kwargs):
        counts.append(len(value))
        return original(value, *args, **kwargs)
    monkeypatch.setattr(np.linalg, 'svd', tracked)
    f = np.tile(np.eye(3), (8, 1, 1))
    envelope().validate_deformation(f)
    assert sum(counts) == 0
    # Endpoint tolerance remains exactly the SVD contract, not Gershgorin.
    f[-1, 0, 0] = 1.3
    envelope().validate_deformation(f)
    assert sum(counts) == 1
    f[-1, 0, 0] = 1.3 * (1 + 2e-12)
    with pytest.raises(ValueError, match='domain'):
        envelope().validate_deformation(f)


def test_stretch_guard_matches_svd_oracle_for_rotated_random_states():
    rng = np.random.default_rng(912)
    f = rng.normal(size=(300, 3, 3)) * .2 + np.eye(3)
    item = envelope()
    singular = np.linalg.svd(f, compute_uv=False)
    expected = ((np.linalg.det(f) > 0)
                & (singular[:, -1] >= .8 * (1 - 1e-12))
                & (singular[:, 0] <= 1.3 * (1 + 1e-12)))
    for matrix, admitted in zip(f, expected):
        if admitted:
            item.validate_deformation(matrix[None])
        else:
            with pytest.raises(ValueError, match='domain'):
                item.validate_deformation(matrix[None])
