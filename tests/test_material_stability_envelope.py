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
