# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Validation describes the immutable payload actually handed to a material."""
import numpy as np
import pytest

from agentfem.constitutive import material_array_batch as arrays


def test_snapshot_is_frozen_before_validation_of_mutable_input(monkeypatch):
    raw = np.arange(4.)
    def check(snapshot):
        raw[:] = np.nan
        return bool(np.isfinite(snapshot).all())
    monkeypatch.setattr(arrays, "_native_all_finite", check)
    result = arrays._array(raw, (4,), "snapshot")
    np.testing.assert_array_equal(result, np.arange(4.))
    assert np.isnan(raw).all()
    with pytest.raises(ValueError):
        result.setflags(write=True)


def test_native_validator_cannot_invalidate_a_previously_checked_live_buffer(monkeypatch):
    raw = np.arange(4.)
    def check(snapshot):
        finite = bool(np.isfinite(snapshot).all())
        raw[:] = np.inf
        return finite
    monkeypatch.setattr(arrays, "_native_all_finite", check)
    np.testing.assert_array_equal(arrays._array(raw, (4,), "snapshot"), np.arange(4.))
