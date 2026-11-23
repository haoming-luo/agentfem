from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from tools import p92_recoverable_spectrum as kernel
from tools.p92_recoverable_frozen import decode_material
from tools.p92_spectrum_recovery_joint import decode, decode_recovery_shape, identify
from tools.p92_two_rate_reference import Branch, Material


def card():
    branch = Branch(20., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    return asdict(kernel.Material(Material(10000., (branch, branch)),
                                  (20000., 40000.), (.5, 10.)))


def test_zero_identity_and_exact_substitution():
    seed = card()
    z = np.zeros(22)
    original = decode(seed, z, (0., 0.))
    assert decode_recovery_shape(seed, z) == original
    z[7:9] = [.1, -.2]
    changed = decode_recovery_shape(seed, z)
    assert changed.backbone.branches[0] == original.backbone.branches[0]
    b = changed.backbone.branches[1]
    np.testing.assert_allclose(b.ratcheting, 3.*10.**z[7:9])
    assert b.static_recovery == original.backbone.branches[1].static_recovery
    assert changed.moduli == original.moduli and changed.times == original.times
    assert seed == card()
    _, restored = decode_material(asdict(changed), kernel.MODEL_ID)
    assert restored == changed


@pytest.mark.parametrize("other", ["shared_history", "directional", "primary",
    "recoverable_kinetics", "transient_strength", "replacement_space",
    "recoverable_mobility", "critical_recovery"])
def test_shape_cannot_combine_mechanisms(other):
    with pytest.raises(ValueError, match="isolate"):
        identify({"model_id": kernel.MODEL_ID}, Path("unused"), (0., 0.),
                 recovery_shape=True, **{other: True})


@pytest.mark.parametrize("z", [np.zeros(21), np.full(22, np.nan)])
def test_invalid_coordinates(z):
    with pytest.raises(ValueError):
        decode_recovery_shape(card(), z)


def test_changed_shape_continuation_and_independent_integration():
    z = np.zeros(22)
    z[7:9] = [.1, -.2]
    mat = decode_recovery_shape(card(), z)
    old = np.zeros(12)
    first = kernel.integrate(old, 2., .01, mat).y[:, -1]
    whole = kernel.integrate(first, 2., -.01, mat).y[:, -1]
    independent = kernel.integrate(first, 2., -.01, mat, method="BDF", rtol=1e-10).y[:, -1]
    halfway = kernel.integrate(first, 1., -.01, mat).y[:, -1]
    split = kernel.integrate(halfway, 1., -.01, mat).y[:, -1]
    np.testing.assert_allclose(whole, independent, atol=1e-6, rtol=1e-7)
    np.testing.assert_allclose(whole, split, atol=1e-7, rtol=1e-7)
    assert whole[9] >= first[9] >= 0
