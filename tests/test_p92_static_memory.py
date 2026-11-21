from dataclasses import asdict, replace

import numpy as np
import pytest

from tools import p92_recoverable_spectrum as kernel
from tools.p92_recoverable_frozen import decode_material
from tools.p92_static_memory_probe import candidates
from tools.p92_two_rate_reference import Branch, Material


def material():
    b = Branch(1000., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    return kernel.Material(Material(10000., (b, b)), (20000., 40000.), (.5, 10.))


def state():
    return np.array([0., 1., -2., .001, 3., -1., .004, 0., 0., 0., 0., 0.])


def test_defaults_and_decoder_preserve_history_semantics():
    m, y = material(), state()
    baseline = kernel.rates(y, 0., m)
    for definition in ("own", "shared"):
        np.testing.assert_array_equal(kernel.rates(y, 0., replace(m,
            static_history=definition, static_history_rates=(250., 250.))), baseline)
        changed = replace(m, static_ratios=(.5, .1), static_history_rates=(250., 250.), static_history=definition)
        _, decoded = decode_material(asdict(changed), kernel.MODEL_ID)
        np.testing.assert_array_equal(kernel.rates(y, 0., decoded), kernel.rates(y, 0., changed))


@pytest.mark.parametrize("definition", ["own", "shared"])
def test_exact_inactive_decay_power_and_split(definition):
    m = replace(material(), static_ratios=(.5, .1), static_history_rates=(250., 250.), static_history=definition)
    y = state(); before = y.copy()
    p = [y[3], y[6]] if definition == "own" else [y[3]+y[6]]*2
    for offset, ratio, accumulated in zip((1, 4), m.static_ratios, p, strict=True):
        factor = ratio+(1-ratio)*np.exp(-250*accumulated)
        v = kernel.rates(y, 0., m)
        np.testing.assert_allclose(v[offset:offset+2], -np.array([.01, .02])*factor*y[offset:offset+2])
    v = kernel.rates(y, 0., m); h=1e-5
    psi = (kernel.stored_energy(y+h*v, m)-kernel.stored_energy(y-h*v, m))/(2*h)
    assert v[8] == pytest.approx(psi+v[9], abs=1e-10)
    full = kernel.integrate(y, 10., 0., m).y[:, -1]
    for offset, ratio, accumulated in zip((1, 4), m.static_ratios, p, strict=True):
        factor = ratio+(1-ratio)*np.exp(-250*accumulated)
        np.testing.assert_allclose(full[offset:offset+2], y[offset:offset+2]*np.exp(-10*np.array([.01, .02])*factor), atol=1e-9)
    bdf = kernel.integrate(y, 10., 0., m, method="BDF", rtol=1e-10).y[:, -1]
    mid = kernel.integrate(y, 4., 0., m).y[:, -1]
    split = kernel.integrate(mid, 6., 0., m).y[:, -1]
    np.testing.assert_allclose(full, bdf, atol=1e-8)
    np.testing.assert_allclose(full, split, atol=1e-8)
    np.testing.assert_array_equal(y, before)


def test_shared_history_couples_other_branch_own_does_not():
    m = replace(material(), static_ratios=(.5, .5), static_history_rates=(250., 250.))
    y = state(); z=y.copy(); z[3] += .002
    assert kernel.rates(y, 0., m)[4] == kernel.rates(z, 0., m)[4]
    shared = replace(m, static_history="shared")
    assert abs(kernel.rates(z, 0., shared)[4]) < abs(kernel.rates(y, 0., shared)[4])
    sign=np.array([-1., -1., -1., 1., -1., -1., 1., -1., 1., 1., -1., -1.])
    np.testing.assert_allclose(kernel.rates(y*sign, 0., shared), kernel.rates(y, 0., shared)*sign)


@pytest.mark.parametrize("update", [dict(static_ratios=(0., 1.)), dict(static_ratios=(1.1, 1.)),
    dict(static_history_rates=(-1., 0.)), dict(static_history_rates=(np.nan, 0.)), dict(static_history="time")])
def test_invalid_history_rejected(update):
    with pytest.raises(ValueError):
        replace(material(), **update)


def test_prespecified_candidates_no_parameter_search():
    rows = candidates(material())
    assert [name for name, _ in rows] == ["control", "own-0.5", "own-0.1", "shared-0.5", "shared-0.1"]
    assert all(m.backbone == material().backbone for _, m in rows)


def test_fixed_card_does_not_forge_successful_identification(monkeypatch, tmp_path):
    from tools import p92_recoverable_frozen as frozen
    card = dict(model_id=kernel.MODEL_ID, parameters=asdict(material()),
                source_sha256={p: p for p in ("RF", "SLR", "SLC")})
    monkeypatch.setattr(frozen, "read_source", lambda path, protocol: dict(sha256=protocol))
    monkeypatch.setattr(frozen, "measured_path", lambda source, cycles: {})
    monkeypatch.setattr(frozen, "attach_waveforms", lambda path, source: path)
    def fail(*args, **kwargs):
        raise RuntimeError("rejected continuation")
    monkeypatch.setattr(frozen, "replay", fail)
    result = frozen.evaluate(card, tmp_path, fixed_card=True)
    assert result["fitted"] is False
    assert all(not p["numerical"]["passed"] for p in result["protocols"].values())
    with pytest.raises(ValueError, match="explicit boolean"):
        frozen.evaluate(card, tmp_path, fixed_card="true")
    with pytest.raises(ValueError, match="converged"):
        frozen.evaluate(dict(training_cycles=[1, 2, 3], fit=dict(success=False)), tmp_path)
