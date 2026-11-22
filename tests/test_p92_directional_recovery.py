from dataclasses import asdict, replace

import numpy as np
import pytest

from tools import p92_recoverable_spectrum as kernel
from tools.p92_two_rate_reference import Branch, Material
from tools.p92_recoverable_frozen import decode_material


def material(directional=True):
    b = Branch(10., .001, 3., (100., 200.), (10., 20.), (.01, .02))
    return kernel.Material(Material(10000., (b, b)), (20000., 40000.), (.5, 10.),
                           directional_recovery=directional)


def test_reverse_flow_only_changes_recall_and_preserves_power():
    m = material()
    y = np.array([-40., 5., 10., .01, 5., 10., .02, 0., 0., 0., 0., 0.])
    a = kernel.rates(y, .01, m)
    b = kernel.rates(y, .01, material(False))
    assert a[1] > b[1]  # Removing recall slows erasure of the old positive backstress.
    np.testing.assert_array_equal(a[[0, 3, 6, 7, 8, 10, 11]], b[[0, 3, 6, 7, 8, 10, 11]])
    energy_rate = y[0]*a[0]/m.backbone.modulus
    for offset, branch in zip((1, 4), m.backbone.branches):
        energy_rate += sum(y[offset+i]*a[offset+i]/h for i, h in enumerate(branch.hardening))
    energy_rate += sum(h*x*v for h, x, v in zip(m.moduli, y[10:], a[10:]))
    assert a[8] == pytest.approx(energy_rate+a[9], abs=1e-12)
    assert a[9] >= 0
    reflected = y.copy(); reflected[[0, 1, 2, 4, 5, 7, 10, 11]] *= -1
    c = kernel.rates(reflected, -.01, m)
    signs = np.ones(12); signs[[0, 1, 2, 4, 5, 7, 10, 11]] = -1
    np.testing.assert_allclose(c, signs*a, atol=1e-14)


@pytest.mark.parametrize("stress", [15., 50.])
def test_zero_flow_or_forward_flow_matches_original(stress):
    y = np.zeros(12); y[0] = stress; y[[1, 2, 4, 5]] = [5., 10., 5., 10.]
    np.testing.assert_array_equal(kernel.rates(y, 0., material()), kernel.rates(y, 0., material(False)))


def test_restart_identity_and_independent_integration():
    m = material(); y = np.zeros(12); y[0] = -40.; y[[1, 2, 4, 5]] = [5., 10., 5., 10.]
    before = y.copy()
    a = kernel.integrate(y, 2., -.002, m)
    b = kernel.integrate(y, 2., -.002, m, method="BDF", rtol=1e-10)
    c = kernel.integrate(y, .7, -.002, m)
    d = kernel.integrate(c.y[:, -1], 1.3, -.002, m)
    np.testing.assert_allclose(a.y[:, -1], b.y[:, -1], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(a.y[:, -1], d.y[:, -1], atol=1e-7, rtol=1e-7)
    np.testing.assert_array_equal(y, before)
    module, decoded = decode_material(asdict(m), kernel.MODEL_ID)
    assert module is kernel and decoded.directional_recovery
    with pytest.raises(ValueError):
        replace(m, directional_recovery=1)


@pytest.mark.parametrize("stress", [-100., 0., 100.])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_saturation_boundary_points_inward(stress, sign):
    m = material(); y = np.zeros(12); y[0] = stress
    for offset, b in zip((1, 4), m.backbone.branches):
        y[offset:offset+2] = sign*np.array(b.saturation)
    rate = kernel.rates(y, 0., m)
    assert np.all(sign*rate[[1, 2, 4, 5]] <= 0.)


def test_old_card_defaults_to_magnitude_and_seed_cannot_silently_drop_direction():
    from tools.p92_spectrum_recovery_joint import decode
    card = asdict(material(False)); card.pop("directional_recovery")
    _, decoded = decode_material(card, kernel.MODEL_ID)
    assert not decoded.directional_recovery
    assert not decode(card, np.zeros(22), (0., 0.)).directional_recovery
    card["directional_recovery"] = True
    with pytest.raises(ValueError, match="baseline seed"):
        decode(card, np.zeros(22), (0., 0.))


def test_forbid_combining_mechanisms():
    from pathlib import Path
    from tools.p92_spectrum_recovery_joint import identify
    seed = dict(model_id=kernel.MODEL_ID)
    with pytest.raises(ValueError, match="isolate"):
        identify(seed, Path("unused"), (0., 0.), shared_history=True, directional=True)
    with pytest.raises(ValueError, match="isolate"):
        identify(seed, Path("unused"), (0., 1.), directional=True)


@pytest.mark.parametrize("x,v", [(5., .01), (5., -.01), (-5., .01), (-5., -.01)])
def test_original_tensor_model_ii_reduces_to_scalar_directional_term(x, v):
    # Ohno--Wang 1992 Eq. 10a, with incompressible uniaxial inelastic flow.
    h, a, chi = 100., 10., 3.
    alpha = np.diag([2*x/3, -x/3, -x/3])
    flow = np.diag([v, -v/2, -v/2])
    magnitude = np.sqrt(1.5*np.sum(alpha*alpha))
    projection = np.sum(flow*alpha/magnitude)
    tensor_rate = h*(2*flow/3-(magnitude/a)**chi*max(projection, 0.)*alpha/a)
    scalar_rate = h*v-h/a*(abs(x)/a)**chi*max(np.sign(x)*v, 0.)*x
    assert 1.5*tensor_rate[0, 0] == pytest.approx(scalar_rate)
