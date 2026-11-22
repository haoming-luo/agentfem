from itertools import product

import numpy as np
import pytest

from tools.p92_control_resolution_audit import endpoint_budget


def test_endpoint_interval_equals_exhaustive_box_corners():
    e, s, modulus = np.array([.001, .00101]), np.array([80., 79.]), 150000.
    result = endpoint_budget(e, s, 300., modulus, strain_quantum=1e-5, stress_quantum=.01)
    deviations = []
    for a, b, c, d in product((-1., 1.), repeat=4):
        ep, sp = e+np.array([a,b])*1e-5, s+np.array([c,d])*.01
        value = ep[1]-ep[0]-(sp[1]-sp[0])/modulus
        deviations.append(value-result['inferred_nonelastic_increment'])
    assert max(deviations) == pytest.approx(result['endpoint_quantum_sensitivity'], abs=1e-18)
    assert min(deviations) == pytest.approx(-result['endpoint_quantum_sensitivity'], abs=1e-18)
    assert not result['sign_robust_to_quantum']


def test_stress_resolution_alone_does_not_resolve_inferred_flow():
    result = endpoint_budget([0., 0.], [48.09,49.], 352.848, 155176.471,
                             strain_quantum=1e-5, stress_quantum=.01)
    assert result['stress_only_resolution_units'] > 90
    assert not result['sign_robust_to_quantum']
    assert result['inferred_nonelastic_increment'] < 0


def test_signed_and_invalid_inputs():
    a = endpoint_budget([0.,0.], [10.,0.], 10.,100., strain_quantum=0.,stress_quantum=0.)
    b = endpoint_budget([0.,0.], [-10.,0.], 10.,100., strain_quantum=0.,stress_quantum=0.)
    assert a['inferred_nonelastic_increment'] == -b['inferred_nonelastic_increment']
    assert a['sign_robust_to_quantum'] and b['sign_robust_to_quantum']
    with pytest.raises(ValueError):
        endpoint_budget([0.,0.], [10.,0.], 0.,100., strain_quantum=0.,stress_quantum=0.)
