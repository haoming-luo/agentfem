from dataclasses import replace

import numpy as np
import pytest

from tools.p92_hold_mechanisms import audit_hold
from tools.p92_published_recovery_reference import published_parameters
from tools.p92_two_mechanism import Mechanism


@pytest.mark.parametrize("sign", [-1., 1.])
def test_flow_decomposition_matches_signed_analytical_creep(sign):
    p = replace(published_parameters(600), C1=0., C2=0., Q=0., H=0.,
                gamma1=0., gamma2=0., gamma_r1=0., gamma_r2=0.)
    mechanisms = (Mechanism(p), Mechanism(replace(p, Z=p.Z*2)))
    old = np.zeros(10)
    old[0] = sign*300.
    result = audit_hold(old, 12., mechanisms)
    expected = [sign*12*((300-p.k)/m.parameters.Z)**p.n for m in mechanisms]
    np.testing.assert_allclose(result["signed_inelastic_strain_by_branch"], expected, atol=1e-11)
    assert result["numerical_closure_passed"]
    assert old[9] == 0.


def test_no_flow_is_not_damage_or_energy():
    p = published_parameters(600)
    result = audit_hold(np.zeros(10), 5., (Mechanism(p, enabled=False),)*2)
    assert result["signed_inelastic_strain_by_branch"] == [0., 0.]
    assert result["strain_closure_error"] == 0.
    assert "not stored energy" in result["scope"]
