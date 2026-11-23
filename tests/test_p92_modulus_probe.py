from dataclasses import asdict

import numpy as np
import pytest

from tools.p92_modulus_probe import variants
from tools.p92_observed_recovery_budget import linear_memory, window_budget
from tools.p92_recoverable_spectrum import Material
from tools.p92_two_rate_reference import Branch, Material as Backbone


def test_only_existing_elastic_modulus_changes():
    b = Branch(10., .001, 3., (100., 200.), (10., 20.), (.01, .02))
    material = Material(Backbone(155176., (b, b)), (20000., 40000.), (.5, 10.))
    control, alternate = variants(material)
    assert control[0] == 155176. and alternate[0] == 162000.
    original = asdict(material)
    assert asdict(control[1]) == original
    changed = asdict(alternate[1])
    changed["backbone"]["modulus"] = original["backbone"]["modulus"]
    assert changed == original
    source = variants(material, source_modulus=True)
    assert source[1][0] == 151000.
    assert source[1][1].moduli == material.moduli
    with pytest.raises(ValueError):
        variants(material, source_modulus=1)


def test_observed_partition_modulus_difference_is_exact():
    t, s, e = np.array([0., 1., 10.]), np.array([300., 240., 200.]), np.array([.002]*3)
    k, tau = np.array([1e6, 8e5]), np.array([.5, 80.])
    memory = linear_memory(t, s, k, tau, [0., 0.])
    values = [window_budget(t, s, e, memory, 1, 2, modulus=modulus, moduli=k,
                           times=tau, initial_bounds=np.zeros(2)) for modulus in (155176., 162000.)]
    assert values[1]["required_remainder_increment"]-values[0]["required_remainder_increment"] == pytest.approx(
        (s[2]-s[1])*(1/155176.-1/162000.), abs=1e-18)
