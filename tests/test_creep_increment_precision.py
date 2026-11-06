"""Independent high-precision checks of the existing constant-stress K-R law."""

from decimal import Decimal, localcontext
from dataclasses import replace

import numpy as np
import pytest

from agentfem.constitutive import CreepDamageState, KachanovRabotnovCreep, PowerLawCreep


def decimal_reference(law, stress, duration, damage):
    with localcontext() as ctx:
        ctx.prec = 90
        d = lambda v: Decimal(str(v))
        q = d(stress) / d(law.reference_stress)
        dt = d(duration) / d(law.reference_time)
        a, b = d(law.creep_coefficient), d(law.damage_coefficient)
        n, m, phi = d(law.creep_exponent), d(law.damage_exponent), d(law.damage_power)
        integrity = 1 - d(damage)
        new = (integrity**(phi + 1) - (phi + 1) * b * q**m * dt)**(1 / (phi + 1))
        exponent = phi + 1 - n
        integral = (integrity / new).ln() if exponent == 0 else (integrity**exponent - new**exponent) / exponent
        return float(a / b * q**(n - m) * integral), float(integrity - new)


@pytest.mark.parametrize("dt", [1e-12, 1e-9, 1e-6, 1., 100.])
def test_published_scale_keeps_tiny_positive_increments(dt):
    law = KachanovRabotnovCreep(1.6e-31, 12.7, 2.6e-26, 10.3, 16.)
    expected_strain, expected_damage = decimal_reference(law, 110., dt, 0.)
    update = law.update(110., dt)
    assert update.equivalent_increment == pytest.approx(expected_strain, rel=2e-14, abs=0)
    assert update.damage_increment == pytest.approx(expected_damage, rel=2e-14, abs=0)
    assert update.equivalent_increment > 0 and update.damage_increment > 0


@pytest.mark.parametrize("n", [2., 4., 6.])
@pytest.mark.parametrize("damage", [0., .3, .8])
def test_power_and_logarithmic_integrals_at_nonzero_prior_damage(n, damage):
    law = KachanovRabotnovCreep(2e-5, n, 1e-4, 5., 3., reference_stress=100.)
    dt = 1e-10
    expected, _ = decimal_reference(law, 80., dt, damage)
    state = CreepDamageState(damage=damage)
    update = law.update(80., dt, state)
    assert update.equivalent_increment == pytest.approx(expected, rel=2e-14, abs=0)
    assert update.state.damage >= damage
    assert law.update(80., 0., state).state == state


def test_repeated_tiny_increments_recover_one_interval():
    law = KachanovRabotnovCreep(1.6e-31, 12.7, 2.6e-26, 10.3, 16.)
    state = None
    for _ in range(1000):
        state = law.update(110., 1e-12, state).state
    one = law.update(110., 1e-9).state
    assert state.damage == pytest.approx(one.damage, rel=1e-12, abs=0)
    assert state.equivalent_creep_strain == pytest.approx(one.equivalent_creep_strain, rel=1e-12, abs=0)


def test_tiny_tensor_flow_retains_deviatoric_direction():
    law = KachanovRabotnovCreep(1.6e-31, 12.7, 2.6e-26, 10.3, 16.)
    scalar = law.update(110., 1e-12)
    tensor = law.update(np.diag([110., 0., 0.]), 1e-12)
    np.testing.assert_allclose(tensor.state.creep_strain,
                               scalar.equivalent_increment * np.diag([1., -.5, -.5]), rtol=1e-14, atol=0)


def test_terminal_state_does_not_heal_or_accumulate_more_strain():
    law = KachanovRabotnovCreep(2e-5, 4., 1e-4, 5., 3., failure_damage=.9)
    for damage in (.9, .95):
        old = CreepDamageState(equivalent_creep_strain=.2, damage=damage)
        update = law.update(1., 1., old)
        assert update.state == old
        assert update.failed
        assert update.damage_increment == update.equivalent_increment == 0


def test_small_declared_failure_damage_has_nonzero_life_and_zero_duration_is_identity():
    law = KachanovRabotnovCreep(2., 1., 1., 1., 0., failure_damage=1e-20)
    assert law.rupture_time(1.) == pytest.approx(1e-20, rel=1e-14, abs=0)
    zero = law.update(1., 0.)
    assert not zero.failed and zero.state.damage == 0 and zero.equivalent_increment == 0
    before = law.update(1., 5e-21)
    assert not before.failed
    assert before.state.damage == pytest.approx(5e-21, rel=1e-14, abs=0)
    after = law.update(1., 2e-20)
    assert after.failed and after.state.damage == law.failure_damage
    assert after.equivalent_increment == pytest.approx(2e-20, rel=1e-14, abs=0)


def test_update_clamps_at_same_failure_state_under_subdivision():
    law = KachanovRabotnovCreep(2e-5, 2., 1e-4, 3., 3., failure_damage=.9)
    life = law.rupture_time(2.)
    direct = law.update(2., life * 2)
    split = law.update(2., life * .5)
    split = law.update(2., life * 1.5, split.state)
    assert direct.failed and split.failed
    assert direct.state.damage == split.state.damage == .9
    assert split.state.equivalent_creep_strain == pytest.approx(direct.state.equivalent_creep_strain, rel=1e-13)


def test_reference_scales_preserve_same_physical_interval():
    hours = KachanovRabotnovCreep(1.6e-31, 12.7, 2.6e-26, 10.3, 16.)
    seconds_pa = replace(hours, reference_stress=1e6, reference_time=3600.)
    for dt in (1e-12, 1., 100.):
        first = hours.update(110., dt)
        second = seconds_pa.update(110e6, dt * 3600.)
        assert second.equivalent_increment == pytest.approx(first.equivalent_increment, rel=1e-14, abs=0)
        assert second.state.damage == pytest.approx(first.state.damage, rel=1e-14, abs=0)
    assert seconds_pa.rupture_time(110e6) == pytest.approx(hours.rupture_time(110.) * 3600.)


@pytest.mark.parametrize("exponent", [-.999999, -.5, 0., .5, 2.])
@pytest.mark.parametrize("interval", [(0., 3.), (1., 4.), (1e10, 1e10 + 1e-5), (1e-100, 1e100)])
def test_power_law_short_and_wide_intervals(exponent, interval):
    law = PowerLawCreep(1e-12, 3., time_exponent=exponent, reference_time=3600.)
    start, end = interval
    with localcontext() as ctx:
        ctx.prec = 90
        # Use the actually representable endpoints, not a reconstructed decimal
        # duration: a solver cannot recover time digits absent from its inputs.
        d = Decimal.from_float
        p = d(exponent) + 1
        reference = (d(law.coefficient) * 100**3 * d(law.reference_time)
                     * ((d(end) / d(law.reference_time))**p
                        - (d(start) / d(law.reference_time))**p) / p)
    actual = law.constant_stress_increment(100., start, end)
    assert actual == pytest.approx(float(reference), rel=3e-14, abs=0)


def test_power_law_large_clock_tensor_direction_and_subdivision():
    law = PowerLawCreep(1e-12, 3.)
    start, end = 1e10, 1e10 + 1e-5
    expected = law.coefficient * 100**3 * (end - start)
    middle = start + (end - start) / 2
    split = (law.constant_stress_increment(100., start, middle)
             + law.constant_stress_increment(100., middle, end))
    assert split == pytest.approx(expected, rel=1e-14, abs=0)
    np.testing.assert_allclose(
        law.tensor_increment(np.diag([100., 0., 0.]), start, end),
        expected * np.diag([1., -.5, -.5]), rtol=1e-14, atol=0,
    )


@pytest.mark.parametrize("start,end", [(-1., 0.), (2., 1.), (float("nan"), 1.), (0., float("inf"))])
def test_power_law_interval_rejects_invalid_clock(start, end):
    with pytest.raises(ValueError):
        PowerLawCreep(1., 2.).constant_stress_increment(1., start, end)


def test_power_law_zero_interval_is_identity_at_singular_initial_rate():
    assert PowerLawCreep(1., 2., time_exponent=-.5).constant_stress_increment(1., 0., 0.) == 0.
