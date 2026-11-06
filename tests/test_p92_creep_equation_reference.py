import math

import pytest
from scipy.integrate import solve_ivp

from tools.p92_creep_equation_reference import integrate, inverse_rate, states


def test_exact_state_reductions_satisfy_original_equations():
    e, t, h = 0.05, 100.0, 1e-6
    s, dl, dp, dn = states(e, t)
    plus, minus = states(e + h, t), states(e - h, t)
    derivatives = [(a - b) / (2 * h) for a, b in zip(plus, minus)]
    assert derivatives[0] == pytest.approx(2400 * (1 - s / 6.2), abs=1e-8)
    assert derivatives[1] == pytest.approx((1 - dl)**2 / 0.375)
    assert derivatives[3] == pytest.approx(1.5)
    assert (states(e, t + h)[2] - states(e, t - h)[2]) / (2 * h) == pytest.approx(1.5e-5 / 3 * (1 - dp)**4)
    assert states(0, 0) == pytest.approx((3, 0, 0, 0))


def test_independent_time_domain_system_agrees_before_singularity():
    # Directly integrate all five original state equations, not the reductions.
    def rhs(t, y):
        e, s, dl, dp, dn = y
        rate = 8e-10 * math.sinh((130 - 65 * (1 - dl)) / (s * (1 - dp) * (1 - dn)))
        return [rate, 2400 * (1 - s / 6.2) * rate,
                (1 - dl)**2 / 0.375 * rate, 1.5e-5 / 3 * (1 - dp)**4, 1.5 * rate]
    def event(t, y):
        return y[4] - 0.3
    event.terminal = True
    event.direction = 1
    direct = solve_ivp(rhs, (0, 2000), [0, 3, 0, 0, 0], method="Radau",
                       events=event, rtol=1e-9, atol=1e-11)
    assert direct.success and len(direct.t_events[0]) == 1
    reduced = integrate(damage_endpoint=0.3)
    assert reduced["time_h"] == pytest.approx(direct.t[-1], rel=1e-7)
    assert direct.y[1:, -1] == pytest.approx(states(direct.y[0, -1], direct.t[-1]), rel=1e-7)


def test_methods_cutoffs_and_tolerances_converge_without_claiming_source_reproduction():
    reference = integrate()
    for options in ({"method": "Radau"}, {"damage_endpoint": 0.99}, {"rtol": 1e-7, "atol_h": 1e-7}):
        assert integrate(**options)["time_h"] == pytest.approx(reference["time_h"], rel=1e-6)
    assert reference["time_h"] == pytest.approx(1188.0432345, rel=1e-8)
    assert abs(reference["time_h"] / 1146 - 1) > 0.03  # Retain unresolved manuscript discrepancy.
    assert not reference["engineering_qualified"] and not reference["fitted"]
    assert reference["termination"] == "declared_damage_endpoint"


def test_equation_disagreement_is_visible_and_large():
    assert integrate(convention="eq15_literal_diagnostic")["time_h"] < 0.05
    assert inverse_rate(0.66, 1188, 130, "eq9_stress_scaled") == 0  # benign underflow


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_inputs(value):
    with pytest.raises(ValueError):
        integrate(stress_mpa=value)


def test_invalid_endpoint_and_convention():
    with pytest.raises(ValueError):
        integrate(damage_endpoint=1)
    with pytest.raises(ValueError):
        integrate(convention="best_fit")


@pytest.mark.parametrize("parameters", [{"density_ratio": 0}, {"density_ratio": 1.1},
                                       {"lath_um": 0.1}, {"lath_um": float("nan")}])
def test_invalid_prior_microstructure(parameters):
    with pytest.raises(ValueError):
        integrate(**parameters)


def test_conditional_prior_state_reduction_and_solver_agreement():
    parameters = dict(lath_um=0.766, density_ratio=0.5)
    e, t, h = 0.002, 100.0, 1e-8
    s, dl, dp, dn = states(e, t, **parameters)
    plus, minus = states(e + h, t, **parameters), states(e - h, t, **parameters)
    derivatives = [(a - b) / (2 * h) for a, b in zip(plus, minus)]
    assert derivatives[0] == pytest.approx(2400 / 0.5 * (1 - s / 6.2))
    assert derivatives[1] == pytest.approx((1 - dl)**2 / 0.766)
    assert derivatives[3] == pytest.approx(1.5 + 6.418 * (0.766 - 0.375))
    result = integrate(**parameters)
    assert result["time_h"] == pytest.approx(integrate(method="Radau", **parameters)["time_h"], rel=1e-6)
    assert result["density_ratio"] == 0.5
    assert not result["engineering_qualified"]
