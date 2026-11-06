"""Bounded, as-received P92 equation audit, not an engineering material card.

Source: doi:10.1016/j.tafmec.2019.102269, equations 14--21 and Table 2.
Fixed 650 C, positive constant stress, initial damage zero, initial sigma0=3 MPa.
Lengths use the manuscript's numerical micrometre convention. Its K_L and A
unit labels are inconsistent with the equations. Equation 15 omits the stress
factor present in equation 9; both readings are exposed, never fitted away.
"""

import json
import math

from scipy.integrate import solve_ivp

CONVENTIONS = ("eq9_stress_scaled", "eq15_literal_diagnostic")


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def states(strain, time_h, *, lath_um=0.375, density_ratio=1.0):
    """Exact reductions with explicit prior microstructure and zero creep damage."""
    for value in (strain, time_h):
        if not math.isfinite(value) or value < 0:
            raise ValueError("strain and time_h must be finite and nonnegative")
    return _state_values(strain, time_h, lath_um, density_ratio)


def _state_values(strain, time_h, lath_um, density_ratio):
    # Smooth mathematical extension for implicit integrator trial iterates.
    # No clipping: an accepted negative physical time is checked separately.
    if not math.isfinite(time_h) or 1 + 1.5e-5 * time_h <= 0:
        raise ValueError("Time outside the analytical precipitate-state domain")
    return (6.2 - 3.2 * math.exp(-2400 * strain / (density_ratio * 6.2)),
            strain / (lath_um + strain),
            -math.expm1(-math.log1p(1.5e-5 * time_h) / 3),
            (1.5 + 6.418 * (lath_um - 0.375)) * strain)


def inverse_rate(strain, time_h, stress_mpa, convention, *, lath_um=0.375, density_ratio=1.0):
    """dt/depsilon, evaluated without overflow as the damage endpoint approaches."""
    if convention not in CONVENTIONS:
        raise ValueError("Unknown backstress convention")
    sigma0, dl, dp, dn = _state_values(strain, time_h, lath_um, density_ratio)
    if dn >= 1:
        raise ValueError("D_N must remain below one")
    backstress = 0.5 * (1 - dl)
    if convention == "eq9_stress_scaled":
        backstress *= stress_mpa
    x = (stress_mpa - backstress) / (sigma0 * (1 - dp) * (1 - dn))
    if x <= 0:
        raise ValueError("Strain-coordinate integration requires positive creep rate")
    log_inverse = math.log(2 / 8e-10) - x - math.log(-math.expm1(-2 * x))
    return math.exp(log_inverse)


def integrate(*, stress_mpa=130.0, convention="eq9_stress_scaled",
              damage_endpoint=1 - 1e-8, rtol=1e-9, atol_h=1e-9, method="DOP853",
              lath_um=0.375, density_ratio=1.0):
    """Integrate to a declared damage endpoint; success does not define rupture."""
    for value, name in ((stress_mpa, "stress_mpa"), (rtol, "rtol"),
                        (atol_h, "atol_h"), (damage_endpoint, "damage_endpoint"),
                        (lath_um, "lath_um"), (density_ratio, "density_ratio")):
        _positive(value, name)
    if lath_um < 0.375 or density_ratio > 1:
        raise ValueError("Prior recovery requires lath_um >= 0.375 and density_ratio <= 1")
    if damage_endpoint >= 1:
        raise ValueError("damage_endpoint must be below one")
    if convention not in CONVENTIONS:
        raise ValueError("Unknown backstress convention")
    if method not in ("DOP853", "Radau", "RK45"):
        raise ValueError("Unsupported reference integrator")
    endpoint = damage_endpoint / (1.5 + 6.418 * (lath_um - 0.375))
    solution = solve_ivp(lambda e, t: [inverse_rate(e, float(t[0]), stress_mpa, convention,
                                                  lath_um=lath_um, density_ratio=density_ratio)],
                         (0, endpoint), [0.0], method=method, rtol=rtol, atol=atol_h)
    if not solution.success or solution.t[-1] != endpoint:
        raise RuntimeError(f"Integration failed; not a rupture prediction: {solution.message}")
    if any(t < 0 for t in solution.y[0]) or any(b < a - atol_h for a, b in zip(solution.y[0][:-1], solution.y[0][1:])):
        raise RuntimeError("Accepted physical time must be nonnegative and nondecreasing")
    time_h = float(solution.y[0, -1])
    if not math.isfinite(time_h) or time_h <= 0:
        raise RuntimeError("Invalid endpoint time")
    return dict(schema="agentfem.research.p92-equation-reference.v1", convention=convention,
                stress_mpa=stress_mpa, temperature_c=650, time_h=time_h,
                lath_um=lath_um, density_ratio=density_ratio,
                strain=endpoint, damage_endpoint=damage_endpoint, evaluations=solution.nfev,
                termination="declared_damage_endpoint", method=method, rtol=rtol, atol_h=atol_h,
                engineering_qualified=False, fitted=False,
                assumptions=["zero initial creep damage", "constant stress",
                             "supplied microstructure, not inferred from observed lifetime",
                             "Table 2 numerical parameters",
                             "initial sigma0=3 MPa", "epsilon_f=0.2",
                             "damage endpoint is a numerical convention, not a validated failure criterion"],
                source="https://doi.org/10.1016/j.tafmec.2019.102269")


if __name__ == "__main__":
    print(json.dumps([integrate(convention=c) for c in CONVENTIONS], indent=2, allow_nan=False))
