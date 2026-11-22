"""Research-only one irreversible flow and three recoverable modes."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from tools.p92_two_rate_reference import Branch

MODEL_ID = "research.scalar-single-flow-three-recovery.v1"
STATE_LAYOUT = ("stress_MPa", "backstress_B1_MPa", "backstress_B2_MPa", "accumulated_B",
                "recoverable_1_strain", "recoverable_2_strain", "recoverable_3_strain",
                "strain", "work_density_MPa", "dissipation_density_MPa")


@dataclass(frozen=True)
class Material:
    modulus: float
    branch: Branch
    moduli: tuple[float, float, float]
    times: tuple[float, float, float]

    def __post_init__(self):
        if (len(self.moduli) != 3 or len(self.times) != 3
                or not np.isfinite([self.modulus, *self.moduli, *self.times]).all()
                or min(self.modulus, *self.moduli, *self.times) <= 0
                or self.branch.saturation_ratio != 1.):
            raise ValueError("Require positive elastic/recovery constants and fixed saturation")


def replace_low_stress_flow(spectrum):
    if (spectrum.strength_ratios != (1., 1.) or spectrum.hardening_ratios != (1., 1.)
            or spectrum.static_powers != (0., 0.) or spectrum.static_ratios != (1., 1.)
            or spectrum.recoverable_powers != (1., 1.) or spectrum.directional_recovery
            or any(b.saturation_ratio != 1. for b in spectrum.backbone.branches)):
        raise ValueError("Require isolated original spectrum")
    a, b = spectrum.backbone.branches
    if a.strength >= 120.:
        raise ValueError("Reference stress must activate the replaced flow")
    h = sum(a.hardening)
    speed = a.reference_flow*((120.-a.strength)/120.)**a.exponent
    return Material(spectrum.backbone.modulus, b, (*spectrum.moduli, h),
                    (*spectrum.times, 120./(h*speed)))


def stored_energy(state, material):
    return float(state[0]**2/(2*material.modulus)
        +sum(state[i+1]**2/(2*h) for i, h in enumerate(material.branch.hardening))
        +sum(h*a*a/2 for h, a in zip(material.moduli, state[4:7], strict=True)))


def rates(state, strain_rate, material, *, stress_control=False):
    b = material.branch
    effective = state[0]-sum(state[1:3])
    speed = b.reference_flow*(max(abs(effective)-b.strength, 0.)/120.)**b.exponent
    flow = speed*np.sign(effective)
    result = np.zeros(10)
    dissipation = effective*flow
    for i, (h, a, r, chi) in enumerate(zip(b.hardening, b.saturation, b.static_recovery,
                                           b.ratcheting, strict=True)):
        x = state[i+1]
        decay = h/a*(abs(x)/a)**chi*speed+r
        result[i+1] = h*flow-decay*x
        dissipation += decay*x*x/h
    result[3] = speed
    result[4:7] = (state[0]/np.asarray(material.moduli)-state[4:7])/material.times
    dissipation += sum(np.asarray(material.moduli)*material.times*result[4:7]**2)
    total = flow+sum(result[4:7])
    result[7] = total if stress_control else strain_rate
    result[0] = 0. if stress_control else material.modulus*(strain_rate-total)
    result[8], result[9] = state[0]*result[7], dissipation
    return result


def integrate(old, duration, strain_rate, material, *, stress_control=False,
              method="Radau", rtol=1e-9):
    if np.iscomplexobj(old):
        raise ValueError("Require real state")
    state = np.asarray(old, dtype=float)
    if (state.shape != (10,) or not np.isfinite(state).all() or np.any(state[[3, 9]] < 0)
            or type(stress_control) is not bool or method not in {"Radau", "BDF"}
            or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid replacement state or controls")
    saturation = np.asarray(material.branch.saturation)
    if np.any(abs(state[1:3]) > saturation+1e-7):
        raise ValueError("Initial backstress outside invariant domain")
    sol = solve_ivp(lambda t, y: rates(y, strain_rate, material, stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13, 1e-11, 1e-11])
    if not sol.success or not np.isfinite(sol.y).all():
        raise RuntimeError("Replacement integration failed; input retained")
    if np.any(abs(sol.y[1:3]) > saturation[:, None]+1e-6):
        raise RuntimeError("Backstress invariant violated")
    accumulated = sol.y[[3, 9]]
    tolerance = 10*(np.array([1e-13, 1e-11])[:, None]+rtol*abs(accumulated))
    if np.any(np.maximum.accumulate(accumulated, axis=1)-accumulated > tolerance):
        raise RuntimeError("Accumulated state decreased")
    end = sol.y[:, -1]
    balance = end[8]-state[8]-stored_energy(end, material)+stored_energy(state, material)-end[9]+state[9]
    if abs(balance) > 1e-7+1e-7*abs(end[8]-state[8]):
        raise RuntimeError("Mechanical balance failed")
    return sol
