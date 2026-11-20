"""Uncalibrated isothermal scalar two-rate reference with mechanical accounting.

State: sigma, x_A1, x_A2, p_A, x_B1, x_B2, p_B, strain, work, dissipation.
MPa, seconds, dimensionless strain; work/energy densities are MPa.
This layout is NOT compatible with the earlier P92 two-mechanism state.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


MODEL_ID = "research.scalar-two-rate-kinematic.v2"
STATE_LAYOUT = ("stress_MPa", "backstress_A1_MPa", "backstress_A2_MPa", "accumulated_A",
                "backstress_B1_MPa", "backstress_B2_MPa", "accumulated_B", "strain",
                "work_density_MPa", "dissipation_density_MPa")


@dataclass(frozen=True)
class Branch:
    strength: float
    reference_flow: float
    exponent: float
    hardening: tuple[float, float]
    saturation: tuple[float, float]
    static_recovery: tuple[float, float]
    ratcheting: tuple[float, float] = (3., 3.)
    saturation_ratio: float = 1.
    softening_rate: float = 0.

    def __post_init__(self):
        pairs = (self.hardening, self.saturation, self.static_recovery, self.ratcheting)
        if (any(len(v) != 2 for v in pairs)
                or not np.isfinite([self.strength, self.reference_flow, self.exponent,
                                   self.saturation_ratio, self.softening_rate,
                                   *(x for p in pairs for x in p)]).all()
                or self.strength < 0 or self.reference_flow <= 0 or self.exponent < 1
                or min(*self.hardening, *self.saturation) <= 0
                or min(*self.static_recovery, *self.ratcheting) < 0
                or not 0 < self.saturation_ratio <= 1 or self.softening_rate < 0):
            raise ValueError("Invalid scalar two-rate branch")


@dataclass(frozen=True)
class Material:
    modulus: float
    branches: tuple[Branch, Branch]

    def __post_init__(self):
        if not np.isfinite(self.modulus) or self.modulus <= 0 or len(self.branches) != 2:
            raise ValueError("Require positive shared modulus and two branches")


def stored_energy(state, material):
    value = state[0]**2/(2*material.modulus)
    for offset, branch in zip((1, 4), material.branches, strict=True):
        value += sum(state[offset+i]**2/(2*h) for i, h in enumerate(branch.hardening))
    return float(value)


def saturation_at(branch, accumulated):
    # Protect solver trial stages below p=0 without modifying accepted history.
    factor = branch.saturation_ratio+(1-branch.saturation_ratio)*np.exp(
        -branch.softening_rate*max(float(accumulated), 0.))
    return np.asarray(branch.saturation)*factor


def rates(state, strain_rate, material, *, stress_control=False):
    result = np.zeros(10)
    total_flow = dissipation = 0.
    for offset, branch in zip((1, 4), material.branches, strict=True):
        effective = state[0]-sum(state[offset:offset+2])
        speed = branch.reference_flow*(max(abs(effective)-branch.strength, 0.)/120.)**branch.exponent
        flow = speed*np.sign(effective)
        total_flow += flow
        dissipation += effective*flow
        for i, (h, a, recovery, chi) in enumerate(zip(branch.hardening, saturation_at(branch, state[offset+2]),
                                                    branch.static_recovery, branch.ratcheting, strict=True)):
            x = state[offset+i]
            decay = h/a*(abs(x)/a)**chi*speed+recovery
            result[offset+i] = h*flow-decay*x
            dissipation += decay*x*x/h
        result[offset+2] = speed
    result[7] = total_flow if stress_control else strain_rate
    result[0] = 0. if stress_control else material.modulus*(strain_rate-total_flow)
    result[8] = state[0]*result[7]
    result[9] = dissipation
    return result


def integrate(old, duration, strain_rate, material, *, stress_control=False,
              method="Radau", rtol=1e-9):
    if np.iscomplexobj(old):
        raise ValueError("Require real state")
    state = np.asarray(old, dtype=float)
    if (state.shape != (10,) or not np.isfinite(state).all() or np.any(state[[3, 6, 9]] < 0)
            or type(stress_control) is not bool or method not in {"Radau", "BDF"}
            or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid scalar reference state or controls")
    for offset, branch in zip((1, 4), material.branches, strict=True):
        if np.any(abs(state[offset:offset+2]) > np.asarray(branch.saturation)+1e-7):
            raise ValueError("Initial backstress lies outside its invariant domain")
    solution = solve_ivp(lambda t, y: rates(y, strain_rate, material, stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11])
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("Reference integration failed; input state retained")
    for offset, branch in zip((1, 4), material.branches, strict=True):
        if np.any(abs(solution.y[offset:offset+2]) > np.asarray(branch.saturation)[:, None]+1e-6):
            raise RuntimeError("Backstress invariant violated")
    accumulated = solution.y[[3, 6, 9]]
    # BDF's local error target does not imply exact plateau monotonicity.
    # Reserve ten local-error scales for this guard; never clip the trajectory.
    # Separate split-path and Radau/BDF response checks remain required.
    tolerance = 10*(np.array([1e-13, 1e-13, 1e-11])[:, None] + rtol*np.abs(accumulated))
    if np.any(np.maximum.accumulate(accumulated, axis=1)-accumulated > tolerance):
        raise RuntimeError("Accumulated state decreased")
    end = solution.y[:, -1]
    balance = (end[8]-state[8])-(stored_energy(end, material)-stored_energy(state, material))-(end[9]-state[9])
    if abs(balance) > 1e-7+1e-7*abs(end[8]-state[8]):
        raise RuntimeError("Integrated mechanical balance failed")
    return solution
