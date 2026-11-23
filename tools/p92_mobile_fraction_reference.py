"""Research mobility population, not a published or validated P92 material."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_recoverable_spectrum as spectrum
from tools.p92_recoverable_mobility_reference import Material as NeutralMaterial

MODEL_ID = "research.scalar-mobile-fraction.v1"
STATE_LAYOUT = (*spectrum.STATE_LAYOUT, "immobilized_fraction")


@dataclass(frozen=True)
class Material:
    spectrum: spectrum.Material
    immobilization: float
    remobilization: float

    def __post_init__(self):
        NeutralMaterial(self.spectrum)
        if (not np.isfinite([self.immobilization, self.remobilization]).all()
                or min(self.immobilization, self.remobilization) < 0):
            raise ValueError("Require finite nonnegative population rates")


def stored_energy(state, material):
    return spectrum.stored_energy(state[:12], material.spectrum)


def rates(state, strain_rate, material, *, stress_control=False):
    # Clamp only the RHS's extension outside the invariant interval for implicit
    # solver trials. Accepted states are independently checked by integrate.
    fraction = np.clip(state[12], 0., 1.)
    base = material.spectrum.backbone
    result = spectrum.rates(state[:12], strain_rate, material.spectrum, stress_control=stress_control)
    flow = dissipation = 0.
    # Isolate terms proportional to irreversible speed. Static recovery and
    # Kelvin motion remain active even at exactly zero mobile population.
    for offset, branch in zip((1, 4), base.branches, strict=True):
        effective = state[0]-sum(state[offset:offset+2])
        speed = result[offset+2]
        signed = speed*np.sign(effective)
        flow += signed
        dissipation += effective*signed
        for i, (h, limit, static, power) in enumerate(zip(branch.hardening,
                branch.saturation, branch.static_recovery, branch.ratcheting, strict=True)):
            x = state[offset+i]
            decay = h/limit*(abs(x)/limit)**power*speed
            dissipation += decay*x*x/h
            result[offset+i] = (1.-fraction)*(result[offset+i]+static*x)-static*x
        result[offset+2] *= 1.-fraction
    if stress_control:
        result[7] -= fraction*flow
    else:
        result[0] += base.modulus*fraction*flow
    result[8] = state[0]*result[7]
    result[9] -= fraction*dissipation
    population = material.immobilization*(1.-fraction)-material.remobilization*(result[3]+result[6])*fraction
    return np.r_[result, population]


def integrate(old, duration, strain_rate, material, *, stress_control=False,
              method="Radau", rtol=1e-9):
    if np.iscomplexobj(old):
        raise ValueError("Require real state")
    state = np.asarray(old, dtype=float)
    if (state.shape != (13,) or not np.isfinite(state).all()
            or np.any(state[[3, 6, 9]] < 0) or not 0 <= state[12] <= 1
            or type(stress_control) is not bool or method not in {"Radau", "BDF"}
            or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid mobile-fraction state or controls")
    for offset, branch in zip((1, 4), material.spectrum.backbone.branches, strict=True):
        if np.any(abs(state[offset:offset+2]) > np.asarray(branch.saturation)+1e-7):
            raise ValueError("Initial backstress outside invariant domain")
    sol = solve_ivp(lambda t, y: rates(y, strain_rate, material, stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13, 1e-12])
    if not sol.success or not np.isfinite(sol.y).all():
        raise RuntimeError("Mobile-fraction integration failed; input retained")
    tol = 10*(1e-12+rtol)
    if np.any(sol.y[12] < -tol) or np.any(sol.y[12] > 1.+tol):
        raise RuntimeError("Mobile-fraction invariant violated")
    for offset, branch in zip((1, 4), material.spectrum.backbone.branches, strict=True):
        if np.any(abs(sol.y[offset:offset+2]) > np.asarray(branch.saturation)[:, None]+1e-6):
            raise RuntimeError("Backstress invariant violated")
    accumulated = sol.y[[3, 6, 9]]
    tolerance = 10*(np.array([1e-13, 1e-13, 1e-11])[:, None]+rtol*abs(accumulated))
    if np.any(np.maximum.accumulate(accumulated, axis=1)-accumulated > tolerance):
        raise RuntimeError("Accumulated state decreased")
    end = sol.y[:, -1]
    balance = end[8]-state[8]-stored_energy(end, material)+stored_energy(state, material)-end[9]+state[9]
    if abs(balance) > 1e-7+1e-7*abs(end[8]-state[8]):
        raise RuntimeError("Mechanical balance failed")
    return sol
