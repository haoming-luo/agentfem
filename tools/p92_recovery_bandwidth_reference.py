"""Research-only linear recovery bandwidth, retaining two irreversible flows."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_recoverable_spectrum as spectrum

MODEL_ID = "research.scalar-two-rate-recovery-bandwidth.v1"
STATE_LAYOUT = (*spectrum.STATE_LAYOUT, "recoverable_intermediate_strain")


@dataclass(frozen=True)
class Material:
    spectrum: spectrum.Material
    extra_modulus: float
    extra_time: float

    def __post_init__(self):
        if (not np.isfinite([self.extra_modulus, self.extra_time]).all()
                or min(self.extra_modulus, self.extra_time) <= 0):
            raise ValueError("Require positive recovery modulus and time")
        s = self.spectrum
        if not s.times[0] <= self.extra_time <= s.times[1]:
            raise ValueError("Require the extra time between the existing fast and slow times")
        if (s.strength_ratios != (1., 1.) or s.hardening_ratios != (1., 1.)
                or s.static_powers != (0., 0.) or s.static_ratios != (1., 1.)
                or s.recoverable_powers != (1., 1.) or s.directional_recovery
                or any(b.saturation_ratio != 1 for b in s.backbone.branches)):
            raise ValueError("Require isolated linear-recovery baseline")


def split_slow(material, spread):
    if not np.isfinite(spread) or not 0 <= spread < 1:
        raise ValueError("Require spread in [0,1)")
    h, tau = material.moduli[1], material.times[1]
    return Material(replace(material, moduli=(material.moduli[0], 2*h),
                            times=(material.times[0], tau/(1-spread))),
                    2*h, tau/(1+spread))


def stored_energy(state, material):
    return spectrum.stored_energy(state[:12], material.spectrum)+material.extra_modulus*state[12]**2/2


def rates(state, strain_rate, material, *, stress_control=False):
    rate = (state[0]/material.extra_modulus-state[12])/material.extra_time
    result = np.r_[spectrum.rates(state[:12], strain_rate, material.spectrum,
                                 stress_control=stress_control), rate]
    if stress_control:
        result[7] += rate
    else:
        result[0] -= material.spectrum.backbone.modulus*rate
    result[8] = state[0]*result[7]
    result[9] += material.extra_modulus*material.extra_time*rate**2
    return result


def integrate(old, duration, strain_rate, material, *, stress_control=False,
              method="Radau", rtol=1e-9):
    if np.iscomplexobj(old):
        raise ValueError("Require real state")
    state = np.asarray(old, dtype=float)
    if (state.shape != (13,) or not np.isfinite(state).all()
            or np.any(state[[3, 6, 9]] < 0) or type(stress_control) is not bool
            or method not in {"Radau", "BDF"}
            or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid recovery-bandwidth state or controls")
    for offset, branch in zip((1, 4), material.spectrum.backbone.branches, strict=True):
        if np.any(abs(state[offset:offset+2]) > np.asarray(branch.saturation)+1e-7):
            raise ValueError("Initial backstress outside invariant domain")
    sol = solve_ivp(lambda t, y: rates(y, strain_rate, material, stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13, 1e-13])
    if not sol.success or not np.isfinite(sol.y).all():
        raise RuntimeError("Recovery-bandwidth integration failed; input retained")
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
