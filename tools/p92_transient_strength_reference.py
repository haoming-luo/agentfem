"""Research-only bounded shared isotropic hardening, with recoverability."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_recoverable_spectrum as spectrum

MODEL_ID = "research.scalar-two-rate-transient-strength.v1"
STATE_LAYOUT = (*spectrum.STATE_LAYOUT, "shared_strength_increment")


@dataclass(frozen=True)
class Material:
    spectrum: spectrum.Material
    maximum: float
    build: float
    recovery: float

    def __post_init__(self):
        if (not np.isfinite([self.maximum, self.build, self.recovery]).all()
                or not 0 <= self.maximum <= 1 or min(self.build, self.recovery) < 0):
            raise ValueError("Require bounded nonnegative strength kinetics")
        if (self.spectrum.strength_ratios != (1., 1.) or self.spectrum.hardening_ratios != (1., 1.)
                or self.spectrum.recoverable_powers != (1., 1.)
                or self.spectrum.static_powers != (0., 0.) or self.spectrum.static_ratios != (1., 1.)
                or self.spectrum.directional_recovery
                or any(b.saturation_ratio != 1. for b in self.spectrum.backbone.branches)):
            raise ValueError("Require isolated magnitude-recovery spectrum")


def stored_energy(state, material):
    return spectrum.stored_energy(state[:12], material.spectrum)


def rates(state, strain_rate, material, *, stress_control=False):
    h = state[12]
    backbone = material.spectrum.backbone
    current = replace(material.spectrum, backbone=replace(backbone,
        branches=tuple(replace(b, strength=b.strength*(1+h)) for b in backbone.branches)))
    r = spectrum.rates(state[:12], strain_rate, current, stress_control=stress_control)
    return np.r_[r, material.build*(r[3]+r[6])*(material.maximum-h)-material.recovery*h]


def integrate(old, duration, strain_rate, material, *, stress_control=False,
              method="Radau", rtol=1e-9):
    if np.iscomplexobj(old):
        raise ValueError("Require real state")
    state = np.asarray(old, dtype=float)
    if (state.shape != (13,) or not np.isfinite(state).all()
            or np.any(state[[3, 6, 9]] < 0) or not 0 <= state[12] <= material.maximum
            or type(stress_control) is not bool or method not in {"Radau", "BDF"}
            or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid transient-strength state or controls")
    for offset, branch in zip((1, 4), material.spectrum.backbone.branches, strict=True):
        if np.any(abs(state[offset:offset+2]) > np.asarray(branch.saturation)+1e-7):
            raise ValueError("Initial backstress outside invariant domain")
    sol = solve_ivp(lambda t, y: rates(y, strain_rate, material, stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13, 1e-12])
    if not sol.success or not np.isfinite(sol.y).all():
        raise RuntimeError("Transient-strength integration failed; input retained")
    tol = 10*(1e-12+rtol*material.maximum)
    if np.any(sol.y[12] < -tol) or np.any(sol.y[12] > material.maximum+tol):
        raise RuntimeError("Strength invariant violated")
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
