"""Research-only bounded kinetic resistance interaction, not a P92 card."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_recoverable_spectrum as spectrum

MODEL_ID = "research.scalar-two-rate-cross-drag.v1"
STATE_LAYOUT = (*spectrum.STATE_LAYOUT, "kinetic_resistance_history")


@dataclass(frozen=True)
class Material:
    spectrum: spectrum.Material
    maximum: float
    build: float
    erase: float
    recovery: float

    def __post_init__(self):
        if (not np.isfinite([self.maximum, self.build, self.erase, self.recovery]).all()
                or min(self.maximum, self.build, self.erase, self.recovery) < 0
                or self.maximum > 100):
            raise ValueError("Require finite nonnegative bounded kinetic parameters")


def stored_energy(state, material):
    # h modifies kinetics only. No unaccounted hardening energy is introduced.
    return spectrum.stored_energy(state[:12], material.spectrum)


def rates(state, strain_rate, material, *, stress_control=False):
    h = state[12]
    a, b = material.spectrum.backbone.branches
    modified = replace(material.spectrum, backbone=replace(material.spectrum.backbone,
        branches=(a, replace(b, reference_flow=b.reference_flow*np.exp(-h)))))
    rate = spectrum.rates(state[:12], strain_rate, modified, stress_control=stress_control)
    history = material.build*rate[3]*(material.maximum-h)-(material.erase*rate[6]+material.recovery)*h
    return np.r_[rate, history]


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
        raise ValueError("Invalid cross-drag state or controls")
    for offset, branch in zip((1, 4), material.spectrum.backbone.branches, strict=True):
        if np.any(abs(state[offset:offset+2]) > np.asarray(branch.saturation)+1e-7):
            raise ValueError("Initial backstress outside invariant domain")
    solution = solve_ivp(lambda t, y: rates(y, strain_rate, material, stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13, 1e-12])
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("Cross-drag integration failed; input retained")
    for offset, branch in zip((1, 4), material.spectrum.backbone.branches, strict=True):
        if np.any(abs(solution.y[offset:offset+2]) > np.asarray(branch.saturation)[:, None]+1e-6):
            raise RuntimeError("Backstress invariant violated")
    history = solution.y[12]
    tolerance_h = 10*(1e-12+rtol*material.maximum)
    if np.any(history < -tolerance_h) or np.any(history > material.maximum+tolerance_h):
        raise RuntimeError("Kinetic history left its invariant interval")
    accumulated = solution.y[[3, 6, 9]]
    tolerance = 10*(np.array([1e-13, 1e-13, 1e-11])[:, None]+rtol*abs(accumulated))
    if np.any(np.maximum.accumulate(accumulated, axis=1)-accumulated > tolerance):
        raise RuntimeError("Accumulated state decreased")
    end = solution.y[:, -1]
    balance = end[8]-state[8]-stored_energy(end, material)+stored_energy(state, material)-end[9]+state[9]
    if abs(balance) > 1e-7+1e-7*abs(end[8]-state[8]):
        raise RuntimeError("Integrated mechanical balance failed")
    return solution
