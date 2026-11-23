"""Reduced rate-target resistance memory; research hypothesis, not a P92 card.

Inspired by aging-stress evolution, DOI 10.1299/mej.24-00369, Eqs. 19--21.
The shared normalization and equal build/decay rates are our reductions, not
the published P91 model. No damage or recoverable strain is removed.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_recoverable_spectrum as spectrum
from tools.p92_recoverable_mobility_reference import Material as NeutralMaterial

MODEL_ID = "research.scalar-rate-target-resistance.v1"
STATE_LAYOUT = (*spectrum.STATE_LAYOUT, "normalized_aging_resistance_increment")


@dataclass(frozen=True)
class Material:
    spectrum: spectrum.Material
    sensitivity: float
    build: float

    def __post_init__(self):
        NeutralMaterial(self.spectrum)
        if not np.isfinite([self.sensitivity, self.build]).all() or min(self.sensitivity, self.build) < 0:
            raise ValueError("Require nonnegative finite resistance-memory coefficients")


def stored_energy(state, material):
    return spectrum.stored_energy(state[:12], material.spectrum)


def memory_rate(speed, increment, reference, sensitivity, build):
    if speed == 0.:
        return 0.  # lim v log(v) = 0; do not introduce an arbitrary rate floor.
    target = max(-1., sensitivity*(np.log10(reference)-np.log10(speed)))
    return build*speed*(target-increment)


def rates(state, strain_rate, material, *, stress_control=False):
    factor = max(0., 1.+state[12])
    base = material.spectrum.backbone
    current = replace(material.spectrum, backbone=replace(base,
        branches=tuple(replace(b, strength=b.strength*factor) for b in base.branches)))
    result = spectrum.rates(state[:12], strain_rate, current, stress_control=stress_control)
    speed = result[3]+result[6]
    return np.r_[result, memory_rate(speed, state[12], sum(b.reference_flow for b in base.branches),
                                    material.sensitivity, material.build)]


def integrate(old, duration, strain_rate, material, *, stress_control=False,
              method="Radau", rtol=1e-9):
    if np.iscomplexobj(old):
        raise ValueError("Require real state")
    state = np.asarray(old, dtype=float)
    if (state.shape != (13,) or not np.isfinite(state).all()
            or np.any(state[[3, 6, 9]] < 0) or state[12] < -1.
            or type(stress_control) is not bool or method not in {"Radau", "BDF"}
            or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid resistance-memory state or controls")
    for offset, branch in zip((1, 4), material.spectrum.backbone.branches, strict=True):
        if np.any(abs(state[offset:offset+2]) > np.asarray(branch.saturation)+1e-7):
            raise ValueError("Initial backstress outside invariant domain")
    sol = solve_ivp(lambda t, y: rates(y, strain_rate, material, stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13, 1e-12])
    if not sol.success or not np.isfinite(sol.y).all():
        raise RuntimeError("Resistance-memory integration failed; input retained")
    if np.any(sol.y[12] < -1.-10*(1e-12+rtol)):
        raise RuntimeError("Resistance positivity invariant violated")
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
