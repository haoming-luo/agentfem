"""Research-only recoverable strain in series with the two-rate reference."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_two_rate_reference as base

MODEL_ID = "research.scalar-two-rate-recoverable.v1"
STATE_LAYOUT = (*base.STATE_LAYOUT, "recoverable_strain")


@dataclass(frozen=True)
class Material:
    backbone: base.Material
    recoverable_modulus: float
    relaxation_time: float
    strength_ratio: float = 1.
    strength_softening_rate: float = 0.

    def __post_init__(self):
        if (not np.isfinite([self.recoverable_modulus, self.relaxation_time,
                            self.strength_ratio, self.strength_softening_rate]).all()
                or min(self.recoverable_modulus, self.relaxation_time) <= 0
                or not 0 < self.strength_ratio <= 1 or self.strength_softening_rate < 0):
            raise ValueError("Require positive finite recoverable modulus and time")


def stored_energy(state, material):
    return base.stored_energy(state[:10], material.backbone)+material.recoverable_modulus*state[10]**2/2


def rates(state, strain_rate, material, *, stress_control=False):
    rate = (state[0]/material.recoverable_modulus-state[10])/material.relaxation_time
    backbone = material.backbone
    if material.strength_ratio != 1.:
        a, b = backbone.branches
        factor = material.strength_ratio+(1-material.strength_ratio)*np.exp(
            -material.strength_softening_rate*max(float(state[3]), 0.))
        backbone = replace(backbone, branches=(replace(a, strength=a.strength*factor), b))
    result = np.r_[base.rates(state[:10], strain_rate, backbone,
                             stress_control=stress_control), rate]
    if stress_control:
        result[7] += rate
    else:
        result[0] -= material.backbone.modulus*rate
    result[8] = state[0]*result[7]
    result[9] += material.recoverable_modulus*material.relaxation_time*rate**2
    return result


def integrate(old, duration, strain_rate, material, *, stress_control=False,
              method="Radau", rtol=1e-9):
    if np.iscomplexobj(old):
        raise ValueError("Require real state")
    state = np.asarray(old, dtype=float)
    if (state.shape != (11,) or not np.isfinite(state).all()
            or np.any(state[[3, 6, 9]] < 0) or type(stress_control) is not bool
            or method not in {"Radau", "BDF"}
            or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid recoverable reference state or controls")
    for offset, branch in zip((1, 4), material.backbone.branches, strict=True):
        if np.any(abs(state[offset:offset+2]) > np.asarray(branch.saturation)+1e-7):
            raise ValueError("Initial backstress lies outside its invariant domain")
    solution = solve_ivp(lambda t, y: rates(y, strain_rate, material,
                                          stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13])
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("Recoverable reference integration failed; input retained")
    for offset, branch in zip((1, 4), material.backbone.branches, strict=True):
        if np.any(abs(solution.y[offset:offset+2]) > np.asarray(branch.saturation)[:, None]+1e-6):
            raise RuntimeError("Backstress invariant violated")
    accumulated = solution.y[[3, 6, 9]]
    tolerance = 10*(np.array([1e-13, 1e-13, 1e-11])[:, None]+rtol*abs(accumulated))
    if np.any(np.maximum.accumulate(accumulated, axis=1)-accumulated > tolerance):
        raise RuntimeError("Accumulated state decreased")
    end = solution.y[:, -1]
    balance = end[8]-state[8]-stored_energy(end, material)+stored_energy(state, material)-end[9]+state[9]
    if abs(balance) > 1e-7+1e-7*abs(end[8]-state[8]):
        raise RuntimeError("Integrated mechanical balance failed")
    return solution
