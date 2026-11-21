"""Research-only matched mobility/strength history; not a qualified P92 law."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_recoverable_spectrum as spectrum

MODEL_ID = "research.scalar-two-rate-resistance-recovery.v1"
STATE_LAYOUT = (*spectrum.STATE_LAYOUT, "resistance_loss_A", "resistance_loss_B")


@dataclass(frozen=True)
class Material:
    spectrum: spectrum.Material
    kind: str
    minimum: float
    dynamic: float
    recovery: float
    anchor_MPa: float = 120.

    def __post_init__(self):
        if (self.kind not in {"strength", "mobility"}
                or not np.isfinite([self.minimum, self.dynamic, self.recovery, self.anchor_MPa]).all()
                or not 0 < self.minimum <= 1 or min(self.dynamic, self.recovery) < 0
                or self.anchor_MPa <= 0):
            raise ValueError("Invalid resistance recovery parameters")
        if (self.spectrum.strength_ratios != (1., 1.)
                or self.spectrum.hardening_ratios != (1., 1.)):
            raise ValueError("Do not combine with existing strength/hardening evolution")


def effective_material(state, material):
    branches = []
    for s, branch in zip(1-state[12:], material.spectrum.backbone.branches, strict=True):
        if material.kind == "strength":
            branches.append(replace(branch, strength=branch.strength*s))
        else:
            factor = (1+branch.strength*(1-s)/material.anchor_MPa)**branch.exponent
            branches.append(replace(branch, reference_flow=branch.reference_flow*factor))
    return replace(material.spectrum, backbone=replace(material.spectrum.backbone, branches=tuple(branches)))


def stored_energy(state, material):
    return spectrum.stored_energy(state[:12], material.spectrum)


def rates(state, strain_rate, material, *, stress_control=False):
    r = spectrum.rates(state[:12], strain_rate, effective_material(state, material),
                       stress_control=stress_control)
    loss = state[12:]
    return np.r_[r, material.dynamic*(1-material.minimum-loss)*r[[3, 6]]-material.recovery*loss]


def integrate(old, duration, strain_rate, material, *, stress_control=False,
              method="Radau", rtol=1e-9):
    if np.iscomplexobj(old):
        raise ValueError("Require real state")
    y = np.asarray(old, dtype=float)
    if (y.shape != (14,) or not np.isfinite(y).all() or np.any(y[[3, 6, 9]] < 0)
            or np.any(y[12:] < 0) or np.any(y[12:] > 1-material.minimum)
            or type(stress_control) is not bool or method not in {"Radau", "BDF"}
            or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid resistance state or integration controls")
    for offset, branch in zip((1, 4), material.spectrum.backbone.branches, strict=True):
        if np.any(abs(y[offset:offset+2]) > np.asarray(branch.saturation)+1e-7):
            raise ValueError("Initial backstress outside invariant domain")
    sol = solve_ivp(lambda t, v: rates(v, strain_rate, material, stress_control=stress_control),
        (0., duration), y.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13, 1e-12, 1e-12])
    if not sol.success or not np.isfinite(sol.y).all():
        raise RuntimeError("Resistance integration failed; input retained")
    tol = 10*(1e-12+rtol)
    if np.any(sol.y[12:] < -tol) or np.any(sol.y[12:] > 1-material.minimum+tol):
        raise RuntimeError("Resistance invariant violated")
    for offset, branch in zip((1, 4), material.spectrum.backbone.branches, strict=True):
        if np.any(abs(sol.y[offset:offset+2]) > np.asarray(branch.saturation)[:, None]+1e-6):
            raise RuntimeError("Backstress invariant violated")
    accumulated = sol.y[[3, 6, 9]]
    tolerance = 10*(np.array([1e-13, 1e-13, 1e-11])[:, None]+rtol*abs(accumulated))
    if np.any(np.maximum.accumulate(accumulated, axis=1)-accumulated > tolerance):
        raise RuntimeError("Accumulated state decreased")
    end = sol.y[:, -1]
    balance = end[8]-y[8]-stored_energy(end, material)+stored_energy(y, material)-end[9]+y[9]
    if abs(balance) > 1e-7+1e-7*abs(end[8]-y[8]):
        raise RuntimeError("Mechanical balance failed")
    return sol
