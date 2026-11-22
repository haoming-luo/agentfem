"""Parameter-neutral research discriminator of recoverable-memory mobility."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_recoverable_spectrum as spectrum

MODEL_ID = "research.scalar-recoverable-memory-mobility.v1"
STATE_LAYOUT = spectrum.STATE_LAYOUT


@dataclass(frozen=True)
class Material:
    spectrum: spectrum.Material
    feedback: str = "none"

    def __post_init__(self):
        if self.feedback not in {"none", "flow", "static_recovery", "strength"}:
            raise ValueError("Unknown predeclared mobility replacement")
        s = self.spectrum
        if (s.strength_ratios != (1., 1.) or s.hardening_ratios != (1., 1.)
                or s.static_ratios != (1., 1.) or s.static_powers != (0., 0.)
                or s.recoverable_powers != (1., 1.) or s.directional_recovery
                or any(b.saturation_ratio != 1. for b in s.backbone.branches)):
            raise ValueError("Isolate mobility from other history hypotheses")


def factor(state, material):
    x = material.spectrum.moduli[1]*state[11]/120.
    return (1./np.hypot(1., x))**2


def stored_energy(state, material):
    return spectrum.stored_energy(state, material.spectrum)


def rates(state, strain_rate, material, *, stress_control=False):
    s = material.spectrum
    a, b = s.backbone.branches
    f = factor(state, material)
    if material.feedback == "flow":
        b = replace(b, reference_flow=b.reference_flow*f)
    elif material.feedback == "static_recovery":
        b = replace(b, static_recovery=tuple(r*f for r in b.static_recovery))
    elif material.feedback == "strength":
        b = replace(b, strength=float(np.hypot(b.strength, s.moduli[1]*state[11])))
    current = replace(s, backbone=replace(s.backbone, branches=(a, b)))
    return spectrum.rates(state, strain_rate, current, stress_control=stress_control)


def integrate(old, duration, strain_rate, material, *, stress_control=False,
              method="Radau", rtol=1e-9):
    if np.iscomplexobj(old):
        raise ValueError("Require real state")
    state = np.asarray(old, dtype=float)
    if (state.shape != (12,) or not np.isfinite(state).all()
            or np.any(state[[3, 6, 9]] < 0) or type(stress_control) is not bool
            or method not in {"Radau", "BDF"}
            or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid memory-mobility state or controls")
    for offset, branch in zip((1, 4), material.spectrum.backbone.branches, strict=True):
        if np.any(abs(state[offset:offset+2]) > np.asarray(branch.saturation)+1e-7):
            raise ValueError("Initial backstress outside invariant domain")
    solution = solve_ivp(lambda t, y: rates(y, strain_rate, material, stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13])
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("Memory-mobility integration failed; input retained")
    for offset, branch in zip((1, 4), material.spectrum.backbone.branches, strict=True):
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
