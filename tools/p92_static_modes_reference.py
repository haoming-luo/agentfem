"""Isolated dissipative memory modes with unchanged recovery eigenvalues."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_recoverable_spectrum as spectrum

MODEL_ID = "research.scalar-static-recovery-modes.v1"
STATE_LAYOUT = spectrum.STATE_LAYOUT


@dataclass(frozen=True)
class Material:
    spectrum: spectrum.Material
    mode: str = "diagonal"

    def __post_init__(self):
        s = self.spectrum
        if self.mode not in {"diagonal", "shared_slow", "difference_slow", "exchanged_slow",
                             "recoverable_shared", "recoverable_difference", "recoverable_exchanged"}:
            raise ValueError("Unknown static recovery mode")
        if (s.strength_ratios != (1., 1.) or s.hardening_ratios != (1., 1.)
                or s.static_ratios != (1., 1.) or s.static_powers != (0., 0.)
                or s.recoverable_powers != (1., 1.) or s.directional_recovery
                or any(b.saturation_ratio != 1. for b in s.backbone.branches)):
            raise ValueError("Isolate static modes from other hypotheses")
        if self.mode.startswith("recoverable_") and s.backbone.branches[1].static_recovery[1] <= 0:
            raise ValueError("Recoverable reassignment requires two positive decay rates")


def matrix(material):
    a, b = (branch.static_recovery[1] for branch in material.spectrum.backbone.branches)
    if material.mode.startswith("recoverable_"):
        a, b = b, 1./material.spectrum.times[1]
    if material.mode == "diagonal":
        return np.diag([a, b])
    if material.mode in {"exchanged_slow", "recoverable_exchanged"}:
        return np.diag([b, a])
    average, cross = (a+b)/2, (a-b)/2
    if material.mode in {"difference_slow", "recoverable_difference"}:
        cross = -cross
    return np.array([[average, cross], [cross, average]])


def stored_energy(state, material):
    return spectrum.stored_energy(state, material.spectrum)


def rates(state, strain_rate, material, *, stress_control=False):
    result = spectrum.rates(state, strain_rate, material.spectrum, stress_control=stress_control)
    if material.mode == "diagonal":
        return result
    if material.mode.startswith("recoverable_"):
        s = material.spectrum
        branch = s.backbone.branches[1]
        sqrt_h, sqrt_k = np.sqrt([branch.hardening[1], s.moduli[1]])
        force = np.array([-state[5]/sqrt_h, state[0]/sqrt_k-sqrt_k*state[11]])
        difference = matrix(material)-np.diag([branch.static_recovery[1], 1./s.times[1]])
        correction = difference@force
        result[5] += sqrt_h*correction[0]
        strain_correction = correction[1]/sqrt_k
        result[11] += strain_correction
        if stress_control:
            result[7] += strain_correction
        else:
            result[0] -= s.backbone.modulus*strain_correction
        result[8] = state[0]*result[7]
        result[9] += float(force@difference@force)
        return result
    branches = material.spectrum.backbone.branches
    sqrt_h = np.sqrt([b.hardening[1] for b in branches])
    q = state[[2, 5]]/sqrt_h
    difference = matrix(material)-np.diag([b.static_recovery[1] for b in branches])
    result[[2, 5]] -= sqrt_h*(difference@q)
    result[9] += float(q@difference@q)
    return result


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
        raise ValueError("Invalid static-mode state or controls")
    solution = solve_ivp(lambda t, y: rates(y, strain_rate, material, stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13])
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("Static-mode integration failed; input retained")
    history = solution.y[[3, 6, 9]]
    tolerance = 10*(np.array([1e-13, 1e-13, 1e-11])[:, None]+rtol*abs(history))
    if np.any(np.maximum.accumulate(history, axis=1)-history > tolerance):
        raise RuntimeError("Accumulated state decreased")
    end = solution.y[:, -1]
    balance = end[8]-state[8]-stored_energy(end, material)+stored_energy(state, material)-end[9]+state[9]
    if abs(balance) > 1e-7+1e-7*abs(end[8]-state[8]):
        raise RuntimeError("Integrated mechanical balance failed")
    return solution
