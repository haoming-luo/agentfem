"""Rate-and-slope-matched activated-flow research discriminator."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from tools import p92_recoverable_spectrum as spectrum

MODEL_ID = "research.scalar-matched-activated-flow.v1"
STATE_LAYOUT = spectrum.STATE_LAYOUT


@lru_cache(maxsize=256)
def activation(exponent):
    if not np.isfinite(exponent) or exponent < 1.:
        raise ValueError("Require finite exponent >= 1")
    if exponent == 1.:
        return 0.
    def residual(x):
        value = 1.+x*x/3.-x**4/45. if abs(x) < 1e-3 else x/np.tanh(x)
        return value-exponent
    return brentq(residual, 0., exponent, xtol=1e-14)


def speed(overstress, branch):
    u = max(overstress, 0.)/120.
    x = activation(branch.exponent)
    if x == 0.:
        return branch.reference_flow*u
    with np.errstate(over="raise", invalid="raise"):
        try:
            return branch.reference_flow*np.sinh(x*u)/np.sinh(x)
        except FloatingPointError as exc:
            raise ValueError("Activated flow exceeded finite numerical range") from exc


@dataclass(frozen=True)
class Material:
    spectrum: spectrum.Material
    activated: bool = True

    def __post_init__(self):
        s = self.spectrum
        if type(self.activated) is not bool:
            raise ValueError("Require explicit activated-flow selection")
        if (s.strength_ratios != (1., 1.) or s.hardening_ratios != (1., 1.)
                or s.static_ratios != (1., 1.) or s.static_powers != (0., 0.)
                or s.recoverable_powers != (1., 1.) or s.directional_recovery
                or any(b.saturation_ratio != 1. for b in s.backbone.branches)):
            raise ValueError("Require isolated original spectrum")


def stored_energy(state, material):
    return spectrum.stored_energy(state, material.spectrum)


def rates(state, strain_rate, material, *, stress_control=False):
    s = material.spectrum
    result = spectrum.rates(state, strain_rate, s, stress_control=stress_control)
    if not material.activated:
        return result
    branch = s.backbone.branches[1]
    x = state[4:6]
    effective = state[0]-sum(x)
    new_speed = speed(abs(effective)-branch.strength, branch)
    delta_speed = new_speed-result[6]
    delta_flow = np.sign(effective)*delta_speed
    h, a = np.asarray(branch.hardening), np.asarray(branch.saturation)
    dynamic = h/a*(abs(x)/a)**branch.ratcheting
    result[4:6] += h*delta_flow-dynamic*delta_speed*x
    result[6] = new_speed
    if stress_control:
        result[7] += delta_flow
    else:
        result[0] -= s.backbone.modulus*delta_flow
    result[8] = state[0]*result[7]
    result[9] += effective*delta_flow+sum(dynamic*delta_speed*x*x/h)
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
        raise ValueError("Invalid activated-flow state or controls")
    for offset, branch in zip((1, 4), material.spectrum.backbone.branches, strict=True):
        if np.any(abs(state[offset:offset+2]) > np.asarray(branch.saturation)+1e-7):
            raise ValueError("Initial backstress outside invariant domain")
    solution = solve_ivp(lambda t, y: rates(y, strain_rate, material, stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13])
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("Activated-flow integration failed; input retained")
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
        raise RuntimeError("Mechanical balance failed")
    return solution
