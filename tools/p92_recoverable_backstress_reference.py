"""Research-only recoverable/backstress interaction with a common potential."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_recoverable_spectrum as spectrum

MODEL_ID = "research.scalar-recoverable-backstress-coupling.v1"
STATE_LAYOUT = spectrum.STATE_LAYOUT


@dataclass(frozen=True)
class Material:
    spectrum: spectrum.Material
    couplings: tuple[float, float] = (0., 0.)

    def __post_init__(self):
        s = self.spectrum
        if (len(self.couplings) != 2 or not np.isfinite(self.couplings).all()
                or any(abs(k) >= 1 for k in self.couplings)):
            raise ValueError("Require a positive-definite interaction potential")
        if (s.strength_ratios != (1., 1.) or s.hardening_ratios != (1., 1.)
                or s.static_powers != (0., 0.) or s.static_ratios != (1., 1.)
                or s.recoverable_powers != (1., 1.) or s.directional_recovery
                or any(b.saturation_ratio != 1. for b in s.backbone.branches)):
            raise ValueError("Reference requires constant coefficients and original recovery")
        if any(self.couplings) and any(s.backbone.branches[0].static_recovery):
            raise ValueError("Couplings replace, rather than add to, the two A static rates")


def forces(state, material):
    s = material.spectrum
    h, k = np.asarray(s.backbone.branches[1].hardening), np.asarray(s.moduli)
    coupling = np.asarray(material.couplings)
    c = coupling*np.sqrt(h*k)
    y = c*state[4:6]/h+k*(1-coupling**2)*state[10:12]
    return y, c


def stored_energy(state, material):
    s = material.spectrum
    return spectrum.base.stored_energy(state[:10], s.backbone)+float(np.sum(
        np.asarray(s.moduli)*(1-np.asarray(material.couplings)**2)*state[10:12]**2/2))


def rates(state, strain_rate, material, *, stress_control=False):
    s = material.spectrum
    result = spectrum.rates(state, strain_rate, s, stress_control=stress_control)
    original = result[10:12].copy()
    y, c = forces(state, material)
    mobility = np.asarray(s.moduli)*s.times
    recovery = (state[0]-y)/mobility
    difference = sum(recovery-original)
    result[10:12] = recovery
    result[4:6] += c*recovery
    if stress_control:
        result[7] += difference
    else:
        result[0] -= s.backbone.modulus*difference
    result[8] = state[0]*result[7]
    result[9] += sum(mobility*(recovery**2-original**2))
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
        raise ValueError("Invalid coupled reference state or controls")
    solution = solve_ivp(lambda t, y: rates(y, strain_rate, material, stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13])
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("Interaction integration failed; input retained")
    accumulated = solution.y[[3, 6, 9]]
    tolerance = 10*(np.array([1e-13, 1e-13, 1e-11])[:, None]+rtol*abs(accumulated))
    if np.any(np.maximum.accumulate(accumulated, axis=1)-accumulated > tolerance):
        raise RuntimeError("Accumulated state decreased")
    end = solution.y[:, -1]
    balance = end[8]-state[8]-stored_energy(end, material)+stored_energy(state, material)-end[9]+state[9]
    if abs(balance) > 1e-7+1e-7*abs(end[8]-state[8]):
        raise RuntimeError("Integrated mechanical balance failed")
    return solution
