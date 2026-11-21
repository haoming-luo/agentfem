"""Research-only energetic coupling; no calibrated P92 material claim.

Conjugate backstresses have a coupled quadratic energy. Their rectangular
uncoupled saturation bounds do not apply. See the prospective design record.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_recoverable_spectrum as spectrum

MODEL_ID = "research.scalar-energetic-two-mechanism.v1"
STATE_LAYOUT = spectrum.STATE_LAYOUT


@dataclass(frozen=True)
class Material:
    spectrum: spectrum.Material
    couplings: tuple[float, float] = (0., 0.)

    def __post_init__(self):
        if (len(self.couplings) != 2 or not np.isfinite(self.couplings).all()
                or any(abs(k) >= 1 for k in self.couplings)):
            raise ValueError("Coupling matrices must be positive definite")
        s = self.spectrum
        if (any(r != 1 for r in (*s.strength_ratios, *s.hardening_ratios))
                or any(b.saturation_ratio != 1 for b in s.backbone.branches)):
            raise ValueError("First coupling reference requires constant coefficients")


def matrices(material):
    a, b = material.spectrum.backbone.branches
    return tuple(np.array([[ha, k*np.sqrt(ha*hb)], [k*np.sqrt(ha*hb), hb]])
                 for ha, hb, k in zip(a.hardening, b.hardening, material.couplings, strict=True))


def stored_energy(state, material):
    s = material.spectrum
    value = state[0]**2/(2*s.backbone.modulus)
    value += sum(h*x*x/2 for h, x in zip(s.moduli, state[10:], strict=True))
    for i, matrix in enumerate(matrices(material)):
        x = state[[1+i, 4+i]]
        value += .5*x@np.linalg.solve(matrix, x)
    return float(value)


def rates(state, strain_rate, material, *, stress_control=False):
    result = spectrum.rates(state, strain_rate, material.spectrum, stress_control=stress_control)
    # Independent rates are H*qdot. Couple the conjugate forces using C*qdot;
    # dissipation remains sum(X*recovery) plus the unchanged flow dissipation.
    for i, matrix in enumerate(matrices(material)):
        indices = [1+i, 4+i]
        result[indices] = matrix@(result[indices]/np.diag(matrix))
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
        raise ValueError("Invalid coupled state or controls")
    solution = solve_ivp(lambda t, y: rates(y, strain_rate, material, stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13])
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("Coupled integration failed; input retained")
    accumulated = solution.y[[3, 6, 9]]
    tolerance = 10*(np.array([1e-13, 1e-13, 1e-11])[:, None]+rtol*abs(accumulated))
    if np.any(np.maximum.accumulate(accumulated, axis=1)-accumulated > tolerance):
        raise RuntimeError("Accumulated state decreased")
    end = solution.y[:, -1]
    balance = end[8]-state[8]-stored_energy(end, material)+stored_energy(state, material)-end[9]+state[9]
    if abs(balance) > 1e-7+1e-7*abs(end[8]-state[8]):
        raise RuntimeError("Integrated mechanical balance failed")
    return solution
