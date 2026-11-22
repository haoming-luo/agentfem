"""Isolated research primary-creep replacement; not an admitted material."""

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_recoverable_spectrum as spectrum

MODEL_ID = "research.scalar-two-rate-primary-creep.v1"
STATE_LAYOUT = (*spectrum.STATE_LAYOUT[:11], "accumulated_primary_strain")
Material = spectrum.Material


def validate_material(material):
    if (material.directional_recovery or material.static_history != "own"
            or tuple(material.static_powers) != (0., 0.)
            or tuple(material.static_ratios) != (1., 1.)
            or tuple(material.strength_ratios) != (1., 1.)
            or tuple(material.hardening_ratios) != (1., 1.)):
        raise ValueError("Primary-creep comparison must isolate one mechanism")


def primary_flow(stress, accumulated, material):
    h, tau = material.moduli[1], material.times[1]
    return stress/(h*tau)*np.exp(-h*max(float(accumulated), 0.)/120.)


def stored_energy(state, material):
    # Slot 11 has no recoverable energy; avoid subtracting large fictitious terms.
    y = np.array(state, dtype=float, copy=True)
    y[11] = 0.
    return spectrum.stored_energy(y, material)


def rates(state, strain_rate, material, *, stress_control=False):
    # Reuse unchanged backbone and fast Kelvin evolution. Remove the slow
    # Kelvin's signed flow and dissipation, then substitute irreversible flow.
    y = np.array(state, dtype=float, copy=True)
    y[11] = 0.
    result = spectrum.rates(y, strain_rate, material, stress_control=stress_control)
    slow = result[11]
    flow = primary_flow(state[0], state[11], material)
    if stress_control:
        result[7] += flow-slow
    else:
        result[0] -= material.backbone.modulus*(flow-slow)
    result[8] = state[0]*result[7]
    result[9] += state[0]*flow-material.moduli[1]*material.times[1]*slow**2
    result[11] = abs(flow)
    return result


def integrate(old, duration, strain_rate, material, *, stress_control=False,
              method="Radau", rtol=1e-9):
    validate_material(material)
    if np.iscomplexobj(old):
        raise ValueError("Require real state")
    state = np.asarray(old, dtype=float)
    if (state.shape != (12,) or not np.isfinite(state).all()
            or np.any(state[[3, 6, 9, 11]] < 0) or type(stress_control) is not bool
            or method not in {"Radau", "BDF"}
            or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid primary-creep state or controls")
    for offset, branch in zip((1, 4), material.backbone.branches, strict=True):
        if np.any(abs(state[offset:offset+2]) > np.asarray(branch.saturation)+1e-7):
            raise ValueError("Initial backstress outside invariant domain")
    atol = [1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13]
    solution = solve_ivp(lambda t, y: rates(y, strain_rate, material, stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, atol=atol, dense_output=True)
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("Primary-creep integration failed; input retained")
    for offset, branch in zip((1, 4), material.backbone.branches, strict=True):
        if np.any(abs(solution.y[offset:offset+2]) > np.asarray(branch.saturation)[:, None]+1e-6):
            raise RuntimeError("Backstress invariant violated")
    indices = [3, 6, 9, 11]
    history = solution.y[indices]
    tolerance = 10*(np.asarray(atol)[indices, None]+rtol*abs(history))
    if np.any(np.maximum.accumulate(history, axis=1)-history > tolerance):
        raise RuntimeError("Accumulated state decreased")
    end = solution.y[:, -1]
    balance = end[8]-state[8]-stored_energy(end, material)+stored_energy(state, material)-end[9]+state[9]
    if abs(balance) > 1e-7+1e-7*abs(end[8]-state[8]):
        raise RuntimeError("Integrated mechanical balance failed")
    return solution
