"""Research scalar shared-backstress/drag reference, Li et al. (2021), eqs 12–17.

MPa and seconds. Optional Kelvin components are an explicit additional model,
not part of the published card. No P92 engineering or life admission.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

MODEL_ID = "research.scalar-shared-drag.v1"
STATE_LAYOUT = ("stress_MPa", "backstress_1_MPa", "backstress_2_MPa", "accumulated_flow",
    "backstress_3_MPa", "log_relative_drag", "signed_flow_strain", "strain",
    "work_density_MPa", "dissipation_density_MPa", "recoverable_fast_strain", "recoverable_slow_strain")
X_INDICES = (1, 2, 4)


@dataclass(frozen=True)
class Material:
    modulus: float
    initial_drag: float
    flow_coefficients: tuple[float, float]
    exponents: tuple[float, float]
    hardening: tuple[float, float, float]
    dynamic_initial: tuple[float, float, float]
    dynamic_final: tuple[float, float, float]
    dynamic_rates: tuple[float, float, float]
    static_recovery: tuple[float, float, float]
    drag_hardening: float
    drag_dynamic_initial: float
    drag_dynamic_final: float
    drag_dynamic_rate: float
    drag_static_recovery: float
    compliances: tuple[float, float] = (0., 0.)
    times: tuple[float, float] = (1., 100.)

    def __post_init__(self):
        pairs = (self.flow_coefficients, self.exponents, self.compliances, self.times)
        triples = (self.hardening, self.dynamic_initial, self.dynamic_final,
                   self.dynamic_rates, self.static_recovery)
        scalars = (self.modulus, self.initial_drag, self.drag_hardening,
            self.drag_dynamic_initial, self.drag_dynamic_final, self.drag_dynamic_rate, self.drag_static_recovery)
        if (any(len(x) != 2 for x in pairs) or any(len(x) != 3 for x in triples)
                or not np.isfinite([*scalars, *(v for x in (*pairs, *triples) for v in x)]).all()
                or min(self.modulus, self.initial_drag, *self.flow_coefficients, *self.hardening,
                    *self.dynamic_initial, *self.dynamic_final, self.drag_dynamic_initial,
                    self.drag_dynamic_final, *self.times) <= 0 or min(self.exponents) < 1
                or min(self.drag_hardening, self.drag_dynamic_rate, self.drag_static_recovery,
                       *self.dynamic_rates, *self.static_recovery, *self.compliances) < 0):
            raise ValueError("Invalid shared-drag reference parameters")


def unverified_table2_transcription():
    """Unverified PDF-text values; exponent extraction conflicts with HTML.

    Preserved to reproduce a failed diagnostic, not an approved paper card.
    Original-table visual verification and loading units remain unresolved.
    """
    return Material(132000., 327., (.141, 1.46e-18), (12.1, 278.),
        (82000., 21200., 10100.), (1870., 95.2, .673), (2270., 1110., 287.),
        (3760., 174., 30.9), (.0246, 6.10e-6, 5.66e-6),
        758., 3.36, 3.75, .773, 2.13e-4)


def stored_energy(state, material):
    value = state[0]**2/(2*material.modulus)+sum(
        state[i]**2/(2*h) for i, h in zip(X_INDICES, material.hardening, strict=True))
    for a, compliance in zip(state[10:12], material.compliances, strict=True):
        if compliance == 0:
            if a != 0:
                raise ValueError("Disabled recoverable component has nonzero state")
        else:
            value += a*a/(2*compliance)
    return float(value)


def rates(state, strain_rate, material, *, stress_control=False):
    effective = state[0]-sum(state[list(X_INDICES)])
    log_drag = np.log(material.initial_drag)+state[5]
    if effective == 0:
        speed = 0.
    else:
        log_rates = np.log(material.flow_coefficients)+np.asarray(material.exponents)*(np.log(abs(effective))-log_drag)
        if max(log_rates) > np.log(np.finfo(float).max)-2:
            raise ValueError("Flow rate exceeds numerical domain; input is not clipped")
        speed = float(sum(np.exp(log_rates)))
    flow = np.sign(effective)*speed
    gamma = np.asarray(material.dynamic_final)+(np.asarray(material.dynamic_initial)-material.dynamic_final)*np.exp(
        -np.asarray(material.dynamic_rates)*abs(state[6]))
    decay = gamma*speed+material.static_recovery
    result = np.zeros(12)
    result[list(X_INDICES)] = np.asarray(material.hardening)*flow-decay*state[list(X_INDICES)]
    result[3], result[6] = speed, flow
    drag_gamma = material.drag_dynamic_final+(material.drag_dynamic_initial-material.drag_dynamic_final)*np.exp(
        -material.drag_dynamic_rate*max(float(state[3]), 0.))
    result[5] = material.drag_hardening*np.exp(-log_drag)*speed-drag_gamma*speed-material.drag_static_recovery
    recoverable = (np.asarray(material.compliances)*state[0]-state[10:12])/material.times
    result[10:12] = recoverable
    result[7] = flow+sum(recoverable) if stress_control else strain_rate
    result[0] = 0. if stress_control else material.modulus*(strain_rate-flow-sum(recoverable))
    result[8] = state[0]*result[7]
    result[9] = effective*flow+sum(decay*state[list(X_INDICES)]**2/material.hardening)
    result[9] += sum(t*v*v/c for t, v, c in zip(material.times, recoverable, material.compliances, strict=True) if c > 0)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite constitutive rate; input is not clipped")
    return result


def integrate(old, duration, strain_rate, material, *, stress_control=False,
              method="Radau", rtol=1e-9):
    if np.iscomplexobj(old):
        raise ValueError("Require real state")
    state = np.asarray(old, dtype=float)
    if (state.shape != (12,) or not np.isfinite(state).all() or state[3] < 0 or state[9] < 0
            or abs(state[6]) > state[3]+1e-11 or type(stress_control) is not bool
            or method not in {"Radau", "BDF"} or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid shared-drag state or controls")
    stored_energy(state, material)
    envelope = np.asarray(material.hardening)/np.minimum(material.dynamic_initial, material.dynamic_final)
    if np.any(abs(state[list(X_INDICES)]) > envelope+1e-7):
        raise ValueError("Initial shared backstress outside invariant envelope")
    solution = solve_ivp(lambda t, y: rates(y, strain_rate, material, stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-11, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13])
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("Shared-drag integration failed; input retained")
    if np.any(abs(solution.y[list(X_INDICES)]) > envelope[:, None]+1e-6):
        raise RuntimeError("Shared backstress envelope violated")
    accumulated = solution.y[[3, 9]]
    tolerance = 10*(np.array([1e-13, 1e-11])[:, None]+rtol*abs(accumulated))
    if np.any(np.maximum.accumulate(accumulated, axis=1)-accumulated > tolerance):
        raise RuntimeError("Accumulated state decreased")
    if np.any(abs(solution.y[6])-solution.y[3] > 1e-10):
        raise RuntimeError("Signed flow exceeds accumulated flow")
    end = solution.y[:, -1]
    balance = end[8]-state[8]-stored_energy(end, material)+stored_energy(state, material)-end[9]+state[9]
    if abs(balance) > 1e-7+1e-7*abs(end[8]-state[8]):
        raise RuntimeError("Integrated shared-drag mechanical balance failed")
    return solution
