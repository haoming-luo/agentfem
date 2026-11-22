"""Research-only two-time recoverable spectrum; no P92 material admission."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_two_rate_reference as base

MODEL_ID = "research.scalar-two-rate-recoverable-spectrum.v1"
STATE_LAYOUT = (*base.STATE_LAYOUT, "recoverable_fast_strain", "recoverable_slow_strain")


@dataclass(frozen=True)
class Material:
    backbone: base.Material
    moduli: tuple[float, float]
    times: tuple[float, float]
    strength_ratios: tuple[float, float] = (1., 1.)
    strength_rates: tuple[float, float] = (0., 0.)
    hardening_ratios: tuple[float, float] = (1., 1.)
    hardening_rates: tuple[float, float] = (0., 0.)
    static_powers: tuple[float, float] = (0., 0.)
    static_ratios: tuple[float, float] = (1., 1.)
    static_history_rates: tuple[float, float] = (0., 0.)
    static_history: str = "own"
    directional_recovery: bool = False

    def __post_init__(self):
        pairs = (self.moduli, self.times, self.strength_ratios, self.strength_rates,
                 self.hardening_ratios, self.hardening_rates, self.static_powers,
                 self.static_ratios, self.static_history_rates)
        if (any(len(v) != 2 for v in pairs)
                or not np.isfinite([x for pair in pairs for x in pair]).all()
                or min(*self.moduli, *self.times) <= 0 or self.times[0] > self.times[1]
                or not all(0 < r <= 1 for r in (*self.strength_ratios, *self.hardening_ratios))
                or min(*self.strength_rates, *self.hardening_rates, *self.static_powers,
                       *self.static_history_rates) < 0
                or not all(0 < r <= 1 for r in self.static_ratios)
                or self.static_history not in {"own", "shared"}
                or type(self.directional_recovery) is not bool):
            raise ValueError("Require two positive moduli and ordered positive times")


def evolving_backbone(state, material):
    """Current positive coefficients and d(log H)/dp, without updating history."""
    branches, logarithmic_derivatives = [], []
    for branch, p, r, w, hratio, hrate in zip(
            material.backbone.branches, state[[3, 6]], material.strength_ratios,
            material.strength_rates, material.hardening_ratios, material.hardening_rates, strict=True):
        accumulated = max(float(p), 0.)
        strength_factor = r+(1-r)*np.exp(-w*accumulated)
        transient = (1-hratio)*np.exp(-hrate*accumulated)
        hardening_factor = hratio+transient
        branches.append(replace(branch, strength=branch.strength*strength_factor,
            hardening=tuple(h*hardening_factor for h in branch.hardening),
            saturation=tuple(a*hardening_factor for a in branch.saturation)))
        logarithmic_derivatives.append(-hrate*transient/hardening_factor)
    return replace(material.backbone, branches=tuple(branches)), logarithmic_derivatives


def stored_energy(state, material):
    backbone, _ = evolving_backbone(state, material)
    return base.stored_energy(state[:10], backbone)+sum(
        h*a*a/2 for h, a in zip(material.moduli, state[10:], strict=True))


def rates(state, strain_rate, material, *, stress_control=False, dynamic_multipliers=(1., 1.)):
    if len(dynamic_multipliers) != 2 or not np.isfinite(dynamic_multipliers).all() or min(dynamic_multipliers) <= 0:
        raise ValueError("Require two positive dynamic-recovery multipliers")
    recovery = (state[0]/np.asarray(material.moduli)-state[10:])/material.times
    backbone, logarithmic_derivatives = evolving_backbone(state, material)
    # Polynomial static recovery: r_eff=r*(|X|/a)^m.  m=0 is the
    # original linear law; every exponent agrees at nominal saturation.
    # This changes kinetics, not stored energy or a prescribed loading path.
    branches = []
    for offset, branch, power, ratio, rate in zip((1, 4), backbone.branches,
            material.static_powers, material.static_ratios, material.static_history_rates, strict=True):
        saturation = base.saturation_at(branch, state[offset+2])
        factor = (abs(state[offset:offset+2])/saturation)**power
        history = (sum(max(float(state[i]), 0.) for i in (3, 6))
                   if material.static_history == "shared" else max(float(state[offset+2]), 0.))
        factor *= ratio+(1-ratio)*np.exp(-rate*history)
        branches.append(replace(branch, static_recovery=tuple(np.asarray(branch.static_recovery)*factor)))
    backbone = replace(backbone, branches=tuple(branches))
    result = np.r_[base.rates(state[:10], strain_rate, backbone,
                              stress_control=stress_control), recovery]
    if stress_control:
        result[7] += sum(recovery)
    else:
        result[0] -= material.backbone.modulus*sum(recovery)
    result[8] = state[0]*result[7]
    result[9] += sum(np.asarray(material.moduli)*material.times*recovery**2)
    for offset, branch, derivative in zip((1, 4), backbone.branches,
                                           logarithmic_derivatives, strict=True):
        logarithmic_rate = derivative*result[offset+2]
        for i, h in enumerate(branch.hardening):
            x = state[offset+i]
            result[offset+i] += logarithmic_rate*x
            result[9] -= logarithmic_rate*x*x/(2*h)
    for offset, branch, multiplier in zip((1, 4), backbone.branches, dynamic_multipliers, strict=True):
        effective = state[0]-sum(state[offset:offset+2])
        for i, (h, a, chi) in enumerate(zip(branch.hardening, base.saturation_at(branch, state[offset+2]),
                                           branch.ratcheting, strict=True)):
            x = state[offset+i]
            active = not material.directional_recovery or x*effective > 0.
            correction = (multiplier*active-1)*h/a*(abs(x)/a)**chi*result[offset+2]
            result[offset+i] -= correction*x
            result[9] += correction*x*x/h
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
        raise ValueError("Invalid spectrum state or controls")
    for offset, branch in zip((1, 4), material.backbone.branches, strict=True):
        if np.any(abs(state[offset:offset+2]) > np.asarray(branch.saturation)+1e-7):
            raise ValueError("Initial backstress outside invariant domain")
    solution = solve_ivp(lambda t, y: rates(y, strain_rate, material, stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13])
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("Spectrum integration failed; input retained")
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
