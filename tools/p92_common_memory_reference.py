"""Research-only common effective stress, preserving recoverable strains."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_recoverable_spectrum as spectrum

MODEL_ID = "research.scalar-common-kinematic-memory.v1"
STATE_LAYOUT = spectrum.STATE_LAYOUT


@dataclass(frozen=True)
class Material:
    spectrum: spectrum.Material
    shared: bool | str = True

    def __post_init__(self):
        s = self.spectrum
        if ((type(self.shared) is not bool and self.shared != "slow") or s.directional_recovery
                or any(v != 1 for v in (*s.strength_ratios, *s.hardening_ratios,
                                        *s.static_ratios, *s.recoverable_powers))
                or any(v != 0 for v in (*s.static_powers, *s.strength_rates,
                                        *s.hardening_rates, *s.static_history_rates))
                or any(b.saturation_ratio != 1 or b.softening_rate != 0 for b in s.backbone.branches)):
            raise ValueError("Common memory comparison requires the unmodified constant spectrum")


def stored_energy(state, material):
    return spectrum.stored_energy(state, material.spectrum)


def rates(state, strain_rate, material, *, stress_control=False):
    s = material.spectrum
    if not material.shared:
        return spectrum.rates(state, strain_rate, s, stress_control=stress_control)
    result = np.zeros(12)
    effective = (np.array([state[0]-state[1]-state[2]-state[5],
                           state[0]-state[4]-state[2]-state[5]]) if material.shared == "slow"
                 else np.full(2, state[0]-sum(state[[1, 2, 4, 5]])))
    speeds = np.array([b.reference_flow*(max(abs(v)-b.strength, 0.)/120.)**b.exponent
                       for b, v in zip(s.backbone.branches, effective, strict=True)])
    flows = np.sign(effective)*speeds
    speed = sum(speeds)
    flow = sum(flows)
    dissipation = float(effective@flows)
    for j, (offset, branch) in enumerate(zip((1, 4), s.backbone.branches, strict=True)):
        for i, (h, a, r, chi) in enumerate(zip(branch.hardening, branch.saturation,
                                              branch.static_recovery, branch.ratcheting, strict=True)):
            x = state[offset+i]
            own = material.shared == "slow" and i == 0
            decay = h/a*(abs(x)/a)**chi*(speeds[j] if own else speed)+r
            result[offset+i] = h*(flows[j] if own else flow)-decay*x
            dissipation += decay*x*x/h
    result[[3, 6]] = speeds
    recovery = (state[0]/np.asarray(s.moduli)-state[10:])/s.times
    result[10:] = recovery
    result[7] = flow+sum(recovery) if stress_control else strain_rate
    result[0] = 0. if stress_control else s.backbone.modulus*(strain_rate-flow-sum(recovery))
    result[8] = state[0]*result[7]
    result[9] = dissipation+sum(np.asarray(s.moduli)*np.asarray(s.times)*recovery**2)
    return result


def integrate(old, duration, strain_rate, material, *, stress_control=False,
              method="Radau", rtol=1e-9):
    if not material.shared:
        return spectrum.integrate(old, duration, strain_rate, material.spectrum,
                                  stress_control=stress_control, method=method, rtol=rtol)
    if np.iscomplexobj(old):
        raise ValueError("Require real state")
    state = np.asarray(old, dtype=float)
    if (state.shape != (12,) or not np.isfinite(state).all() or np.any(state[[3, 6, 9]] < 0)
            or type(stress_control) is not bool or method not in {"Radau", "BDF"}
            or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid common-memory state or controls")
    saturation = np.array([a for b in material.spectrum.backbone.branches for a in b.saturation])
    if np.any(abs(state[[1, 2, 4, 5]]) > saturation+1e-7):
        raise ValueError("Initial backstress outside invariant domain")
    sol = solve_ivp(lambda t, y: rates(y, strain_rate, material, stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13])
    if not sol.success or not np.isfinite(sol.y).all():
        raise RuntimeError("Common-memory integration failed; original input retained")
    if np.any(abs(sol.y[[1, 2, 4, 5]]) > saturation[:, None]+1e-6):
        raise RuntimeError("Backstress invariant violated")
    accumulated = sol.y[[3, 6, 9]]
    tolerance = 10*(np.array([1e-13, 1e-13, 1e-11])[:, None]+rtol*abs(accumulated))
    if np.any(np.maximum.accumulate(accumulated, axis=1)-accumulated > tolerance):
        raise RuntimeError("Accumulated state decreased")
    end = sol.y[:, -1]
    balance = end[8]-state[8]-stored_energy(end, material)+stored_energy(state, material)-end[9]+state[9]
    if abs(balance) > 1e-7+1e-7*abs(end[8]-state[8]):
        raise RuntimeError("Integrated mechanical balance failed")
    return sol
