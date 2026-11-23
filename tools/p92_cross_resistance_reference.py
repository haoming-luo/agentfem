"""Replace one signed memory by bounded shared resistance; research only."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_recoverable_spectrum as spectrum

MODEL_ID = "research.scalar-cross-resistance-replacement.v1"
STATE_LAYOUT = (*spectrum.STATE_LAYOUT[:5], "shared_resistance_MPa", *spectrum.STATE_LAYOUT[6:])


@dataclass(frozen=True)
class Material:
    spectrum: spectrum.Material

    def __post_init__(self):
        s = self.spectrum
        if (s.directional_recovery or any(v != 1 for v in
                (*s.strength_ratios, *s.hardening_ratios, *s.static_ratios, *s.recoverable_powers))
                or any(v != 0 for v in (*s.strength_rates, *s.hardening_rates,
                                       *s.static_powers, *s.static_history_rates))
                or any(b.saturation_ratio != 1 or b.softening_rate != 0 for b in s.backbone.branches)):
            raise ValueError("Resistance replacement requires the constant original spectrum")


def stored_energy(state, material):
    # The same quadratic coefficient now belongs to nonnegative R, not X_B2.
    return spectrum.stored_energy(state, material.spectrum)


def rates(state, strain_rate, material, *, stress_control=False):
    s = material.spectrum
    result = np.zeros(12)
    resistance = state[5]
    effective = [state[0]-state[1]-state[2], state[0]-state[4]]
    speeds = np.array([b.reference_flow*(max(abs(v)-b.strength-resistance, 0.)/120.)**b.exponent
                       for b, v in zip(s.backbone.branches, effective, strict=True)])
    flows = speeds*np.sign(effective)
    dissipation = float(np.dot(effective, flows))
    for j, (offset, branch) in enumerate(zip((1, 4), s.backbone.branches, strict=True)):
        for i in range(2 if j == 0 else 1):
            h, a, r, chi = (values[i] for values in
                (branch.hardening, branch.saturation, branch.static_recovery, branch.ratcheting))
            x = state[offset+i]
            decay = h/a*(abs(x)/a)**chi*speeds[j]+r
            result[offset+i] = h*flows[j]-decay*x
            dissipation += decay*x*x/h
    b = s.backbone.branches[1]
    h, a, r = b.hardening[1], b.saturation[1], b.static_recovery[1]
    result[5] = h*speeds[0]-(h/a*sum(speeds)+r)*resistance
    dissipation -= resistance*result[5]/h
    result[[3, 6]] = speeds
    recovery = (state[0]/np.asarray(s.moduli)-state[10:])/s.times
    result[10:] = recovery
    result[7] = sum(flows)+sum(recovery) if stress_control else strain_rate
    result[0] = 0. if stress_control else s.backbone.modulus*(strain_rate-sum(flows)-sum(recovery))
    result[8] = state[0]*result[7]
    result[9] = dissipation+sum(np.asarray(s.moduli)*np.asarray(s.times)*recovery**2)
    return result


def integrate(old, duration, strain_rate, material, *, stress_control=False,
              method="Radau", rtol=1e-9):
    if np.iscomplexobj(old):
        raise ValueError("Require real state")
    y = np.asarray(old, dtype=float)
    if (y.shape != (12,) or not np.isfinite(y).all() or np.any(y[[3, 5, 6, 9]] < 0)
            or type(stress_control) is not bool or method not in {"Radau", "BDF"}
            or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid resistance state or controls")
    bounds = np.array([a for b in material.spectrum.backbone.branches for a in b.saturation])
    if np.any(abs(y[[1, 2, 4, 5]]) > bounds+1e-7):
        raise ValueError("State outside memory invariant domain")
    sol = solve_ivp(lambda t, state: rates(state, strain_rate, material, stress_control=stress_control),
        (0., duration), y.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13])
    if not sol.success or not np.isfinite(sol.y).all():
        raise RuntimeError("Resistance integration failed; original state retained")
    if (np.any(abs(sol.y[[1, 2, 4, 5]]) > bounds[:, None]+1e-6)
            or np.any(sol.y[5] < -1e-8)):
        raise RuntimeError("Memory invariant domain violated")
    accumulated = sol.y[[3, 6, 9]]
    tolerance = 10*(np.array([1e-13, 1e-13, 1e-11])[:, None]+rtol*abs(accumulated))
    if np.any(np.maximum.accumulate(accumulated, axis=1)-accumulated > tolerance):
        raise RuntimeError("Accumulated state decreased")
    end = sol.y[:, -1]
    balance = end[8]-y[8]-stored_energy(end, material)+stored_energy(y, material)-end[9]+y[9]
    if abs(balance) > 1e-7+1e-7*abs(end[8]-y[8]):
        raise RuntimeError("Integrated mechanical balance failed")
    return sol
