"""Research uniaxial two-mechanism integrator, not an installed P92 law.

One elastic stress couples separate inelastic histories. Uses the existing
bounded-recovery/Jiang hypothesis per branch, not a published parameter card.
State: sigma, (x1,x2,R,p)_A, (x1,x2,R,p)_B, total axial strain.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from tools.p92_published_recovery_reference import Parameters, isotropic_strength_lower_bound
from tools.p92_ratcheting_screen import corrected_rates


@dataclass(frozen=True)
class Mechanism:
    parameters: Parameters
    exponent: float = 3.
    enabled: bool = True

    def __post_init__(self):
        p = self.parameters
        if (type(self.enabled) is not bool or not np.isfinite(self.exponent)
                or self.exponent < 0 or not np.isfinite(list(vars(p).values())).all()
                or min(p.E, p.Z, p.a1, p.a2) <= 0 or p.n < 1
                or min(p.C1, p.C2, p.gamma1, p.gamma2, p.q) < 0
                or p.m < 1 or isotropic_strength_lower_bound(p) <= 0):
            raise ValueError("Invalid mechanism parameters")


def rates(state, strain_rate, mechanisms, *, stress_control=False):
    """Evaluate simultaneous rates; never sum separately updated stresses."""
    if len(mechanisms) != 2 or mechanisms[0].parameters.E != mechanisms[1].parameters.E:
        raise ValueError("Two mechanisms must share one elastic modulus")
    values = np.zeros(10)
    flow = 0.
    for offset, mechanism in zip((1, 5), mechanisms, strict=True):
        if not mechanism.enabled:
            continue
        local = np.r_[state[0], state[offset:offset+4]]
        v = corrected_rates(local, 0., mechanism.parameters, exponent=mechanism.exponent)
        flow -= v[0]/mechanism.parameters.E
        values[offset:offset+4] = v[1:]
    E = mechanisms[0].parameters.E
    values[0] = 0. if stress_control else E*(strain_rate-flow)
    values[9] = flow if stress_control else strain_rate
    return values


def integrate(old, duration, strain_rate, mechanisms, *, stress_control=False,
              rtol=1e-8, method="Radau"):
    if np.iscomplexobj(old):
        raise ValueError("Require real state")
    state = np.asarray(old, dtype=float)
    if (state.shape != (10,) or not np.isfinite(state).all()
            or np.any(state[[4, 8]] < 0) or type(stress_control) is not bool
            or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1 or method not in {"Radau", "BDF"}):
        raise ValueError("Invalid state or integration controls")
    rates(state, strain_rate, mechanisms, stress_control=stress_control)
    for offset, m in zip((1, 5), mechanisms, strict=True):
        if m.enabled and isotropic_strength_lower_bound(m.parameters, state[offset+2]) <= 0:
            raise ValueError("Nonpositive branch strength bound")
    sol = solve_ivp(lambda t, y: rates(y, strain_rate, mechanisms,
                                     stress_control=stress_control),
                    (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
                    atol=[1e-8, 1e-8, 1e-8, 1e-8, 1e-12,
                          1e-8, 1e-8, 1e-8, 1e-12, 1e-12])
    if not sol.success or not np.isfinite(sol.y).all():
        raise RuntimeError("Two-mechanism integration failed; old state retained")
    for offset, m in zip((1, 5), mechanisms, strict=True):
        if (m.enabled and (np.any(np.diff(sol.y[offset+3]) < -1e-10)
                or sol.y[offset+3, -1] < state[offset+3]
                or np.any(m.parameters.k+sol.y[offset+2] <= 0))):
            raise RuntimeError("Two-mechanism state violates branch domain")
    if stress_control and not np.allclose(sol.y[0], state[0], atol=1e-8, rtol=0):
        raise RuntimeError("Prescribed stress not preserved")
    return sol
