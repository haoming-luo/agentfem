"""Research discriminator: signed versus accumulated dynamic-recovery memory.

This is not the published Li model or a P92 qualification. It adds an explicit
signed-flow state rather than mistaking experimental strain offset for history.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_recoverable_spectrum as spectrum

MODEL_ID = "research.scalar-two-rate-reversal-memory.v1"
STATE_LAYOUT = (*spectrum.STATE_LAYOUT, "signed_total_flow_strain")


@dataclass(frozen=True)
class Material:
    spectrum: spectrum.Material
    ratios: tuple[float, float] = (1., 1.)
    rate: float = 1000.
    coordinate: str = "signed"

    def __post_init__(self):
        if (len(self.ratios) != 2 or not np.isfinite([*self.ratios, self.rate]).all()
                or min(self.ratios) <= 0 or self.rate < 0
                or self.coordinate not in {"signed", "accumulated"}):
            raise ValueError("Invalid positive recovery memory")


def stored_energy(state, material):
    return spectrum.stored_energy(state[:12], material.spectrum)


def rates(state, strain_rate, material, *, stress_control=False):
    z = abs(state[12]) if material.coordinate == "signed" else max(state[3]+state[6], 0.)
    multipliers = 1+(np.asarray(material.ratios)-1)*(-np.expm1(-material.rate*z))
    result = spectrum.rates(state[:12], strain_rate, material.spectrum,
        stress_control=stress_control, dynamic_multipliers=multipliers)
    flow = result[7]-result[0]/material.spectrum.backbone.modulus-sum(result[10:12])
    return np.r_[result, flow]


def integrate(old, duration, strain_rate, material, *, stress_control=False,
              method="Radau", rtol=1e-9):
    if np.iscomplexobj(old):
        raise ValueError("Require real state")
    state = np.asarray(old, dtype=float)
    if (state.shape != (13,) or not np.isfinite(state).all()
            or np.any(state[[3, 6, 9]] < 0) or abs(state[12]) > state[3]+state[6]+1e-11
            or type(stress_control) is not bool or method not in {"Radau", "BDF"}
            or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid reversal-memory state or controls")
    # m<1 expands the former saturation envelope; do not reuse its old bound.
    for offset, branch, ratio in zip((1, 4), material.spectrum.backbone.branches, material.ratios, strict=True):
        bound = np.asarray(branch.saturation)/min(1., ratio)**(1/(1+np.asarray(branch.ratcheting)))
        if np.any(abs(state[offset:offset+2]) > bound+1e-7):
            raise ValueError("Initial backstress outside memory envelope")
    solution = solve_ivp(lambda t, y: rates(y, strain_rate, material, stress_control=stress_control),
        (0., duration), state.copy(), method=method, rtol=rtol, dense_output=True,
        atol=[1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13, 1e-13])
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("Memory integration failed; input retained")
    for offset, branch, ratio in zip((1, 4), material.spectrum.backbone.branches, material.ratios, strict=True):
        bound = np.asarray(branch.saturation)/min(1., ratio)**(1/(1+np.asarray(branch.ratcheting)))
        if np.any(abs(solution.y[offset:offset+2]) > bound[:, None]+1e-6):
            raise RuntimeError("Memory backstress envelope violated")
    accumulated = solution.y[[3, 6, 9]]
    tolerance = 10*(np.array([1e-13, 1e-13, 1e-11])[:, None]+rtol*abs(accumulated))
    if np.any(np.maximum.accumulate(accumulated, axis=1)-accumulated > tolerance):
        raise RuntimeError("Accumulated state decreased")
    if np.any(abs(solution.y[12])-solution.y[3]-solution.y[6] > 1e-10):
        raise RuntimeError("Signed flow exceeds accumulated flow")
    end = solution.y[:, -1]
    balance = end[8]-state[8]-stored_energy(end, material)+stored_energy(state, material)-end[9]+state[9]
    if abs(balance) > 1e-7+1e-7*abs(end[8]-state[8]):
        raise RuntimeError("Integrated mechanical balance failed")
    return solution
