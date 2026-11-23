"""Parameter-neutral rate-sensitive recovery hypothesis, not a published P92 law.

Prior-rate-dependent kinematic hardening motivates the question, not this formula.
The bounded interpolation doubles the existing recovery exponent at zero flow
and recovers it at high flow. No fitted constant, state or recoverable mode added.
"""

from dataclasses import dataclass, replace

import numpy as np

from tools import p92_recoverable_spectrum as spectrum
from tools.p92_recoverable_mobility_reference import (
    Material as MobilityMaterial, integrate_response,
)

MODEL_ID = "research.scalar-rate-sensitive-recovery.v1"
STATE_LAYOUT = spectrum.STATE_LAYOUT


@dataclass(frozen=True)
class Material:
    spectrum: spectrum.Material

    def __post_init__(self):
        # Same neutral baseline restriction; no compounded mechanism hypotheses.
        MobilityMaterial(self.spectrum)


def current_material(state, material):
    branches = []
    for offset, branch in zip((1, 4), material.spectrum.backbone.branches, strict=True):
        effective = state[0]-np.sum(state[offset:offset+2])
        normalized = max(abs(effective)-branch.strength, 0.)/120.
        # reference_flow/(reference_flow+p_dot), evaluated without division by
        # a possibly tiny reference_flow; exactly the existing power-law speed.
        fraction = 1./(1.+normalized**branch.exponent)
        branches.append(replace(branch,
            ratcheting=tuple(chi*(1.+fraction) for chi in branch.ratcheting)))
    return replace(material.spectrum, backbone=replace(
        material.spectrum.backbone, branches=tuple(branches)))


def stored_energy(state, material):
    return spectrum.stored_energy(state, material.spectrum)


def rates(state, strain_rate, material, *, stress_control=False):
    return spectrum.rates(state, strain_rate, current_material(state, material),
                          stress_control=stress_control)


def integrate(old, duration, strain_rate, material, *, stress_control=False,
              method="Radau", rtol=1e-9):
    return integrate_response(old, duration, strain_rate, material, rates, stored_energy,
                              stress_control=stress_control, method=method, rtol=rtol)
