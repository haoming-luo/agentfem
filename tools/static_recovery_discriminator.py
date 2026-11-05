# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Synthetic tensile recovery discriminator, not a runtime material or P92 fit."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Parameters:
    young: float
    hardening: float
    yield_stress: float
    recovery_rate: float

    def __post_init__(self):
        values = (self.young, self.hardening, self.yield_stress, self.recovery_rate)
        if (any(isinstance(v, (bool, np.bool_)) or np.iscomplexobj(v) for v in values)
                or not np.isfinite(values).all() or min(values[:3]) <= 0 or values[3] < 0):
            raise ValueError("Require positive finite E, H, Y and nonnegative recovery rate.")


@dataclass(frozen=True)
class State:
    stress: float
    strength: float
    plastic_strain: float = 0.

    def __post_init__(self):
        values = (self.stress, self.strength, self.plastic_strain)
        if (any(isinstance(v, (bool, np.bool_)) or np.iscomplexobj(v) for v in values)
                or not np.isfinite(values).all() or min(values) < 0):
            raise ValueError("The discriminator supports finite nonnegative tensile states only.")


def update(old, strain_increment, dt, parameters):
    """Return new state and d(sigma_new)/d(epsilon_new), holding old fixed."""
    if (any(isinstance(v, (bool, np.bool_)) or np.iscomplexobj(v)
            for v in (strain_increment, dt))
            or not np.isfinite([strain_increment, dt]).all() or dt <= 0):
        raise ValueError("Require real finite strain increment and positive physical dt.")
    E, H, Y, k = (parameters.young, parameters.hardening,
                  parameters.yield_stress, parameters.recovery_rate)
    if old.stress > Y + old.strength + 1.e-12 * max(Y, old.stress):
        raise ValueError("Initial state is outside the yield surface.")
    predictor = old.stress + E * strain_increment
    if predictor < 0:
        raise ValueError("Compression requires a different branch; this test is tensile only.")
    factor = 1. + k * dt
    relaxed = old.strength / factor
    if predictor <= Y + relaxed:
        return State(predictor, relaxed, old.plastic_strain), E
    dp = (predictor - Y - relaxed) / (E + H / factor)
    return State(predictor - E * dp, (old.strength + H * dp) / factor,
                 old.plastic_strain + dp), E * H / (E * factor + H)


def activation_time(initial, parameters):
    """Exact first activation under a total-strain hold with nonzero recovery."""
    Y, k = parameters.yield_stress, parameters.recovery_rate
    if initial.stress > Y + initial.strength:
        raise ValueError("Initial state is outside the yield surface.")
    if k == 0 or initial.stress <= Y:
        return math.inf
    return math.log(initial.strength / (initial.stress - Y)) / k


def exact_hold(times, initial, parameters):
    """Independent continuous-time solution; no calls to the BE update."""
    if np.iscomplexobj(times):
        raise ValueError("Times must be real.")
    times = np.asarray(times, dtype=float)
    if not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("Times must be finite and nonnegative.")
    ta = activation_time(initial, parameters)
    E, H, Y, k = (parameters.young, parameters.hardening,
                  parameters.yield_stress, parameters.recovery_rate)
    strength = np.array(initial.strength * np.exp(-k * times), copy=True)
    stress = np.full_like(times, initial.stress)
    active = times >= ta
    strength[active] = (initial.stress - Y) * np.exp(-k * E / (E + H) * (times[active] - ta))
    stress[active] = Y + strength[active]
    plastic = initial.plastic_strain + (initial.stress - stress) / E
    return stress, strength, plastic


def refinement_report():
    parameters = Parameters(200000., 10000., 100., .02)
    initial = State(150., 100., .01)
    rows = []
    for count in (100, 200, 400, 800):
        times = np.linspace(0., 100., count + 1)
        current = initial
        states = [initial]
        for dt in np.diff(times):
            current, _ = update(current, 0., dt, parameters)
            states.append(current)
        exact = exact_hold(times, initial, parameters)
        rows.append({"increments": count,
                     "maximum_stress_error": float(np.max(np.abs(
                         np.array([s.stress for s in states]) - exact[0]))),
                     "final_stress": current.stress})
    return {"scope": "synthetic tensile material point; not P92 validation",
            "activation_time": activation_time(initial, parameters), "refinement": rows}


if __name__ == "__main__":
    import json
    print(json.dumps(refinement_report(), indent=2, allow_nan=False))
