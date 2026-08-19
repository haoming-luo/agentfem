"""Stress-life fatigue utilities for post-processing FEM results."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FatigueBlock:
    """One constant-amplitude block for cumulative-damage assessment."""

    stress_amplitude: float
    cycles: float
    label: str = ""

    def __post_init__(self) -> None:
        if (
            not isfinite(float(self.stress_amplitude))
            or self.stress_amplitude <= 0.0
        ):
            raise ValueError("FatigueBlock.stress_amplitude must be positive.")
        if not isfinite(float(self.cycles)) or self.cycles < 0.0:
            raise ValueError("FatigueBlock.cycles must be finite and nonnegative.")


@dataclass(frozen=True)
class StressCycle:
    """One rainflow-counted stress cycle or residual half-cycle."""

    stress_range: float
    mean_stress: float
    count: float

    def __post_init__(self) -> None:
        if not isfinite(float(self.stress_range)) or self.stress_range < 0.0:
            raise ValueError("StressCycle.stress_range must be finite and nonnegative.")
        if not isfinite(float(self.mean_stress)):
            raise ValueError("StressCycle.mean_stress must be finite.")
        if self.count not in {0.5, 1.0}:
            raise ValueError("StressCycle.count must be 0.5 or 1.0.")

    @property
    def stress_amplitude(self) -> float:
        return 0.5 * self.stress_range


@dataclass(frozen=True)
class FatigueAssessment:
    """Auditable stress-life assessment derived from one scalar history."""

    cycles: tuple[StressCycle, ...]
    damage: float
    repeated_history_life: float
    source: str = "array"
    mean_stress_correction: str = "none"

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "stress_life_fatigue_assessment",
            "source": self.source,
            "cycle_count": len(self.cycles),
            "counted_repetitions": float(
                sum(cycle.count for cycle in self.cycles)
            ),
            "damage": self.damage,
            "repeated_history_life": self.repeated_history_life,
            "mean_stress_correction": self.mean_stress_correction,
            "cycles": [
                {
                    "stress_range": cycle.stress_range,
                    "mean_stress": cycle.mean_stress,
                    "count": cycle.count,
                }
                for cycle in self.cycles
            ],
        }


@dataclass(frozen=True)
class BasquinCurve:
    """Fully reversed stress-life curve ``sigma_a = sigma_f' (2N)^b``."""

    fatigue_strength_coefficient: float
    fatigue_strength_exponent: float
    endurance_limit: float | None = None
    name: str = "Basquin S-N curve"

    def __post_init__(self) -> None:
        if (
            not isfinite(float(self.fatigue_strength_coefficient))
            or self.fatigue_strength_coefficient <= 0.0
        ):
            raise ValueError("fatigue_strength_coefficient must be positive.")
        if (
            not isfinite(float(self.fatigue_strength_exponent))
            or self.fatigue_strength_exponent >= 0.0
        ):
            raise ValueError("fatigue_strength_exponent must be finite and negative.")
        if self.endurance_limit is not None and (
            not isfinite(float(self.endurance_limit))
            or self.endurance_limit <= 0.0
        ):
            raise ValueError("endurance_limit must be positive when set.")

    def cycles_to_failure(self, stress_amplitude: float) -> float:
        amplitude = float(stress_amplitude)
        if not isfinite(amplitude) or amplitude <= 0.0:
            raise ValueError("stress_amplitude must be finite and positive.")
        if self.endurance_limit is not None and amplitude <= self.endurance_limit:
            return float("inf")
        reversals = (
            amplitude / self.fatigue_strength_coefficient
        ) ** (1.0 / self.fatigue_strength_exponent)
        return float(0.5 * reversals)

    def stress_amplitude(self, cycles_to_failure: float) -> float:
        cycles = float(cycles_to_failure)
        if not isfinite(cycles) or cycles <= 0.0:
            raise ValueError("cycles_to_failure must be finite and positive.")
        return float(
            self.fatigue_strength_coefficient
            * (2.0 * cycles) ** self.fatigue_strength_exponent
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": "basquin_stress_life",
            "fatigue_strength_coefficient": self.fatigue_strength_coefficient,
            "fatigue_strength_exponent": self.fatigue_strength_exponent,
            "endurance_limit": self.endurance_limit,
            "maturity": "postprocessor",
            "mean_stress_correction": False,
            "cycle_counting": False,
        }


@dataclass(frozen=True)
class TabulatedSNCurve:
    """Log-log interpolated S-N data with explicit extrapolation policy."""

    stress_amplitudes: tuple[float, ...]
    cycles: tuple[float, ...]
    extrapolation: str = "error"
    name: str = "tabulated S-N curve"

    def __post_init__(self) -> None:
        stress = np.asarray(self.stress_amplitudes, dtype=float)
        cycles = np.asarray(self.cycles, dtype=float)
        if stress.ndim != 1 or cycles.ndim != 1 or stress.size != cycles.size:
            raise ValueError("S-N stress and cycle arrays must have equal one-dimensional shape.")
        if stress.size < 2:
            raise ValueError("TabulatedSNCurve requires at least two points.")
        if np.any(~np.isfinite(stress)) or np.any(stress <= 0.0):
            raise ValueError("S-N stress amplitudes must be finite and positive.")
        if np.any(~np.isfinite(cycles)) or np.any(cycles <= 0.0):
            raise ValueError("S-N cycles must be finite and positive.")
        if np.any(np.diff(stress) >= 0.0) or np.any(np.diff(cycles) <= 0.0):
            raise ValueError(
                "TabulatedSNCurve requires decreasing stress and increasing cycles."
            )
        policy = self.extrapolation.lower()
        if policy not in {"error", "clip", "linear"}:
            raise ValueError("extrapolation must be 'error', 'clip', or 'linear'.")
        object.__setattr__(self, "stress_amplitudes", tuple(float(x) for x in stress))
        object.__setattr__(self, "cycles", tuple(float(x) for x in cycles))
        object.__setattr__(self, "extrapolation", policy)

    def cycles_to_failure(self, stress_amplitude: float) -> float:
        amplitude = float(stress_amplitude)
        if not isfinite(amplitude) or amplitude <= 0.0:
            raise ValueError("stress_amplitude must be finite and positive.")
        stress = np.asarray(self.stress_amplitudes)
        cycles = np.asarray(self.cycles)
        if self.extrapolation == "error" and not stress[-1] <= amplitude <= stress[0]:
            raise ValueError(
                f"stress_amplitude={amplitude} lies outside the tabulated range "
                f"[{stress[-1]}, {stress[0]}]."
            )
        if self.extrapolation == "clip":
            amplitude = float(np.clip(amplitude, stress[-1], stress[0]))
        log_cycles = np.interp(
            np.log(amplitude),
            np.log(stress[::-1]),
            np.log(cycles[::-1]),
        )
        if self.extrapolation == "linear":
            if amplitude > stress[0]:
                log_cycles = _linear_log_segment(
                    amplitude,
                    stress[:2],
                    cycles[:2],
                )
            elif amplitude < stress[-1]:
                log_cycles = _linear_log_segment(
                    amplitude,
                    stress[-2:],
                    cycles[-2:],
                )
        return float(np.exp(log_cycles))


def miner_damage(blocks: Iterable[FatigueBlock], curve) -> float:
    """Return Palmgren-Miner cumulative damage ``sum(n_i / N_i)``."""

    damage = 0.0
    for block in blocks:
        life = float(curve.cycles_to_failure(block.stress_amplitude))
        if np.isinf(life):
            continue
        damage += block.cycles / life
    return float(damage)


def life_scale_factor(blocks: Iterable[FatigueBlock], curve) -> float:
    """Return the number of repeated block sequences to Miner damage one."""

    damage = miner_damage(tuple(blocks), curve)
    return float("inf") if damage == 0.0 else 1.0 / damage


def turning_points(history) -> np.ndarray:
    """Return endpoints and local reversals from a scalar stress history.

    Consecutive duplicate samples are removed before reversal detection.  The
    returned sequence is the appropriate input to ASTM-style rainflow stack
    counting; time values are intentionally irrelevant to cycle counting.
    """

    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("stress history must be a one-dimensional array of length >= 2.")
    if not np.all(np.isfinite(values)):
        raise ValueError("stress history must contain only finite values.")
    keep = np.concatenate(([True], np.diff(values) != 0.0))
    selected = values[keep]
    if selected.size == 1:
        return selected.copy()
    slopes = np.diff(selected)
    reversals = np.concatenate(
        (
            [0],
            np.flatnonzero(slopes[:-1] * slopes[1:] < 0.0) + 1,
            [selected.size - 1],
        )
    )
    return selected[reversals]


def rainflow_cycles(history) -> tuple[StressCycle, ...]:
    """Count full and residual half-cycles from a scalar stress history.

    This is the standard three-point stack formulation.  It returns individual
    cycles rather than prematurely binning them, so callers can inspect means,
    apply a documented mean-stress correction, or choose their own S-N curve.
    """

    reversals = turning_points(history)
    if reversals.size < 2:
        return ()
    stack: list[float] = []
    cycles: list[StressCycle] = []
    for point in reversals:
        stack.append(float(point))
        while len(stack) >= 3:
            older_range = abs(stack[-2] - stack[-3])
            newer_range = abs(stack[-1] - stack[-2])
            if newer_range < older_range:
                break
            mean = 0.5 * (stack[-3] + stack[-2])
            if len(stack) == 3:
                cycles.append(StressCycle(older_range, mean, 0.5))
                stack.pop(0)
            else:
                cycles.append(StressCycle(older_range, mean, 1.0))
                last = stack.pop()
                stack.pop()
                stack.pop()
                stack.append(last)
    for first, second in zip(stack[:-1], stack[1:]):
        cycles.append(
            StressCycle(
                abs(second - first),
                0.5 * (first + second),
                0.5,
            )
        )
    return tuple(cycle for cycle in cycles if cycle.stress_range > 0.0)


def goodman_amplitude(
    stress_amplitude: float,
    mean_stress: float,
    ultimate_strength: float,
) -> float:
    """Return fully reversed amplitude using the linear Goodman correction."""

    amplitude = float(stress_amplitude)
    mean = float(mean_stress)
    strength = float(ultimate_strength)
    if amplitude < 0.0 or not isfinite(amplitude):
        raise ValueError("stress_amplitude must be finite and nonnegative.")
    if not isfinite(mean):
        raise ValueError("mean_stress must be finite.")
    if not isfinite(strength) or strength <= 0.0:
        raise ValueError("ultimate_strength must be finite and positive.")
    denominator = 1.0 - mean / strength
    if denominator <= 0.0:
        raise ValueError(
            "Linear Goodman correction is undefined for mean_stress >= "
            "ultimate_strength."
        )
    return amplitude / denominator


def damage_from_history(
    history,
    curve,
    *,
    ultimate_strength: float | None = None,
) -> float:
    """Rainflow count a stress history and apply Palmgren-Miner damage.

    When ``ultimate_strength`` is omitted, counted amplitudes are treated as
    already compatible with the supplied S-N curve.  Passing it applies a
    linear Goodman mean-stress correction cycle by cycle.
    """

    blocks = []
    for cycle in rainflow_cycles(history):
        amplitude = cycle.stress_amplitude
        if ultimate_strength is not None:
            amplitude = goodman_amplitude(
                amplitude,
                cycle.mean_stress,
                ultimate_strength,
            )
        if amplitude > 0.0:
            blocks.append(FatigueBlock(amplitude, cycle.count))
    return miner_damage(blocks, curve)


def assess_history(
    history,
    curve,
    *,
    ultimate_strength: float | None = None,
    source: str | None = None,
) -> FatigueAssessment:
    """Return cycles, Miner damage, and repeated-history life together.

    ``history`` may be an array or an AgentFEM ``HistoryResult``. Accepting the
    result object preserves a meaningful source name while keeping the fatigue
    algorithm independent of the FE solver.
    """

    values = getattr(history, "values", history)
    selected = np.asarray(values, dtype=float)
    if selected.ndim != 1:
        raise ValueError(
            "Stress-life assessment currently requires one scalar stress history."
        )
    cycles = rainflow_cycles(selected)
    blocks = []
    for cycle in cycles:
        amplitude = cycle.stress_amplitude
        if ultimate_strength is not None:
            amplitude = goodman_amplitude(
                amplitude,
                cycle.mean_stress,
                ultimate_strength,
            )
        if amplitude > 0.0:
            blocks.append(FatigueBlock(amplitude, cycle.count))
    damage = miner_damage(blocks, curve)
    return FatigueAssessment(
        cycles=cycles,
        damage=damage,
        repeated_history_life=float("inf") if damage == 0.0 else 1.0 / damage,
        source=source or getattr(history, "name", "array"),
        mean_stress_correction=(
            "none" if ultimate_strength is None else "linear_goodman"
        ),
    )


def assess_result_history(
    result,
    history_name: str,
    curve,
    *,
    ultimate_strength: float | None = None,
) -> FatigueAssessment:
    """Assess one named ``SimulationResult`` history with provenance."""

    try:
        history = result.histories[history_name]
    except KeyError as exc:
        raise KeyError(
            f"History {history_name!r} is absent; "
            f"available={tuple(result.histories)!r}."
        ) from exc
    return assess_history(
        history,
        curve,
        ultimate_strength=ultimate_strength,
        source=f"{getattr(result, 'name', 'result')}:{history_name}",
    )


def _linear_log_segment(amplitude: float, stress, cycles) -> float:
    x = np.log(np.asarray(stress, dtype=float))
    y = np.log(np.asarray(cycles, dtype=float))
    slope = (y[1] - y[0]) / (x[1] - x[0])
    return float(y[0] + slope * (np.log(amplitude) - x[0]))
