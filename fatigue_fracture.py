"""Cycle-coordinate and cohesive-fatigue building blocks.

Fatigue cycles are a physical coordinate in their own right.  They are not
aliases for transient time increments or output frames.  This module owns the
solver-neutral contracts used by the global cyclic-fatigue lifecycle:

* a force-cycle history that can also drive ordinary time integration;
* an auditable adaptive cycle-jump decision;
* a cyclic Mode-I cohesive law layered on the monotonic bilinear envelope;
* trial/commit/rollback state for cohesive quadrature points.

The reference fatigue evolution law is intentionally replaceable.  The
transaction and cycle lifecycle are the platform contract; one calibration
equation is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from math import cos, floor, isfinite, pi
from pathlib import Path

import numpy as np

from . import amplitudes
from .interfaces import BilinearCohesiveLaw, CohesiveResponse


def _finite_array(value, *, name: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(selected)):
        raise ValueError(f"{name} must contain only finite values.")
    return selected


def _cycle_manifest_path(path, *, suffix: str) -> Path:
    selected = Path(path)
    ending = f".{suffix}.json"
    return selected if str(selected).endswith(ending) else Path(f"{selected}{ending}")


def _field_partition_identity(value) -> dict[str, object]:
    from . import checkpointing

    try:
        identity = checkpointing.function_partition_identity(value)
    except (AttributeError, TypeError):
        identity = {
            "schema": "agentfem.array-layout-identity.v1",
            "local_size": int(np.asarray(value.x.array).size),
            "shape": list(np.asarray(value.x.array).shape),
        }
    return json.loads(json.dumps(identity, sort_keys=True))


@dataclass(frozen=True)
class ForceCycle:
    """One scalar cyclic-load definition expressed in test parameters.

    ``minimum`` and ``maximum`` are physical load magnitudes.  The cycle starts
    at the minimum, reaches the maximum at phase 0.5, and returns to the
    minimum at phase 1.  ``frequency`` is used only when converting the cycle
    to a physical-time amplitude; cycle-jump execution consumes the extrema
    directly and never invents elapsed dynamic time.
    """

    minimum: float
    maximum: float
    frequency: float = 1.0
    waveform: str = "sine"
    hold_minimum_fraction: float = 0.0
    hold_maximum_fraction: float = 0.0
    table: tuple[tuple[float, float], ...] = ()
    name: str = "force cycle"

    def __post_init__(self) -> None:
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        frequency = float(self.frequency)
        waveform = str(self.waveform).strip().lower().replace("-", "_")
        if any(not isfinite(value) for value in (minimum, maximum, frequency)):
            raise ValueError("Force-cycle values must be finite.")
        if maximum <= minimum:
            raise ValueError("ForceCycle.maximum must exceed minimum.")
        if frequency <= 0.0:
            raise ValueError("ForceCycle.frequency must be positive.")
        if waveform not in {"sine", "triangle", "tabular"}:
            raise ValueError(
                "ForceCycle waveform must be 'sine', 'triangle', or 'tabular'."
            )
        hold_minimum = float(self.hold_minimum_fraction)
        hold_maximum = float(self.hold_maximum_fraction)
        if (
            not isfinite(hold_minimum)
            or not isfinite(hold_maximum)
            or hold_minimum < 0.0
            or hold_maximum < 0.0
            or hold_minimum + hold_maximum >= 1.0
        ):
            raise ValueError(
                "Force-cycle hold fractions must be nonnegative and sum below one."
            )
        table = tuple((float(phase), float(value)) for phase, value in self.table)
        if waveform == "tabular":
            if len(table) < 2:
                raise ValueError("A tabular force cycle requires at least two points.")
            phases = np.asarray([item[0] for item in table], dtype=float)
            values = np.asarray([item[1] for item in table], dtype=float)
            if (
                np.any(~np.isfinite(phases))
                or np.any(~np.isfinite(values))
                or phases[0] != 0.0
                or phases[-1] != 1.0
                or np.any(np.diff(phases) <= 0.0)
            ):
                raise ValueError(
                    "Tabular force-cycle phase must increase strictly from 0 to 1."
                )
            if np.min(values) != minimum or np.max(values) != maximum:
                raise ValueError(
                    "Tabular force-cycle values must attain the declared minimum "
                    "and maximum."
                )
            if hold_minimum > 0.0 or hold_maximum > 0.0:
                raise ValueError(
                    "Encode holds directly in a tabular cycle instead of combining "
                    "table and hold fractions."
                )
        elif table:
            raise ValueError("ForceCycle.table is only valid for waveform='tabular'.")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "waveform", waveform)
        object.__setattr__(self, "hold_minimum_fraction", hold_minimum)
        object.__setattr__(self, "hold_maximum_fraction", hold_maximum)
        object.__setattr__(self, "table", table)

    @property
    def load_ratio(self) -> float:
        if self.maximum == 0.0:
            raise ZeroDivisionError("A force cycle with maximum=0 has no load ratio.")
        return self.minimum / self.maximum

    def at_phase(self, phase: float) -> float:
        """Evaluate the physical load at a periodic phase in cycles."""

        selected = float(phase) % 1.0
        if self.waveform == "tabular":
            phases = np.asarray([item[0] for item in self.table], dtype=float)
            values = np.asarray([item[1] for item in self.table], dtype=float)
            return float(np.interp(selected, phases, values))

        minimum_half_hold = 0.5 * self.hold_minimum_fraction
        active_half = 0.5 * (
            1.0 - self.hold_minimum_fraction - self.hold_maximum_fraction
        )
        rise_end = minimum_half_hold + active_half
        maximum_hold_end = rise_end + self.hold_maximum_fraction
        if selected <= minimum_half_hold or selected >= 1.0 - minimum_half_hold:
            blend = 0.0
        elif selected < rise_end:
            local = (selected - minimum_half_hold) / active_half
            blend = (
                0.5 * (1.0 - cos(pi * local))
                if self.waveform == "sine"
                else local
            )
        elif selected <= maximum_hold_end:
            blend = 1.0
        else:
            local = (selected - maximum_hold_end) / active_half
            blend = (
                0.5 * (1.0 + cos(pi * local))
                if self.waveform == "sine"
                else 1.0 - local
            )
        return self.minimum + (self.maximum - self.minimum) * blend

    def at_time(self, physical_time: float) -> float:
        return self.at_phase(float(physical_time) * self.frequency)

    def normalized_amplitude(self) -> amplitudes.Amplitude:
        """Return a time amplitude scaled to a load declared at ``maximum``."""

        if self.maximum == 0.0:
            raise ValueError("Cannot normalize a force cycle with maximum=0.")
        return amplitudes.Amplitude(
            name=self.name,
            kind=f"cyclic_{self.waveform}",
            value=lambda time: self.at_time(time) / self.maximum,
            metadata={
                "minimum": self.minimum,
                "maximum": self.maximum,
                "load_ratio": self.load_ratio,
                "frequency": self.frequency,
                "hold_minimum_fraction": self.hold_minimum_fraction,
                "hold_maximum_fraction": self.hold_maximum_fraction,
                "coordinate": "physical_time",
                "cycle_coordinate": "frequency * physical_time",
            },
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "force_cycle",
            "minimum": self.minimum,
            "maximum": self.maximum,
            "load_ratio": self.load_ratio,
            "frequency": self.frequency,
            "waveform": self.waveform,
            "hold_minimum_fraction": self.hold_minimum_fraction,
            "hold_maximum_fraction": self.hold_maximum_fraction,
            "table": [list(item) for item in self.table],
            "cycle_coordinate_independent_of_output_frames": True,
        }


def force_cycle(
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    frequency: float = 1.0,
    waveform: str = "sine",
    hold_minimum_fraction: float = 0.0,
    hold_maximum_fraction: float = 0.0,
    table=(),
    name: str = "force cycle",
) -> ForceCycle:
    """Create a cyclic force from ``minimum/maximum`` or ``fmin/fmax``."""

    low = minimum if minimum is not None else fmin
    high = maximum if maximum is not None else fmax
    if low is None or high is None:
        raise ValueError("force_cycle requires minimum/maximum or fmin/fmax.")
    if minimum is not None and fmin is not None and float(minimum) != float(fmin):
        raise ValueError("minimum and fmin disagree.")
    if maximum is not None and fmax is not None and float(maximum) != float(fmax):
        raise ValueError("maximum and fmax disagree.")
    return ForceCycle(
        low,
        high,
        frequency=frequency,
        waveform=waveform,
        hold_minimum_fraction=hold_minimum_fraction,
        hold_maximum_fraction=hold_maximum_fraction,
        table=tuple(tuple(item) for item in table),
        name=name,
    )


@dataclass(frozen=True)
class CycleJumpDecision:
    """One inspectable proposal for advancing the independent cycle count."""

    start_cycle: int
    cycles: int
    end_cycle: int
    reason: str
    controlling_damage_rate: float
    predicted_damage_increment: float
    controlling_front_advance_rate: float = 0.0
    predicted_front_advance: float = 0.0
    exact_landing_target: int | None = None

    def summary(self) -> dict[str, object]:
        return {
            "start_cycle": self.start_cycle,
            "cycles": self.cycles,
            "end_cycle": self.end_cycle,
            "reason": self.reason,
            "controlling_damage_rate": self.controlling_damage_rate,
            "predicted_damage_increment": self.predicted_damage_increment,
            "controlling_front_advance_rate": self.controlling_front_advance_rate,
            "predicted_front_advance": self.predicted_front_advance,
            "exact_landing_target": self.exact_landing_target,
        }

    def cutback(self, cycles: int, *, reason: str = "global_error_cutback"):
        """Return a smaller decision that preserves the same cycle origin."""

        selected = int(cycles)
        if selected < 1 or selected >= self.cycles:
            raise ValueError("A cycle cutback must lie in [1, current cycles).")
        return replace(
            self,
            cycles=selected,
            end_cycle=self.start_cycle + selected,
            reason=str(reason),
            predicted_damage_increment=(
                self.controlling_damage_rate * selected
            ),
            predicted_front_advance=(
                self.controlling_front_advance_rate * selected
            ),
            exact_landing_target=(
                self.exact_landing_target
                if self.start_cycle + selected == self.exact_landing_target
                else None
            ),
        )


@dataclass(frozen=True)
class CycleJumpPolicy:
    """Bound a cycle block by predicted damage and exact output landings."""

    maximum_damage_increment: float = 0.01
    minimum_cycles: int = 1
    maximum_cycles: int = 100_000
    safety_factor: float = 0.8
    maximum_front_advance: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < float(self.maximum_damage_increment) <= 1.0:
            raise ValueError("maximum_damage_increment must lie in (0, 1].")
        if int(self.minimum_cycles) < 1:
            raise ValueError("minimum_cycles must be at least one.")
        if int(self.maximum_cycles) < int(self.minimum_cycles):
            raise ValueError("maximum_cycles must not be smaller than minimum_cycles.")
        if not 0.0 < float(self.safety_factor) <= 1.0:
            raise ValueError("safety_factor must lie in (0, 1].")
        if self.maximum_front_advance is not None and (
            not isfinite(float(self.maximum_front_advance))
            or float(self.maximum_front_advance) <= 0.0
        ):
            raise ValueError("maximum_front_advance must be finite and positive.")

    def propose(
        self,
        *,
        start_cycle: int,
        damage_rate,
        front_advance_rate=0.0,
        stop_cycle: int,
        landing_cycles=(),
    ) -> CycleJumpDecision:
        """Propose an integer jump without stepping across a required cycle."""

        start = int(start_cycle)
        stop = int(stop_cycle)
        if start < 0 or stop <= start:
            raise ValueError("Cycle range must satisfy 0 <= start_cycle < stop_cycle.")
        rates = _finite_array(damage_rate, name="damage_rate")
        if np.any(rates < 0.0):
            raise ValueError("damage_rate cannot be negative.")
        controlling = float(np.max(rates, initial=0.0))
        front_rates = _finite_array(front_advance_rate, name="front_advance_rate")
        if np.any(front_rates < 0.0):
            raise ValueError("front_advance_rate cannot be negative.")
        controlling_front = float(np.max(front_rates, initial=0.0))
        if controlling == 0.0:
            proposed = int(self.maximum_cycles)
            reason = "maximum_jump_no_active_damage"
        else:
            proposed = floor(
                self.safety_factor * self.maximum_damage_increment / controlling
            )
            proposed = max(int(self.minimum_cycles), int(proposed))
            reason = "damage_increment_limit"
        if self.maximum_front_advance is not None and controlling_front > 0.0:
            front_proposed = floor(
                self.safety_factor
                * float(self.maximum_front_advance)
                / controlling_front
            )
            front_proposed = max(int(self.minimum_cycles), int(front_proposed))
            if front_proposed < proposed:
                proposed = front_proposed
                reason = "front_advance_limit"
        proposed = min(proposed, int(self.maximum_cycles), stop - start)

        targets = sorted(
            {
                int(value)
                for value in landing_cycles
                if start < int(value) <= stop
            }
        )
        landing = None
        if targets and start + proposed >= targets[0]:
            proposed = targets[0] - start
            landing = targets[0]
            reason = "exact_landing"
        proposed = max(1, int(proposed))
        return CycleJumpDecision(
            start_cycle=start,
            cycles=proposed,
            end_cycle=start + proposed,
            reason=reason,
            controlling_damage_rate=controlling,
            predicted_damage_increment=controlling * proposed,
            controlling_front_advance_rate=controlling_front,
            predicted_front_advance=controlling_front * proposed,
            exact_landing_target=landing,
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "adaptive_cycle_jump_policy",
            "maximum_damage_increment": self.maximum_damage_increment,
            "minimum_cycles": self.minimum_cycles,
            "maximum_cycles": self.maximum_cycles,
            "safety_factor": self.safety_factor,
            "maximum_front_advance": self.maximum_front_advance,
            "acceptance": "global consumer must re-solve and may rollback/cut back",
        }


@dataclass(frozen=True)
class CycleJumpRecord:
    """Accepted or rejected cycle-block evidence."""

    decision: CycleJumpDecision
    accepted: bool
    error_estimate: float
    message: str

    def summary(self) -> dict[str, object]:
        return {
            **self.decision.summary(),
            "accepted": self.accepted,
            "error_estimate": self.error_estimate,
            "message": self.message,
        }


class CycleJumpLedger:
    """Record exact cycle progress and every jump/cutback decision."""

    _SCHEMA = "agentfem.cycle-jump-ledger.v1"

    def __init__(self, *, start_cycle: int = 0):
        if int(start_cycle) < 0:
            raise ValueError("start_cycle cannot be negative.")
        self.start_cycle = int(start_cycle)
        self.current_cycle = int(start_cycle)
        self.records: list[CycleJumpRecord] = []
        self._trial: CycleJumpDecision | None = None

    def begin(self, decision: CycleJumpDecision) -> None:
        if self._trial is not None:
            raise RuntimeError("A cycle-jump decision is already active.")
        if decision.start_cycle != self.current_cycle:
            raise ValueError("Cycle-jump decision does not start at current_cycle.")
        self._trial = decision

    def commit(self, *, error_estimate: float, message: str = "accepted") -> None:
        if self._trial is None:
            raise RuntimeError("No cycle-jump decision is active.")
        error = float(error_estimate)
        if not isfinite(error) or error < 0.0:
            raise ValueError("error_estimate must be finite and nonnegative.")
        self.records.append(CycleJumpRecord(self._trial, True, error, str(message)))
        self.current_cycle = self._trial.end_cycle
        self._trial = None

    def rollback(self, *, error_estimate: float, message: str = "cutback") -> None:
        if self._trial is None:
            raise RuntimeError("No cycle-jump decision is active.")
        error = float(error_estimate)
        if not isfinite(error) or error < 0.0:
            raise ValueError("error_estimate must be finite and nonnegative.")
        self.records.append(CycleJumpRecord(self._trial, False, error, str(message)))
        self._trial = None

    def snapshot(self) -> dict[str, object]:
        if self._trial is not None:
            raise RuntimeError("Rollback an active cycle block before checkpointing.")
        return {
            "schema": self._SCHEMA,
            "start_cycle": self.start_cycle,
            "current_cycle": self.current_cycle,
            "records": [record.summary() for record in self.records],
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        if self._trial is not None:
            raise RuntimeError("Rollback the active cycle block before restore.")
        if snapshot.get("schema") != self._SCHEMA:
            raise ValueError("Unsupported cycle-jump ledger schema.")
        start = int(snapshot.get("start_cycle", 0))
        current = int(snapshot.get("current_cycle", -1))
        if start < 0 or current < start:
            raise ValueError("Cycle-jump checkpoint has an invalid cycle count.")
        restored = []
        for item in snapshot.get("records", []):
            decision = CycleJumpDecision(
                start_cycle=int(item["start_cycle"]),
                cycles=int(item["cycles"]),
                end_cycle=int(item["end_cycle"]),
                reason=str(item["reason"]),
                controlling_damage_rate=float(item["controlling_damage_rate"]),
                predicted_damage_increment=float(item["predicted_damage_increment"]),
                controlling_front_advance_rate=float(
                    item.get("controlling_front_advance_rate", 0.0)
                ),
                predicted_front_advance=float(item.get("predicted_front_advance", 0.0)),
                exact_landing_target=(
                    None
                    if item.get("exact_landing_target") is None
                    else int(item["exact_landing_target"])
                ),
            )
            restored.append(
                CycleJumpRecord(
                    decision=decision,
                    accepted=bool(item["accepted"]),
                    error_estimate=float(item["error_estimate"]),
                    message=str(item["message"]),
                )
            )
        accepted_end = [
            record.decision.end_cycle for record in restored if record.accepted
        ]
        expected = accepted_end[-1] if accepted_end else start
        if current != expected:
            raise ValueError("Cycle-jump checkpoint progress differs from its records.")
        self.start_cycle = start
        self.current_cycle = current
        self.records = restored

    def summary(self) -> dict[str, object]:
        return {
            "kind": "cycle_jump_ledger",
            "start_cycle": self.start_cycle,
            "current_cycle": self.current_cycle,
            "accepted_blocks": sum(record.accepted for record in self.records),
            "rejected_blocks": sum(not record.accepted for record in self.records),
            "records": [record.summary() for record in self.records],
        }


@dataclass(frozen=True)
class CyclicCohesiveResponse(CohesiveResponse):
    """Mode-I response with separated monotonic and fatigue evidence."""

    monotonic_damage: np.ndarray
    fatigue_damage: np.ndarray
    opening_minimum: np.ndarray
    opening_maximum: np.ndarray
    opening_range: np.ndarray
    local_load_ratio: np.ndarray
    cumulative_cycles: np.ndarray
    monotonic_dissipated_energy: np.ndarray
    fatigue_dissipated_energy: np.ndarray
    failed: np.ndarray


@dataclass(frozen=True)
class CyclicCohesiveLaw:
    """Replaceable power-law range fatigue layered on a bilinear envelope.

    At fixed cycle extrema the reference evolution is

    ``d D_f / dN = A * (1 - D_f)**p``

    where ``A`` depends on the normalized positive opening range above a
    threshold and, optionally, on peak opening.  Local opening extrema encode
    the local load-ratio effect; a global applied ``R`` is deliberately not
    injected into every integration point.  With no cycle advancement the
    response is exactly the wrapped monotonic bilinear law.

    This is an experimental, auditable reference model rather than a universal
    metal/polymer/composite fatigue law.  Alternative laws should implement
    the same transaction contract.
    """

    monotonic: BilinearCohesiveLaw
    fatigue_coefficient: float
    fatigue_exponent: float
    range_threshold: float
    peak_exponent: float = 0.0
    residual_exponent: float = 0.0
    name: str = "power-law cyclic Mode-I cohesive law"

    def __post_init__(self) -> None:
        values = {
            "fatigue_coefficient": self.fatigue_coefficient,
            "fatigue_exponent": self.fatigue_exponent,
            "peak_exponent": self.peak_exponent,
            "residual_exponent": self.residual_exponent,
        }
        if any(not isfinite(float(value)) or float(value) < 0.0 for value in values.values()):
            raise ValueError("Cyclic cohesive coefficients/exponents must be finite and nonnegative.")
        if float(self.fatigue_coefficient) <= 0.0:
            raise ValueError("fatigue_coefficient must be positive.")
        if not 0.0 <= float(self.range_threshold) < 1.0:
            raise ValueError("range_threshold must lie in [0, 1).")

    @property
    def strength(self) -> float:
        return self.monotonic.strength

    @property
    def fracture_energy(self) -> float:
        return self.monotonic.fracture_energy

    @property
    def initial_stiffness(self) -> float:
        return self.monotonic.initial_stiffness

    @property
    def closure_stiffness(self) -> float:
        return self.monotonic.closure_stiffness

    @property
    def peak_opening(self) -> float:
        return self.monotonic.peak_opening

    @property
    def failure_opening(self) -> float:
        return self.monotonic.failure_opening

    def characteristic_length(self, elastic_modulus: float) -> float:
        return self.monotonic.characteristic_length(elastic_modulus)

    def transaction(self, size: int) -> "CyclicCohesiveTransaction":
        return CyclicCohesiveTransaction(self, size)

    def damage_rate(self, minimum_opening, maximum_opening, fatigue_damage=0.0):
        minimum = np.maximum(_finite_array(minimum_opening, name="minimum_opening"), 0.0)
        maximum = np.maximum(_finite_array(maximum_opening, name="maximum_opening"), 0.0)
        fatigue = _finite_array(fatigue_damage, name="fatigue_damage")
        minimum, maximum, fatigue = np.broadcast_arrays(minimum, maximum, fatigue)
        if np.any(maximum < minimum):
            raise ValueError("maximum_opening must not be smaller than minimum_opening.")
        if np.any((fatigue < 0.0) | (fatigue > 1.0)):
            raise ValueError("fatigue_damage must lie in [0, 1].")
        opening_range = maximum - minimum
        range_ratio = opening_range / self.failure_opening
        active_range = np.maximum(
            (range_ratio - self.range_threshold) / (1.0 - self.range_threshold),
            0.0,
        )
        peak_ratio = np.clip(maximum / self.failure_opening, 0.0, 1.0)
        rate = (
            self.fatigue_coefficient
            * active_range**self.fatigue_exponent
            * peak_ratio**self.peak_exponent
            * np.maximum(1.0 - fatigue, 0.0) ** self.residual_exponent
        )
        return np.where((maximum > 0.0) & (active_range > 0.0), rate, 0.0)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": "cohesive_traction_separation",
            "mode": "normal",
            "envelope": "bilinear",
            "cyclic_evolution": "power_law_positive_opening_range",
            "fatigue_coefficient_per_cycle": self.fatigue_coefficient,
            "fatigue_exponent": self.fatigue_exponent,
            "range_threshold_fraction_of_failure_opening": self.range_threshold,
            "peak_exponent": self.peak_exponent,
            "residual_exponent": self.residual_exponent,
            "load_ratio_effect": "local_positive_opening_minimum_over_maximum",
            "monotonic_limit": self.monotonic.summary(),
            "state": [
                "maximum_opening",
                "fatigue_damage",
                "opening_minimum",
                "opening_maximum",
                "cumulative_cycles",
                "fatigue_dissipated_energy",
            ],
            "maturity": "experimental_material_point_and_facet_consumer",
        }


class CyclicCohesiveTransaction:
    """Atomic monotonic trials and cycle-block trials for cohesive points."""

    _SCHEMA = "agentfem.cyclic-cohesive-state.v1"

    def __init__(self, law: CyclicCohesiveLaw, size: int):
        if int(size) <= 0:
            raise ValueError("CyclicCohesiveTransaction.size must be positive.")
        self.law = law
        count = int(size)
        self._maximum = np.zeros(count)
        self._fatigue = np.zeros(count)
        self._minimum = np.zeros(count)
        self._cycle_maximum = np.zeros(count)
        self._cycles = np.zeros(count)
        self._fatigue_dissipation = np.zeros(count)
        self._trial_response: CyclicCohesiveResponse | None = None
        self._trial_state: dict[str, np.ndarray] | None = None

    @property
    def size(self) -> int:
        return int(self._maximum.size)

    @property
    def committed_maximum(self) -> np.ndarray:
        return self._maximum.copy()

    @property
    def fatigue_damage(self) -> np.ndarray:
        return self._fatigue.copy()

    @property
    def cumulative_cycles(self) -> np.ndarray:
        return self._cycles.copy()

    @property
    def trial(self) -> CyclicCohesiveResponse | None:
        return self._trial_response

    def _response(
        self,
        opening,
        *,
        maximum=None,
        fatigue=None,
        dissipation=None,
        opening_minimum=None,
        opening_maximum=None,
        cumulative_cycles=None,
    ):
        values = _finite_array(opening, name="opening")
        if values.shape != self._maximum.shape:
            raise ValueError(f"opening must have shape {self._maximum.shape}.")
        history = self._maximum if maximum is None else maximum
        fatigue_state = self._fatigue if fatigue is None else fatigue
        fatigue_energy = self._fatigue_dissipation if dissipation is None else dissipation
        base = self.law.monotonic.update(values, history)
        fatigue_state = np.asarray(fatigue_state, dtype=float)
        tensile = values >= 0.0
        scale = np.where(tensile, 1.0 - fatigue_state, 1.0)
        total_damage = 1.0 - (1.0 - base.damage) * (1.0 - fatigue_state)
        minimum_history = self._minimum if opening_minimum is None else opening_minimum
        maximum_history = (
            self._cycle_maximum if opening_maximum is None else opening_maximum
        )
        cycle_history = self._cycles if cumulative_cycles is None else cumulative_cycles
        positive_minimum = np.maximum(minimum_history, 0.0)
        positive_maximum = np.maximum(maximum_history, positive_minimum)
        opening_range = positive_maximum - positive_minimum
        ratio = np.divide(
            positive_minimum,
            positive_maximum,
            out=np.zeros_like(positive_minimum),
            where=positive_maximum > 0.0,
        )
        return CyclicCohesiveResponse(
            opening=base.opening,
            traction=base.traction * scale,
            tangent=base.tangent * scale,
            maximum_opening=base.maximum_opening,
            damage=total_damage,
            stored_energy=base.stored_energy * scale,
            dissipated_energy=base.dissipated_energy + fatigue_energy,
            monotonic_damage=base.damage,
            fatigue_damage=fatigue_state.copy(),
            opening_minimum=positive_minimum.copy(),
            opening_maximum=positive_maximum.copy(),
            opening_range=opening_range,
            local_load_ratio=ratio,
            cumulative_cycles=np.asarray(cycle_history, dtype=float).copy(),
            monotonic_dissipated_energy=base.dissipated_energy,
            fatigue_dissipated_energy=np.asarray(fatigue_energy, dtype=float).copy(),
            failed=total_damage >= 1.0 - 1.0e-12,
        )

    def evaluate(self, opening) -> CyclicCohesiveResponse:
        return self._response(opening)

    def begin(self, opening) -> CyclicCohesiveResponse:
        if self._trial_state is not None:
            raise RuntimeError("A cycle-block trial is already active.")
        self._trial_response = self._response(opening)
        return self._trial_response

    def commit(self) -> None:
        if self._trial_response is None:
            raise RuntimeError("No cohesive trial state is available to commit.")
        self._maximum[:] = self._trial_response.maximum_opening
        self._trial_response = None

    def begin_cycle(self, minimum_opening, maximum_opening, *, cycles: int = 1):
        """Create a replaceable fatigue update for one accepted cycle block."""

        if self._trial_response is not None or self._trial_state is not None:
            raise RuntimeError("Commit or rollback the active cohesive trial first.")
        count = int(cycles)
        if count < 1:
            raise ValueError("cycles must be a positive integer.")
        minimum = _finite_array(minimum_opening, name="minimum_opening")
        maximum = _finite_array(maximum_opening, name="maximum_opening")
        if minimum.shape != self._maximum.shape or maximum.shape != self._maximum.shape:
            raise ValueError("Cycle extrema must match the cohesive transaction size.")
        positive_minimum = np.maximum(minimum, 0.0)
        positive_maximum = np.maximum(maximum, 0.0)
        if np.any(positive_maximum < positive_minimum):
            raise ValueError("Cycle maximum opening must not be below its minimum.")

        proposed_maximum = np.maximum(self._maximum, positive_maximum)
        monotonic_damage = self.law.monotonic.damage_from_maximum(proposed_maximum)
        amplitude = self.law.damage_rate(
            positive_minimum,
            positive_maximum,
            fatigue_damage=np.zeros_like(self._fatigue),
        )
        new_fatigue = _integrate_residual_power(
            self._fatigue,
            amplitude,
            count,
            self.law.residual_exponent,
        )
        new_fatigue = np.minimum(
            new_fatigue,
            np.where(monotonic_damage >= 1.0 - 1.0e-12, 1.0, 1.0),
        )
        increment = np.maximum(new_fatigue - self._fatigue, 0.0)
        release_rate = (
            0.5
            * (1.0 - monotonic_damage)
            * self.law.initial_stiffness
            * positive_maximum**2
        )
        new_dissipation = self._fatigue_dissipation + release_rate * increment
        new_cycles = self._cycles + count
        self._trial_state = {
            "maximum_opening": proposed_maximum,
            "fatigue_damage": new_fatigue,
            "opening_minimum": positive_minimum,
            "opening_maximum": positive_maximum,
            "cumulative_cycles": new_cycles,
            "fatigue_dissipated_energy": new_dissipation,
        }
        self._trial_response = self._response(
            positive_maximum,
            maximum=proposed_maximum,
            fatigue=new_fatigue,
            dissipation=new_dissipation,
            opening_minimum=positive_minimum,
            opening_maximum=positive_maximum,
            cumulative_cycles=new_cycles,
        )
        return self._trial_response

    def commit_cycle(self) -> None:
        if self._trial_state is None:
            raise RuntimeError("No cycle-block trial is available to commit.")
        self._maximum[:] = self._trial_state["maximum_opening"]
        self._fatigue[:] = self._trial_state["fatigue_damage"]
        self._minimum[:] = self._trial_state["opening_minimum"]
        self._cycle_maximum[:] = self._trial_state["opening_maximum"]
        self._cycles[:] = self._trial_state["cumulative_cycles"]
        self._fatigue_dissipation[:] = self._trial_state[
            "fatigue_dissipated_energy"
        ]
        self._trial_state = None
        self._trial_response = None

    def rollback(self) -> None:
        self._trial_state = None
        self._trial_response = None

    def initialize(self, maximum_opening) -> None:
        values = np.broadcast_to(
            _finite_array(maximum_opening, name="maximum_opening"),
            self._maximum.shape,
        )
        if np.any(values < 0.0):
            raise ValueError("Initial cohesive maximum opening cannot be negative.")
        self._maximum[:] = values
        self._trial_state = None
        self._trial_response = None

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": self._SCHEMA,
            "law": self.law.summary(),
            **{name: values.tolist() for name, values in self.state_arrays().items()},
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        if snapshot.get("schema") != self._SCHEMA:
            raise ValueError("Unsupported cyclic cohesive-state schema.")
        if snapshot.get("law") != self.law.summary():
            raise ValueError("Cyclic cohesive-state law differs from current law.")
        self.restore_state_arrays(
            {
                name: snapshot.get(name)
                for name in self.state_arrays()
            }
        )

    def state_arrays(self) -> dict[str, np.ndarray]:
        return {
            "maximum_opening": self._maximum.copy(),
            "fatigue_damage": self._fatigue.copy(),
            "opening_minimum": self._minimum.copy(),
            "opening_maximum": self._cycle_maximum.copy(),
            "cumulative_cycles": self._cycles.copy(),
            "fatigue_dissipated_energy": self._fatigue_dissipation.copy(),
        }

    def restore_state_arrays(self, arrays: dict[str, object]) -> None:
        expected = self.state_arrays()
        if set(arrays) != set(expected):
            raise ValueError("Cyclic cohesive-state fields differ from this law.")
        selected = {}
        for name in expected:
            values = _finite_array(arrays[name], name=name)
            if values.shape != self._maximum.shape:
                raise ValueError(f"Cyclic cohesive-state field {name!r} has wrong shape.")
            selected[name] = values
        if np.any(selected["maximum_opening"] < 0.0):
            raise ValueError("maximum_opening cannot be negative.")
        if np.any((selected["fatigue_damage"] < 0.0) | (selected["fatigue_damage"] > 1.0)):
            raise ValueError("fatigue_damage must lie in [0, 1].")
        if np.any(selected["cumulative_cycles"] < 0.0):
            raise ValueError("cumulative_cycles cannot be negative.")
        self._maximum[:] = selected["maximum_opening"]
        self._fatigue[:] = selected["fatigue_damage"]
        self._minimum[:] = selected["opening_minimum"]
        self._cycle_maximum[:] = selected["opening_maximum"]
        self._cycles[:] = selected["cumulative_cycles"]
        self._fatigue_dissipation[:] = selected["fatigue_dissipated_energy"]
        self.rollback()


class FieldStateTransaction:
    """In-memory rollback for bulk fields and other transactional assets.

    This adapter is deliberately small: a global cycle block must be able to
    restore every equilibrium unknown, while durable MPI-portable persistence
    remains the responsibility of the ordinary AgentFEM checkpoint layer.
    """

    _SCHEMA = "agentfem.field-state-transaction.v1"

    def __init__(self, fields, *, assets=None):
        records = dict(fields)
        if not records:
            raise ValueError("FieldStateTransaction requires at least one field.")
        self.fields = {}
        for name, value in records.items():
            selected = getattr(value, "value", value)
            if not hasattr(selected, "x") or not hasattr(selected.x, "array"):
                raise TypeError(f"Transactional field {name!r} has no x.array state.")
            self.fields[str(name)] = selected
        self.assets = dict(() if assets is None else assets)
        for name, asset in self.assets.items():
            if not callable(getattr(asset, "snapshot", None)) or not callable(
                getattr(asset, "restore", None)
            ):
                raise TypeError(
                    f"Transactional asset {name!r} needs snapshot() and restore()."
                )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": self._SCHEMA,
            "fields": {
                name: np.asarray(value.x.array).copy()
                for name, value in self.fields.items()
            },
            "assets": {
                str(name): asset.snapshot()
                for name, asset in self.assets.items()
            },
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        if snapshot.get("schema") != self._SCHEMA:
            raise ValueError("Unsupported field-state transaction schema.")
        stored_fields = snapshot.get("fields", {})
        stored_assets = snapshot.get("assets", {})
        if set(stored_fields) != set(self.fields):
            raise ValueError("Transactional field names differ from the snapshot.")
        if set(stored_assets) != {str(name) for name in self.assets}:
            raise ValueError("Transactional asset names differ from the snapshot.")
        for name, value in self.fields.items():
            restored = _finite_array(stored_fields[name], name=name)
            if restored.shape != value.x.array.shape:
                raise ValueError(f"Transactional field {name!r} layout differs.")
            value.x.array[:] = restored
            if callable(getattr(value.x, "scatter_forward", None)):
                value.x.scatter_forward()
        for name, asset in self.assets.items():
            asset.restore(stored_assets[str(name)])

    def summary(self) -> dict[str, object]:
        return {
            "kind": "field_state_transaction",
            "fields": tuple(self.fields),
            "assets": tuple(str(name) for name in self.assets),
            "persistence": "in_memory_rollback",
        }

    def save_checkpoint(self, path) -> Path:
        """Persist field shards for a same-partition cycle restart."""

        if self.assets:
            raise NotImplementedError(
                "Durable FieldStateTransaction checkpoints currently accept "
                "fields only; persist auxiliary constitutive assets with their "
                "own physical-identity checkpoint adapter."
            )
        from . import checkpointing

        first = next(iter(self.fields.values()))
        comm = first.function_space.mesh.comm
        manifest = _cycle_manifest_path(path, suffix="field-state")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        shard = manifest.with_name(
            f"{manifest.name.removesuffix('.json')}.rank-{comm.rank:05d}.npz"
        )
        checkpointing.atomic_savez(
            shard,
            **{name: value.x.array for name, value in self.fields.items()},
        )
        local = {
            "path": shard.name,
            "size": int(shard.stat().st_size),
            "identity": {
                name: _field_partition_identity(value)
                for name, value in self.fields.items()
            },
        }
        shards = comm.gather(local, root=0)
        error = None
        if comm.rank == 0:
            try:
                checkpointing.atomic_write_text(
                    manifest,
                    json.dumps(
                        {
                            "schema": "agentfem.cyclic-field-checkpoint.v1",
                            "rank_count": int(comm.size),
                            "fields": list(self.fields),
                            "shards": shards,
                            "portability": "same MPI partition and rank count",
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                )
            except Exception as exc:  # pragma: no cover - filesystem failure
                error = f"{type(exc).__name__}: {exc}"
        error = comm.bcast(error, root=0)
        if error is not None:
            raise RuntimeError(f"Cyclic field checkpoint write failed: {error}")
        comm.barrier()
        return manifest

    def load_checkpoint(self, path) -> dict[str, object]:
        """Restore a same-partition field checkpoint after identity checks."""

        first = next(iter(self.fields.values()))
        comm = first.function_space.mesh.comm
        manifest = _cycle_manifest_path(path, suffix="field-state")
        payload = None
        if comm.rank == 0:
            try:
                payload = {"metadata": json.loads(manifest.read_text()), "error": None}
            except Exception as exc:
                payload = {"metadata": None, "error": f"{type(exc).__name__}: {exc}"}
        payload = comm.bcast(payload, root=0)
        if payload["error"] is not None:
            raise RuntimeError(
                f"Cyclic field checkpoint read failed: {payload['error']}"
            )
        metadata = payload["metadata"]
        if metadata.get("schema") != "agentfem.cyclic-field-checkpoint.v1":
            raise ValueError("Unsupported cyclic field checkpoint schema.")
        if int(metadata.get("rank_count", -1)) != int(comm.size):
            raise ValueError("Cyclic field checkpoint MPI rank count differs.")
        if metadata.get("fields") != list(self.fields):
            raise ValueError("Cyclic field checkpoint field names differ.")
        shard = metadata["shards"][comm.rank]
        current_identity = {
            name: _field_partition_identity(value)
            for name, value in self.fields.items()
        }
        if shard.get("identity") != current_identity:
            raise ValueError("Cyclic field checkpoint partition identity differs.")
        selected = manifest.with_name(shard["path"])
        if int(selected.stat().st_size) != int(shard["size"]):
            raise ValueError("Cyclic field checkpoint shard size differs.")
        with np.load(selected, allow_pickle=False) as arrays:
            if set(arrays.files) != set(self.fields):
                raise ValueError("Cyclic field checkpoint shard fields differ.")
            for name, value in self.fields.items():
                restored = np.asarray(arrays[name])
                if restored.shape != value.x.array.shape:
                    raise ValueError(
                        f"Cyclic field checkpoint layout differs for {name!r}."
                    )
                value.x.array[:] = restored
                if callable(getattr(value.x, "scatter_forward", None)):
                    value.x.scatter_forward()
        return metadata


@dataclass(frozen=True)
class CyclicEquilibriumPoint:
    """Evidence returned by one converged peak or valley equilibrium solve."""

    branch: str
    load: float
    cycle: int
    converged: bool = True
    iterations: int = 0
    reaction: float | None = None
    control_displacement: float | None = None
    external_work: float | None = None
    bulk_strain_energy: float | None = None
    energy_balance_error: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        branch = str(self.branch).strip().lower()
        if branch not in {"minimum", "maximum", "verification_maximum", "closing"}:
            raise ValueError("Unknown cyclic-equilibrium branch.")
        if not isfinite(float(self.load)):
            raise ValueError("Equilibrium load must be finite.")
        if int(self.cycle) < 0 or int(self.iterations) < 0:
            raise ValueError("Equilibrium cycle and iteration counts cannot be negative.")
        for name in (
            "reaction",
            "control_displacement",
            "external_work",
            "bulk_strain_energy",
            "energy_balance_error",
        ):
            value = getattr(self, name)
            if value is not None and not isfinite(float(value)):
                raise ValueError(f"Equilibrium evidence {name!r} must be finite.")
        if self.energy_balance_error is not None and self.energy_balance_error < 0.0:
            raise ValueError("energy_balance_error cannot be negative.")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "load", float(self.load))
        object.__setattr__(self, "cycle", int(self.cycle))
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def coerce(cls, value, *, branch: str, load: float, cycle: int):
        if isinstance(value, cls):
            if value.branch != branch or value.load != float(load) or value.cycle != int(cycle):
                raise ValueError("Equilibrium callback returned mismatched coordinates.")
            return value
        if value is None:
            return cls(branch=branch, load=load, cycle=cycle)
        if isinstance(value, bool):
            return cls(branch=branch, load=load, cycle=cycle, converged=value)
        if isinstance(value, dict):
            selected = dict(value)
            selected.setdefault("branch", branch)
            selected.setdefault("load", load)
            selected.setdefault("cycle", cycle)
            return cls(**selected)
        raise TypeError(
            "Equilibrium callback must return None, bool, a mapping, or "
            "CyclicEquilibriumPoint."
        )

    def summary(self) -> dict[str, object]:
        return {
            "branch": self.branch,
            "load": self.load,
            "cycle": self.cycle,
            "converged": bool(self.converged),
            "iterations": self.iterations,
            "reaction": self.reaction,
            "control_displacement": self.control_displacement,
            "external_work": self.external_work,
            "bulk_strain_energy": self.bulk_strain_energy,
            "energy_balance_error": self.energy_balance_error,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CyclicFatigueBlock:
    """Accepted structure-level cycle block and its error evidence."""

    decision: CycleJumpDecision
    minimum: CyclicEquilibriumPoint
    maximum: CyclicEquilibriumPoint
    verification_maximum: CyclicEquilibriumPoint
    closing: CyclicEquilibriumPoint
    maximum_damage_increment: float
    opening_feedback_error: float
    energy_balance_error: float
    observations: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "accepted_cyclic_fatigue_block",
            "decision": self.decision.summary(),
            "minimum": self.minimum.summary(),
            "maximum": self.maximum.summary(),
            "verification_maximum": self.verification_maximum.summary(),
            "closing": self.closing.summary(),
            "maximum_damage_increment": self.maximum_damage_increment,
            "opening_feedback_error": self.opening_feedback_error,
            "energy_balance_error": self.energy_balance_error,
            "observations": {
                name: value.summary() if hasattr(value, "summary") else value
                for name, value in self.observations.items()
            },
        }


class GlobalCyclicFatigueStep:
    """Quasi-static peak/valley fatigue loop with global rollback and cutback.

    ``solve_equilibrium`` owns the finite-element nonlinear solve but must not
    commit irreversible cohesive history. A replaceable equilibrium trial may
    remain because the controller immediately evaluates and rolls back the
    accepted opening. This controller owns the larger cycle-block transaction:
    bulk fields, every named interface, the cycle ledger, exact output
    landings, and the post-damage verification equilibrium are accepted
    together or not at all.
    """

    _SCHEMA = "agentfem.global-cyclic-fatigue-step.v1"

    def __init__(
        self,
        *,
        cycle: ForceCycle,
        stop_cycle: int,
        interfaces,
        state,
        solve_equilibrium,
        jump: CycleJumpPolicy | None = None,
        landing_cycles=(),
        maximum_opening_feedback: float = 0.02,
        maximum_energy_balance_error: float | None = None,
        observe=None,
        name: str = "cyclic fatigue",
    ):
        if not isinstance(cycle, ForceCycle):
            raise TypeError("GlobalCyclicFatigueStep requires a ForceCycle.")
        if int(stop_cycle) < 1:
            raise ValueError("stop_cycle must be a positive integer.")
        required = (
            "names",
            "snapshot",
            "restore",
            "begin_cycle",
            "commit_cycle",
            "rollback",
            "cycle_openings",
            "material_point_responses",
        )
        missing = [item for item in required if not hasattr(interfaces, item)]
        if missing:
            raise TypeError(f"Named cohesive interfaces are missing {missing}.")
        if not callable(getattr(state, "snapshot", None)) or not callable(
            getattr(state, "restore", None)
        ):
            raise TypeError("Global cyclic state needs snapshot() and restore().")
        if not callable(solve_equilibrium):
            raise TypeError("solve_equilibrium must be callable.")
        opening_limit = float(maximum_opening_feedback)
        if not isfinite(opening_limit) or opening_limit <= 0.0:
            raise ValueError("maximum_opening_feedback must be finite and positive.")
        energy_limit = (
            None
            if maximum_energy_balance_error is None
            else float(maximum_energy_balance_error)
        )
        if energy_limit is not None and (
            not isfinite(energy_limit) or energy_limit < 0.0
        ):
            raise ValueError(
                "maximum_energy_balance_error must be finite and nonnegative."
            )
        targets = tuple(sorted({int(value) for value in landing_cycles}))
        if any(value < 1 or value > int(stop_cycle) for value in targets):
            raise ValueError("landing_cycles must lie in [1, stop_cycle].")
        self.name = str(name)
        self.cycle = cycle
        self.stop_cycle = int(stop_cycle)
        self.interfaces = interfaces
        self.state = state
        self.solve_equilibrium = solve_equilibrium
        self.jump = CycleJumpPolicy() if jump is None else jump
        self.landing_cycles = targets
        self.maximum_opening_feedback = opening_limit
        self.maximum_energy_balance_error = energy_limit
        self.observe = observe
        self.ledger = CycleJumpLedger()
        self.history: list[CyclicFatigueBlock] = []
        self._restored_history: list[dict[str, object]] = []
        self._damage_rate: float | None = None
        self._front_rate = 0.0

    @property
    def current_cycle(self) -> int:
        return self.ledger.current_cycle

    def run(self, *, until_cycle: int | None = None, maximum_blocks: int | None = None):
        target = self.stop_cycle if until_cycle is None else int(until_cycle)
        if target < self.current_cycle or target > self.stop_cycle:
            raise ValueError("until_cycle must lie between current and stop cycle.")
        if maximum_blocks is not None and int(maximum_blocks) < 1:
            raise ValueError("maximum_blocks must be positive when supplied.")
        accepted_here = 0
        while self.current_cycle < target:
            decision = self._propose(target)
            while True:
                accepted = self._attempt(decision)
                if accepted:
                    accepted_here += 1
                    break
                if decision.cycles == 1:
                    raise RuntimeError(
                        "The global fatigue equilibrium failed after cutback to one cycle."
                    )
                decision = decision.cutback(
                    max(1, decision.cycles // 2),
                    reason="global_feedback_cutback",
                )
            if maximum_blocks is not None and accepted_here >= int(maximum_blocks):
                break
        return self

    def _propose(self, target: int) -> CycleJumpDecision:
        if self._damage_rate is None:
            return CycleJumpDecision(
                start_cycle=self.current_cycle,
                cycles=1,
                end_cycle=self.current_cycle + 1,
                reason="exact_bootstrap_cycle",
                controlling_damage_rate=0.0,
                predicted_damage_increment=0.0,
            )
        return self.jump.propose(
            start_cycle=self.current_cycle,
            damage_rate=self._damage_rate,
            front_advance_rate=self._front_rate,
            stop_cycle=target,
            landing_cycles=self.landing_cycles,
        )

    def _solve(self, *, load: float, branch: str, cycle: int):
        result = self.solve_equilibrium(load=load, branch=branch, cycle=cycle)
        point = CyclicEquilibriumPoint.coerce(
            result,
            branch=branch,
            load=load,
            cycle=cycle,
        )
        if not point.converged:
            raise RuntimeError(
                f"Global equilibrium did not converge at {branch} of cycle {cycle}."
            )
        return point

    def _attempt(self, decision: CycleJumpDecision) -> bool:
        bulk_snapshot = self.state.snapshot()
        interface_snapshot = self.interfaces.snapshot()
        before_damage = self._fatigue_damage()
        self.ledger.begin(decision)
        try:
            minimum = self._solve(
                load=self.cycle.minimum,
                branch="minimum",
                cycle=decision.start_cycle,
            )
            minimum_opening = self.interfaces.cycle_openings()
            maximum = self._solve(
                load=self.cycle.maximum,
                branch="maximum",
                cycle=decision.end_cycle,
            )
            maximum_opening = self.interfaces.cycle_openings()
            self.interfaces.begin_cycle(
                minimum_opening,
                maximum_opening,
                cycles=decision.cycles,
            )
            self.interfaces.commit_cycle()
            verification = self._solve(
                load=self.cycle.maximum,
                branch="verification_maximum",
                cycle=decision.end_cycle,
            )
            verified_opening = self.interfaces.cycle_openings()
            closing = self._solve(
                load=self.cycle.minimum,
                branch="closing",
                cycle=decision.end_cycle,
            )
            after_damage = self._fatigue_damage()
            damage_increment = self._maximum_damage_increment(
                before_damage,
                after_damage,
            )
            opening_error = self._opening_feedback_error(
                maximum_opening,
                verified_opening,
            )
            energy_error = max(
                (
                    float(point.energy_balance_error)
                    for point in (minimum, maximum, verification, closing)
                    if point.energy_balance_error is not None
                ),
                default=0.0,
            )
            accepted = (
                damage_increment
                <= self.jump.maximum_damage_increment + 1.0e-14
                and opening_error <= self.maximum_opening_feedback
                and (
                    self.maximum_energy_balance_error is None
                    or energy_error <= self.maximum_energy_balance_error
                )
            )
            error = max(
                damage_increment / self.jump.maximum_damage_increment,
                opening_error / self.maximum_opening_feedback,
                (
                    0.0
                    if self.maximum_energy_balance_error in {None, 0.0}
                    else energy_error / self.maximum_energy_balance_error
                ),
            )
            if not accepted:
                self.interfaces.restore(interface_snapshot)
                self.state.restore(bulk_snapshot)
                self.ledger.rollback(
                    error_estimate=error,
                    message=(
                        "cycle block exceeded damage, structural-feedback, or "
                        "energy acceptance tolerance"
                    ),
                )
                return False
            observations = (
                {}
                if self.observe is None
                else dict(self.observe(cycle=decision.end_cycle, step=self))
            )
            block = CyclicFatigueBlock(
                decision=decision,
                minimum=minimum,
                maximum=maximum,
                verification_maximum=verification,
                closing=closing,
                maximum_damage_increment=damage_increment,
                opening_feedback_error=opening_error,
                energy_balance_error=energy_error,
                observations=observations,
            )
            self.history.append(block)
            self.ledger.commit(error_estimate=error)
            self._damage_rate = damage_increment / decision.cycles
            self._front_rate = self._front_advance_rate(block)
            return True
        except Exception as exc:
            self.interfaces.rollback()
            self.interfaces.restore(interface_snapshot)
            self.state.restore(bulk_snapshot)
            if self.ledger._trial is not None:
                self.ledger.rollback(
                    error_estimate=np.finfo(float).max,
                    message=f"{type(exc).__name__}: {exc}",
                )
            if decision.cycles == 1:
                raise
            return False

    def _fatigue_damage(self) -> dict[str, np.ndarray]:
        responses = self.interfaces.material_point_responses()
        result = {}
        for name, response in responses.items():
            if not hasattr(response, "fatigue_damage"):
                raise TypeError(
                    f"Cohesive interface {name!r} does not expose fatigue damage."
                )
            result[name] = np.asarray(response.fatigue_damage, dtype=float).copy()
        return result

    def _maximum_damage_increment(self, before, after) -> float:
        if set(before) != set(after):
            raise ValueError("Named fatigue-damage fields changed during a cycle block.")
        local = max(
            (
                float(np.max(np.asarray(after[name]) - np.asarray(before[name]), initial=0.0))
                for name in before
            ),
            default=0.0,
        )
        force = self.interfaces[self.interfaces.names[0]]
        comm = force.displacement.function_space.mesh.comm
        try:
            from mpi4py import MPI

            return float(comm.allreduce(local, op=MPI.MAX))
        except AttributeError:
            return local

    def _opening_feedback_error(self, before, after) -> float:
        if set(before) != set(after):
            raise ValueError("Named interface openings changed identity.")
        local = 0.0
        for name in before:
            first = np.asarray(before[name], dtype=float)
            second = np.asarray(after[name], dtype=float)
            if first.shape != second.shape:
                raise ValueError(f"Interface {name!r} opening layout changed.")
            law = self.interfaces[name].assembler.law
            scale = max(float(law.failure_opening), np.finfo(float).eps)
            local = max(local, float(np.max(np.abs(second - first), initial=0.0)) / scale)
        force = self.interfaces[self.interfaces.names[0]]
        comm = force.displacement.function_space.mesh.comm
        try:
            from mpi4py import MPI

            return float(comm.allreduce(local, op=MPI.MAX))
        except AttributeError:
            return local

    @staticmethod
    def _front_advance_rate(block: CyclicFatigueBlock) -> float:
        values = []
        for observation in block.observations.values():
            rate = getattr(observation, "front_advance", None)
            if rate is not None:
                values.append(float(rate) / block.decision.cycles)
        return max(values, default=0.0)

    def snapshot(self) -> dict[str, object]:
        """Return an atomic in-memory restart envelope at an accepted cycle."""

        return {
            "schema": self._SCHEMA,
            "name": self.name,
            "cycle": self.cycle.summary(),
            "stop_cycle": self.stop_cycle,
            "landing_cycles": list(self.landing_cycles),
            "ledger": self.ledger.snapshot(),
            "history": [
                *self._restored_history,
                *(item.summary() for item in self.history),
            ],
            "damage_rate": self._damage_rate,
            "front_rate": self._front_rate,
            "bulk_state": self.state.snapshot(),
            "interfaces": self.interfaces.snapshot(),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        """Restore bulk, interface and cycle identity from an accepted state."""

        if snapshot.get("schema") != self._SCHEMA:
            raise ValueError("Unsupported global cyclic-fatigue checkpoint schema.")
        checks = {
            "name": (snapshot.get("name"), self.name),
            "cycle": (snapshot.get("cycle"), self.cycle.summary()),
            "stop_cycle": (snapshot.get("stop_cycle"), self.stop_cycle),
            "landing_cycles": (
                snapshot.get("landing_cycles"),
                list(self.landing_cycles),
            ),
        }
        for label, (stored, current) in checks.items():
            if stored != current:
                raise ValueError(
                    f"Global cyclic-fatigue checkpoint {label} differs."
                )
        self.state.restore(snapshot["bulk_state"])
        self.interfaces.restore(snapshot["interfaces"])
        self.ledger.restore(snapshot["ledger"])
        self._damage_rate = snapshot.get("damage_rate")
        self._front_rate = float(snapshot.get("front_rate", 0.0))
        self._restored_history = [dict(item) for item in snapshot.get("history", [])]
        # Typed block objects are regenerated by subsequent accepted blocks.
        self.history = []

    def save_checkpoint(self, path) -> Path:
        """Persist bulk fields, named interface state and cycle evidence."""

        saver = getattr(self.state, "save_checkpoint", None)
        if not callable(saver):
            raise TypeError(
                "The selected global cycle state has no durable checkpoint adapter."
            )
        from . import checkpointing

        field_manifest = saver(path)
        force = self.interfaces[self.interfaces.names[0]]
        comm = force.displacement.function_space.mesh.comm
        manifest = _cycle_manifest_path(path, suffix="cyclic-fatigue")
        payload = self.snapshot()
        payload["bulk_state"] = {
            "manifest": Path(field_manifest).name,
            "portability": "same MPI partition and rank count",
        }
        error = None
        if comm.rank == 0:
            try:
                checkpointing.atomic_write_text(
                    manifest,
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                )
            except Exception as exc:  # pragma: no cover - filesystem failure
                error = f"{type(exc).__name__}: {exc}"
        error = comm.bcast(error, root=0)
        if error is not None:
            raise RuntimeError(f"Global cyclic checkpoint write failed: {error}")
        comm.barrier()
        return manifest

    def load_checkpoint(self, path) -> None:
        """Load a durable accepted-cycle checkpoint."""

        loader = getattr(self.state, "load_checkpoint", None)
        if not callable(loader):
            raise TypeError(
                "The selected global cycle state has no durable checkpoint adapter."
            )
        force = self.interfaces[self.interfaces.names[0]]
        comm = force.displacement.function_space.mesh.comm
        manifest = _cycle_manifest_path(path, suffix="cyclic-fatigue")
        payload = None
        if comm.rank == 0:
            try:
                payload = {"snapshot": json.loads(manifest.read_text()), "error": None}
            except Exception as exc:
                payload = {"snapshot": None, "error": f"{type(exc).__name__}: {exc}"}
        payload = comm.bcast(payload, root=0)
        if payload["error"] is not None:
            raise RuntimeError(
                f"Global cyclic checkpoint read failed: {payload['error']}"
            )
        snapshot = payload["snapshot"]
        state_record = snapshot.get("bulk_state", {})
        field_manifest = manifest.with_name(str(state_record.get("manifest", "")))
        loader(field_manifest)
        in_memory = dict(snapshot)
        in_memory["bulk_state"] = self.state.snapshot()
        self.restore(in_memory)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "global_cyclic_fatigue_step",
            "name": self.name,
            "cycle": self.cycle.summary(),
            "current_cycle": self.current_cycle,
            "stop_cycle": self.stop_cycle,
            "landing_cycles": self.landing_cycles,
            "jump": self.jump.summary(),
            "maximum_opening_feedback": self.maximum_opening_feedback,
            "maximum_energy_balance_error": self.maximum_energy_balance_error,
            "accepted_blocks": len(self._restored_history) + len(self.history),
            "ledger": self.ledger.summary(),
            "procedure": "quasi_static_peak_valley_with_post_damage_equilibrium",
            "maturity": "experimental_global_cycle_consumer",
        }


@dataclass(frozen=True)
class SurfaceCrackComponent:
    """One connected failed component in a surface-crack observation."""

    local_label: int
    facet_keys: tuple[str, ...]
    failed_area: float
    front_segments: np.ndarray
    front_length: float
    maximum_cod: float
    mean_cod: float
    centroid: np.ndarray

    @property
    def front_points(self) -> np.ndarray:
        if self.front_segments.size == 0:
            return np.empty((0, 3), dtype=float)
        return np.unique(self.front_segments.reshape((-1, 3)), axis=0)

    def summary(self) -> dict[str, object]:
        return {
            "local_label": self.local_label,
            "facet_keys": list(self.facet_keys),
            "failed_facets": len(self.facet_keys),
            "failed_area": self.failed_area,
            "front_length": self.front_length,
            "maximum_cod": self.maximum_cod,
            "mean_cod": self.mean_cod,
            "centroid": self.centroid.tolist(),
        }


@dataclass(frozen=True)
class SurfaceCrackObservation:
    """One cycle's geometric evidence on a triangular cohesive surface."""

    cycle: int
    name: str
    failed_facets: np.ndarray
    component_labels: np.ndarray
    front_segments: np.ndarray
    failed_area: float
    front_length: float
    maximum_cod: float
    mean_cod: float
    damage_threshold: float
    components: tuple[SurfaceCrackComponent, ...] = ()
    facet_identity: str = "local_index"

    @property
    def component_count(self) -> int:
        labels = self.component_labels[self.component_labels >= 0]
        return 0 if labels.size == 0 else int(np.max(labels)) + 1

    @property
    def front_points(self) -> np.ndarray:
        if self.front_segments.size == 0:
            return np.empty((0, 3), dtype=float)
        return np.unique(self.front_segments.reshape((-1, 3)), axis=0)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "surface_crack_observation",
            "cycle": self.cycle,
            "name": self.name,
            "failed_facets": int(np.count_nonzero(self.failed_facets)),
            "components": self.component_count,
            "failed_area": self.failed_area,
            "front_length": self.front_length,
            "maximum_cod": self.maximum_cod,
            "mean_cod": self.mean_cod,
            "damage_threshold": self.damage_threshold,
            "front_definition": "failed_intact_shared_edges",
            "facet_identity": self.facet_identity,
            "component_records": [component.summary() for component in self.components],
        }


@dataclass(frozen=True)
class CrackTopologyEvent:
    """Auditable identity change between two accepted crack observations."""

    cycle: int
    kind: str
    parent_ids: tuple[str, ...] = ()
    result_ids: tuple[str, ...] = ()

    def summary(self) -> dict[str, object]:
        return {
            "cycle": self.cycle,
            "kind": self.kind,
            "parent_ids": list(self.parent_ids),
            "result_ids": list(self.result_ids),
        }


@dataclass(frozen=True)
class TrackedSurfaceCrack:
    """A connected crack component with identity stable across cycle blocks."""

    cycle: int
    crack_id: str
    interface_name: str
    facet_keys: tuple[str, ...]
    failed_area: float
    front_segments: np.ndarray
    front_length: float
    maximum_cod: float
    mean_cod: float
    centroid: np.ndarray
    parent_ids: tuple[str, ...] = ()
    area_growth_rate: float | None = None
    front_length_growth_rate: float | None = None

    @property
    def front_points(self) -> np.ndarray:
        if self.front_segments.size == 0:
            return np.empty((0, 3), dtype=float)
        return np.unique(self.front_segments.reshape((-1, 3)), axis=0)

    def summary(self) -> dict[str, object]:
        return {
            "cycle": self.cycle,
            "crack_id": self.crack_id,
            "interface_name": self.interface_name,
            "facet_keys": list(self.facet_keys),
            "failed_facets": len(self.facet_keys),
            "failed_area": self.failed_area,
            "front_length": self.front_length,
            "maximum_cod": self.maximum_cod,
            "mean_cod": self.mean_cod,
            "centroid": self.centroid.tolist(),
            "parent_ids": list(self.parent_ids),
            "area_growth_rate_per_cycle": self.area_growth_rate,
            "front_length_growth_rate_per_cycle": self.front_length_growth_rate,
        }


@dataclass(frozen=True)
class SurfaceCrackTrackingFrame:
    """Persistent component identities and topology events at one cycle."""

    cycle: int
    interface_name: str
    cracks: tuple[TrackedSurfaceCrack, ...]
    events: tuple[CrackTopologyEvent, ...] = ()

    def interactions(
        self, *, coalescence_tolerance: float = 0.0
    ) -> tuple["CrackInteractionObservation", ...]:
        """Return every same-surface crack-pair ligament without manual splitting."""

        tolerance = float(coalescence_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("coalescence_tolerance must be finite and nonnegative.")
        records = []
        for first_index, first in enumerate(self.cracks):
            for second in self.cracks[first_index + 1 :]:
                first_points = first.front_points
                second_points = second.front_points
                if first_points.size == 0 or second_points.size == 0:
                    ligament = float("inf")
                else:
                    ligament = float(
                        np.min(
                            np.linalg.norm(
                                first_points[:, None, :] - second_points[None, :, :],
                                axis=2,
                            )
                        )
                    )
                records.append(
                    CrackInteractionObservation(
                        cycle=self.cycle,
                        first_name=first.crack_id,
                        second_name=second.crack_id,
                        minimum_ligament=ligament,
                        coalesced=ligament <= tolerance,
                    )
                )
        return tuple(records)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "surface_crack_tracking_frame",
            "cycle": self.cycle,
            "interface_name": self.interface_name,
            "cracks": [crack.summary() for crack in self.cracks],
            "events": [event.summary() for event in self.events],
        }


class SurfaceCrackTracker:
    """Track cracks on one fixed cohesive surface by physical facet identity.

    Identity is inherited only for an unambiguous one-to-one continuation.
    A merge or split creates new identities and records its parents, avoiding
    an arbitrary choice of which pre-existing crack supposedly survives.
    """

    _SCHEMA = "agentfem.surface-crack-tracker.v1"

    def __init__(self, *, interface_name: str, id_prefix: str | None = None):
        name = str(interface_name).strip()
        if not name:
            raise ValueError("SurfaceCrackTracker.interface_name cannot be empty.")
        self.interface_name = name
        self.id_prefix = str(id_prefix or name).strip()
        if not self.id_prefix:
            raise ValueError("SurfaceCrackTracker.id_prefix cannot be empty.")
        self._next_id = 1
        self._active: dict[str, TrackedSurfaceCrack] = {}
        self._last_cycle: int | None = None

    @property
    def active(self) -> tuple[TrackedSurfaceCrack, ...]:
        return tuple(self._active[name] for name in sorted(self._active))

    def _new_id(self) -> str:
        selected = f"{self.id_prefix}:{self._next_id:04d}"
        self._next_id += 1
        return selected

    def observe(self, observation: SurfaceCrackObservation) -> SurfaceCrackTrackingFrame:
        if observation.name != self.interface_name:
            raise ValueError("Tracked observation belongs to a different interface.")
        if self._last_cycle is not None and int(observation.cycle) <= self._last_cycle:
            raise ValueError("Tracked crack cycles must increase strictly.")
        if observation.component_count and not observation.components:
            raise ValueError(
                "Persistent tracking requires component records from "
                "observe_surface_crack()."
            )

        previous = dict(self._active)
        previous_keys = {
            crack_id: set(crack.facet_keys) for crack_id, crack in previous.items()
        }
        current_keys = [set(component.facet_keys) for component in observation.components]
        parents_by_current = [
            tuple(
                sorted(
                    crack_id
                    for crack_id, keys in previous_keys.items()
                    if keys & selected
                )
            )
            for selected in current_keys
        ]
        children_by_previous = {
            crack_id: tuple(
                index
                for index, selected in enumerate(current_keys)
                if keys & selected
            )
            for crack_id, keys in previous_keys.items()
        }

        tracked: list[TrackedSurfaceCrack] = []
        events: list[CrackTopologyEvent] = []
        for index, component in enumerate(observation.components):
            parents = parents_by_current[index]
            if not parents:
                crack_id = self._new_id()
                events.append(
                    CrackTopologyEvent(observation.cycle, "birth", (), (crack_id,))
                )
            elif len(parents) == 1 and len(children_by_previous[parents[0]]) == 1:
                crack_id = parents[0]
            else:
                crack_id = self._new_id()
                kind = "merge" if len(parents) > 1 else "split"
                events.append(
                    CrackTopologyEvent(
                        observation.cycle,
                        kind,
                        parents,
                        (crack_id,),
                    )
                )
            tracked.append(
                TrackedSurfaceCrack(
                    cycle=int(observation.cycle),
                    crack_id=crack_id,
                    interface_name=self.interface_name,
                    facet_keys=component.facet_keys,
                    failed_area=component.failed_area,
                    front_segments=component.front_segments.copy(),
                    front_length=component.front_length,
                    maximum_cod=component.maximum_cod,
                    mean_cod=component.mean_cod,
                    centroid=component.centroid.copy(),
                    parent_ids=parents if crack_id not in previous else (),
                    area_growth_rate=(
                        None
                        if crack_id not in previous
                        else (
                            component.failed_area - previous[crack_id].failed_area
                        )
                        / (int(observation.cycle) - previous[crack_id].cycle)
                    ),
                    front_length_growth_rate=(
                        None
                        if crack_id not in previous
                        else (
                            component.front_length - previous[crack_id].front_length
                        )
                        / (int(observation.cycle) - previous[crack_id].cycle)
                    ),
                )
            )

        continued_parents = {parent for parents in parents_by_current for parent in parents}
        for crack_id in sorted(set(previous) - continued_parents):
            events.append(
                CrackTopologyEvent(observation.cycle, "death", (crack_id,), ())
            )
        self._active = {crack.crack_id: crack for crack in tracked}
        self._last_cycle = int(observation.cycle)
        return SurfaceCrackTrackingFrame(
            cycle=int(observation.cycle),
            interface_name=self.interface_name,
            cracks=tuple(tracked),
            events=tuple(events),
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": self._SCHEMA,
            "interface_name": self.interface_name,
            "id_prefix": self.id_prefix,
            "next_id": self._next_id,
            "last_cycle": self._last_cycle,
            "active": [
                {
                    **crack.summary(),
                    "front_segments": crack.front_segments.tolist(),
                }
                for crack in self.active
            ],
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        if snapshot.get("schema") != self._SCHEMA:
            raise ValueError("Unsupported surface-crack tracker schema.")
        if snapshot.get("interface_name") != self.interface_name:
            raise ValueError("Surface-crack tracker interface differs from checkpoint.")
        if snapshot.get("id_prefix") != self.id_prefix:
            raise ValueError("Surface-crack tracker ID prefix differs from checkpoint.")
        active = {}
        for record in snapshot.get("active", []):
            crack = TrackedSurfaceCrack(
                cycle=int(record["cycle"]),
                crack_id=str(record["crack_id"]),
                interface_name=str(record["interface_name"]),
                facet_keys=tuple(str(value) for value in record["facet_keys"]),
                failed_area=float(record["failed_area"]),
                front_segments=_finite_array(
                    record.get("front_segments", []), name="front_segments"
                ).reshape((-1, 2, 3)),
                front_length=float(record["front_length"]),
                maximum_cod=float(record["maximum_cod"]),
                mean_cod=float(record["mean_cod"]),
                centroid=_finite_array(record["centroid"], name="centroid"),
                parent_ids=tuple(str(value) for value in record.get("parent_ids", [])),
                area_growth_rate=(
                    None
                    if record.get("area_growth_rate_per_cycle") is None
                    else float(record["area_growth_rate_per_cycle"])
                ),
                front_length_growth_rate=(
                    None
                    if record.get("front_length_growth_rate_per_cycle") is None
                    else float(record["front_length_growth_rate_per_cycle"])
                ),
            )
            active[crack.crack_id] = crack
        self._active = active
        self._next_id = int(snapshot["next_id"])
        self._last_cycle = (
            None if snapshot.get("last_cycle") is None else int(snapshot["last_cycle"])
        )


@dataclass(frozen=True)
class CrackInteractionObservation:
    """Two-crack geometry and growth evidence at one exact cycle."""

    cycle: int
    first_name: str
    second_name: str
    minimum_ligament: float
    coalesced: bool
    first_growth_ratio: float | None = None
    second_growth_ratio: float | None = None

    def summary(self) -> dict[str, object]:
        return {
            "kind": "two_surface_crack_interaction",
            "cycle": self.cycle,
            "interfaces": [self.first_name, self.second_name],
            "minimum_ligament": self.minimum_ligament,
            "coalesced": self.coalesced,
            "first_shielding_or_amplification": self.first_growth_ratio,
            "second_shielding_or_amplification": self.second_growth_ratio,
            "ratio_convention": "double_crack_growth_rate / single_crack_baseline",
        }


def observe_surface_crack(
    coordinates,
    facets,
    damage,
    opening,
    *,
    cycle: int,
    name: str = "surface crack",
    damage_threshold: float = 0.95,
    include_boundary_front: bool = False,
    facet_ids=None,
) -> SurfaceCrackObservation:
    """Recover connected failed area and a three-dimensional crack front.

    The primary front is the set of mesh edges shared by one failed and one
    intact cohesive facet.  Boundary edges are excluded by default because a
    surface-crack mouth is not its propagating embedded front.
    """

    points = _finite_array(coordinates, name="coordinates")
    triangles = np.asarray(facets, dtype=int)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Surface-crack coordinates must have shape (nodes, 3).")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Surface-crack facets must be triangular.")
    if np.any(triangles < 0) or np.any(triangles >= points.shape[0]):
        raise ValueError("Surface-crack facet connectivity is invalid.")
    selected_damage = _facet_values(damage, triangles.shape[0], name="damage")
    selected_opening = _facet_values(opening, triangles.shape[0], name="opening")
    threshold = float(damage_threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("damage_threshold must lie in (0, 1].")
    failed = selected_damage >= threshold
    if facet_ids is None:
        selected_facet_keys = tuple(str(index) for index in range(triangles.shape[0]))
        facet_identity = "local_index"
    else:
        if len(facet_ids) != triangles.shape[0]:
            raise ValueError("facet_ids must provide one stable identity per facet.")
        selected_facet_keys = tuple(_canonical_facet_key(value) for value in facet_ids)
        if len(set(selected_facet_keys)) != len(selected_facet_keys):
            raise ValueError("facet_ids must be unique on the observed surface.")
        facet_identity = "declared_physical_key"

    edge_to_facets: dict[tuple[int, int], list[int]] = {}
    for facet_index, triangle in enumerate(triangles):
        for left, right in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(left), int(right))))
            edge_to_facets.setdefault(edge, []).append(facet_index)

    adjacency = [set() for _ in range(triangles.shape[0])]
    front_edges: list[tuple[tuple[int, int], int]] = []
    for edge, attached in edge_to_facets.items():
        if len(attached) == 2:
            left, right = attached
            adjacency[left].add(right)
            adjacency[right].add(left)
            if failed[left] != failed[right]:
                front_edges.append((edge, left if failed[left] else right))
        elif len(attached) == 1:
            if include_boundary_front and failed[attached[0]]:
                front_edges.append((edge, attached[0]))
        else:
            raise ValueError("Surface-crack mesh contains a non-manifold edge.")

    labels = np.full(triangles.shape[0], -1, dtype=int)
    next_label = 0
    for start in np.flatnonzero(failed):
        if labels[start] >= 0:
            continue
        labels[start] = next_label
        stack = [int(start)]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if failed[neighbor] and labels[neighbor] < 0:
                    labels[neighbor] = next_label
                    stack.append(neighbor)
        next_label += 1

    facet_points = points[triangles]
    cross = np.cross(
        facet_points[:, 1] - facet_points[:, 0],
        facet_points[:, 2] - facet_points[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    if np.any(areas <= np.finfo(float).eps):
        raise ValueError("Surface-crack mesh contains a degenerate triangle.")
    segments = (
        points[np.asarray([edge for edge, _owner in front_edges], dtype=int)]
        if front_edges
        else np.empty((0, 2, 3), dtype=float)
    )
    front_length = float(
        np.sum(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1))
    )
    failed_area = float(np.sum(areas[failed]))
    maximum_cod = (
        0.0 if not np.any(failed) else float(np.max(selected_opening[failed]))
    )
    mean_cod = (
        0.0
        if failed_area == 0.0
        else float(np.sum(selected_opening[failed] * areas[failed]) / failed_area)
    )
    facet_centroids = np.mean(facet_points, axis=1)
    components = []
    for label in range(next_label):
        component_facets = np.flatnonzero(labels == label)
        component_area = float(np.sum(areas[component_facets]))
        component_segments = [
            points[np.asarray(edge, dtype=int)]
            for edge, owner in front_edges
            if labels[owner] == label
        ]
        component_front = (
            np.asarray(component_segments, dtype=float)
            if component_segments
            else np.empty((0, 2, 3), dtype=float)
        )
        component_front_length = float(
            np.sum(
                np.linalg.norm(
                    component_front[:, 1] - component_front[:, 0], axis=1
                )
            )
        )
        component_cod = selected_opening[component_facets]
        component_areas = areas[component_facets]
        centroid = np.sum(
            facet_centroids[component_facets] * component_areas[:, None], axis=0
        ) / component_area
        components.append(
            SurfaceCrackComponent(
                local_label=label,
                facet_keys=tuple(
                    sorted(selected_facet_keys[index] for index in component_facets)
                ),
                failed_area=component_area,
                front_segments=component_front,
                front_length=component_front_length,
                maximum_cod=float(np.max(component_cod)),
                mean_cod=float(np.sum(component_cod * component_areas) / component_area),
                centroid=centroid,
            )
        )
    return SurfaceCrackObservation(
        cycle=int(cycle),
        name=str(name),
        failed_facets=failed,
        component_labels=labels,
        front_segments=segments,
        failed_area=failed_area,
        front_length=front_length,
        maximum_cod=maximum_cod,
        mean_cod=mean_cod,
        damage_threshold=threshold,
        components=tuple(components),
        facet_identity=facet_identity,
    )


def _canonical_facet_key(value) -> str:
    """Return a deterministic JSON key for a physical facet identity."""

    def portable(selected):
        if isinstance(selected, np.generic):
            return selected.item()
        if isinstance(selected, np.ndarray):
            return [portable(item) for item in selected.tolist()]
        if isinstance(selected, (tuple, list)):
            return [portable(item) for item in selected]
        if isinstance(selected, dict):
            return {
                str(key): portable(item)
                for key, item in sorted(selected.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(selected, (str, int, float, bool)) or selected is None:
            return selected
        raise TypeError("facet_ids must be JSON-compatible physical identities.")

    selected = portable(value)
    if isinstance(selected, str):
        return selected
    return json.dumps(selected, sort_keys=True, separators=(",", ":"), allow_nan=False)


def surface_crack_interaction(
    first: SurfaceCrackObservation,
    second: SurfaceCrackObservation,
    *,
    first_single_growth_rate: float | None = None,
    second_single_growth_rate: float | None = None,
    first_double_growth_rate: float | None = None,
    second_double_growth_rate: float | None = None,
    coalescence_tolerance: float = 0.0,
) -> CrackInteractionObservation:
    """Compare two named fronts without hiding the single-crack baseline."""

    if int(first.cycle) != int(second.cycle):
        raise ValueError("Crack interaction requires observations at the same cycle.")
    tolerance = float(coalescence_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("coalescence_tolerance must be finite and nonnegative.")
    first_points = first.front_points
    second_points = second.front_points
    if first_points.size == 0 or second_points.size == 0:
        ligament = float("inf")
    else:
        distances = np.linalg.norm(
            first_points[:, None, :] - second_points[None, :, :], axis=2
        )
        ligament = float(np.min(distances))

    def ratio(double, single):
        if double is None and single is None:
            return None
        if double is None or single is None:
            raise ValueError("Both double- and single-crack growth rates are required.")
        baseline = float(single)
        selected = float(double)
        if not isfinite(baseline) or baseline <= 0.0 or not isfinite(selected):
            raise ValueError("Growth rates must be finite and baselines positive.")
        return selected / baseline

    return CrackInteractionObservation(
        cycle=int(first.cycle),
        first_name=first.name,
        second_name=second.name,
        minimum_ligament=ligament,
        coalesced=ligament <= tolerance,
        first_growth_ratio=ratio(first_double_growth_rate, first_single_growth_rate),
        second_growth_ratio=ratio(second_double_growth_rate, second_single_growth_rate),
    )


@dataclass(frozen=True)
class ParisEvidence:
    """Postprocessed Paris-region evidence; never a crack-growth solver law."""

    cycles: np.ndarray
    crack_size: np.ndarray
    driving_force: np.ndarray
    growth_rate: np.ndarray
    fit_mask: np.ndarray
    coefficient: float
    exponent: float
    coefficient_of_determination: float
    logarithmic_root_mean_square_error: float
    driving_force_name: str
    crack_size_name: str
    driving_force_unit: str
    crack_size_unit: str
    derivative_window: int
    fit_selection: str

    def predict(self, driving_force) -> np.ndarray:
        selected = _finite_array(driving_force, name="driving_force")
        if np.any(selected <= 0.0):
            raise ValueError("Paris driving force must be positive.")
        return self.coefficient * selected**self.exponent

    def summary(self, *, include_series: bool = False) -> dict[str, object]:
        record = {
            "kind": "paris_law_postprocessing_evidence",
            "role": "postprocessing_not_solver_input",
            "relation": "da/dN = C * driving_force**m",
            "driving_force_name": self.driving_force_name,
            "crack_size_name": self.crack_size_name,
            "driving_force_unit": self.driving_force_unit,
            "crack_size_unit": self.crack_size_unit,
            "coefficient": self.coefficient,
            "exponent": self.exponent,
            "coefficient_of_determination": self.coefficient_of_determination,
            "logarithmic_root_mean_square_error": (
                self.logarithmic_root_mean_square_error
            ),
            "samples": int(np.count_nonzero(self.fit_mask)),
            "derivative": f"local_linear_window_{self.derivative_window}",
            "fit_selection": self.fit_selection,
            "coefficient_unit": (
                f"{self.crack_size_unit}/cycle/"
                f"({self.driving_force_unit})^{self.exponent:.8g}"
            ),
        }
        if include_series:
            record["series"] = {
                "cycles": self.cycles.tolist(),
                "crack_size": self.crack_size.tolist(),
                "driving_force": self.driving_force.tolist(),
                "growth_rate": self.growth_rate.tolist(),
                "fit_mask": self.fit_mask.tolist(),
            }
        return record


def paris_evidence(
    cycles,
    crack_size,
    driving_force,
    *,
    fit_cycle_range: tuple[float, float] | None = None,
    fit_mask=None,
    derivative_window: int = 3,
    driving_force_name: str = "Delta K",
    crack_size_name: str = "a",
    driving_force_unit: str = "declared",
    crack_size_unit: str = "declared",
) -> ParisEvidence:
    """Fit a Paris relation after simulation from ``a(N)`` and a driver.

    The function deliberately does not select a propagation law for the
    solver.  It derives ``da/dN`` from accepted observations and records the
    exact samples used for the log--log fit.  Publication workflows should
    predeclare ``fit_cycle_range`` or ``fit_mask`` rather than selecting a
    visually convenient interval after seeing the result.
    """

    selected_cycles = _finite_array(cycles, name="cycles")
    selected_size = _finite_array(crack_size, name="crack_size")
    selected_driver = _finite_array(driving_force, name="driving_force")
    if (
        selected_cycles.ndim != 1
        or selected_size.ndim != 1
        or selected_driver.ndim != 1
        or not (
            selected_cycles.size == selected_size.size == selected_driver.size
        )
    ):
        raise ValueError("Paris evidence inputs must be equal-length one-dimensional arrays.")
    if selected_cycles.size < 3:
        raise ValueError("Paris evidence requires at least three observations.")
    if np.any(np.diff(selected_cycles) <= 0.0):
        raise ValueError("Paris evidence cycles must increase strictly.")
    tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(selected_size))))
    if np.any(np.diff(selected_size) < -tolerance):
        raise ValueError("Observed crack size cannot decrease.")
    if np.any(selected_driver <= 0.0):
        raise ValueError("Paris driving force must be positive.")
    window = int(derivative_window)
    if window < 3 or window % 2 == 0 or window > selected_cycles.size:
        raise ValueError("derivative_window must be an odd integer within the series.")

    growth_rate = _local_linear_growth_rate(
        selected_cycles, selected_size, window=window
    )
    selected_mask = np.ones(selected_cycles.size, dtype=bool)
    selection_parts = []
    if fit_cycle_range is not None:
        lower, upper = (float(value) for value in fit_cycle_range)
        if not isfinite(lower) or not isfinite(upper) or upper <= lower:
            raise ValueError("fit_cycle_range must contain finite increasing bounds.")
        selected_mask &= (selected_cycles >= lower) & (selected_cycles <= upper)
        selection_parts.append("declared_cycle_range")
    if fit_mask is not None:
        declared_mask = np.asarray(fit_mask, dtype=bool)
        if declared_mask.shape != selected_mask.shape:
            raise ValueError("fit_mask must match the observation series.")
        selected_mask &= declared_mask
        selection_parts.append("declared_mask")
    selected_mask &= growth_rate > 0.0
    if np.count_nonzero(selected_mask) < 3:
        raise ValueError("Paris evidence requires at least three positive-growth fit samples.")

    logarithmic_driver = np.log(selected_driver[selected_mask])
    logarithmic_rate = np.log(growth_rate[selected_mask])
    design = np.column_stack(
        (np.ones(logarithmic_driver.size), logarithmic_driver)
    )
    intercept, exponent = np.linalg.lstsq(design, logarithmic_rate, rcond=None)[0]
    fitted = intercept + exponent * logarithmic_driver
    residual = logarithmic_rate - fitted
    total = logarithmic_rate - np.mean(logarithmic_rate)
    residual_sum = float(np.dot(residual, residual))
    total_sum = float(np.dot(total, total))
    r_squared = 1.0 if total_sum == 0.0 else 1.0 - residual_sum / total_sum
    return ParisEvidence(
        cycles=selected_cycles.copy(),
        crack_size=selected_size.copy(),
        driving_force=selected_driver.copy(),
        growth_rate=growth_rate,
        fit_mask=selected_mask,
        coefficient=float(np.exp(intercept)),
        exponent=float(exponent),
        coefficient_of_determination=float(r_squared),
        logarithmic_root_mean_square_error=float(
            np.sqrt(np.mean(residual**2))
        ),
        driving_force_name=str(driving_force_name),
        crack_size_name=str(crack_size_name),
        driving_force_unit=str(driving_force_unit),
        crack_size_unit=str(crack_size_unit),
        derivative_window=window,
        fit_selection=(
            "+".join(selection_parts) if selection_parts else "all_positive_growth_samples"
        ),
    )


def _local_linear_growth_rate(cycles: np.ndarray, crack_size: np.ndarray, *, window: int):
    """Differentiate irregular cycle observations with a declared local fit."""

    radius = window // 2
    rates = np.empty(cycles.size, dtype=float)
    for index in range(cycles.size):
        start = max(0, index - radius)
        end = min(cycles.size, start + window)
        start = max(0, end - window)
        local_cycle = cycles[start:end]
        local_size = crack_size[start:end]
        centered = local_cycle - np.mean(local_cycle)
        rates[index] = float(np.polyfit(centered, local_size, 1)[0])
    tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(rates))))
    rates[np.abs(rates) <= tolerance] = 0.0
    return rates


def _facet_values(values, number_of_facets: int, *, name: str) -> np.ndarray:
    selected = _finite_array(values, name=name)
    if selected.shape == (number_of_facets,):
        return selected
    if selected.ndim == 2 and selected.shape[0] == number_of_facets:
        return np.mean(selected, axis=1)
    raise ValueError(
        f"{name} must contain one value or a quadrature row per surface facet."
    )


def cyclic_cohesive(
    *,
    monotonic: BilinearCohesiveLaw,
    fatigue_coefficient: float,
    fatigue_exponent: float,
    range_threshold: float,
    peak_exponent: float = 0.0,
    residual_exponent: float = 0.0,
    name: str = "power-law cyclic Mode-I cohesive law",
) -> CyclicCohesiveLaw:
    """Create the experimental reference cyclic cohesive law."""

    return CyclicCohesiveLaw(
        monotonic=monotonic,
        fatigue_coefficient=fatigue_coefficient,
        fatigue_exponent=fatigue_exponent,
        range_threshold=range_threshold,
        peak_exponent=peak_exponent,
        residual_exponent=residual_exponent,
        name=name,
    )


def field_state(fields=None, *, assets=None, **named_fields) -> FieldStateTransaction:
    """Create rollback state from a field mapping or named field arguments."""

    records = {} if fields is None else dict(fields)
    overlap = set(records) & set(named_fields)
    if overlap:
        raise ValueError(f"Duplicate transactional field names: {sorted(overlap)}.")
    records.update(named_fields)
    return FieldStateTransaction(records, assets=assets)


def global_cyclic_fatigue_step(**kwargs) -> GlobalCyclicFatigueStep:
    """Create a reusable force-controlled peak/valley fatigue controller."""

    return GlobalCyclicFatigueStep(**kwargs)


def _integrate_residual_power(initial, amplitude, cycles: int, exponent: float):
    """Integrate ``dD/dN=A(1-D)^p`` exactly for constant cycle extrema."""

    damage = np.asarray(initial, dtype=float)
    rate = np.asarray(amplitude, dtype=float)
    duration = float(cycles)
    remaining = np.maximum(1.0 - damage, 0.0)
    power = float(exponent)
    if power == 0.0:
        updated = damage + rate * duration
    elif power == 1.0:
        updated = 1.0 - remaining * np.exp(-rate * duration)
    else:
        base = remaining ** (1.0 - power) - (1.0 - power) * rate * duration
        if power < 1.0:
            next_remaining = np.maximum(base, 0.0) ** (1.0 / (1.0 - power))
        else:
            next_remaining = np.maximum(base, np.finfo(float).tiny) ** (
                1.0 / (1.0 - power)
            )
        updated = 1.0 - next_remaining
    return np.clip(updated, damage, 1.0)


__all__ = [
    "CrackTopologyEvent",
    "CrackInteractionObservation",
    "CyclicEquilibriumPoint",
    "CyclicFatigueBlock",
    "CycleJumpLedger",
    "CycleJumpRecord",
    "CycleJumpDecision",
    "CycleJumpPolicy",
    "CyclicCohesiveLaw",
    "CyclicCohesiveResponse",
    "CyclicCohesiveTransaction",
    "FieldStateTransaction",
    "ForceCycle",
    "GlobalCyclicFatigueStep",
    "ParisEvidence",
    "SurfaceCrackComponent",
    "SurfaceCrackObservation",
    "SurfaceCrackTracker",
    "SurfaceCrackTrackingFrame",
    "TrackedSurfaceCrack",
    "cyclic_cohesive",
    "field_state",
    "force_cycle",
    "global_cyclic_fatigue_step",
    "observe_surface_crack",
    "paris_evidence",
    "surface_crack_interaction",
]
