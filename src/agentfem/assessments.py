"""Engineering assessments that consume, but do not alter, FEM solutions.

The module deliberately separates solver-integrated constitutive evolution
from code- or experiment-defined life assessment.  Normative interaction
curves remain explicit user inputs with a source; AgentFEM does not embed or
silently reinterpret proprietary design-code data.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Iterable

import numpy as np

from .constitutive.fatigue import FatigueAssessment


@dataclass(frozen=True)
class SequentialEnergyLedger:
    """Layered evidence for a one-way heat-to-mechanics workflow.

    Thermal and mechanical residuals retain their own equations and units.
    They are never summed into a fictitious monolithic conservation error.
    """

    thermal: dict[str, float]
    mechanical: dict[str, float]
    transfer: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "sequential_thermo_mechanical_energy_ledger",
            "coupling": "one_way_sequential",
            "full_coupled_conservation_claim": False,
            "thermal": dict(self.thermal),
            "mechanical": dict(self.mechanical),
            "transfer": dict(self.transfer),
        }

    def attach(self, result, *, name: str = "sequential_energy"):
        selected = str(name).strip().lower().replace(" ", "_")
        if not selected:
            raise ValueError("Sequential energy-ledger name must be nonempty.")
        quantities = {
            f"{selected}_thermal_residual_max_abs": self.thermal[
                "residual_max_abs"
            ],
            f"{selected}_mechanical_residual_max_abs": self.mechanical[
                "residual_max_abs"
            ],
        }
        result.add_quantities(quantities, kind="diagnostic")
        result.metadata.setdefault("energy_ledgers", {})[selected] = self.as_dict()
        return result


def _required_history(result, name: str):
    try:
        return result.histories[name]
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f"SimulationResult lacks required history {name!r}."
        ) from exc


def sequential_energy_ledger(
    thermal_result,
    mechanical_result,
    *,
    field_history=None,
) -> SequentialEnergyLedger:
    """Audit thermal and mechanical energy channels across sequential steps."""

    thermal_content = _required_history(thermal_result, "thermal_content")
    applied_rate = _required_history(thermal_result, "applied_heat_rate")
    outward_rate = _required_history(thermal_result, "outward_heat_rate")
    thermal_residual = _required_history(
        thermal_result, "heat_balance_residual"
    )
    mechanical_residual_name = (
        "mechanical_energy_residual"
        if "mechanical_energy_residual" in mechanical_result.histories
        else "energy_balance_error"
    )
    mechanical_residual = _required_history(
        mechanical_result, mechanical_residual_name
    )
    internal = _required_history(mechanical_result, "internal_energy")
    external = _required_history(mechanical_result, "external_work")
    transfer = {
        "field": None,
        "history_identity": None,
        "statement": (
            "Temperature is prescribed from accepted thermal frames; "
            "mechanical dissipation and deformation are not returned to heat."
        ),
    }
    if field_history is not None:
        transfer.update(
            {
                "field": getattr(field_history, "name", None),
                "history_identity": field_history.scientific_identity(),
            }
        )
    return SequentialEnergyLedger(
        thermal={
            "final_content": float(thermal_content.latest),
            "final_applied_heat_rate": float(applied_rate.latest),
            "final_outward_heat_rate": float(outward_rate.latest),
            "residual_max_abs": float(
                np.max(np.abs(thermal_residual.values), initial=0.0)
            ),
        },
        mechanical={
            "final_internal_energy": float(internal.latest),
            "final_external_work": float(external.latest),
            "residual_name": mechanical_residual_name,
            "residual_max_abs": float(
                np.max(np.abs(mechanical_residual.values), initial=0.0)
            ),
        },
        transfer=transfer,
    )


@dataclass(frozen=True)
class CreepDamageBlock:
    """One dwell or service block for the time-fraction rule."""

    duration: float
    rupture_time: float
    repetitions: float = 1.0
    label: str = ""
    source: str = ""
    stress: float | None = None
    temperature: float | None = None
    start_time: float | None = None
    end_time: float | None = None

    def __post_init__(self) -> None:
        for name in ("duration", "rupture_time", "repetitions"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(
                    f"CreepDamageBlock.{name} must be finite and nonnegative."
                )
        if self.rupture_time <= 0.0:
            raise ValueError("CreepDamageBlock.rupture_time must be positive.")
        if not str(self.source).strip():
            raise ValueError(
                "CreepDamageBlock.source must identify the rupture relation, "
                "test data, design curve, or reviewed calculation."
            )
        for name in ("stress", "temperature", "start_time", "end_time"):
            value = getattr(self, name)
            if value is not None and not isfinite(float(value)):
                raise ValueError(f"CreepDamageBlock.{name} must be finite when set.")
        if (
            self.start_time is not None
            and self.end_time is not None
            and float(self.end_time) <= float(self.start_time)
        ):
            raise ValueError("CreepDamageBlock end_time must exceed start_time.")

    @property
    def damage(self) -> float:
        return float(self.repetitions * self.duration / self.rupture_time)

    def as_dict(self) -> dict[str, object]:
        return {
            "duration": float(self.duration),
            "rupture_time": float(self.rupture_time),
            "repetitions": float(self.repetitions),
            "damage": self.damage,
            "label": self.label,
            "source": self.source,
            "stress": self.stress,
            "temperature": self.temperature,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass(frozen=True)
class DwellInterval:
    """One explicitly declared hold interval in a result time history."""

    start: float
    end: float
    repetitions: float = 1.0
    label: str = "dwell"

    def __post_init__(self) -> None:
        if not isfinite(float(self.start)) or not isfinite(float(self.end)):
            raise ValueError("DwellInterval bounds must be finite.")
        if float(self.end) <= float(self.start):
            raise ValueError("DwellInterval.end must exceed start.")
        if not isfinite(float(self.repetitions)) or self.repetitions < 0.0:
            raise ValueError("DwellInterval.repetitions must be nonnegative.")

    @property
    def duration(self) -> float:
        return float(self.end - self.start)


def _history_window(history, start: float, end: float) -> np.ndarray:
    coordinates = np.asarray(history.abscissa, dtype=float)
    values = np.asarray(history.values, dtype=float)
    if values.ndim != 1:
        raise ValueError(
            f"History {history.name!r} must be scalar for dwell assessment."
        )
    tolerance = 1.0e-12 * max(1.0, abs(start), abs(end))
    if start < coordinates[0] - tolerance or end > coordinates[-1] + tolerance:
        raise ValueError(
            f"Dwell [{start:g}, {end:g}] lies outside history "
            f"[{coordinates[0]:g}, {coordinates[-1]:g}]."
        )
    inside = (coordinates > start) & (coordinates < end)
    return np.concatenate(
        (
            [np.interp(start, coordinates, values)],
            values[inside],
            [np.interp(end, coordinates, values)],
        )
    )


def creep_blocks_from_result(
    result,
    *,
    stress_history: str,
    temperature_history: str,
    dwells: Iterable[DwellInterval],
    rupture_time: Callable[[float, float], float],
    rupture_source: str,
    stress_reducer: str = "maximum_absolute",
    temperature_reducer: str = "maximum",
) -> tuple[CreepDamageBlock, ...]:
    """Create source-identified creep blocks from named result histories.

    The rupture relation stays a user-owned callable.  AgentFEM owns only the
    reproducible extraction of governing stress and temperature per declared
    dwell; it does not embed licensed code curves or material data.
    """

    if not callable(rupture_time):
        raise TypeError("rupture_time must be callable(stress, temperature).")
    if not str(rupture_source).strip():
        raise ValueError("rupture_source must identify the supplied relation.")
    try:
        stress = result.histories[stress_history]
        temperature = result.histories[temperature_history]
    except KeyError as exc:
        raise KeyError(
            f"Required result history is absent; available={tuple(result.histories)!r}."
        ) from exc
    stress_reducers = {
        "maximum": np.max,
        "maximum_absolute": lambda values: np.max(np.abs(values)),
        "mean": np.mean,
    }
    temperature_reducers = {"maximum": np.max, "mean": np.mean}
    if stress_reducer not in stress_reducers:
        raise ValueError(
            f"stress_reducer must be one of {tuple(stress_reducers)}."
        )
    if temperature_reducer not in temperature_reducers:
        raise ValueError(
            f"temperature_reducer must be one of {tuple(temperature_reducers)}."
        )
    blocks = []
    for dwell in tuple(dwells):
        if not isinstance(dwell, DwellInterval):
            raise TypeError("dwells must contain DwellInterval objects.")
        selected_stress = float(
            stress_reducers[stress_reducer](
                _history_window(stress, dwell.start, dwell.end)
            )
        )
        selected_temperature = float(
            temperature_reducers[temperature_reducer](
                _history_window(temperature, dwell.start, dwell.end)
            )
        )
        selected_rupture = float(
            rupture_time(selected_stress, selected_temperature)
        )
        blocks.append(
            CreepDamageBlock(
                duration=dwell.duration,
                rupture_time=selected_rupture,
                repetitions=dwell.repetitions,
                label=dwell.label,
                source=rupture_source,
                stress=selected_stress,
                temperature=selected_temperature,
                start_time=dwell.start,
                end_time=dwell.end,
            )
        )
    if not blocks:
        raise ValueError("At least one dwell interval is required.")
    return tuple(blocks)


@dataclass(frozen=True)
class CreepDamageAssessment:
    """Auditable linear time-fraction assessment over service blocks."""

    blocks: tuple[CreepDamageBlock, ...]
    damage: float

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "creep_time_fraction_assessment",
            "maturity": "postprocessor",
            "damage": float(self.damage),
            "block_count": len(self.blocks),
            "blocks": [block.as_dict() for block in self.blocks],
        }


@dataclass(frozen=True)
class InteractionDiagram:
    """Declared creep/fatigue allowable boundary in damage coordinates.

    Points are ordered as ``(creep_damage, allowable_fatigue_damage)``.  The
    polyline must move right and down.  This compact contract can represent a
    linear damage sum, a licensed design-code curve, company practice, or a
    research hypothesis without putting any of those data inside AgentFEM.
    """

    points: tuple[tuple[float, float], ...]
    name: str
    source: str

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError(
                "InteractionDiagram requires at least two (Dc, Df) points."
            )
        if np.any(~np.isfinite(points)) or np.any(points < 0.0) or np.any(points > 1.0):
            raise ValueError(
                "InteractionDiagram damage coordinates must lie in [0, 1]."
            )
        if np.any(np.diff(points[:, 0]) <= 0.0):
            raise ValueError("InteractionDiagram creep damage must increase strictly.")
        if np.any(np.diff(points[:, 1]) >= 0.0):
            raise ValueError(
                "InteractionDiagram allowable fatigue damage must decrease strictly."
            )
        if not str(self.name).strip() or not str(self.source).strip():
            raise ValueError("InteractionDiagram name and source must be nonempty.")
        object.__setattr__(
            self,
            "points",
            tuple((float(x), float(y)) for x, y in points),
        )

    def allowable_fatigue_damage(self, creep_damage: float) -> float:
        selected = float(creep_damage)
        if not isfinite(selected) or selected < 0.0:
            raise ValueError("creep_damage must be finite and nonnegative.")
        points = np.asarray(self.points, dtype=float)
        if selected <= points[0, 0]:
            return float(points[0, 1])
        if selected > points[-1, 0]:
            return 0.0
        return float(np.interp(selected, points[:, 0], points[:, 1]))

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "creep_fatigue_interaction_diagram",
            "name": self.name,
            "source": self.source,
            "points": [list(point) for point in self.points],
            "interpolation": "piecewise_linear",
        }


@dataclass(frozen=True)
class CreepFatigueAssessment:
    """Combined engineering assessment from independent damage consumers."""

    fatigue: FatigueAssessment
    creep: CreepDamageAssessment
    interaction: InteractionDiagram
    allowable_fatigue_damage: float
    margin: float
    acceptable: bool

    @property
    def fatigue_damage(self) -> float:
        return float(self.fatigue.damage)

    @property
    def creep_damage(self) -> float:
        return float(self.creep.damage)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "creep_fatigue_engineering_assessment",
            "maturity": "postprocessor",
            "acceptable": bool(self.acceptable),
            "creep_damage": self.creep_damage,
            "fatigue_damage": self.fatigue_damage,
            "allowable_fatigue_damage": float(self.allowable_fatigue_damage),
            "margin": float(self.margin),
            "creep": self.creep.as_dict(),
            "fatigue": self.fatigue.as_dict(),
            "interaction": self.interaction.as_dict(),
        }

    def attach(self, result, *, prefix: str = "creep_fatigue"):
        """Attach scalar decisions and the complete assessment to a result."""

        selected = str(prefix).strip().lower().replace(" ", "_")
        if not selected:
            raise ValueError("assessment result prefix must be nonempty.")
        result.add_quantities(
            {
                f"{selected}_creep_damage": self.creep_damage,
                f"{selected}_fatigue_damage": self.fatigue_damage,
                f"{selected}_allowable_fatigue_damage": self.allowable_fatigue_damage,
                f"{selected}_margin": self.margin,
                f"{selected}_acceptable": int(self.acceptable),
            },
            kind="assessment",
        )
        result.metadata.setdefault("assessments", {})[selected] = self.as_dict()
        return result


def creep_time_fraction(
    blocks: Iterable[CreepDamageBlock],
) -> CreepDamageAssessment:
    """Evaluate the linear creep time-fraction rule for declared blocks."""

    selected = tuple(blocks)
    if not selected:
        raise ValueError("creep_time_fraction requires at least one block.")
    if not all(isinstance(block, CreepDamageBlock) for block in selected):
        raise TypeError("creep_time_fraction expects CreepDamageBlock objects.")
    return CreepDamageAssessment(
        blocks=selected,
        damage=float(sum(block.damage for block in selected)),
    )


def interaction_diagram(*, points, name: str, source: str) -> InteractionDiagram:
    """Create an explicit piecewise-linear creep/fatigue interaction curve."""

    return InteractionDiagram(tuple(tuple(point) for point in points), name, source)


def linear_interaction() -> InteractionDiagram:
    """Return the transparent reference boundary ``Dc + Df = 1``."""

    return InteractionDiagram(
        ((0.0, 1.0), (1.0, 0.0)),
        name="linear damage interaction",
        source="AgentFEM transparent reference: Dc + Df = 1",
    )


def creep_fatigue(
    *,
    fatigue: FatigueAssessment,
    creep: CreepDamageAssessment,
    interaction: InteractionDiagram | None = None,
) -> CreepFatigueAssessment:
    """Combine existing fatigue and creep assessments against one boundary."""

    if not isinstance(fatigue, FatigueAssessment):
        raise TypeError("fatigue must be a constitutive.fatigue.FatigueAssessment.")
    if not isinstance(creep, CreepDamageAssessment):
        raise TypeError("creep must be a CreepDamageAssessment.")
    selected = linear_interaction() if interaction is None else interaction
    if not isinstance(selected, InteractionDiagram):
        raise TypeError("interaction must be an InteractionDiagram.")
    allowable = selected.allowable_fatigue_damage(creep.damage)
    margin = allowable - float(fatigue.damage)
    tolerance = 1.0e-12 * max(1.0, allowable, abs(float(fatigue.damage)))
    return CreepFatigueAssessment(
        fatigue=fatigue,
        creep=creep,
        interaction=selected,
        allowable_fatigue_damage=allowable,
        margin=margin,
        acceptable=bool(margin >= -tolerance),
    )


def creep_fatigue_from_result(
    result,
    *,
    fatigue_history: str,
    fatigue_curve,
    stress_history: str,
    temperature_history: str,
    dwells: Iterable[DwellInterval],
    rupture_time: Callable[[float, float], float],
    rupture_source: str,
    interaction: InteractionDiagram | None = None,
    ultimate_strength: float | None = None,
    stress_reducer: str = "maximum_absolute",
    temperature_reducer: str = "maximum",
) -> CreepFatigueAssessment:
    """Build the engineering V1 assessment from named result histories."""

    from .constitutive.fatigue import assess_result_history

    fatigue = assess_result_history(
        result,
        fatigue_history,
        fatigue_curve,
        ultimate_strength=ultimate_strength,
    )
    creep = creep_time_fraction(
        creep_blocks_from_result(
            result,
            stress_history=stress_history,
            temperature_history=temperature_history,
            dwells=dwells,
            rupture_time=rupture_time,
            rupture_source=rupture_source,
            stress_reducer=stress_reducer,
            temperature_reducer=temperature_reducer,
        )
    )
    return creep_fatigue(
        fatigue=fatigue,
        creep=creep,
        interaction=interaction,
    )


__all__ = [
    "CreepDamageAssessment",
    "CreepDamageBlock",
    "CreepFatigueAssessment",
    "DwellInterval",
    "InteractionDiagram",
    "SequentialEnergyLedger",
    "creep_fatigue",
    "creep_fatigue_from_result",
    "creep_blocks_from_result",
    "creep_time_fraction",
    "interaction_diagram",
    "linear_interaction",
    "sequential_energy_ledger",
]
