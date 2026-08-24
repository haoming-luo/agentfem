"""Engineering assessments that consume, but do not alter, FEM solutions.

The module deliberately separates solver-integrated constitutive evolution
from code- or experiment-defined life assessment.  Normative interaction
curves remain explicit user inputs with a source; AgentFEM does not embed or
silently reinterpret proprietary design-code data.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np

from .constitutive.fatigue import FatigueAssessment


@dataclass(frozen=True)
class CreepDamageBlock:
    """One dwell or service block for the time-fraction rule."""

    duration: float
    rupture_time: float
    repetitions: float = 1.0
    label: str = ""
    source: str = ""

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
        }


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


__all__ = [
    "CreepDamageAssessment",
    "CreepDamageBlock",
    "CreepFatigueAssessment",
    "InteractionDiagram",
    "creep_fatigue",
    "creep_time_fraction",
    "interaction_diagram",
    "linear_interaction",
]
