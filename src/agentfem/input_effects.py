# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Neutral contract for data that changes during a Procedure.

The Model owns the physical assets and declares their effects.  A Procedure
consumes the same declaration to choose a safe assembly lifecycle.  Keeping
this dependency-free vocabulary outside both owners avoids making Model depend
on a time integrator.  The contract does not inspect UFL expressions or guess
from field names.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
import json


class TimeInputEffect(str, Enum):
    """Numerical consequence of updating one time-dependent input."""

    RIGHT_HAND_SIDE = "right_hand_side"
    OPERATOR = "operator"
    STATE = "state"
    OUTPUT = "output"


_EFFECT_ALIASES = {
    "rhs": TimeInputEffect.RIGHT_HAND_SIDE,
    "right_hand_side": TimeInputEffect.RIGHT_HAND_SIDE,
    "load": TimeInputEffect.RIGHT_HAND_SIDE,
    "operator": TimeInputEffect.OPERATOR,
    "matrix": TimeInputEffect.OPERATOR,
    "state": TimeInputEffect.STATE,
    "history": TimeInputEffect.STATE,
    "output": TimeInputEffect.OUTPUT,
    "observation": TimeInputEffect.OUTPUT,
}
_CONSERVATIVE_EFFECTS = frozenset(TimeInputEffect)


def normalize_effects(
    effects: TimeInputEffect | str | Iterable[TimeInputEffect | str],
) -> frozenset[TimeInputEffect]:
    """Return a non-empty, canonical effect set."""

    selected = (effects,) if isinstance(effects, (str, TimeInputEffect)) else effects
    normalized = []
    for effect in selected:
        if isinstance(effect, TimeInputEffect):
            normalized.append(effect)
            continue
        key = str(effect).strip().lower().replace("-", "_").replace(" ", "_")
        try:
            normalized.append(_EFFECT_ALIASES[key])
        except KeyError as error:
            choices = ", ".join(item.value for item in TimeInputEffect)
            raise ValueError(
                f"Unknown time-input effect {effect!r}; expected one of {choices}."
            ) from error
    result = frozenset(normalized)
    if not result:
        raise ValueError("A time input must declare at least one effect.")
    return result


@dataclass(frozen=True)
class TimeInputUpdate:
    """One callable update with explicit numerical effects."""

    callback: Callable[[float], object]
    effects: frozenset[TimeInputEffect]
    name: str = "time_input"
    declaration: str = "explicit"
    identity: object | None = None

    def __post_init__(self) -> None:
        if not callable(self.callback):
            raise TypeError("TimeInputUpdate.callback must be callable.")
        object.__setattr__(self, "effects", normalize_effects(self.effects))
        object.__setattr__(self, "name", str(self.name).strip() or "time_input")
        if self.identity is not None:
            try:
                identity = json.loads(
                    json.dumps(
                        self.identity,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "A time-input identity must be finite JSON data so it can "
                    "survive result and checkpoint archives."
                ) from error
            object.__setattr__(self, "identity", identity)

    def __call__(self, time_value: float):
        return self.callback(float(time_value))

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "effects": tuple(sorted(item.value for item in self.effects)),
            "declaration": self.declaration,
            "identity": self.identity,
        }


@dataclass(frozen=True)
class TimeInputPlan:
    """Ordered updates and their combined invalidation contract."""

    updates: tuple[TimeInputUpdate, ...]

    def __post_init__(self) -> None:
        selected = tuple(self.updates)
        if any(not isinstance(item, TimeInputUpdate) for item in selected):
            raise TypeError("TimeInputPlan requires TimeInputUpdate entries.")
        object.__setattr__(self, "updates", selected)

    @property
    def effects(self) -> frozenset[TimeInputEffect]:
        return frozenset(effect for update in self.updates for effect in update.effects)

    @property
    def changes_operator(self) -> bool:
        """Return whether a fixed assembled operator is unsafe to reuse."""

        return bool(self.effects & {TimeInputEffect.OPERATOR, TimeInputEffect.STATE})

    def __call__(self, time_value: float) -> tuple[object, ...]:
        return tuple(update(float(time_value)) for update in self.updates)

    def summary(self) -> dict[str, object]:
        effects = self.effects
        return {
            "kind": "time_input_plan",
            "effects": tuple(sorted(item.value for item in effects)),
            "changes_right_hand_side": (TimeInputEffect.RIGHT_HAND_SIDE in effects),
            "changes_operator": self.changes_operator,
            "changes_state": TimeInputEffect.STATE in effects,
            "changes_output": TimeInputEffect.OUTPUT in effects,
            "restart_identity_bound": all(
                item.identity is not None for item in self.updates
            ),
            "updates": tuple(item.summary() for item in self.updates),
        }


def update(
    callback: Callable[[float], object],
    *,
    effects: TimeInputEffect | str | Iterable[TimeInputEffect | str],
    name: str | None = None,
    identity: object | None = None,
) -> TimeInputUpdate:
    """Declare the numerical effects of a custom time callback."""

    return TimeInputUpdate(
        callback=callback,
        effects=normalize_effects(effects),
        name=name or getattr(callback, "__name__", "time_input"),
        identity=identity,
    )


def from_asset(asset: object) -> TimeInputUpdate:
    """Create an update from a Model asset, conservatively when untyped."""

    callback = getattr(asset, "update", None)
    if callback is None or not callable(callback):
        raise TypeError("A time-dependent asset must provide update(time).")
    declared = getattr(asset, "time_effects", None)
    return TimeInputUpdate(
        callback=callback,
        effects=(
            _CONSERVATIVE_EFFECTS if declared is None else normalize_effects(declared)
        ),
        name=str(getattr(asset, "name", type(asset).__name__)),
        declaration="conservative" if declared is None else "asset",
        identity=(asset.summary() if hasattr(asset, "summary") else None),
    )


def as_update(value: object, *, name: str | None = None) -> TimeInputUpdate:
    """Normalize a callback or typed update.

    Bare callbacks are deliberately conservative: they may mutate any
    coefficient or history value, so a Procedure must not assume a fixed
    operator.  Users can call :func:`update` to declare a narrower contract.
    """

    if isinstance(value, TimeInputUpdate):
        return value
    if not callable(value):
        raise TypeError("A time input must be callable or a TimeInputUpdate.")
    return TimeInputUpdate(
        callback=value,
        effects=_CONSERVATIVE_EFFECTS,
        name=name or getattr(value, "__name__", "time_input"),
        declaration="conservative",
    )


def compose(*values: object) -> TimeInputPlan | None:
    """Compose typed updates without losing order or invalidation effects."""

    updates = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, TimeInputPlan):
            updates.extend(value.updates)
        else:
            updates.append(as_update(value))
    return None if not updates else TimeInputPlan(tuple(updates))


def effects_of(value: object | None) -> frozenset[TimeInputEffect]:
    """Return declared effects, conservatively classifying bare callbacks."""

    if value is None:
        return frozenset()
    if isinstance(value, TimeInputPlan | TimeInputUpdate):
        return value.effects
    return _CONSERVATIVE_EFFECTS if callable(value) else frozenset()


def summary_of(value: object | None) -> dict[str, object]:
    """Return a stable description for Procedure and result evidence."""

    if value is None:
        return {
            "kind": "time_input_plan",
            "effects": (),
            "changes_right_hand_side": False,
            "changes_operator": False,
            "changes_state": False,
            "changes_output": False,
            "restart_identity_bound": True,
            "updates": (),
        }
    if isinstance(value, TimeInputPlan):
        return value.summary()
    selected = as_update(value)
    return TimeInputPlan((selected,)).summary()


__all__ = [
    "TimeInputEffect",
    "TimeInputPlan",
    "TimeInputUpdate",
    "as_update",
    "compose",
    "effects_of",
    "from_asset",
    "normalize_effects",
    "summary_of",
    "update",
]
