"""Typed parameters and deterministic design-of-experiment plans."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import exp, isfinite, log
from numbers import Integral, Real
from typing import Iterable, Mapping, Sequence

import numpy as np


def _require_name(name: str) -> str:
    selected = str(name).strip()
    if not selected:
        raise ValueError("Parameter names must not be empty.")
    return selected


@dataclass(frozen=True)
class RealParameter:
    """Bounded continuous parameter with optional units and log scaling."""

    name: str
    lower: float
    upper: float
    unit: str | None = None
    description: str = ""
    scale: str = "linear"
    nominal: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_name(self.name))
        lower = float(self.lower)
        upper = float(self.upper)
        if not isfinite(lower) or not isfinite(upper) or lower >= upper:
            raise ValueError(
                f"RealParameter {self.name!r} requires finite lower < upper."
            )
        scale = self.scale.lower().replace("-", "_")
        if scale not in {"linear", "log"}:
            raise ValueError("RealParameter.scale must be 'linear' or 'log'.")
        if scale == "log" and lower <= 0.0:
            raise ValueError("Log-scaled parameters require lower > 0.")
        if self.nominal is not None:
            self.validate(self.nominal)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "scale", scale)

    def validate(self, value) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"Parameter {self.name!r} requires a real number.")
        selected = float(value)
        if not isfinite(selected) or not self.lower <= selected <= self.upper:
            raise ValueError(
                f"Parameter {self.name!r}={selected!r} is outside "
                f"[{self.lower}, {self.upper}]."
            )
        return selected

    def from_unit_interval(self, value: float) -> float:
        fraction = _unit_fraction(value)
        if fraction == 0.0:
            return self.lower
        if fraction == 1.0:
            return self.upper
        if self.scale == "log":
            return exp(log(self.lower) + fraction * (log(self.upper) - log(self.lower)))
        return self.lower + fraction * (self.upper - self.lower)

    def to_unit_interval(self, value) -> float:
        selected = self.validate(value)
        if self.scale == "log":
            return (log(selected) - log(self.lower)) / (
                log(self.upper) - log(self.lower)
            )
        return (selected - self.lower) / (self.upper - self.lower)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "real",
            "lower": self.lower,
            "upper": self.upper,
            "unit": self.unit,
            "description": self.description,
            "scale": self.scale,
            "nominal": self.nominal,
        }


@dataclass(frozen=True)
class IntegerParameter:
    """Bounded integer parameter."""

    name: str
    lower: int
    upper: int
    unit: str | None = None
    description: str = ""
    nominal: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_name(self.name))
        if isinstance(self.lower, bool) or isinstance(self.upper, bool):
            raise TypeError("IntegerParameter bounds must be integers.")
        if not isinstance(self.lower, Integral) or not isinstance(self.upper, Integral):
            raise TypeError("IntegerParameter bounds must be integers.")
        if int(self.lower) > int(self.upper):
            raise ValueError("IntegerParameter requires lower <= upper.")
        object.__setattr__(self, "lower", int(self.lower))
        object.__setattr__(self, "upper", int(self.upper))
        if self.nominal is not None:
            self.validate(self.nominal)

    def validate(self, value) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"Parameter {self.name!r} requires an integer.")
        selected = int(value)
        if not self.lower <= selected <= self.upper:
            raise ValueError(
                f"Parameter {self.name!r}={selected!r} is outside "
                f"[{self.lower}, {self.upper}]."
            )
        return selected

    def from_unit_interval(self, value: float) -> int:
        fraction = _unit_fraction(value)
        width = self.upper - self.lower + 1
        return min(self.upper, self.lower + int(fraction * width))

    def to_unit_interval(self, value) -> float:
        selected = self.validate(value)
        width = self.upper - self.lower
        return 0.0 if width == 0 else (selected - self.lower) / width

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "integer",
            "lower": self.lower,
            "upper": self.upper,
            "unit": self.unit,
            "description": self.description,
            "nominal": self.nominal,
        }


@dataclass(frozen=True)
class ChoiceParameter:
    """Finite categorical or policy parameter."""

    name: str
    choices: tuple[object, ...]
    description: str = ""
    nominal: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_name(self.name))
        choices = tuple(self.choices)
        if not choices:
            raise ValueError("ChoiceParameter requires at least one choice.")
        for index, choice in enumerate(choices):
            if not isinstance(choice, (str, bool, Integral, Real)):
                raise TypeError(
                    "ChoiceParameter choices must be JSON-safe scalar "
                    "strings, booleans, integers, or real numbers."
                )
            if isinstance(choice, Real) and not isfinite(float(choice)):
                raise ValueError("ChoiceParameter choices must be finite.")
            for other in choices[:index]:
                if choice == other:
                    raise ValueError(
                        f"ChoiceParameter {self.name!r} has duplicate choice {choice!r}."
                    )
        object.__setattr__(self, "choices", choices)
        if self.nominal is not None:
            self.validate(self.nominal)

    def validate(self, value):
        for choice in self.choices:
            if value == choice:
                return choice
        raise ValueError(
            f"Parameter {self.name!r}={value!r} is not one of {self.choices!r}."
        )

    def from_unit_interval(self, value: float):
        fraction = _unit_fraction(value)
        return self.choices[min(len(self.choices) - 1, int(fraction * len(self.choices)))]

    def to_unit_interval(self, value) -> float:
        selected = self.validate(value)
        index = self.choices.index(selected)
        return 0.0 if len(self.choices) == 1 else index / (len(self.choices) - 1)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "choice",
            "choices": self.choices,
            "description": self.description,
            "nominal": self.nominal,
        }


Parameter = RealParameter | IntegerParameter | ChoiceParameter


@dataclass(frozen=True)
class ParameterSpace:
    """Ordered scientific input schema for a campaign."""

    parameters: tuple[Parameter, ...]
    name: str = "parameter_space"

    def __post_init__(self) -> None:
        parameters = tuple(self.parameters)
        if not parameters:
            raise ValueError("ParameterSpace requires at least one parameter.")
        names = tuple(parameter.name for parameter in parameters)
        if len(set(names)) != len(names):
            raise ValueError(f"Parameter names must be unique, got {names!r}.")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "name", _require_name(self.name))

    @classmethod
    def create(cls, *parameters: Parameter, name: str = "parameter_space"):
        return cls(parameters=tuple(parameters), name=name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return non-ordinal learning-feature names.

        Continuous/integer parameters contribute one normalized feature.
        Choices contribute one feature per category so learning adapters do not
        invent an ordering or distance between categorical values.
        """

        names = []
        for parameter in self.parameters:
            if isinstance(parameter, ChoiceParameter):
                names.extend(
                    f"{parameter.name}={type(choice).__name__}:{choice!r}"
                    for choice in parameter.choices
                )
            else:
                names.append(parameter.name)
        return tuple(names)

    def validate(self, values: Mapping[str, object]) -> dict[str, object]:
        missing = [name for name in self.names if name not in values]
        extra = [name for name in values if name not in self.names]
        if missing or extra:
            raise ValueError(
                "Parameter sample keys do not match the space; "
                f"missing={missing}, extra={extra}."
            )
        return {
            parameter.name: parameter.validate(values[parameter.name])
            for parameter in self.parameters
        }

    def normalize(self, values: Mapping[str, object]) -> np.ndarray:
        """Map one sample to the unit coordinates used by sampling plans."""

        selected = self.validate(values)
        return np.asarray(
            [
                parameter.to_unit_interval(selected[parameter.name])
                for parameter in self.parameters
            ],
            dtype=float,
        )

    def denormalize(self, values: Sequence[float]) -> dict[str, object]:
        """Restore one sample from sampling-plan unit coordinates."""

        if len(values) != len(self.parameters):
            raise ValueError(
                f"Expected {len(self.parameters)} normalized values, got {len(values)}."
            )
        return {
            parameter.name: parameter.from_unit_interval(value)
            for parameter, value in zip(self.parameters, values, strict=True)
        }

    def encode(self, values: Mapping[str, object]) -> np.ndarray:
        """Encode one sample for learning without ordinal categorical bias."""

        selected = self.validate(values)
        features = []
        for parameter in self.parameters:
            value = selected[parameter.name]
            if isinstance(parameter, ChoiceParameter):
                features.extend(
                    1.0 if value == choice else 0.0
                    for choice in parameter.choices
                )
            else:
                features.append(parameter.to_unit_interval(value))
        return np.asarray(features, dtype=float)

    def decode(self, features: Sequence[float]) -> dict[str, object]:
        """Restore one sample from the learning-feature encoding."""

        values = np.asarray(features, dtype=float).reshape(-1)
        if values.size != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} encoded features, "
                f"got {values.size}."
            )
        result = {}
        offset = 0
        for parameter in self.parameters:
            if isinstance(parameter, ChoiceParameter):
                width = len(parameter.choices)
                block = values[offset : offset + width]
                if not np.all(np.isfinite(block)):
                    raise ValueError(
                        f"Encoded choice {parameter.name!r} contains non-finite values."
                    )
                one_hot = np.isclose(block, 1.0, rtol=0.0, atol=1.0e-12)
                zero_hot = np.isclose(block, 0.0, rtol=0.0, atol=1.0e-12)
                if np.count_nonzero(one_hot) != 1 or not np.all(one_hot | zero_hot):
                    raise ValueError(
                        f"Encoded choice {parameter.name!r} is not a valid one-hot block."
                    )
                result[parameter.name] = parameter.choices[int(np.argmax(block))]
                offset += width
            else:
                result[parameter.name] = parameter.from_unit_interval(values[offset])
                offset += 1
        return result

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "parameter_space",
            "feature_encoding": "normalized_numeric_plus_one_hot_choices",
            "feature_names": self.feature_names,
            "parameters": [parameter.summary() for parameter in self.parameters],
        }


@dataclass(frozen=True)
class SamplingPlan:
    """Immutable, validated collection of parameter samples."""

    space: ParameterSpace
    samples: tuple[Mapping[str, object], ...]
    method: str
    seed: int | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        selected = tuple(self.space.validate(sample) for sample in self.samples)
        if not selected:
            raise ValueError("SamplingPlan requires at least one sample.")
        object.__setattr__(self, "samples", selected)
        object.__setattr__(self, "method", _require_name(self.method))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def summary(self) -> dict[str, object]:
        return {
            "kind": "sampling_plan",
            "method": self.method,
            "seed": self.seed,
            "sample_count": len(self.samples),
            "parameter_space": self.space.summary(),
            "metadata": dict(self.metadata or {}),
        }


def explicit(
    space: ParameterSpace,
    samples: Iterable[Mapping[str, object]],
    *,
    metadata: Mapping[str, object] | None = None,
) -> SamplingPlan:
    """Create a plan from caller-supplied samples."""

    return SamplingPlan(
        space=space,
        samples=tuple(samples),
        method="explicit",
        metadata=metadata,
    )


def random(space: ParameterSpace, count: int, *, seed: int = 0) -> SamplingPlan:
    """Draw reproducible independent uniform samples in normalized space."""

    selected_count = _require_count(count)
    rng = np.random.default_rng(seed)
    values = rng.random((selected_count, len(space.parameters)))
    return SamplingPlan(
        space=space,
        samples=tuple(space.denormalize(row) for row in values),
        method="random_uniform",
        seed=int(seed),
    )


def latin_hypercube(
    space: ParameterSpace,
    count: int,
    *,
    seed: int = 0,
) -> SamplingPlan:
    """Draw a reproducible Latin-hypercube design."""

    selected_count = _require_count(count)
    dimensions = len(space.parameters)
    rng = np.random.default_rng(seed)
    values = np.empty((selected_count, dimensions), dtype=float)
    for dimension in range(dimensions):
        strata = (np.arange(selected_count) + rng.random(selected_count)) / selected_count
        values[:, dimension] = strata[rng.permutation(selected_count)]
    return SamplingPlan(
        space=space,
        samples=tuple(space.denormalize(row) for row in values),
        method="latin_hypercube",
        seed=int(seed),
    )


def full_factorial(
    space: ParameterSpace,
    levels: int | Mapping[str, int] = 3,
) -> SamplingPlan:
    """Create a full-factorial design in normalized coordinates."""

    axes = []
    resolved_levels: dict[str, int] = {}
    for parameter in space.parameters:
        count = levels.get(parameter.name, 3) if isinstance(levels, Mapping) else levels
        selected = _require_count(count)
        resolved_levels[parameter.name] = selected
        axes.append(np.linspace(0.0, 1.0, selected))
    samples = tuple(space.denormalize(point) for point in product(*axes))
    return SamplingPlan(
        space=space,
        samples=samples,
        method="full_factorial",
        metadata={"levels": resolved_levels},
    )


def _unit_fraction(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("Normalized parameter values must be real numbers.")
    selected = float(value)
    if not isfinite(selected) or not 0.0 <= selected <= 1.0:
        raise ValueError("Normalized parameter values must lie in [0, 1].")
    return selected


def _require_count(count) -> int:
    if isinstance(count, bool) or not isinstance(count, Integral) or int(count) <= 0:
        raise ValueError("Sample count/levels must be a positive integer.")
    return int(count)


__all__ = [
    "ChoiceParameter",
    "IntegerParameter",
    "Parameter",
    "ParameterSpace",
    "RealParameter",
    "SamplingPlan",
    "explicit",
    "full_factorial",
    "latin_hypercube",
    "random",
]
