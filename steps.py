"""Analysis-step controls expressed in finite-element language.

Incrementation belongs to an analysis step, not to a Newton or linear solver.
The controls in this module therefore describe how a normalized step interval
``0 <= load_factor <= 1`` is traversed.  Solver tolerances remain in
``agentfem.solvers`` and field-output frequency remains in
``agentfem.results``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class AutomaticIncrementation:
    """Adaptive load/time incrementation for one analysis step.

    ``max_increments`` is an upper bound on accepted increments, not a requested
    number of increments.  Failed attempts are governed separately by
    ``max_cutbacks``.
    """

    initial: float = 0.1
    minimum: float = 1.0e-5
    maximum: float = 0.25
    max_increments: int = 100
    max_cutbacks: int = 5
    cutback_factor: float = 0.25
    growth_factor: float = 1.5
    fast_iterations: int = 4
    slow_iterations: int = 10

    def __post_init__(self) -> None:
        sizes = (self.initial, self.minimum, self.maximum)
        if any(not isfinite(value) or value <= 0.0 for value in sizes):
            raise ValueError("Automatic increment sizes must be finite and positive.")
        if self.minimum > self.initial or self.initial > self.maximum:
            raise ValueError(
                "Automatic increments require minimum <= initial <= maximum."
            )
        if self.maximum > 1.0:
            raise ValueError(
                "Automatic maximum cannot exceed the normalized step interval 1.0."
            )
        if self.max_increments <= 0:
            raise ValueError("max_increments must be positive.")
        if self.max_cutbacks < 0:
            raise ValueError("max_cutbacks cannot be negative.")
        if not 0.0 < self.cutback_factor < 1.0:
            raise ValueError("cutback_factor must lie between zero and one.")
        if self.growth_factor <= 1.0:
            raise ValueError("growth_factor must be greater than one.")
        if self.fast_iterations <= 0:
            raise ValueError("fast_iterations must be positive.")
        if self.slow_iterations <= self.fast_iterations:
            raise ValueError("slow_iterations must exceed fast_iterations.")

    @property
    def automatic(self) -> bool:
        return True

    def after_convergence(self, size: float, iterations: int) -> float:
        """Return the proposed size of the next increment."""

        selected = float(size)
        if iterations <= self.fast_iterations:
            selected *= self.growth_factor
        elif iterations >= self.slow_iterations:
            selected *= 0.5
        return min(self.maximum, max(self.minimum, selected))

    def after_failure(self, size: float) -> float:
        """Return a cut-back size after a failed attempt."""

        return float(size) * self.cutback_factor

    def summary(self) -> dict[str, object]:
        return {
            "kind": "automatic_incrementation",
            "initial": self.initial,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "max_increments": self.max_increments,
            "max_cutbacks": self.max_cutbacks,
            "cutback_factor": self.cutback_factor,
            "growth_factor": self.growth_factor,
            "fast_iterations": self.fast_iterations,
            "slow_iterations": self.slow_iterations,
        }


@dataclass(frozen=True)
class FixedIncrementation:
    """A prescribed monotone load-factor path for one analysis step."""

    load_factors: tuple[float, ...]

    def __post_init__(self) -> None:
        factors = tuple(float(value) for value in self.load_factors)
        if not factors or any(not isfinite(value) or value <= 0.0 for value in factors):
            raise ValueError("Fixed load factors must be finite and positive.")
        if any(right <= left for left, right in zip(factors, factors[1:])):
            raise ValueError("Fixed load factors must be strictly increasing.")
        if abs(factors[-1] - 1.0) > 1.0e-12:
            raise ValueError("A fixed load path must end at load factor 1.0.")
        object.__setattr__(self, "load_factors", factors)

    @property
    def automatic(self) -> bool:
        return False

    @property
    def increments(self) -> int:
        return len(self.load_factors)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "fixed_incrementation",
            "increments": self.increments,
            "load_factors": self.load_factors,
        }


def automatic(
    *,
    initial: float = 0.1,
    minimum: float = 1.0e-5,
    maximum: float = 0.25,
    max_increments: int = 100,
    max_cutbacks: int = 5,
    cutback_factor: float = 0.25,
    growth_factor: float = 1.5,
    fast_iterations: int = 4,
    slow_iterations: int = 10,
) -> AutomaticIncrementation:
    """Create inspectable Abaqus-style automatic incrementation."""

    return AutomaticIncrementation(
        initial=initial,
        minimum=minimum,
        maximum=maximum,
        max_increments=max_increments,
        max_cutbacks=max_cutbacks,
        cutback_factor=cutback_factor,
        growth_factor=growth_factor,
        fast_iterations=fast_iterations,
        slow_iterations=slow_iterations,
    )


def fixed(increments: int) -> FixedIncrementation:
    """Divide the normalized step interval into exactly ``increments`` parts."""

    selected = int(increments)
    if selected <= 0:
        raise ValueError("Fixed increments must be positive.")
    return FixedIncrementation(
        tuple(index / selected for index in range(1, selected + 1))
    )


def at(*load_factors: float) -> FixedIncrementation:
    """Create a prescribed, nonuniform load-factor path."""

    return FixedIncrementation(tuple(load_factors))


def normalize(
    value=None,
    *,
    increments: int | None = None,
    load_factors=None,
):
    """Normalize public and compatibility incrementation inputs."""

    supplied = sum(
        item is not None for item in (value, increments, load_factors)
    )
    if supplied > 1:
        raise ValueError(
            "Specify only one of incrementation, increments, or load_factors."
        )
    if value is None and increments is None and load_factors is None:
        return automatic()
    if increments is not None:
        return fixed(increments)
    if load_factors is not None:
        return at(*tuple(load_factors))
    if not isinstance(value, (AutomaticIncrementation, FixedIncrementation)):
        raise TypeError(
            "incrementation must be steps.automatic(...) or steps.fixed(...)."
        )
    return value


__all__ = [
    "AutomaticIncrementation",
    "FixedIncrementation",
    "at",
    "automatic",
    "fixed",
    "normalize",
]
