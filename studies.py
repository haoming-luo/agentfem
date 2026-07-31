"""Study contexts for finite-element analyses.

A study is the early modeling context: analysis type, physics, dimension, and
modeling assumptions. It is intentionally lightweight. Operators and problems
may consult it, but it does not assemble or solve anything by itself.
"""

from __future__ import annotations

from dataclasses import dataclass


ANALYSIS_TYPES = {
    "linear_static",
    "first_order_transient",
    "second_order_dynamics",
    "modal",
    "nonlinear_static",
    "nonlinear_transient",
}

PHYSICS_TYPES = {
    "solid_mechanics",
    "heat_transfer",
    "acoustics",
    "electromagnetics",
    "multiphysics",
}

SOLID_2D_ASSUMPTIONS = {
    "plane_stress",
    "plane_strain",
    "axisymmetric",
}


@dataclass(frozen=True)
class Study:
    """Early analysis context for a finite-element workflow."""

    analysis: str
    physics: str
    dimension: int
    assumption: str | None = None
    time_domain: str | None = None
    linearity: str | None = None
    preferred_procedure: str | None = None
    name: str = "study"

    def __post_init__(self) -> None:
        analysis = _normalize(self.analysis)
        physics = _normalize(self.physics)
        assumption = _normalize_optional(self.assumption)
        time_domain = self.time_domain or _default_time_domain(analysis)
        linearity = self.linearity or _default_linearity(analysis)

        object.__setattr__(self, "analysis", analysis)
        object.__setattr__(self, "physics", physics)
        object.__setattr__(self, "assumption", assumption)
        object.__setattr__(self, "time_domain", time_domain)
        object.__setattr__(self, "linearity", linearity)
        object.__setattr__(
            self,
            "preferred_procedure",
            _normalize_optional(self.preferred_procedure),
        )
        self.validate()

    @property
    def is_static(self) -> bool:
        """Return true for static analysis contexts."""

        return self.time_domain == "static"

    @property
    def is_transient(self) -> bool:
        """Return true for time-domain analysis contexts."""

        return self.time_domain == "transient"

    @property
    def is_solid_mechanics(self) -> bool:
        """Return true for solid-mechanics physics."""

        return self.physics == "solid_mechanics"

    @property
    def is_heat_transfer(self) -> bool:
        """Return true for heat-transfer physics."""

        return self.physics == "heat_transfer"

    def validate(self) -> None:
        """Validate the context before it influences forms or operators."""

        if self.analysis not in ANALYSIS_TYPES:
            raise ValueError(
                f"Unknown analysis type {self.analysis!r}. "
                f"Expected one of {sorted(ANALYSIS_TYPES)}."
            )
        if self.physics not in PHYSICS_TYPES:
            raise ValueError(
                f"Unknown physics type {self.physics!r}. "
                f"Expected one of {sorted(PHYSICS_TYPES)}."
            )
        if self.dimension not in (1, 2, 3):
            raise ValueError("Study dimension must be 1, 2, or 3.")
        if self.physics == "solid_mechanics" and self.dimension == 2:
            if self.assumption not in SOLID_2D_ASSUMPTIONS:
                raise ValueError(
                    "2D solid mechanics requires assumption='plane_stress', "
                    "'plane_strain', or 'axisymmetric'."
                )
        if self.dimension == 3 and self.assumption in SOLID_2D_ASSUMPTIONS:
            raise ValueError("plane_stress, plane_strain, and axisymmetric are 2D assumptions.")

    def require(self, *, analysis: str | None = None, physics: str | None = None) -> None:
        """Raise a modeling error if this study does not match a requirement."""

        if analysis is not None and self.analysis != _normalize(analysis):
            raise ValueError(f"Expected analysis={analysis!r}, got {self.analysis!r}.")
        if physics is not None and self.physics != _normalize(physics):
            raise ValueError(f"Expected physics={physics!r}, got {self.physics!r}.")

    def summary(self) -> dict[str, object]:
        """Return an agent-readable study summary."""

        return {
            "name": self.name,
            "analysis": self.analysis,
            "physics": self.physics,
            "dimension": self.dimension,
            "assumption": self.assumption,
            "time_domain": self.time_domain,
            "linearity": self.linearity,
            "preferred_procedure": self.preferred_procedure,
        }


def define(
    *,
    analysis: str,
    physics: str,
    dimension: int,
    assumption: str | None = None,
    name: str | None = None,
    preferred_procedure: str | None = None,
) -> Study:
    """Define a general finite-element study context."""

    return Study(
        analysis=analysis,
        physics=physics,
        dimension=dimension,
        assumption=assumption,
        name=name or f"{_normalize(analysis)}_{_normalize(physics)}",
        preferred_procedure=preferred_procedure,
    )


def linear_static(
    *,
    physics: str,
    dimension: int,
    assumption: str | None = None,
    name: str | None = None,
) -> Study:
    """Define a linear static study."""

    return define(
        analysis="linear_static",
        physics=physics,
        dimension=dimension,
        assumption=assumption,
        name=name,
    )


def nonlinear_static(
    *,
    physics: str,
    dimension: int,
    assumption: str | None = None,
    name: str | None = None,
) -> Study:
    """Define a nonlinear static study."""

    return define(
        analysis="nonlinear_static",
        physics=physics,
        dimension=dimension,
        assumption=assumption,
        name=name,
    )


def first_order_transient(
    *,
    physics: str,
    dimension: int,
    assumption: str | None = None,
    name: str | None = None,
) -> Study:
    """Define a first-order transient study."""

    return define(
        analysis="first_order_transient",
        physics=physics,
        dimension=dimension,
        assumption=assumption,
        name=name,
    )


def transient(
    *,
    physics: str,
    dimension: int,
    assumption: str | None = None,
    name: str | None = None,
) -> Study:
    """Compatibility alias for ``first_order_transient``."""

    return first_order_transient(
        physics=physics,
        dimension=dimension,
        assumption=assumption,
        name=name,
    )


def second_order_dynamics(
    *,
    physics: str,
    dimension: int,
    assumption: str | None = None,
    name: str | None = None,
    procedure: str | None = None,
) -> Study:
    """Define a second-order dynamics study."""

    return define(
        analysis="second_order_dynamics",
        physics=physics,
        dimension=dimension,
        assumption=assumption,
        name=name,
        preferred_procedure=procedure,
    )


def implicit_dynamics(
    *,
    physics: str,
    dimension: int,
    assumption: str | None = None,
    method: str = "newmark",
    name: str | None = None,
) -> Study:
    """Define second-order dynamics with a Standard/implicit preference."""

    normalized = _normalize(method)
    if normalized not in {"newmark", "generalized_alpha"}:
        raise ValueError("Implicit dynamics method must be 'newmark' or 'generalized_alpha'.")
    return second_order_dynamics(
        physics=physics,
        dimension=dimension,
        assumption=assumption,
        procedure=normalized,
        name=name,
    )


def explicit_dynamics(
    *,
    physics: str,
    dimension: int,
    assumption: str | None = None,
    name: str | None = None,
) -> Study:
    """Define second-order dynamics with an Explicit preference."""

    return second_order_dynamics(
        physics=physics,
        dimension=dimension,
        assumption=assumption,
        procedure="central_difference",
        name=name,
    )


def _normalize(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_optional(value: str | None) -> str | None:
    return None if value is None else _normalize(value)


def _default_time_domain(analysis: str) -> str:
    if analysis in {"linear_static", "nonlinear_static", "modal"}:
        return "static"
    return "transient"


def _default_linearity(analysis: str) -> str:
    if analysis.startswith("nonlinear"):
        return "nonlinear"
    return "linear"
