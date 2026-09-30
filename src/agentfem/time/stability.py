# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Auditable stability evidence for explicit time integration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, sqrt


@dataclass(frozen=True)
class ExplicitStabilityContribution:
    """One conservative contribution to an explicit spectral bound.

    ``unsafed_time_increment`` and ``spectral_radius_upper_bound`` describe
    the same bound through ``dt = 2 / sqrt(rho)``.  Keeping both values makes
    the numerical meaning inspectable without asking downstream code to infer
    whether a number is a frequency, an eigenvalue, or a time increment.
    """

    name: str
    unsafed_time_increment: float
    spectral_radius_upper_bound: float
    method: str = "equivalent_spectral_upper_bound"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        dt = float(self.unsafed_time_increment)
        spectral = float(self.spectral_radius_upper_bound)
        method = str(self.method).strip()
        if not name:
            raise ValueError("Explicit stability contribution name cannot be empty.")
        if not method:
            raise ValueError("Explicit stability contribution method cannot be empty.")
        if not isfinite(dt) or dt <= 0.0:
            raise ValueError(
                "Explicit stability time increment must be finite and positive."
            )
        if not isfinite(spectral) or spectral <= 0.0:
            raise ValueError(
                "Explicit stability spectral bound must be finite and positive."
            )
        expected = 2.0 / sqrt(spectral)
        if not isclose(dt, expected, rel_tol=1.0e-12, abs_tol=0.0):
            raise ValueError(
                "Explicit stability time increment is inconsistent with its "
                "spectral bound."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "unsafed_time_increment", dt)
        object.__setattr__(self, "spectral_radius_upper_bound", spectral)
        object.__setattr__(self, "method", method)

    @classmethod
    def from_time_increment(
        cls,
        name: str,
        unsafed_time_increment: float,
        *,
        method: str = "equivalent_spectral_upper_bound",
    ) -> "ExplicitStabilityContribution":
        """Create a contribution from an unsafed central-difference limit."""

        dt = float(unsafed_time_increment)
        if not isfinite(dt) or dt <= 0.0:
            raise ValueError(
                "Explicit stability time increment must be finite and positive."
            )
        return cls(
            name=name,
            unsafed_time_increment=dt,
            spectral_radius_upper_bound=4.0 / (dt * dt),
            method=method,
        )

    @classmethod
    def from_spectral_bound(
        cls,
        name: str,
        spectral_radius_upper_bound: float,
        *,
        method: str = "assembled_spectral_upper_bound",
    ) -> "ExplicitStabilityContribution":
        """Create a contribution from an upper bound on ``rho(M^-1 K)``."""

        spectral = float(spectral_radius_upper_bound)
        if not isfinite(spectral) or spectral <= 0.0:
            raise ValueError(
                "Explicit stability spectral bound must be finite and positive."
            )
        return cls(
            name=name,
            unsafed_time_increment=2.0 / sqrt(spectral),
            spectral_radius_upper_bound=spectral,
            method=method,
        )

    def summary(self) -> dict[str, object]:
        """Return a stable, machine-readable contribution summary."""

        return {
            "name": self.name,
            "unsafed_time_increment": self.unsafed_time_increment,
            "spectral_radius_upper_bound": self.spectral_radius_upper_bound,
            "method": self.method,
        }


@dataclass(frozen=True)
class ExplicitStabilityEstimate:
    """Conservative whole-system estimate from additive spectral bounds."""

    selected: float
    unsafed_time_increment: float
    spectral_radius_upper_bound: float
    safety_factor: float
    controller: str
    contributions: tuple[ExplicitStabilityContribution, ...]

    def __post_init__(self) -> None:
        selected = float(self.selected)
        unsafed = float(self.unsafed_time_increment)
        spectral = float(self.spectral_radius_upper_bound)
        factor = float(self.safety_factor)
        contributions = tuple(self.contributions)
        if not contributions:
            raise ValueError("Explicit stability estimate requires contributions.")
        names = tuple(item.name for item in contributions)
        if len(set(names)) != len(names):
            raise ValueError("Explicit stability contribution names must be unique.")
        if self.controller not in names:
            raise ValueError("Explicit stability controller must name a contribution.")
        if not isfinite(factor) or not (0.0 < factor <= 1.0):
            raise ValueError("safety_factor must satisfy 0 < value <= 1.")
        if not all(isfinite(value) and value > 0.0 for value in (selected, unsafed, spectral)):
            raise ValueError("Explicit stability estimate values must be positive.")
        expected_spectral = sum(
            item.spectral_radius_upper_bound for item in contributions
        )
        expected_unsafed = 2.0 / sqrt(expected_spectral)
        if not isclose(spectral, expected_spectral, rel_tol=1.0e-12, abs_tol=0.0):
            raise ValueError("Explicit stability spectral total is inconsistent.")
        if not isclose(unsafed, expected_unsafed, rel_tol=1.0e-12, abs_tol=0.0):
            raise ValueError("Explicit stability unsafed limit is inconsistent.")
        if not isclose(selected, factor * unsafed, rel_tol=1.0e-12, abs_tol=0.0):
            raise ValueError("Explicit stability selected limit is inconsistent.")
        object.__setattr__(self, "selected", selected)
        object.__setattr__(self, "unsafed_time_increment", unsafed)
        object.__setattr__(self, "spectral_radius_upper_bound", spectral)
        object.__setattr__(self, "safety_factor", factor)
        object.__setattr__(self, "contributions", contributions)

    def contribution(self, name: str) -> ExplicitStabilityContribution | None:
        """Return a named contribution when it exists."""

        selected = str(name)
        return next((item for item in self.contributions if item.name == selected), None)

    def summary(self) -> dict[str, object]:
        """Return a stable, machine-readable whole-system summary."""

        return {
            "selected": self.selected,
            "unsafed_time_increment": self.unsafed_time_increment,
            "spectral_radius_upper_bound": self.spectral_radius_upper_bound,
            "safety_factor": self.safety_factor,
            "controller": self.controller,
            "composition": "additive_spectral_upper_bounds",
            "contributions": [item.summary() for item in self.contributions],
            "maturity": "screening_estimate",
        }


def combine_explicit_stability(
    contributions,
    *,
    safety_factor: float = 0.8,
) -> ExplicitStabilityEstimate:
    """Combine conservative stiffness contributions before choosing ``dt``.

    For ``K = sum(K_i)``, subadditivity gives a conservative bound
    ``rho(M^-1 K) <= sum(rho_i)``.  Therefore separately derived time limits
    cannot generally be combined with ``min(dt_i)``; their equivalent spectral
    upper bounds must be added first.
    """

    items = tuple(contributions)
    if not items:
        raise ValueError("At least one explicit stability contribution is required.")
    if not all(isinstance(item, ExplicitStabilityContribution) for item in items):
        raise TypeError(
            "Explicit stability composition accepts ExplicitStabilityContribution "
            "objects."
        )
    factor = float(safety_factor)
    if not isfinite(factor) or not (0.0 < factor <= 1.0):
        raise ValueError("safety_factor must satisfy 0 < value <= 1.")
    names = tuple(item.name for item in items)
    if len(set(names)) != len(names):
        raise ValueError("Explicit stability contribution names must be unique.")
    total = sum(item.spectral_radius_upper_bound for item in items)
    unsafed = 2.0 / sqrt(total)
    controller = max(items, key=lambda item: item.spectral_radius_upper_bound).name
    return ExplicitStabilityEstimate(
        selected=factor * unsafed,
        unsafed_time_increment=unsafed,
        spectral_radius_upper_bound=total,
        safety_factor=factor,
        controller=controller,
        contributions=items,
    )


__all__ = [
    "ExplicitStabilityContribution",
    "ExplicitStabilityEstimate",
    "combine_explicit_stability",
]
