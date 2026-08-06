"""Solution procedures: how an analysis is advanced and solved.

``Study`` describes the physical problem.  ``SolutionProcedure`` describes
the numerical route used to solve it.  Keeping the two separate prevents
terms such as ``dynamic`` or ``nonlinear`` from silently selecting one
particular algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass


_FAMILIES = {"standard", "explicit"}
_ORDERS = {"static", "first_order", "second_order"}
_CONTROL = {"single_solve", "load_increments", "time_increments"}


@dataclass(frozen=True)
class SolutionProcedure:
    """Inspectable, backend-neutral description of a solution algorithm."""

    name: str
    family: str
    equation_order: str
    control: str
    algorithm: str
    nonlinear: bool = False
    requires_global_solve: bool = True
    stateful: bool = False
    numerical_dissipation: str = "none"

    def __post_init__(self) -> None:
        family = _normalize(self.family)
        equation_order = _normalize(self.equation_order)
        control = _normalize(self.control)
        algorithm = _normalize(self.algorithm)
        if family not in _FAMILIES:
            raise ValueError(f"Unknown solution-procedure family {family!r}.")
        if equation_order not in _ORDERS:
            raise ValueError(f"Unknown equation order {equation_order!r}.")
        if control not in _CONTROL:
            raise ValueError(f"Unknown procedure control {control!r}.")
        if family == "explicit" and self.requires_global_solve:
            raise ValueError("An explicit procedure cannot require a global solve.")
        if equation_order == "static" and control == "time_increments":
            raise ValueError("A static procedure cannot use time increments.")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "equation_order", equation_order)
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "algorithm", algorithm)

    @property
    def implicit(self) -> bool:
        return self.family == "standard"

    @property
    def explicit(self) -> bool:
        return self.family == "explicit"

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "equation_order": self.equation_order,
            "control": self.control,
            "algorithm": self.algorithm,
            "nonlinear": self.nonlinear,
            "requires_global_solve": self.requires_global_solve,
            "stateful": self.stateful,
            "numerical_dissipation": self.numerical_dissipation,
        }


def linear_static() -> SolutionProcedure:
    return SolutionProcedure(
        name="linear static",
        family="standard",
        equation_order="static",
        control="single_solve",
        algorithm="direct_or_iterative_linear",
    )


def nonlinear_static(*, stateful: bool = False) -> SolutionProcedure:
    return SolutionProcedure(
        name="nonlinear static",
        family="standard",
        equation_order="static",
        control="load_increments",
        algorithm="newton",
        nonlinear=True,
        stateful=stateful,
    )


def implicit_euler(*, nonlinear: bool = False, stateful: bool = True) -> SolutionProcedure:
    return SolutionProcedure(
        name="implicit Euler",
        family="standard",
        equation_order="first_order",
        control="time_increments",
        algorithm="implicit_euler",
        nonlinear=nonlinear,
        stateful=stateful,
    )


def implicit_creep() -> SolutionProcedure:
    """Quasi-static backward-Euler creep with global Newton equilibrium."""

    return SolutionProcedure(
        name="implicit creep",
        family="standard",
        equation_order="first_order",
        control="time_increments",
        algorithm="backward_euler_newton",
        nonlinear=True,
        stateful=True,
    )


def newmark() -> SolutionProcedure:
    return SolutionProcedure(
        name="Newmark",
        family="standard",
        equation_order="second_order",
        control="time_increments",
        algorithm="newmark",
        stateful=True,
    )


def generalized_alpha() -> SolutionProcedure:
    return SolutionProcedure(
        name="generalized-alpha",
        family="standard",
        equation_order="second_order",
        control="time_increments",
        algorithm="generalized_alpha",
        stateful=True,
        numerical_dissipation="controllable high-frequency",
    )


def central_difference() -> SolutionProcedure:
    return SolutionProcedure(
        name="central difference",
        family="explicit",
        equation_order="second_order",
        control="time_increments",
        algorithm="central_difference",
        requires_global_solve=False,
        stateful=True,
    )


def for_step(*, analysis: str, method: str | None = None, stateful: bool = False):
    """Resolve a procedure without coupling ``Study`` to one solver route."""

    selected_analysis = _normalize(analysis)
    selected_method = _normalize(method or "")
    if selected_analysis == "linear_static":
        return linear_static()
    if selected_analysis == "nonlinear_static":
        return nonlinear_static(stateful=stateful)
    if selected_analysis == "first_order_transient":
        return implicit_euler(stateful=True)
    if selected_analysis == "nonlinear_transient":
        if selected_method in {"", "implicit_creep", "backward_euler"}:
            return implicit_creep()
    if selected_analysis == "second_order_dynamics":
        if selected_method in {"explicit", "central_difference"}:
            return central_difference()
        if selected_method in {"generalized_alpha", "generalized-alpha"}:
            return generalized_alpha()
        return newmark()
    raise NotImplementedError(
        f"No built-in solution procedure for analysis={analysis!r}, method={method!r}."
    )


def _normalize(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


__all__ = [
    "SolutionProcedure",
    "central_difference",
    "for_step",
    "generalized_alpha",
    "implicit_euler",
    "implicit_creep",
    "linear_static",
    "newmark",
    "nonlinear_static",
]
