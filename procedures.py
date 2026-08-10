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
        if selected_method in {
            "",
            "implicit_creep",
            "backward_euler",
            "backward_euler_newton",
        }:
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


def resolve(
    *,
    analysis: str,
    requested: SolutionProcedure | str | None = None,
    preferred: str | None = None,
    stateful: bool = False,
) -> SolutionProcedure:
    """Resolve and validate the numerical procedure for one analysis request.

    ``Study`` remains the declaration of the physical problem.  This function
    only resolves the numerical route, giving an explicitly supplied
    ``SolutionProcedure`` precedence over a study preference.  It is used by
    the same provider request that later constructs the executable Step, so a
    GUI or agent cannot inspect one route while the solver silently uses
    another.
    """

    selected_analysis = _normalize(analysis)
    if isinstance(requested, SolutionProcedure):
        procedure = requested
    else:
        method = requested if requested is not None else preferred
        _validate_method_name(selected_analysis, method)
        dispatch_analysis = (
            "second_order_dynamics"
            if selected_analysis == "explicit_dynamics"
            else selected_analysis
        )
        if selected_analysis == "explicit_dynamics" and method is None:
            method = "central_difference"
        procedure = for_step(
            analysis=dispatch_analysis,
            method=None if method is None else str(method),
            stateful=stateful,
        )
    _validate_for_analysis(procedure, selected_analysis)
    return procedure


def _validate_method_name(analysis: str, method: str | None) -> None:
    if method is None:
        return
    selected = _normalize(method)
    allowed = {
        "linear_static": {
            "linear",
            "linear_static",
            "direct_or_iterative_linear",
        },
        "nonlinear_static": {"newton", "nonlinear_static"},
        "first_order_transient": {
            "implicit_euler",
            "backward_euler",
        },
        "nonlinear_transient": {
            "implicit_creep",
            "backward_euler",
            "backward_euler_newton",
        },
        "second_order_dynamics": {
            "implicit",
            "newmark",
            "generalized_alpha",
            "explicit",
            "central_difference",
        },
        "explicit_dynamics": {"explicit", "central_difference"},
    }.get(analysis)
    if allowed is not None and selected not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(
            f"Unknown numerical method {method!r} for analysis={analysis!r}; "
            f"expected one of: {choices}."
        )


def _validate_for_analysis(
    procedure: SolutionProcedure,
    analysis: str,
) -> None:
    expected_order = {
        "linear_static": "static",
        "nonlinear_static": "static",
        "first_order_transient": "first_order",
        "nonlinear_transient": "first_order",
        "second_order_dynamics": "second_order",
        "explicit_dynamics": "second_order",
    }.get(analysis)
    if expected_order is None:
        return
    if procedure.equation_order != expected_order:
        raise ValueError(
            f"Procedure {procedure.algorithm!r} has equation_order="
            f"{procedure.equation_order!r}, but analysis={analysis!r} requires "
            f"{expected_order!r}."
        )
    if analysis in {
        "linear_static",
        "nonlinear_static",
        "first_order_transient",
        "nonlinear_transient",
    } and not procedure.implicit:
        raise ValueError(
            f"Analysis {analysis!r} currently requires a Standard procedure; "
            "use a dynamics Study for an Explicit route."
        )
    if analysis == "linear_static" and procedure.nonlinear:
        raise ValueError("A linear-static Study cannot use a nonlinear procedure.")
    if analysis in {"nonlinear_static", "nonlinear_transient"} and not procedure.nonlinear:
        raise ValueError(
            f"Analysis {analysis!r} requires a nonlinear SolutionProcedure."
        )
    if analysis == "explicit_dynamics" and not procedure.explicit:
        raise ValueError(
            "An explicit-dynamics request requires an Explicit SolutionProcedure."
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
    "resolve",
]
