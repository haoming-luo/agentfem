"""Reusable mechanical weak boundary models."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import ufl

from agentfem.ir.values import describe_value
from agentfem.kernel import constants
from agentfem.operators import OperatorForm


@dataclass(frozen=True)
class ElasticFoundation:
    """Distributed linear spring support on a solid boundary."""

    stiffness: object
    location: object
    mode: str = "isotropic"
    normal: object | None = None
    name: str = "elastic_foundation"

    def __post_init__(self) -> None:
        if isinstance(self.stiffness, Real) and self.stiffness < 0.0:
            raise ValueError("Foundation stiffness must be non-negative.")
        if self.location is None or not hasattr(self.location, "measure"):
            raise ValueError("ElasticFoundation requires a boundary region.")
        selected = str(self.mode).lower().replace("-", "_")
        if selected not in {"isotropic", "normal"}:
            raise ValueError("Foundation mode must be isotropic or normal.")
        object.__setattr__(self, "mode", selected)

    def operator(self, displacement):
        trial, test = displacement.trial, displacement.test
        if self.mode == "normal":
            normal = self.normal or ufl.FacetNormal(self.location.domain)
            expression = (
                self.stiffness
                * ufl.dot(trial, normal)
                * ufl.dot(test, normal)
                * self.location.measure
            )
        else:
            expression = (
                self.stiffness
                * ufl.inner(trial, test)
                * self.location.measure
            )
        return OperatorForm(
            name=f"K_{self.name}",
            expression=expression,
            kind="elastic_foundation_operator",
            role="matrix",
            family="mechanical_boundary",
            metadata={"mode": self.mode},
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "elastic_foundation",
            "location": getattr(self.location, "name", None),
            "mode": self.mode,
            "stiffness": describe_value(self.stiffness),
        }


def elastic_foundation(
    *, on=None, location=None, stiffness, mode: str = "isotropic", normal=None,
    name: str = "elastic_foundation",
) -> ElasticFoundation:
    selected = location if location is not None else on
    if on is not None and location is not None:
        raise ValueError("Pass either on=... or location=..., not both.")
    if selected is None or not hasattr(selected, "domain"):
        raise ValueError("elastic_foundation requires a boundary region.")
    return ElasticFoundation(
        stiffness=constants.constant(selected.domain, stiffness),
        location=selected,
        mode=mode,
        normal=normal,
        name=name,
    )


__all__ = ["ElasticFoundation", "elastic_foundation"]
