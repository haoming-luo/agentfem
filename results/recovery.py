"""Explicit recovery policies for constitutive integration-point fields."""

from __future__ import annotations

from dataclasses import dataclass

from .core import FieldResult


@dataclass(frozen=True)
class FieldRecovery:
    """A reviewable conversion from constitutive evidence to a field.

    Recovery is deliberately narrower than visualization smoothing.  The
    first supported policy produces one weighted average per cell and never
    mixes neighboring elements or material regions.
    """

    method: str = "cell_average"
    target_family: str = "DG"
    target_degree: int = 0
    material_boundary_policy: str = "preserve"

    def __post_init__(self) -> None:
        if self.method != "cell_average":
            raise NotImplementedError(
                "Only weighted cell-average integration-point recovery is "
                "currently supported."
            )
        if self.target_family.upper() != "DG" or int(self.target_degree) != 0:
            raise NotImplementedError(
                "Integration-point recovery currently targets DG0 cell fields."
            )
        if self.material_boundary_policy != "preserve":
            raise ValueError(
                "Scientific recovery must preserve material boundaries."
            )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "field_recovery",
            "method": "quadrature_weighted_cell_average",
            "source_position": "integration_points",
            "target_position": "cells",
            "target_space": "DG0",
            "material_boundary_policy": self.material_boundary_policy,
            "nodal_extrapolation": False,
            "interelement_smoothing": False,
            "material_boundary_averaging": False,
        }


def cell_average_recovery() -> FieldRecovery:
    """Return the standard scientific integration-point recovery policy."""

    return FieldRecovery()


def recover_integration_point_field(
    source,
    *,
    name: str | None = None,
    policy: FieldRecovery | None = None,
    unit: str | None = None,
    description: str = "",
) -> FieldResult:
    """Recover one ``QuadratureField`` without hiding its processing history."""

    selected_policy = policy or cell_average_recovery()
    if not hasattr(source, "cell_average") or not hasattr(source, "weights"):
        raise TypeError(
            "recover_integration_point_field requires an AgentFEM "
            "QuadratureField with points and weights."
        )
    source_name = str(getattr(source.function, "name", "IP_FIELD"))
    selected_name = name or source_name
    recovered = source.cell_average(name=selected_name)
    processing = selected_policy.summary()
    processing.update(
        {
            "source_field": source_name,
            "quadrature_point_count_per_cell": int(len(source.points)),
            "quadrature_weight_sum": float(source.weights.sum()),
        }
    )
    return FieldResult(
        selected_name,
        recovered,
        unit,
        "cells",
        None,
        description or f"Weighted cell average recovered from {source_name}.",
        processing,
    )


__all__ = [
    "FieldRecovery",
    "cell_average_recovery",
    "recover_integration_point_field",
]
