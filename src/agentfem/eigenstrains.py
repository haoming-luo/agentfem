# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Inspectable sources of stress-free strain.

Eigenstrain is a scientific input, not a load guessed from a field name.  A
source owns the kinematic offset; materials and regions remain model assets,
and the operator layer lowers their combination into equivalent virtual work.
"""

from __future__ import annotations

from dataclasses import dataclass

import ufl

from . import fields as field_api


@dataclass(frozen=True)
class ThermalEigenstrain:
    """Isotropic free strain driven by an explicit temperature field."""

    temperature: object
    name: str = "thermal_eigenstrain"
    kind: str = "thermal"

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("ThermalEigenstrain.name must not be empty.")

    def strain(self, properties, *, dimension: int, study=None):
        """Return the visible in-model eigenstrain tensor."""

        from .constitutive import elasticity

        properties = getattr(properties, "material", properties)
        _require_thermal_properties(properties)
        selected_dimension = 3 if _axisymmetric(study) else int(dimension)
        return elasticity.thermal_strain(
            self.temperature,
            properties,
            dimension=selected_dimension,
        )

    def equivalent_stress(self, properties, *, dimension: int, study=None):
        """Return positive ``C:E_eigen`` for the equivalent load operator."""

        from .constitutive import elasticity

        properties = getattr(properties, "material", properties)
        _require_thermal_properties(properties)
        return elasticity.thermal_expansion_stress(
            self.temperature,
            properties,
            study=study,
            dimension=dimension,
        )

    def constitutive_temperature(self):
        """Return the temperature used by temperature-dependent stiffness."""

        return self.temperature

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "eigenstrain_source",
            "source": self.kind,
            "temperature_field": getattr(
                field_api.unwrap(self.temperature),
                "name",
                type(self.temperature).__name__,
            ),
        }

    def to_ir(self) -> dict[str, object]:
        return self.summary()


@dataclass(frozen=True)
class PrescribedEigenstrain:
    """Explicit stress-free strain tensor supplied by an expert workflow.

    This is the low-level escape hatch for cure shrinkage, transformation
    strain and prestrain.  Its tensor dimension is checked during lowering;
    no physical meaning is inferred from the field name.
    """

    value: object
    name: str = "prescribed_eigenstrain"
    source: str = "prescribed"

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("PrescribedEigenstrain.name must not be empty.")
        if not str(self.source).strip():
            raise ValueError("PrescribedEigenstrain.source must not be empty.")

    def strain(self, properties, *, dimension: int, study=None):
        del properties, study
        selected = field_api.unwrap(self.value)
        expression = (
            selected if hasattr(selected, "ufl_shape") else ufl.as_tensor(selected)
        )
        if tuple(expression.ufl_shape) != (int(dimension), int(dimension)):
            raise ValueError(
                "Prescribed eigenstrain must be a square tensor matching the "
                f"model dimension; expected {(dimension, dimension)}, got "
                f"{tuple(expression.ufl_shape)}."
            )
        return expression

    def equivalent_stress(self, properties, *, dimension: int, study=None):
        from .constitutive import elasticity

        return elasticity.stress_from_strain(
            self.strain(properties, dimension=dimension, study=study),
            properties,
            study=study,
        )

    def constitutive_temperature(self):
        return None

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "eigenstrain_source",
            "source": self.source,
            "value_shape": tuple(
                getattr(field_api.unwrap(self.value), "ufl_shape", ())
            ),
        }

    def to_ir(self) -> dict[str, object]:
        return self.summary()


def thermal(temperature, *, name: str = "thermal_eigenstrain") -> ThermalEigenstrain:
    """Create an explicit thermal eigenstrain source."""

    return ThermalEigenstrain(temperature=temperature, name=name)


def prescribed(
    value,
    *,
    name: str = "prescribed_eigenstrain",
    source: str = "prescribed",
) -> PrescribedEigenstrain:
    """Create a checked expert-defined eigenstrain source."""

    return PrescribedEigenstrain(value=value, name=name, source=source)


def _axisymmetric(study) -> bool:
    return getattr(study, "assumption", None) == "axisymmetric"


def _require_thermal_properties(properties) -> None:
    missing = tuple(
        name
        for name in ("thermal_expansion", "reference_temperature")
        if not hasattr(properties, name)
    )
    if missing:
        raise TypeError(
            "Thermal eigenstrain requires material thermal_expansion and "
            f"reference_temperature; missing={missing}."
        )


__all__ = (
    "PrescribedEigenstrain",
    "ThermalEigenstrain",
    "prescribed",
    "thermal",
)
