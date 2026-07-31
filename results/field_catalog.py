"""Standard finite-element result variables and context-aware aliases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldVariable:
    """Stable public meaning of one result variable."""

    key: str
    name: str
    location: str
    tensor_type: str
    description: str
    aliases: tuple[str, ...] = ()


_VARIABLES = {
    "U": FieldVariable("U", "Displacement", "nodes", "vector", "Displacement"),
    "S": FieldVariable("S", "CauchyStress", "cells", "symmetric_tensor", "Cauchy stress"),
    "P": FieldVariable("P", "FirstPiolaStress", "cells", "tensor", "First Piola stress"),
    "F": FieldVariable("F", "DeformationGradient", "cells", "tensor", "Deformation gradient"),
    "LE": FieldVariable("LE", "LogarithmicStrain", "cells", "symmetric_tensor", "Spatial logarithmic strain"),
    "GREEN": FieldVariable("GREEN", "GreenLagrangeStrain", "cells", "symmetric_tensor", "Green--Lagrange strain"),
    "MISES": FieldVariable("MISES", "VonMisesStress", "cells", "scalar", "von Mises equivalent stress"),
    "J": FieldVariable("J", "DeformationJacobian", "cells", "scalar", "det(F)"),
    "SENER": FieldVariable("SENER", "StrainEnergyDensity", "cells", "scalar", "Strain-energy density"),
    "EVOL": FieldVariable("EVOL", "CurrentElementVolume", "cells", "scalar", "Current element volume"),
    "TEMP": FieldVariable("TEMP", "Temperature", "nodes", "scalar", "Temperature", ("NT",)),
    "RF": FieldVariable("RF", "ReactionForce", "nodes", "vector", "Reaction force"),
}


def field_variable(name: str, *, finite_strain: bool = False) -> FieldVariable:
    """Resolve a standard variable, including the context-dependent ``E`` alias."""

    key = str(name).strip().upper()
    if key == "E":
        key = "LE" if finite_strain else "E"
    if key == "E":
        return FieldVariable(
            "E",
            "InfinitesimalStrain",
            "cells",
            "symmetric_tensor",
            "Infinitesimal strain",
        )
    if key in _VARIABLES:
        return _VARIABLES[key]
    for variable in _VARIABLES.values():
        if key == variable.name.upper() or key in variable.aliases:
            return variable
    raise KeyError(
        f"Unknown result variable {name!r}; available={tuple(sorted(_VARIABLES))}."
    )


def resolve_field_variables(names, *, finite_strain: bool = False) -> tuple[FieldVariable, ...]:
    """Resolve aliases, preserve request order, and remove duplicates."""

    selected = []
    seen = set()
    for name in names:
        variable = field_variable(name, finite_strain=finite_strain)
        if variable.key not in seen:
            selected.append(variable)
            seen.add(variable.key)
    return tuple(selected)


def preselected_fields(*, physics: str, finite_strain: bool = False) -> tuple[str, ...]:
    """Return a small, conventional default field set for one physics context."""

    normalized = str(physics).lower().replace("-", "_")
    if normalized == "solid_mechanics":
        return ("U", "S", "LE" if finite_strain else "E")
    if normalized in {"heat", "heat_transfer", "thermal"}:
        return ("TEMP",)
    raise KeyError(f"No preselected field set is registered for physics={physics!r}.")
