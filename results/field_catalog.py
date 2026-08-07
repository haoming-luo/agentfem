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
    derived_from: tuple[str, ...] = ()


_VARIABLES = {
    "U": FieldVariable("U", "Displacement", "nodes", "vector", "Displacement"),
    "S": FieldVariable("S", "CauchyStress", "cells", "symmetric_tensor", "Cauchy stress"),
    "P": FieldVariable("P", "FirstPiolaStress", "cells", "tensor", "First Piola stress"),
    "PRESSURE": FieldVariable(
        "PRESSURE",
        "Pressure",
        "cells",
        "scalar",
        "Independent mixed pressure; positive in compression",
    ),
    "F": FieldVariable("F", "DeformationGradient", "cells", "tensor", "Deformation gradient"),
    "LE": FieldVariable("LE", "LogarithmicStrain", "cells", "symmetric_tensor", "Spatial logarithmic strain"),
    "GREEN": FieldVariable("GREEN", "GreenLagrangeStrain", "cells", "symmetric_tensor", "Green--Lagrange strain"),
    "PE": FieldVariable(
        "PE", "PlasticStrain", "cells", "symmetric_tensor", "Plastic strain"
    ),
    "PEEQ": FieldVariable(
        "PEEQ",
        "EquivalentPlasticStrain",
        "cells",
        "scalar",
        "Equivalent plastic strain",
    ),
    "MISES": FieldVariable(
        "MISES",
        "VonMisesStress",
        "cells",
        "scalar",
        "von Mises equivalent stress",
        derived_from=("S",),
    ),
    "J": FieldVariable("J", "DeformationJacobian", "cells", "scalar", "det(F)"),
    "SENER": FieldVariable(
        "SENER",
        "StrainEnergyDensity",
        "cells",
        "scalar",
        "Strain-energy density",
        derived_from=("S", "E"),
    ),
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
    """Return the engineering-default field set for one physics context.

    Preselected fields are intentionally smaller than the catalog of available
    variables.  For solids, Mises stress is materialized for immediate plotting
    even though it is an invariant of ``S``; strain-energy density remains an
    opt-in diagnostic field.
    """

    normalized = str(physics).lower().replace("-", "_")
    if normalized == "solid_mechanics":
        return ("U", "S", "LE" if finite_strain else "E", "MISES")
    if normalized in {"heat", "heat_transfer", "thermal"}:
        return ("TEMP",)
    raise KeyError(f"No preselected field set is registered for physics={physics!r}.")
