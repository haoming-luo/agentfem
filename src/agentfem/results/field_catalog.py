# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

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
    "V": FieldVariable("V", "Velocity", "nodes", "vector", "Velocity"),
    "A": FieldVariable("A", "Acceleration", "nodes", "vector", "Acceleration"),
    "S": FieldVariable("S", "CauchyStress", "cells", "symmetric_tensor", "Cauchy stress"),
    "S_MATERIAL": FieldVariable(
        "S_MATERIAL",
        "MaterialFrameStress",
        "cells",
        "symmetric_tensor",
        "Small-strain stress components in the declared material frame",
        aliases=("SMATERIAL",),
        derived_from=("S",),
    ),
    "P": FieldVariable("P", "FirstPiolaStress", "cells", "tensor", "First Piola stress"),
    "PRESSURE": FieldVariable(
        "PRESSURE",
        "Pressure",
        "cells",
        "scalar",
        "Independent mixed pressure; positive in compression",
    ),
    "F": FieldVariable("F", "DeformationGradient", "cells", "tensor", "Deformation gradient"),
    "FP": FieldVariable(
        "FP",
        "PlasticDeformationGradient",
        "cells",
        "tensor",
        "Plastic deformation gradient",
    ),
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
        "Provider-defined stored-energy channel; for finite-strain J2 this is "
        "ELENER + HARDENER and retains the provider's ELENER semantics",
    ),
    "ELENER": FieldVariable(
        "ELENER",
        "ElasticStoredEnergyDensity",
        "cells",
        "scalar",
        "Provider-defined elastic-energy channel; primal for displacement-only "
        "finite-strain J2 and condensed, not automatically primal, for mixed J2",
    ),
    "HARDENER": FieldVariable(
        "HARDENER",
        "HardeningStoredEnergyDensity",
        "cells",
        "scalar",
        "Stored isotropic-hardening free-energy density",
        derived_from=("PEEQ",),
    ),
    "PDENER": FieldVariable(
        "PDENER",
        "PlasticDissipationDensity",
        "cells",
        "scalar",
        "Cumulative irrecoverable plastic dissipation density",
    ),
    "KED": FieldVariable(
        "KED",
        "KineticEnergyDensity",
        "cells",
        "scalar",
        "Kinetic-energy density per reference volume",
        derived_from=("V",),
    ),
    "EVOL": FieldVariable("EVOL", "CurrentElementVolume", "cells", "scalar", "Current element volume"),
    "TEMP": FieldVariable("TEMP", "Temperature", "nodes", "scalar", "Temperature", ("NT",)),
    "RF": FieldVariable("RF", "ReactionForce", "nodes", "vector", "Reaction force"),
    "E_MATERIAL": FieldVariable(
        "E_MATERIAL",
        "MaterialFrameStrain",
        "cells",
        "symmetric_tensor",
        "Infinitesimal strain components in the declared material frame",
        aliases=("EMATERIAL",),
        derived_from=("E",),
    ),
    "FABRIC_GENERALIZED_STRAIN": FieldVariable(
        "FABRIC_GENERALIZED_STRAIN",
        "FabricGeneralizedStrain",
        "cells",
        "vector",
        "Warp strain, weft strain, and trellising angle in that order",
        aliases=("FABRIC_STRAIN",),
    ),
    "FABRIC_GENERALIZED_RESULTANT": FieldVariable(
        "FABRIC_GENERALIZED_RESULTANT",
        "FabricGeneralizedResultant",
        "cells",
        "vector",
        "Membrane resultants conjugate to warp strain, weft strain, and "
        "trellising angle",
        aliases=("FABRIC_N",),
        derived_from=("FABRIC_GENERALIZED_STRAIN",),
    ),
    "FABRIC_WARP_DIRECTION": FieldVariable(
        "FABRIC_WARP_DIRECTION",
        "FabricWarpDirection",
        "cells",
        "vector",
        "Current unit direction of the warp yarn family",
        aliases=("FABRIC_WARP",),
    ),
    "FABRIC_WEFT_DIRECTION": FieldVariable(
        "FABRIC_WEFT_DIRECTION",
        "FabricWeftDirection",
        "cells",
        "vector",
        "Current unit direction of the weft yarn family",
        aliases=("FABRIC_WEFT",),
    ),
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
