# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Private, shared metadata for finite-element result fields."""

from __future__ import annotations


def field_location(field) -> str:
    """Return the public result location implied by a finite-element space."""

    element = getattr(field.function_space, "element", None)
    basix_element = getattr(element, "basix_element", None)
    discontinuous = bool(
        getattr(element, "discontinuous", False)
        or getattr(basix_element, "discontinuous", False)
    )
    return "cells" if discontinuous else "nodes"


def projected_field_processing(field) -> dict[str, object]:
    """Describe a constitutive expression projected for result inspection."""

    family, degree = _element_identity(field)
    return {
        "source_position": "constitutive_expression",
        "method": "global_l2_projection",
        "representation": "cell_average" if degree == 0 else "discontinuous_field",
        "space_family": family,
        "space_degree": degree,
        "nodal_extrapolation": False,
        "interelement_smoothing": False,
        "material_boundary_averaging": False,
    }


def primary_subfield_processing(field) -> dict[str, object]:
    """Describe an exact field collapsed from a monolithic mixed unknown."""

    family, degree = _element_identity(field)
    discontinuous = field_location(field) == "cells"
    if discontinuous and degree == 0:
        representation = "cellwise_constant"
    elif discontinuous:
        representation = "discontinuous_cell_moments"
    else:
        representation = "finite_element_dofs"
    return {
        "method": "primary_mixed_finite_element_subfield",
        "representation": representation,
        "space_family": family,
        "space_degree": degree,
        "postprocessed": False,
        "visualization_requires_cell_recovery": bool(
            discontinuous and degree is not None and degree > 0
        ),
    }


def generated_field_processing(step, field) -> dict[str, object]:
    """Apply the generating provider's declared result-field semantics."""

    role = getattr(step, "result_field_role", "primary_subfield")
    if role == "primary_subfield":
        return primary_subfield_processing(field)
    if role == "constitutive_projection":
        return projected_field_processing(field)
    raise ValueError(
        f"Unknown generated result-field role {role!r}; expected "
        "'primary_subfield' or 'constitutive_projection'."
    )


def _element_identity(field) -> tuple[str | None, int | None]:
    element = getattr(field.function_space, "element", None)
    basix_element = getattr(element, "basix_element", None)
    degree = getattr(basix_element, "degree", None)
    family = getattr(basix_element, "family", None)
    return (
        None if family is None else str(getattr(family, "name", family)),
        None if degree is None else int(degree),
    )


__all__ = ()
