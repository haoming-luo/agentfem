# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light predicates shared by Step dispatch and built-in providers."""

from __future__ import annotations


def selected_material(model, request):
    """Return an explicitly selected or unambiguous registered material."""

    selected = request.material
    if selected is None and len(getattr(model, "materials", ())) == 1:
        return model.materials[0].item
    return selected


def registered_materials(model, request) -> tuple[object, ...]:
    """Return executable materials addressed by one request."""

    selected = request.material
    if selected is not None:
        return (selected,)
    return tuple(record.item for record in getattr(model, "materials", ()))


def procedure_method(model, request) -> str | None:
    """Return the normalized requested or study-preferred procedure name."""

    if request.procedure is not None:
        return normalize(request.procedure.algorithm)
    if request.method is not None:
        return normalize(request.method)
    return getattr(getattr(model, "study", None), "preferred_procedure", None)


def same_procedure(left, right) -> bool:
    """Compare the scientific identity of two procedure declarations."""

    names = (
        "family",
        "equation_order",
        "control",
        "algorithm",
        "nonlinear",
        "requires_global_solve",
        "stateful",
    )
    return all(
        getattr(left, name, None) == getattr(right, name, None) for name in names
    )


def target_summary(target) -> dict[str, object] | None:
    """Return a JSON-safe identity for a requested primary field."""

    if target is None:
        return None
    shape = target_shape(target)
    return {
        "name": getattr(target, "name", type(target).__name__),
        "kind": getattr(target, "kind", None),
        "shape": shape,
    }


def target_shape(target) -> tuple[int, ...] | None:
    shape = getattr(target, "ufl_shape", None)
    if shape is None:
        value = getattr(target, "value", None)
        shape = getattr(value, "ufl_shape", None)
    if shape is None:
        return None
    return tuple(int(item) for item in shape)


def is_scalar_target(target) -> bool:
    kind = getattr(target, "kind", None)
    if kind in {"temperature", "scalar_unknown"}:
        return True
    if kind in {"displacement", "vector_unknown"}:
        return False
    shape = target_shape(target)
    return shape in {None, ()}


def is_vector_target(target) -> bool:
    kind = getattr(target, "kind", None)
    if kind in {"displacement", "vector_unknown"}:
        return True
    if kind in {"temperature", "scalar_unknown"}:
        return False
    shape = target_shape(target)
    return shape is None or len(shape) == 1


def has_complete_linear_system(request) -> bool:
    return request.options.get("K") is not None and request.options.get("F") is not None


def supports_elasticity(material) -> bool:
    return hasattr(material, "stiffness_voigt") or (
        hasattr(material, "young") and hasattr(material, "poisson")
    )


def supports_axisymmetric_elasticity(material) -> bool:
    """Require a constitutive record defining the full isotropic hoop response."""

    return (
        hasattr(material, "young")
        and hasattr(material, "poisson")
        and not hasattr(material, "stiffness_voigt")
    )


def supports_dynamics(material) -> bool:
    return (
        supports_elasticity(material) and getattr(material, "density", None) is not None
    )


def supports_conduction(material) -> bool:
    return hasattr(material, "conductivity")


def supports_heat_capacity(material) -> bool:
    return hasattr(material, "volumetric_heat_capacity") or (
        getattr(material, "density", None) is not None
        and hasattr(material, "specific_heat")
    )


def supports_stateful_constitutive(material) -> bool:
    """Return whether a material declares committed constitutive history."""

    if bool(getattr(material, "stateful_constitutive", False)):
        return True
    try:
        from .constitutive.small_strain_user_material import SmallStrainUserMaterial

        if isinstance(material, SmallStrainUserMaterial):
            return True
    except (ImportError, TypeError):
        pass
    regional = getattr(material, "materials", None)
    if regional is None:
        return False
    values = (
        tuple(regional.values()) if hasattr(regional, "values") else tuple(regional)
    )
    return bool(values) and all(
        bool(getattr(item, "stateful_constitutive", False)) for item in values
    )


def all_materials_support(model, request, predicate) -> bool:
    materials = registered_materials(model, request)
    return bool(materials) and all(predicate(item) for item in materials)


def normalize(value: str) -> str:
    normalized = str(value).lower().replace("-", "_").strip()
    aliases = {
        "static": "linear_static",
        "hyperelastic": "nonlinear_static",
        "neo_hookean": "nonlinear_static",
        "explicit": "explicit_dynamics",
    }
    return aliases.get(normalized, normalized)


COMMON_STEP_OPTIONS = (
    "K",
    "F",
    "constraints",
    "solver_options",
    "name",
    "material",
    "output",
)


__all__ = ()
