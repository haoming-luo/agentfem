# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Lower model-registered material assignments into operator contributions.

This private module is the ownership boundary behind the model-first
``model.stiffness(...)``/``mass(...)``/thermal conveniences.  ``Model`` owns
the engineering registry and selects an explicitly requested material;
operators own regional measure resolution, coefficient validation, individual
form construction, and operator composition.

The functions deliberately consume assignment-shaped objects instead of a
``Model``.  This keeps the operator layer independent of the orchestration
facade and makes the lowering reusable without creating an upward dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import ufl

from .. import _axisymmetric
from ..constitutive import hyperelasticity
from ..materials.properties import constant_volumetric_heat_capacity
from . import core
from . import dispatch
from . import elasticity as elasticity_operators


class _MaterialAssignment(Protocol):
    """Structural view of one model-owned material assignment."""

    item: object
    region: object | None


def lower_stiffness(
    target,
    *,
    assignments: Sequence[_MaterialAssignment],
    selected: _MaterialAssignment | None = None,
    study=None,
    measure=None,
    law=None,
    temperature=None,
    name: str = "K",
):
    """Lower registered material assignments into one stiffness operator."""

    records = _records(assignments, selected)
    if not records:
        raise ValueError("model.stiffness requires at least one registered material.")
    if measure is not None and len(records) > 1:
        raise ValueError(
            "model.stiffness with multiple materials cannot use one explicit measure. "
            "Pass material=... or let each material use its registered region."
        )
    if len(records) == 1:
        return _stiffness_contribution(
            target,
            records[0],
            study=study,
            measure=measure,
            law=law,
            temperature=temperature,
            name=name,
        )

    missing = tuple(
        _describe(record.item) for record in records if record.region is None
    )
    if missing:
        raise ValueError(
            "Multiple-material stiffness requires every material to have a region. "
            f"Materials without regions: {list(missing)}."
        )
    parts = tuple(
        _stiffness_contribution(
            target,
            record,
            study=study,
            measure=record.region.measure,
            law=law,
            temperature=temperature,
            name=f"K_{getattr(record.region, 'name', index)}",
        )
        for index, record in enumerate(records)
    )
    return core.combine(*parts, name=name, kind="partitioned_stiffness")


def lower_mass(
    target,
    *,
    assignments: Sequence[_MaterialAssignment],
    selected: _MaterialAssignment | None = None,
    study=None,
    measure=None,
    name: str = "M",
):
    """Lower registered densities into one consistent mass operator."""

    records = _records(assignments, selected)
    if not records:
        raise ValueError("model.mass requires at least one registered material.")
    if measure is not None and len(records) > 1:
        raise ValueError("Pass material=... when using one explicit mass measure.")

    weight = _axisymmetric.integration_weight(target, study)
    parts = []
    for index, record in enumerate(records):
        if len(records) > 1 and record.region is None:
            raise ValueError(
                "Multiple-material mass requires a region for every material."
            )
        selected_measure = _measure(record, explicit=measure)
        parts.append(
            core.mass_operator(
                target,
                material_density(record.item) * weight,
                measure=selected_measure,
            ).renamed(f"{name}_{index}" if len(records) > 1 else name)
        )
    return (
        parts[0]
        if len(parts) == 1
        else core.combine(*parts, name=name, kind="partitioned_mass")
    )


def lower_conduction(
    temperature,
    *,
    assignments: Sequence[_MaterialAssignment],
    selected: _MaterialAssignment | None = None,
    measure=None,
    name: str = "K",
):
    """Lower registered conductivities into one heat-conduction operator."""

    records = _records(assignments, selected)
    if not records:
        raise ValueError("model.conduction requires at least one material.")
    if measure is not None and len(records) > 1:
        raise ValueError(
            "Pass material=... when using one explicit conduction measure."
        )

    parts = []
    for index, record in enumerate(records):
        if not hasattr(record.item, "conductivity"):
            raise ValueError(
                f"Material {_describe(record.item)!r} does not define conductivity."
            )
        if len(records) > 1 and record.region is None:
            raise ValueError(
                "Multiple-material conduction requires a region for every material."
            )
        parts.append(
            core.conduction_operator(
                temperature,
                record.item.conductivity,
                measure=_measure(record, explicit=measure),
            ).renamed(
                name
                if len(records) == 1
                else f"{name}_{getattr(record.region, 'name', index)}"
            )
        )
    return (
        parts[0]
        if len(parts) == 1
        else core.combine(*parts, name=name, kind="partitioned_conduction")
    )


def lower_heat_capacity(
    temperature,
    *,
    assignments: Sequence[_MaterialAssignment],
    selected: _MaterialAssignment | None = None,
    measure=None,
    name: str = "C",
):
    """Lower registered constant ``rho c_p`` values into one operator."""

    records = _records(assignments, selected)
    if not records:
        raise ValueError("model.heat_capacity requires at least one material.")
    if measure is not None and len(records) > 1:
        raise ValueError("Pass material=... when using one explicit capacity measure.")

    parts = []
    for index, record in enumerate(records):
        if len(records) > 1 and record.region is None:
            raise ValueError(
                "Multiple-material heat capacity requires a region for every material."
            )
        parts.append(
            core.capacity_operator(
                temperature,
                constant_volumetric_heat_capacity(record.item),
                measure=_measure(record, explicit=measure),
            ).renamed(
                name
                if len(records) == 1
                else f"{name}_{getattr(record.region, 'name', index)}"
            )
        )
    return (
        parts[0]
        if len(parts) == 1
        else core.combine(*parts, name=name, kind="partitioned_heat_capacity")
    )


def lower_damping(
    target,
    coefficient,
    *,
    study=None,
    measure=None,
    name: str = "C",
):
    """Lower one viscous damping contribution with study integration weight."""

    weight = _axisymmetric.integration_weight(target, study)
    return core.damping_operator(
        target,
        coefficient * weight,
        measure=ufl.dx if measure is None else measure,
    ).renamed(name)


def lower_thermal_expansion(
    target,
    source,
    *,
    assignments: Sequence[_MaterialAssignment],
    selected: _MaterialAssignment | None = None,
    study=None,
    measure=None,
    name: str = "F_thermal",
):
    """Lower an eigenstrain source over a complete material partition."""

    records = _records(assignments, selected)
    if not records:
        raise ValueError("model.thermal_expansion requires a registered material.")
    if measure is not None and len(records) > 1:
        raise ValueError(
            "Multi-material eigenstrain cannot use one explicit measure. "
            "Pass material=... or use registered material regions."
        )
    missing = tuple(
        _describe(record.item) for record in records if record.region is None
    )
    if len(records) > 1 and missing:
        raise ValueError(
            "Multi-material eigenstrain requires a region for every material. "
            f"Materials without regions: {list(missing)}."
        )
    parts = tuple(
        elasticity_operators.eigenstrain_vector(
            target,
            source,
            record.item,
            study=study,
            measure=_measure(record, explicit=measure),
            name=(
                name
                if len(records) == 1
                else f"{name}_{getattr(record.region, 'name', index)}"
            ),
        )
        for index, record in enumerate(records)
    )
    return (
        parts[0]
        if len(parts) == 1
        else core.combine(*parts, name=name, kind="partitioned_eigenstrain")
    )


def lower_lumped_mass(
    target,
    *,
    assignments: Sequence[_MaterialAssignment],
    selected: _MaterialAssignment | None = None,
    study=None,
    measure=None,
    method: str = "row_sum",
):
    """Lower registered densities into a row-sum lumped mass operator."""

    normalized_method = method.lower().replace("-", "_")
    if normalized_method not in {"row_sum", "diagonal", "lumped"}:
        raise ValueError("model.lumped_mass currently supports method='row_sum'.")

    from .. import assembly

    records = _records(assignments, selected)
    if not records:
        raise ValueError("model.lumped_mass requires at least one registered material.")
    if measure is not None and len(records) > 1:
        raise ValueError(
            "model.lumped_mass with multiple materials cannot use one explicit measure. "
            "Pass material=... or let each material use its registered region."
        )

    function_space = _space(target)
    weight = _axisymmetric.integration_weight(target, study)
    mass = None
    missing = []
    for record in records:
        if len(records) > 1 and record.region is None:
            missing.append(_describe(record.item))
            continue
        selected_measure = _measure(record, explicit=measure)
        part = assembly.assemble_lumped_mass(
            function_space,
            density=material_density(record.item) * weight,
            measure=selected_measure,
        )
        mass = part if mass is None else mass + part
    if missing:
        raise ValueError(
            "Multiple-material lumped mass requires every material to have a region. "
            f"Materials without regions: {missing}."
        )
    return core.LumpedMassOperator(
        mass=mass,
        inv_mass=assembly.inverse_diagonal(mass),
    )


def lower_load_vector(target, *, loads=(), load=None, study=None):
    """Lower registered or explicit loads into one external-force operator."""

    return core.load_vector(target, loads, load=load, study=study)


def lower_internal_force(
    displacement,
    test_function,
    *,
    assignments: Sequence[_MaterialAssignment],
    selected: _MaterialAssignment | None = None,
    study=None,
    measure=None,
    name: str = "F_internal",
):
    """Lower registered material assignments into an internal-force operator."""

    records = _records(assignments, selected)
    if not records:
        raise ValueError(
            "model.internal_force_vector requires at least one registered material."
        )
    if measure is not None and len(records) > 1:
        raise ValueError(
            "model.internal_force_vector with multiple materials cannot use one explicit "
            "measure. Pass material=... or let each material use its registered region."
        )
    if len(records) == 1:
        return _internal_force_contribution(
            displacement,
            test_function,
            records[0],
            study=study,
            measure=measure,
            name=name,
        )

    missing = tuple(
        _describe(record.item) for record in records if record.region is None
    )
    if missing:
        raise ValueError(
            "Multiple-material internal force requires every material to have a region. "
            f"Materials without regions: {list(missing)}."
        )
    parts = tuple(
        _internal_force_contribution(
            displacement,
            test_function,
            record,
            study=study,
            measure=record.region.measure,
            name=f"F_internal_{getattr(record.region, 'name', index)}",
        )
        for index, record in enumerate(records)
    )
    return core.combine(*parts, name=name, kind="partitioned_internal_force")


def lower_boundary_force(boundary_model, field, test_function):
    """Lower weak boundary physics into its force contribution."""

    return core.boundary_model_vector(boundary_model, field, test_function)


def lower_force_balance(
    *,
    internal=None,
    external=None,
    damping=None,
    absorbing=None,
    boundary=None,
    name: str = "R",
    convention: str = "internal_minus_external",
):
    """Compose force contributions into one explicitly signed residual."""

    positive = []
    negative = []
    if convention == "internal_minus_external":
        positive.extend(_as_tuple(internal))
        positive.extend(_as_tuple(damping))
        positive.extend(_as_tuple(absorbing))
        positive.extend(_as_tuple(boundary))
        negative.extend(_as_tuple(external))
    elif convention == "external_minus_internal":
        positive.extend(_as_tuple(external))
        negative.extend(_as_tuple(internal))
        negative.extend(_as_tuple(damping))
        negative.extend(_as_tuple(absorbing))
        negative.extend(_as_tuple(boundary))
    else:
        raise ValueError(
            "force_balance convention must be 'internal_minus_external' "
            "or 'external_minus_internal'."
        )

    terms = [*positive, *(core.scale(item, -1.0) for item in negative)]
    if not terms:
        raise ValueError("force_balance requires at least one force contribution.")
    return core.combine(*terms, name=name, kind="force_balance")


def material_density(material) -> float:
    """Resolve one positive constant density for mass-like lowering."""

    if not hasattr(material, "density"):
        raise ValueError(f"Material {_describe(material)!r} does not define density.")
    density = float(material.density)
    if density <= 0.0:
        raise ValueError(
            f"Material {_describe(material)!r} must have positive density."
        )
    return density


def _records(
    assignments: Sequence[_MaterialAssignment],
    selected: _MaterialAssignment | None,
) -> tuple[_MaterialAssignment, ...]:
    return (selected,) if selected is not None else tuple(assignments)


def _measure(record: _MaterialAssignment, *, explicit=None):
    if explicit is not None:
        return explicit
    if record.region is not None:
        return record.region.measure
    return ufl.dx


def _stiffness_contribution(
    target,
    record: _MaterialAssignment,
    *,
    study,
    measure=None,
    law=None,
    temperature=None,
    name: str,
):
    if hyperelasticity.is_finite_strain_hyperelastic(record.item) and law is None:
        raise TypeError(
            "A hyperelastic material has a deformation-dependent tangent, not "
            "one linear stiffness operator. Build its finite-strain residual "
            "and use operators.linearize(...) when a tangent is required."
        )
    selected_measure = measure
    if selected_measure is None and record.region is not None:
        selected_measure = record.region.measure
    kwargs = {"study": study}
    if law is not None:
        kwargs["law"] = law
    if temperature is not None:
        kwargs["temperature"] = getattr(temperature, "value", temperature)
    if selected_measure is not None:
        kwargs["measure"] = selected_measure
    operator = dispatch.stiffness(target, record.item, **kwargs)
    if record.region is not None:
        return operator.renamed(name, kind="regional_stiffness")
    return operator.renamed(name)


def _internal_force_contribution(
    displacement,
    test_function,
    record: _MaterialAssignment,
    *,
    study,
    measure=None,
    name: str,
):
    selected_measure = _measure(record, explicit=measure)
    if hyperelasticity.is_finite_strain_hyperelastic(record.item):
        from .. import fracture

        operator = fracture.finite_strain_internal_force(
            displacement,
            test_function,
            record.item,
            measure=selected_measure,
            name=name,
        )
        if record.region is not None:
            return operator.renamed(
                name,
                kind="regional_finite_strain_internal_force",
            )
        return operator
    operator = elasticity_operators.internal_force_vector(
        displacement,
        test_function,
        record.item,
        study=study,
        measure=selected_measure,
    )
    if record.region is not None:
        return operator.renamed(name, kind="regional_internal_force")
    return operator.renamed(name)


def _space(target):
    if hasattr(target, "space"):
        return target.space
    if hasattr(target, "function_space"):
        return target.function_space
    if hasattr(target, "value") and hasattr(target.value, "function_space"):
        return target.value.function_space
    return target


def _as_tuple(item) -> tuple:
    if item is None:
        return ()
    if isinstance(item, tuple):
        return item
    if isinstance(item, list):
        return tuple(item)
    return (item,)


def _describe(item):
    if hasattr(item, "summary"):
        return item.summary()
    if hasattr(item, "as_dict"):
        return item.as_dict()
    return getattr(item, "name", repr(item))


__all__ = ()
