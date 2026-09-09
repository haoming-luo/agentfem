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
    operator = core.stiffness(target, record.item, **kwargs)
    if record.region is not None:
        return operator.renamed(name, kind="regional_stiffness")
    return operator.renamed(name)


def _describe(item):
    if hasattr(item, "summary"):
        return item.summary()
    if hasattr(item, "as_dict"):
        return item.as_dict()
    return getattr(item, "name", repr(item))


__all__ = ()
