# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Result registration shared by accepted small/finite-strain material Steps.

The caller owns acceptance and the constitutive meaning. This module only
registers supplied quadrature fields and optional cell recovery; it never
updates a material, advances time or commits/restores state.
"""


def attach_material_fields(result, records, *, recovered=(), recover=False):
    from .recovery import recover_integration_point_field

    for record in records:
        data = dict(record)
        source = data.pop("source")
        name = data.pop("name")
        result.add_field(name, source.function, location="quadrature_points", sampling=source, **data)
        if recover:
            item = recover_integration_point_field(source, name=f"{name}_CELL",
                unit=data.get("unit"),
                description=f"Cell average of {name} preserving material discontinuities.")
            _attach_recovered(result, item, description=item.description)
    for item in recovered:
        _attach_recovered(result, item)
    return result


def _attach_recovered(result, item, *, description=""):
    result.add_field(item.name, item.field, unit=item.processing.get("unit", item.unit),
        location=item.location, description=description, processing=item.processing)
