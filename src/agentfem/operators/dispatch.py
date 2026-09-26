# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Scientific operator dispatch above primitive operator records.

The dependency direction is intentional: primitive :mod:`operators.core`
objects do not know concrete elasticity, transport, or flow formulations.
This module may select those formulations for the concise public API.
"""

from __future__ import annotations

import ufl

from .core import OperatorForm
from .elasticity import elastic_stiffness


def stiffness(
    field,
    properties=None,
    *,
    law=None,
    study=None,
    temperature=None,
    measure=ufl.dx,
) -> OperatorForm:
    """Create the primary stiffness-like operator ``K`` for an unknown field."""

    if law is not None:
        if hasattr(law, "stiffness_operator"):
            return law.stiffness_operator(
                field,
                properties,
                study=study,
                measure=measure,
            )
        if callable(law):
            return law(field, properties, study=study, measure=measure)
        raise ValueError("law must be callable or provide stiffness_operator(...).")
    if getattr(field, "kind", None) == "displacement":
        if properties is None:
            raise ValueError(
                "operators.stiffness(displacement, ...) requires material properties."
            )
        if study is not None and hasattr(study, "require"):
            study.require(physics="solid_mechanics")
        return elastic_stiffness(
            field,
            properties,
            study=study,
            temperature=temperature,
            measure=measure,
        )
    raise ValueError(
        "operators.stiffness currently dispatches displacement fields to elastic "
        "stiffness. Use operators.conduction_operator(...) for scalar "
        "diffusion/conduction, or pass law=..."
    )


__all__ = ["stiffness"]
