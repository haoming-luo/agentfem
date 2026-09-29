# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Reference external material registered without modifying AgentFEM core."""

from agentfem.extensions import Extension, ExtensionSpec


MATERIAL_ID = "agentfem_reference_alloy"


def _register(context) -> None:
    context.add_material(
        MATERIAL_ID,
        {
            "id": MATERIAL_ID,
            "display_name": "AgentFEM extension reference alloy",
            "family": "extension_acceptance_fixture",
            "unit_system": "SI",
            "source": "external_extension_acceptance_not_design_data",
            "models": {
                "isotropic_linear_elastic": {
                    "young": 210.0e9,
                    "poisson": 0.30,
                    "density": 7800.0,
                }
            },
        },
    )


extension = Extension(
    spec=ExtensionSpec(
        name="agentfem-reference-material",
        version="0.1.0",
        description="Installed-wheel material provider acceptance fixture.",
        capabilities=("material.isotropic_linear_elastic",),
    ),
    register=_register,
)


__all__ = ["MATERIAL_ID", "extension"]
