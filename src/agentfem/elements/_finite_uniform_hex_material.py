# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private single-point material lowering; commits remain a Procedure decision."""

from dataclasses import dataclass

import numpy as np

from ..state import field_transaction


@dataclass(frozen=True)
class FiniteHexMaterialTrial:
    deformation_gradient: np.ndarray
    element_response: object
    material_response: object


def response_fields(response):
    """Scratch fields covered by a downstream assembly transaction."""
    return {
        "P": response.first_piola_stress.function,
        "S": response.cauchy_stress.function,
        "A": response.tangent.function,
        "W": response.strain_energy_density.function,
        **{
            f"component_{name}": field.function
            for name, field in response.stored_energy_density_components.items()
        },
    }


def evaluate_material_trial(
    operator,
    response,
    material,
    displacement,
    *,
    deformation_gradient_old,
    time,
    time_increment,
):
    """Use the existing quadrature driver without creating another state store.

    Serial, one point per Hex8, matching local cell order. The caller retains
    the accepted F and invokes response.commit()/rollback() after whole-system
    acceptance. A rejected trial restores response fields and discards material
    trial state, including failures after a successful provider update.
    """
    if response.domain.comm.size != 1:
        raise NotImplementedError(
            "Finite Hex8 material lowering has no MPI ownership evidence yet."
        )
    scratch = response_fields(response)
    try:
        if len(response.state.reference_field.points) != 1:
            raise ValueError(
                "Uniform Hex8 requires exactly one material point per cell."
            )
        count = response.domain.topology.index_map(3).size_local
        if (
            len(operator.coordinates) != count
            or len(response.state.reference_field.values) != count
        ):
            raise ValueError("Finite Hex8 cell and material point counts differ.")
        with field_transaction(**scratch):
            gradient = operator.deformation_gradient(displacement)
            updated = response.update(
                material,
                deformation_gradient_old=deformation_gradient_old,
                deformation_gradient_new=gradient,
                time=time,
                time_increment=time_increment,
                commit=False,
            )
            element = operator.response(
                displacement,
                first_piola=response.first_piola_stress.values,
                stored_energy_density=updated.strain_energy_density,
            )
        gradient.setflags(write=False)
        return FiniteHexMaterialTrial(gradient, element, updated)
    except Exception:
        response.rollback()
        raise
