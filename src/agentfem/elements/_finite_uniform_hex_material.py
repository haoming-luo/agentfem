# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private single-point material lowering; commits remain a Procedure decision."""

from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np

from ..state import field_transaction
from ..provenance import collective_call


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


@contextmanager
def material_trial(
    operator,
    response,
    material,
    displacement,
    *,
    deformation_gradient_old,
    time,
    time_increment,
    exchange_gradient=None,
):
    """Use the existing quadrature driver without creating another state store.

    One point per owned Hex8, matching local cell order. The caller retains
    the accepted F and invokes response.commit()/rollback() after whole-system
    acceptance. A rejected trial restores response fields and discards material
    trial state, including failures after a successful provider update.
    """
    comm = response.domain.comm
    scratch = response_fields(response)

    def prepare():
        if len(response.state.reference_field.points) != 1:
            raise ValueError(
                "Uniform Hex8 requires exactly one material point per cell."
            )
        count = response.domain.topology.index_map(3).size_local
        if (
            len(operator.coordinates) != count
            or len(response.state.reference_field.owned_values) != count
        ):
            raise ValueError("Finite Hex8 cell and material point counts differ.")
        # Own a fixed copy for this trial; provider callbacks cannot change
        # the nodal data between geometric admission and force assembly.
        selected_displacement = np.array(displacement, dtype=float, copy=True)
        selected_displacement.setflags(write=False)
        old_gradient = np.asarray(deformation_gradient_old, dtype=float)
        if old_gradient.shape != (count, 3, 3) or not np.isfinite(old_gradient).all():
            raise ValueError("Accepted gradient must match owned finite Hex8 cells.")
        return (
            selected_displacement,
            operator.deformation_gradient(selected_displacement),
            old_gradient,
        )

    try:
        selected_displacement, gradient, old_gradient = collective_call(
            prepare, comm=comm, label="Finite Hex8 kinematics"
        )
        if comm.size > 1 and exchange_gradient is None:
            raise NotImplementedError(
                "Distributed material lowering requires backend gradient exchange."
            )
        new_input = (
            gradient if exchange_gradient is None else exchange_gradient(gradient)
        )
        old_input = (
            old_gradient
            if exchange_gradient is None
            else exchange_gradient(old_gradient)
        )
        with field_transaction(**scratch):
            updated = response.update(
                material,
                deformation_gradient_old=old_input,
                deformation_gradient_new=new_input,
                time=time,
                time_increment=time_increment,
                commit=False,
            )

            def assemble_local():
                if not np.all(updated.strain_energy_density_defined):
                    raise ValueError(
                        "Finite Hex8 energy reporting requires material stored energy."
                    )
                return operator._response_from_checked_displacement(
                    selected_displacement,
                    first_piola=response.first_piola_stress.owned_values,
                    stored_energy_density=updated.strain_energy_density[
                        : len(operator.coordinates)
                    ],
                )

            element = collective_call(
                assemble_local, comm=comm, label="Finite Hex8 material force"
            )
            gradient.setflags(write=False)
            yield FiniteHexMaterialTrial(gradient, element, updated)
    except BaseException:
        response.rollback()
        raise


def evaluate_material_trial(*args, **kwargs):
    """Standalone evaluation; downstream consumers can extend material_trial."""
    with material_trial(*args, **kwargs) as trial:
        return trial
