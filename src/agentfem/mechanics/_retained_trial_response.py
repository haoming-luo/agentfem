# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Ephemeral evidence for retaining an unchanged material trial response.

This is not a response cache or another State owner. Hashes guard the existing
buffers, including writable public aliases. No arrays or constitutive results
are stored here. The owning transaction controls the scope and invalidation.
"""

from hashlib import sha256

import numpy as np


def retained_trial_digest(transaction, old_gradient, new_gradient, *, start_factor, target_factor):
    """Identify the bounded J2 inputs AND the response that remains in memory.

    Evaluated gradients cover live expression coefficients, not merely the
    displacement address. Raw field storage avoids detached ``values`` copies.
    The digest is local, never a checkpoint or a scientific-input identity.
    """
    response = transaction.response
    state = response.state
    digest = sha256()

    def describe(value):
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")

    def array(value):
        selected = np.ascontiguousarray(value)
        describe((selected.dtype.str, selected.shape))
        digest.update(memoryview(selected).cast("B"))

    def field(value):
        describe((id(value), id(value.function), id(value.function.function_space)))
        array(value.function.x.array)

    for callback in (transaction.refresh_trial, response.update,
                     transaction.material.update, transaction.material.update_batch):
        describe((id(getattr(callback, "__func__", callback)),
                  id(getattr(callback, "__self__", None))))
    describe((
        id(transaction.material), repr(transaction.material),
        repr(transaction.material.state_schema), repr(transaction.material.tangent_convention),
        id(response), id(state), repr(state.state_schema), state.degree, state.scheme,
        transaction.accepted_factor, float(start_factor), float(target_factor),
        transaction._response_linearization_generation,
        transaction._response_linearization_origin,
        transaction.last_plastic_points, transaction.last_maximum_plastic_increment,
        transaction.last_maximum_pressure_projection_defect,
    ))
    array(old_gradient)
    array(new_gradient)
    array(transaction.solution.x.array)
    array(transaction.accepted_solution.x.array)
    array(response.domain.geometry.x)
    for fields in (state.committed, state.trial, response.stored_energy_density_components):
        describe(tuple(sorted(fields)))
        for name in sorted(fields):
            field(fields[name])
    for selected in (
        response.first_piola_stress, response.cauchy_stress, response.tangent,
        response.strain_energy_density, transaction.deformation_gradient,
        transaction.equivalent_stress,
    ):
        field(selected)
    return digest.digest()
