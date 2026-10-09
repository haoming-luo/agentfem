# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private serial total-Lagrangian assembly; no time integration or acceptance."""

import numpy as np

from ._finite_uniform_hex import FiniteUniformHexBatch
from ._finite_uniform_hex_material import evaluate_material_trial, response_fields
from ._hex_dolfinx_layout import prepare_layout
from ..state import field_transaction


class FiniteUniformHexResidual:
    """Map existing material transactions to a DOLFINx Q1 residual.

    The owning Procedure must supply accepted F, time and dt, and commit the
    material only after whole-system acceptance. This is not a public Step,
    stability policy, checkpoint implementation or distributed assembly claim.
    """

    def __init__(
        self,
        displacement,
        response,
        *,
        density,
        hourglass_modulus,
        hourglass_scale,
        chunk_size=1024,
    ):
        from ..fields import unwrap

        self.displacement = unwrap(displacement)
        domain = self.displacement.function_space.mesh
        if domain.comm.size != 1:
            raise NotImplementedError("Finite Hex8 global assembly is serial only.")
        if response.domain is not domain:
            raise ValueError("Material response and displacement must share one mesh.")
        self.response = response
        self._tangent_available = False
        self.cell_nodes, coordinates, self._scatter_plans = prepare_layout(
            self.displacement, chunk_size
        )
        self.cells = FiniteUniformHexBatch(
            coordinates,
            density=density,
            hourglass_modulus=hourglass_modulus,
            hourglass_scale=hourglass_scale,
            chunk_size=chunk_size,
        )
        mass = self._scatter(np.repeat(self.cells.lumped_mass[..., None], 3, axis=2))
        try:
            self.mass_diagonal = mass.array.copy()
        finally:
            mass.destroy()
        from ..assembly import inverse_diagonal

        self.inv_mass = inverse_diagonal(self.mass_diagonal, comm=domain.comm)
        self.mass_diagonal.setflags(write=False)
        self.inv_mass.setflags(write=False)

    def _scatter(self, cell_values):
        vector = self.displacement.x.petsc_vec.duplicate()
        try:
            with vector.localForm() as local:
                local.set(0)
                values = local.array.reshape(-1, 3)
                for region, (nodes, inverse) in zip(
                    self.cells._regions(), self._scatter_plans
                ):
                    force = cell_values[region].reshape(-1, 3)
                    for component in range(3):
                        values[nodes, component] += np.bincount(
                            inverse, weights=force[:, component], minlength=len(nodes)
                        )
            return vector
        except Exception:
            vector.destroy()
            raise

    def evaluate(self, material, *, deformation_gradient_old, time, time_increment):
        """Return (owned PETSc force vector, unaccepted material/element trial)."""
        self._tangent_available = False
        try:
            with field_transaction(**response_fields(self.response)):
                trial = evaluate_material_trial(
                    self.cells,
                    self.response,
                    material,
                    self.displacement.x.array.reshape(-1, 3)[self.cell_nodes],
                    deformation_gradient_old=deformation_gradient_old,
                    time=time,
                    time_increment=time_increment,
                )
                vector = self._scatter(trial.element_response.internal_force)
            self._tangent_available = True
            return vector, trial
        except Exception:
            self.response.rollback()
            raise

    def tangent_action(self, direction):
        """Apply the most recently evaluated trial tangent; never update history."""
        if not self._tangent_available:
            raise RuntimeError("Evaluate a valid material trial before tangent action.")
        selected = np.asarray(direction, dtype=float)
        if selected.shape != self.displacement.x.array.shape:
            raise ValueError("Direction must match the displacement coefficient array.")
        action = self.cells.tangent_action(
            selected.reshape(-1, 3)[self.cell_nodes],
            first_piola_tangent=self.response.tangent.values,
        )
        return self._scatter(action)
