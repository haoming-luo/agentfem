# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private total-Lagrangian assembly; DOLFINx owns distributed cells and DOFs."""

import numpy as np
from contextlib import contextmanager

from ._finite_uniform_hex import FiniteUniformHexBatch
from ._finite_uniform_hex_material import material_trial
from ._hex_dolfinx_layout import prepare_layout
from ..provenance import collective_call


class FiniteUniformHexResidual:
    """Map existing material transactions to a DOLFINx Q1 residual.

    The owning Procedure must supply accepted F, time and dt, and commit the
    material only after whole-system acceptance. This is not a public Step,
    stability policy or checkpoint implementation.
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
        self.comm = domain.comm
        self.response = response
        from dolfinx import fem
        from dataclasses import replace

        self._gradient_exchange = replace(
            response.first_piola_stress,
            function=fem.Function(response.first_piola_stress.function.function_space),
        )
        self._tangent_available = False
        collective_call(
            lambda: self._prepare_cells(
                density, hourglass_modulus, hourglass_scale, chunk_size
            ),
            comm=self.comm,
            label="Finite Hex8 local preparation",
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

    def _prepare_cells(self, density, hourglass_modulus, hourglass_scale, chunk_size):
        if self.response.domain is not self.displacement.function_space.mesh:
            raise ValueError("Material response and displacement must share one mesh.")
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

    def _scatter(self, cell_values):
        from petsc4py import PETSc

        vector = self.displacement.x.petsc_vec.duplicate()
        try:

            def local_scatter():
                if (
                    np.shape(cell_values) != (len(self.cell_nodes), 8, 3)
                    or not np.isfinite(cell_values).all()
                ):
                    raise ValueError(
                        "Finite Hex8 scatter requires finite cell vectors."
                    )
                with vector.localForm() as local:
                    local.set(0)
                    values = local.array.reshape(-1, 3)
                    for region, (nodes, inverse) in zip(
                        self.cells._regions(), self._scatter_plans
                    ):
                        force = cell_values[region].reshape(-1, 3)
                        for component in range(3):
                            values[nodes, component] += np.bincount(
                                inverse,
                                weights=force[:, component],
                                minlength=len(nodes),
                            )

            collective_call(local_scatter, comm=self.comm, label="Finite Hex8 scatter")
            vector.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )
            return vector
        except Exception:
            vector.destroy()
            raise

    def evaluate(self, material, *, deformation_gradient_old, time, time_increment):
        """Return (owned PETSc force vector, unaccepted material/element trial)."""
        with self.trial_evaluation(
            material,
            deformation_gradient_old=deformation_gradient_old,
            time=time,
            time_increment=time_increment,
        ) as result:
            return result

    def _exchange_gradient(self, owned):
        """Use the backend's quadrature ghost layout; do not own extra history."""
        exchange = self._gradient_exchange
        # Quadrature dof order need not equal cell order on a partition.
        indices = exchange._array_indices()[: owned.size]
        exchange.function.x.array[indices] = owned.ravel()
        exchange.function.x.scatter_forward()
        return exchange.values

    @contextmanager
    def trial_evaluation(
        self, material, *, deformation_gradient_old, time, time_increment
    ):
        """One rollback scope through material, assembly and caller validation.

        The caller owns the returned vector after successful scope exit; on
        failure this scope destroys it. Material acceptance remains explicit.
        """
        self._tangent_available = False
        vector = None
        try:
            self.displacement.x.scatter_forward()
            with material_trial(
                self.cells,
                self.response,
                material,
                self.displacement.x.array.reshape(-1, 3)[self.cell_nodes],
                deformation_gradient_old=deformation_gradient_old,
                time=time,
                time_increment=time_increment,
                exchange_gradient=self._exchange_gradient,
            ) as trial:
                vector = self._scatter(trial.element_response.internal_force)
                self._tangent_available = True
                yield vector, trial
        except BaseException:
            if vector is not None:
                vector.destroy()
            self._tangent_available = False
            self.response.rollback()
            raise

    def tangent_action(self, direction):
        """Apply the most recently evaluated trial tangent; never update history."""
        if not self._tangent_available:
            raise RuntimeError("Evaluate a valid material trial before tangent action.")

        def local_action():
            selected = np.asarray(direction, dtype=float)
            if selected.shape != self.displacement.x.array.shape:
                raise ValueError(
                    "Direction must match the displacement coefficient array."
                )
            return self.cells.tangent_action(
                selected.reshape(-1, 3)[self.cell_nodes],
                first_piola_tangent=self.response.tangent.owned_values,
            )

        action = collective_call(
            local_action, comm=self.comm, label="Finite Hex8 tangent action"
        )
        return self._scatter(action)
