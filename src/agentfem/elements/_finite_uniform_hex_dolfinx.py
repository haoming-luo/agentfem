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
        material_batch_points=None,
    ):
        from ..fields import unwrap

        self.displacement = unwrap(displacement)
        domain = self.displacement.function_space.mesh
        self.comm = domain.comm
        self.response = response
        selected_batch = chunk_size if material_batch_points is None else material_batch_points
        if (isinstance(selected_batch, (bool, np.bool_))
                or not isinstance(selected_batch, (int, np.integer))
                or selected_batch < 1):
            raise ValueError("material_batch_points must be a positive integer.")
        self.material_batch_points = int(selected_batch)
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
        # Preserve established mass summation and checkpoint identities. Dynamic
        # force scatter can reduce whole owned-cell arrays without chunk loops.
        mass = self._scatter(np.repeat(self.cells.lumped_mass[..., None], 3, axis=2),
                             chunked=True)
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

    def _scatter(self, cell_values, *, chunked=False, additional_cell_values=None):
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
                if additional_cell_values is not None and (
                    np.shape(additional_cell_values) != np.shape(cell_values)
                    or not np.isfinite(additional_cell_values).all()
                ):
                    raise ValueError("Additional Hex8 forces require matching finite cell vectors.")
                with vector.localForm() as local:
                    local.set(0)
                    values = local.array.reshape(-1, 3)
                    if not chunked:
                        force = cell_values.reshape(-1, 3)
                        nodes = self.cell_nodes.ravel()
                        for component in range(3):
                            weights = force[:, component]
                            if additional_cell_values is not None:
                                weights = weights + additional_cell_values.reshape(-1, 3)[:, component]
                            values[:, component] = np.bincount(
                                nodes,
                                weights=weights,
                                minlength=len(values),
                            )
                    else:
                        for region, (nodes, inverse) in zip(
                            self.cells._regions(), self._scatter_plans
                        ):
                            force = cell_values[region].reshape(-1, 3)
                            for component in range(3):
                                weights = force[:, component]
                                if additional_cell_values is not None:
                                    weights = weights + additional_cell_values[region].reshape(-1, 3)[:, component]
                                values[nodes, component] += np.bincount(
                                    inverse, weights=weights,
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

    def assemble_bulk_viscosity(self, velocity, **options):
        """Separate artificial force/power; caller owns vector and acceptance.

        Not automatically enabled in existing Steps. A Procedure must include
        damping in stability, power integration and accepted restart evidence.
        """
        from ..fields import unwrap
        from ._bulk_viscosity import bulk_viscosity_response

        velocity = unwrap(velocity)
        if self.comm.size != 1:
            raise NotImplementedError("Bulk viscosity composition is serial pending MPI acceptance.")
        if velocity.function_space is not self.displacement.function_space:
            raise ValueError("Velocity and displacement must share the function space.")
        response = bulk_viscosity_response(
            self.cells,
            self.displacement.x.array.reshape(-1, 3)[self.cell_nodes],
            velocity.x.array.reshape(-1, 3)[self.cell_nodes], **options)
        return self._scatter(response.internal_force), response

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

    @contextmanager
    def explicit_trial_evaluation(self, material, *, deformation_gradient_old, time, time_increment,
                                  velocity=None, bulk_viscosity=None):
        """Serial response-only trial; does not provide a time-step policy.

        Reuses cell geometry, scatter and quadrature State. The existing public
        finite Step remains on its tangent/storage path until explicit stability
        and the accepted work-only ledger are independently admitted.
        """
        from ..constitutive._explicit_material_driver import (
            ExplicitQuadratureResponse, _prepare_explicit_material_points,
        )
        from ._finite_uniform_hex_material import FiniteHexMaterialTrial
        from ..state import field_transaction

        if self.comm.size != 1 or not isinstance(self.response, ExplicitQuadratureResponse):
            raise NotImplementedError("Expected serial response-only quadrature.")
        if (velocity is None) != (bulk_viscosity is None):
            raise ValueError("Bulk viscosity requires both velocity and explicit coefficients.")
        if len(self.response.state.reference_field.points) != 1:
            raise ValueError("Uniform Hex8 requires one material point per cell.")
        self._tangent_available = False
        vector = None
        try:
            with field_transaction(_kinematics=self.displacement, **self.response.trial_fields()):
                nodal = self.displacement.x.array.copy()
                # Axis-specific gather creates detached cell storage without
                # the generic multi-axis advanced-indexing overhead.
                displacement = np.take(nodal.reshape(-1, 3), self.cell_nodes, axis=0)
                gradient = self.cells.deformation_gradient(displacement)
                prepared = _prepare_explicit_material_points(
                    material, self.response.state, deformation_gradient_old=deformation_gradient_old,
                    deformation_gradient_new=gradient, time=time, time_increment=time_increment,
                    max_batch_points=self.material_batch_points)
                self.response._assign_prepared(prepared)
                response = prepared.response
                if not np.array_equal(self.displacement.x.array, nodal):
                    raise RuntimeError("Material callback changed the admitted Hex8 configuration.")
                element = self.cells._response_from_checked_displacement(
                    displacement, first_piola=self.response.first_piola_stress.owned_values,
                    stored_energy_density=response.stored_energy_density)
                cell_force = element.internal_force
                viscosity = None
                if bulk_viscosity is not None:
                    from ..fields import unwrap
                    from ._bulk_viscosity import _bulk_viscosity_from_checked_gradient
                    from ._bulk_viscosity_policy import BulkViscosityPolicy
                    from .._small_matrix import determinants_3x3

                    velocity_field = unwrap(velocity)
                    if velocity_field.function_space is not self.displacement.function_space:
                        raise ValueError("Velocity and displacement must share the function space.")
                    options = (bulk_viscosity.options(determinants_3x3(gradient))
                               if isinstance(bulk_viscosity, BulkViscosityPolicy)
                               else bulk_viscosity)
                    viscosity = _bulk_viscosity_from_checked_gradient(
                        self.cells, gradient,
                        np.take(velocity_field.x.array.reshape(-1, 3), self.cell_nodes, axis=0), **options)
                    # Both contributions share the same cell-to-node map.
                    # Assemble once, retaining separate response/energy records.
                vector = self._scatter(
                    cell_force,
                    additional_cell_values=None if viscosity is None else viscosity.internal_force,
                )
                gradient.setflags(write=False)
                yield vector, FiniteHexMaterialTrial(gradient, element, response, viscosity)
                if not np.array_equal(self.displacement.x.array, nodal):
                    raise RuntimeError("Hex8 configuration changed before trial scope acceptance.")
        except BaseException:
            if vector is not None:
                vector.destroy()
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
