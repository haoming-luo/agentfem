# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private total-Lagrangian assembly; DOLFINx owns distributed cells and DOFs."""

import numpy as np
from contextlib import contextmanager

from ._finite_uniform_hex import FiniteUniformHexBatch
from ._finite_uniform_hex_material import material_trial
from ._hex_dolfinx_layout import prepare_layout
from ..provenance import collective_call

try:
    from .._hex_native import scatter as _native_scatter
except ImportError:
    _native_scatter = None


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
        execution_kernel=None,
    ):
        from ..fields import unwrap

        self.displacement = unwrap(displacement)
        domain = self.displacement.function_space.mesh
        self.comm = domain.comm
        self.response = response
        from .explicit_execution import ExplicitHexBodyKernel, ExplicitHexGeometry
        if execution_kernel is not None and not isinstance(execution_kernel, ExplicitHexBodyKernel):
            raise TypeError("execution_kernel requires ExplicitHexBodyKernel")
        if execution_kernel is not None and domain.comm.size != 1:
            raise NotImplementedError("Explicit body kernels currently require serial execution")
        self.execution_kernel = execution_kernel
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
        self._execution_geometry = None if execution_kernel is None else ExplicitHexGeometry(
            self.cells.average_gradient, self.cells.hourglass_modes,
            self.cells.volume, self.cells.hourglass_coefficient,
            self.cells.coordinates, self.cells._reference_perturbation_radius,
            self.cells.lumped_mass)
        self._execution_cell_nodes = (None if execution_kernel is None else
            np.frombuffer(self.cell_nodes.tobytes(), dtype=np.int32).reshape(self.cell_nodes.shape))

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
                ):
                    raise ValueError(
                        "Finite Hex8 scatter requires finite cell vectors."
                    )
                if additional_cell_values is not None and (
                    np.shape(additional_cell_values) != np.shape(cell_values)
                ):
                    raise ValueError("Additional Hex8 forces require matching finite cell vectors.")
                with vector.localForm() as local:
                    local.set(0)
                    values = local.array.reshape(-1, 3)
                    native = (_native_scatter is not None and not chunked
                              and cell_values.dtype == np.dtype(float)
                              and cell_values.flags.c_contiguous and cell_values.flags.aligned
                              and self.cell_nodes.flags.c_contiguous and self.cell_nodes.flags.aligned
                              and self.cell_nodes.dtype in (np.dtype('int32'),np.dtype('int64'))
                              and (additional_cell_values is None or (
                                  additional_cell_values.dtype == np.dtype(float)
                                  and additional_cell_values.flags.c_contiguous
                                  and additional_cell_values.flags.aligned)))
                    if native:
                        try:
                            raw = _native_scatter(cell_values, self.cell_nodes,
                                                  additional_cell_values, len(values))
                        except ValueError:
                            if not np.isfinite(cell_values).all():
                                raise ValueError('Finite Hex8 scatter requires finite cell vectors.') from None
                            if additional_cell_values is not None and not np.isfinite(additional_cell_values).all():
                                raise ValueError('Additional Hex8 forces require matching finite cell vectors.') from None
                            raise
                        values[:] = np.frombuffer(raw,dtype=float).reshape(values.shape)
                        return
                    if not np.isfinite(cell_values).all():
                        raise ValueError('Finite Hex8 scatter requires finite cell vectors.')
                    if additional_cell_values is not None and not np.isfinite(additional_cell_values).all():
                        raise ValueError('Additional Hex8 forces require matching finite cell vectors.')
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
            ExplicitQuadratureResponse,
        )
        from ._finite_uniform_hex_material import FiniteHexMaterialTrial
        from ..state import field_transaction
        from contextlib import nullcontext

        ledger = getattr(self, 'performance', None)
        measure = (lambda name: nullcontext()) if ledger is None else ledger.measure

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
                with measure('body_cell_gather'):
                    displacement = self._gather_cell_vectors(nodal.reshape(-1, 3))
                with measure('body_kinematics_admission'):
                    if self.execution_kernel is not None and self.execution_kernel.prepare_kinematics is not None:
                        from .explicit_execution import prepare_kernel_kinematics
                        from ..constitutive.material_array_batch import _array
                        displacement = _array(displacement, displacement.shape, 'cell_displacement')
                        gradient = prepare_kernel_kinematics(self.execution_kernel,
                                                             self._execution_geometry, displacement)
                    else:
                        gradient = self.cells.deformation_gradient(displacement)
                with measure('body_material_and_force'):
                    prepared, element = self._prepare_explicit_response(
                        material, displacement, gradient,
                        deformation_gradient_old=deformation_gradient_old,
                        time=time, time_increment=time_increment)
                    self.response._assign_prepared(prepared)
                response = prepared.response
                if not np.array_equal(self.displacement.x.array, nodal):
                    raise RuntimeError("Material callback changed the admitted Hex8 configuration.")
                cell_force = element.internal_force
                viscosity = None
                if bulk_viscosity is not None:
                    from ..fields import unwrap
                    from ._bulk_viscosity import _bulk_viscosity_from_checked_gradient
                    from ._bulk_viscosity_policy import BulkViscosityPolicy
                    from .._small_matrix import determinants_3x3

                    with measure('body_bulk_viscosity'):
                        velocity_field = unwrap(velocity)
                        if velocity_field.function_space is not self.displacement.function_space:
                            raise ValueError("Velocity and displacement must share the function space.")
                        options = (bulk_viscosity.options(determinants_3x3(gradient))
                                   if isinstance(bulk_viscosity, BulkViscosityPolicy)
                                   else bulk_viscosity)
                        cell_velocity = self._gather_cell_vectors(velocity_field.x.array.reshape(-1, 3))
                        def reference_viscosity():
                            return _bulk_viscosity_from_checked_gradient(
                                self.cells, gradient, cell_velocity, **options)
                        if self.execution_kernel is not None and self.execution_kernel.evaluate_viscosity is not None:
                            from .explicit_execution import prepare_kernel_viscosity
                            viscosity = prepare_kernel_viscosity(self.execution_kernel, self._execution_geometry,
                                gradient, cell_velocity, options, reference_viscosity)
                        else:
                            viscosity = reference_viscosity()
                    # Both contributions share the same cell-to-node map.
                    # Assemble once, retaining separate response/energy records.
                with measure('body_force_scatter'):
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

    def _gather_cell_vectors(self, values):
        """Detached cell vectors; no retained mutable scratch across trials."""
        kernel = getattr(self, 'execution_kernel', None)
        if kernel is not None and kernel.gather_cell_vectors is not None:
            from ..constitutive.material_array_batch import _array
            # Freeze the smaller nodal input once; providers cannot retain a
            # writable solver buffer. The much larger cell result can share
            # immutable native storage instead of another full-mesh copy.
            nodal = _array(values, values.shape, 'nodal_vectors')
            nodes = self._execution_cell_nodes.view()
            return _array(kernel.gather_cell_vectors(nodal, nodes),
                          (len(self.cell_nodes), 8, 3), 'gathered_cell_vectors')
        return np.take(values, self.cell_nodes, axis=0)

    def _prepare_explicit_response(self, material, displacement, gradient, *,
                                   deformation_gradient_old, time, time_increment):
        """Detached body continuation; field publication remains with the owner.

        Execution lowerings may fuse this arithmetic, but cannot commit history
        or omit the material increment policy. No PETSc work belongs here.
        """
        from ..constitutive._explicit_material_driver import _prepare_explicit_material_points

        if self.execution_kernel is not None:
            from .explicit_execution import prepare_kernel_response
            return prepare_kernel_response(self.execution_kernel, self._execution_geometry,
                self, material, displacement, gradient,
                deformation_gradient_old=deformation_gradient_old,
                time=time, time_increment=time_increment)

        prepared = _prepare_explicit_material_points(
            material, self.response.state,
            deformation_gradient_old=deformation_gradient_old,
            deformation_gradient_new=gradient, time=time, time_increment=time_increment,
            max_batch_points=self.material_batch_points,
            accepted_history=self.response.committed_history())
        element = self.cells._response_from_checked_displacement(
            displacement, first_piola=prepared.first_piola_stress,
            stored_energy_density=prepared.response.stored_energy_density)
        return prepared, element

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
