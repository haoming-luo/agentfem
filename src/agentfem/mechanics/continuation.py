# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded hyperelastic adapter to the physics-independent arc-length engine."""

import numpy as np
import ufl
from scipy import sparse
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc
from petsc4py import PETSc
from .. import solvers, loads
from ..constitutive import hyperelasticity
from ..backends._modal import _free_dof_layout
from .modal import _collect_modal_bcs_collectively, _require_homogeneous_modal_bcs


class HyperelasticArcLengthStep:
    """Serial sparse continuation of conservative proportional body/traction loads."""

    def __init__(
        self,
        model,
        target,
        *,
        options,
        increments,
        procedure,
        name="hyperelastic_arc_length",
    ):
        self.model, self.target = model, target
        self.options, self.increments = options, increments
        self.procedure, self.name = procedure, name
        u = target.value
        if u.function_space.mesh.comm.size != 1:
            raise NotImplementedError(
                "AFM-ARC-FEM-001: sparse reference adapter is serial; distributed continuation is not implemented."
            )
        if model.study.physics != "solid_mechanics" or model.study.assumption not in (
            None,
            "plane_strain",
            "plane_stress",
        ):
            raise NotImplementedError(
                "AFM-ARC-FEM-002: only plane and 3D displacement solids are supported."
            )
        if len(model.materials) != 1 or model.boundary_models or model.eigenstrains:
            raise ValueError(
                "AFM-ARC-FEM-002: one stateless hyperelastic material, no boundary models/eigenstrains."
            )
        material = model.materials[0].item
        if not hyperelasticity.is_finite_strain_hyperelastic(
            material
        ) or not hyperelasticity.supports_hyperelastic_study(
            material, dimension=model.study.dimension, assumption=model.study.assumption
        ):
            raise ValueError(
                "AFM-ARC-FEM-002: unsupported hyperelastic material/study."
            )
        if not model.loads or any(
            type(load) not in (loads.BodyLoad, loads.BoundaryLoad)
            for load in model.loads
        ):
            raise ValueError(
                "AFM-ARC-FEM-003: use proportional dead body forces/tractions only; no follower pressure or time amplitudes."
            )
        if (
            isinstance(increments, bool)
            or not isinstance(increments, int)
            or increments < 1
        ):
            raise ValueError("AFM-ARC-FEM-004: positive integer increments required.")
        if not isinstance(options, solvers.ArcLengthOptions):
            raise TypeError("arc_options must be solvers.ArcLengthOptions.")
        bcs = _collect_modal_bcs_collectively(u, constraints=tuple(model.constraints))
        _require_homogeneous_modal_bcs(u, bcs)
        free, _, _ = _free_dof_layout(u.function_space, bcs)
        self.free = free
        region = model.materials[0].region
        internal = hyperelasticity.internal_virtual_work(
            u,
            target.test,
            material,
            measure=ufl.dx if region is None else region.measure,
        )
        self.residual_form = fem.form(internal)
        self.tangent_form = fem.form(ufl.derivative(internal, u, target.trial))
        external = model.external_force(target)
        if u in ufl.algorithms.extract_coefficients(external.expression):
            raise ValueError(
                "AFM-ARC-FEM-003: reference load must not depend on the displacement field."
            )
        force = external.assemble_vector()
        try:
            force.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            self.reference = force.array_r[free].copy()
        finally:
            force.destroy()
        self.path = None

    def _evaluate(self, values):
        u = self.target.value
        u.x.array[:] = 0.0
        u.x.array[self.free] = values
        u.x.scatter_forward()
        residual = fem_petsc.assemble_vector(self.residual_form)
        matrix = None
        try:
            residual.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )
            matrix = fem_petsc.assemble_matrix(self.tangent_form)
            matrix.assemble()
            indptr, indices, data = matrix.getValuesCSR()
            tangent = sparse.csr_matrix(
                (data.copy(), indices.copy(), indptr.copy()), shape=matrix.getSize()
            )
            return residual.array_r[self.free].copy(), tangent[self.free, :][
                :, self.free
            ]
        finally:
            residual.destroy()
            if matrix is not None:
                matrix.destroy()

    def solve(self):
        u = self.target.value
        original = u.x.array.copy()
        try:
            if self.path is None:
                self.path = solvers.ArcLengthPath(
                    self._evaluate,
                    self.reference,
                    initial=original[self.free],
                    options=self.options,
                )
            self.path.run(self.increments)
        finally:
            if self.path is None:
                u.x.array[:] = original
            else:
                u.x.array[:] = 0.0
                u.x.array[self.free] = self.path.displacement
            u.x.scatter_forward()
        return u

    def displacement_at(self, increment=-1):
        """Copy the displacement at a zero-based accepted increment."""
        if self.path is None or not self.path.history:
            raise ValueError("No accepted arc-length increments are available.")
        field = fem.Function(self.target.value.function_space, name="Displacement")
        field.x.array[self.free] = self.path.history[increment]["displacement"]
        field.x.scatter_forward()
        return field

    def summary(self):
        return dict(
            kind="hyperelastic_spherical_arc_length",
            name=self.name,
            scope="serial_stateless_proportional_conservative_loading",
            accepted_increments=0 if self.path is None else len(self.path.history),
            procedure=self.procedure.summary(),
        )

    def solve_result(self, *, output=None, strict_output=False):
        from ..results import SimulationResult
        from ..results.lifecycle import complete_result

        u = self.solve()
        result = SimulationResult(name=self.name, metadata={"problem": self.summary()})
        result.add_field("Displacement", u)
        records = self.path.history
        result.add_history(
            "load_factor",
            np.arange(1, len(records) + 1),
            [r["load_factor"] for r in records],
            abscissa_name="accepted_increment",
        )
        result.add_quantity("equilibrium_residuals", [r["residual"] for r in records])
        result.add_quantity("arc_constraint_errors", [r["arc_error"] for r in records])
        result.add_quantity("load_factor", self.path.load_factor)
        result.metadata["continuation_attempts"] = self.path.attempts
        return complete_result(self, result, output=output, strict_output=strict_output)


def lower_arc_length(model, request):
    options = dict(request.options)
    for key in ("K", "F", "solver_options", "material", "constraints"):
        if options.get(key) is not None:
            raise ValueError(
                f"AFM-ARC-FEM-005: {key} override is not supported; use registered model assets and arc_options."
            )
    step = HyperelasticArcLengthStep(
        model,
        request.target,
        options=options.get("arc_options", solvers.ArcLengthOptions()),
        increments=options["increments"],
        procedure=request.procedure,
        name=options.get("name") or "hyperelastic_arc_length",
    )
    return model.add_step(step)
