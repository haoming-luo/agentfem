# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Reusable mechanical weak boundary models."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import ufl
from mpi4py import MPI

from agentfem.ir.values import describe_value
from agentfem.kernel import constants
from agentfem.operators import OperatorForm
from .rigid import PrescribedRigidMotion, RigidPlaneSurface


@dataclass(frozen=True)
class ElasticFoundation:
    """Distributed linear spring support on a solid boundary."""

    stiffness: object
    location: object
    mode: str = "isotropic"
    normal: object | None = None
    name: str = "elastic_foundation"

    def __post_init__(self) -> None:
        if isinstance(self.stiffness, Real) and self.stiffness < 0.0:
            raise ValueError("Foundation stiffness must be non-negative.")
        if self.location is None or not hasattr(self.location, "measure"):
            raise ValueError("ElasticFoundation requires a boundary region.")
        selected = str(self.mode).lower().replace("-", "_")
        if selected not in {"isotropic", "normal", "matrix"}:
            raise ValueError(
                "Foundation mode must be isotropic, normal, or matrix."
            )
        symbolic_shape = getattr(self.stiffness, "ufl_shape", None)
        shape = tuple(
            symbolic_shape
            if symbolic_shape is not None
            else np.asarray(self.stiffness).shape
        )
        if selected == "matrix":
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError(
                    "Matrix foundation stiffness must be a square rank-two value."
                )
            self._validate_conservative_matrix()
        elif shape:
            raise ValueError(
                f"{selected.capitalize()} foundation stiffness must be scalar."
            )
        object.__setattr__(self, "mode", selected)

    def _validate_conservative_matrix(self) -> None:
        """Reject numerical matrices that cannot own conservative energy."""

        raw = getattr(self.stiffness, "value", self.stiffness)
        try:
            matrix = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Matrix foundation stiffness currently requires an explicit "
                "numerical matrix so conservative energy can be verified."
            ) from exc
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("Matrix foundation stiffness must be finite.")
        scale = max(float(np.linalg.norm(matrix, ord=2)), 1.0)
        tolerance = 256.0 * np.finfo(float).eps * scale
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
            raise ValueError(
                "Conservative matrix foundation stiffness must be symmetric."
            )
        if float(np.min(np.linalg.eigvalsh(matrix))) < -tolerance:
            raise ValueError(
                "Conservative matrix foundation stiffness must be positive "
                "semidefinite."
            )

    def operator(self, displacement):
        trial, test = displacement.trial, displacement.test
        expression = self._expression(trial, test)
        return OperatorForm(
            name=f"K_{self.name}",
            expression=expression,
            kind="elastic_foundation_operator",
            role="matrix",
            family="mechanical_boundary",
            metadata={"mode": self.mode},
        )

    def _expression(self, trial, test):
        """Return the reviewed weak operator for arbitrary trial/test fields."""

        if self.mode == "normal":
            normal = self.normal or ufl.FacetNormal(self.location.domain)
            expression = (
                self.stiffness
                * ufl.dot(trial, normal)
                * ufl.dot(test, normal)
                * self.location.measure
            )
        elif self.mode == "matrix":
            shape = tuple(getattr(trial, "ufl_shape", ()))
            stiffness_shape = tuple(getattr(self.stiffness, "ufl_shape", ()))
            if len(shape) != 1 or stiffness_shape != (shape[0], shape[0]):
                raise ValueError(
                    "Matrix foundation stiffness must match the displacement "
                    "dimension."
                )
            expression = (
                ufl.inner(ufl.dot(self.stiffness, trial), test)
                * self.location.measure
            )
        else:
            expression = (
                self.stiffness
                * ufl.inner(trial, test)
                * self.location.measure
            )
        return expression

    def capabilities(self):
        """Declare the foundation's reaction and stored-energy semantics."""

        from agentfem.constraints import ConstraintCapabilities

        return ConstraintCapabilities(
            kind="weak_constraint",
            enforcement="weak_elastic_boundary_operator",
            analyses=("linear_static",),
            procedures=("direct_linear_solve",),
            strict=True,
            supports_parallel=True,
            reaction_evidence="provider_dual_required",
            work_evidence="internal_energy_operator",
        )

    def dual_evidence(self, problem):
        """Return the foundation reaction without counting its energy twice."""

        solution_getter = getattr(problem, "_solution", None)
        if not callable(solution_getter) or getattr(problem, "last_solve_info", None) is None:
            raise RuntimeError(
                "Elastic-foundation dual evidence requires a converged linear "
                "system problem."
            )

        from dolfinx import fem
        import dolfinx.fem.petsc as fem_petsc
        from petsc4py import PETSc

        from agentfem.constraints import constraint_dual

        solution = solution_getter()
        trial = ufl.TrialFunction(solution.function_space)
        test = ufl.TestFunction(solution.function_space)
        expression = self._expression(trial, test)
        internal = fem_petsc.assemble_vector(
            fem.form(ufl.action(expression, solution))
        )
        try:
            internal.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE,
            )
            distribution = fem.Function(
                solution.function_space,
                name=f"{self.name}_reaction",
            )
            owned = int(
                solution.function_space.dofmap.index_map.size_local
                * solution.function_space.dofmap.index_map_bs
            )
            distribution.x.array[:owned] = -np.asarray(internal.array_r[:owned])
            distribution.x.scatter_forward()
        finally:
            internal.destroy()

        values = np.asarray(distribution.x.array[:owned], dtype=float)
        shape = tuple(getattr(solution, "ufl_shape", ()))
        if len(shape) != 1:
            raise NotImplementedError(
                "Elastic-foundation reaction evidence requires a vector solid field."
            )
        components = int(shape[0])
        if values.size % components:
            raise RuntimeError(
                "Elastic-foundation reaction storage is incompatible with the field."
            )
        local_resultant = np.sum(values.reshape((-1, components)), axis=0)
        resultant = np.empty(components, dtype=float)
        comm = solution.function_space.mesh.comm
        comm.Allreduce(local_resultant, resultant, op=MPI.SUM)
        internal_action = ufl.action(expression, solution)
        stored_local = fem.assemble_scalar(
            fem.form(0.5 * ufl.action(internal_action, solution))
        )
        stored_energy = float(comm.allreduce(float(stored_local), op=MPI.SUM))
        diagnostics = {
            "status": "complete",
            "reaction_distribution": distribution.name,
            "distribution_location": "nodes",
            "resultant_norm": float(np.linalg.norm(resultant)),
            "stored_energy": stored_energy,
            "stored_energy_accounting": "included_in_system_strain_energy",
            "comm_size": int(comm.size),
        }
        return constraint_dual(
            self,
            force=resultant,
            coordinate=None,
            resultant=resultant,
            distribution=distribution,
            diagnostics=diagnostics,
            role="weak_constraint",
            source="elastic_foundation_weak_operator_action",
            complete=True,
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "elastic_foundation",
            "location": getattr(self.location, "name", None),
            "mode": self.mode,
            "stiffness": describe_value(self.stiffness),
        }


@dataclass(frozen=True)
class RigidObstaclePenaltyContact:
    """Frictionless one-sided contact with one analytical rigid plane.

    The compatibility route uses ``normal`` and ``initial_gap`` for a fixed
    plane. The explicit route uses a :class:`RigidPlaneSurface`, optionally
    following :class:`PrescribedRigidMotion`. In both cases the normal points
    from the obstacle towards the admissible half-space. The provider owns the
    conservative penalty potential

    ``0.5 * penalty * min(g, 0)^2``.

    This deliberately bounded model has no surface search, finite sliding,
    friction, free rigid-body dynamics, or two-body coupling. Those require a
    dedicated contact backend rather than hidden extensions of this UFL
    boundary operator.
    """

    penalty: object
    location: object
    normal: object
    initial_gap: object = 0.0
    surface: RigidPlaneSurface | None = None
    motion: PrescribedRigidMotion | None = None
    surface_point: object | None = None
    motion_factor: object | None = None
    name: str = "rigid_obstacle_contact"

    def __post_init__(self) -> None:
        if self.location is None or not hasattr(self.location, "measure"):
            raise ValueError("Rigid obstacle contact requires a boundary region.")
        if isinstance(self.penalty, Real) and self.penalty <= 0.0:
            raise ValueError("Contact penalty must be positive.")
        normal_shape = tuple(getattr(self.normal, "ufl_shape", ()))
        dimension = int(self.location.domain.geometry.dim)
        if normal_shape != (dimension,):
            raise ValueError(
                "Contact normal must match the mesh geometric dimension; "
                f"expected {(dimension,)}, got {normal_shape}."
            )
        if tuple(getattr(self.penalty, "ufl_shape", ())) != ():
            raise ValueError("Contact penalty must be scalar.")
        if tuple(getattr(self.initial_gap, "ufl_shape", ())) != ():
            raise ValueError("Contact initial_gap must be scalar.")
        if isinstance(self.initial_gap, Real) and not np.isfinite(self.initial_gap):
            raise ValueError("Contact initial_gap must be finite.")
        if self.surface is not None:
            if self.surface.dimension != dimension:
                raise ValueError("Rigid contact surface must match the mesh dimension.")
            if self.surface_point is None or self.motion_factor is None:
                raise ValueError(
                    "Moving rigid-plane contact requires point and factor state."
                )
            if self.motion is not None and self.motion.dimension != dimension:
                raise ValueError("Rigid contact motion must match the mesh dimension.")

    @property
    def has_prescribed_motion(self) -> bool:
        if self.motion is None:
            return False
        coordinate = self.motion.generalized_coordinate(1.0)
        return bool(np.any(np.abs(coordinate) > 0.0))

    def update_motion(self, factor: float) -> None:
        """Update analytical rigid-plane kinematics for one trial station."""

        if self.surface is None:
            return
        state = self.surface.transformed(self.motion, factor)
        self.surface_point.value = np.asarray(state["point"], dtype=float)
        self.normal.value = np.asarray(state["normal"], dtype=float)
        self.motion_factor.value = float(factor)

    def _rigid_state(self) -> dict[str, np.ndarray | float] | None:
        if self.surface is None:
            return None
        factor = float(np.asarray(self.motion_factor.value).reshape(-1)[0])
        return self.surface.transformed(self.motion, factor)

    def gap(self, displacement):
        """Return the signed gap; negative values denote penetration."""

        if self.surface is not None:
            position = ufl.SpatialCoordinate(self.location.domain)
            return ufl.dot(position + displacement - self.surface_point, self.normal)
        return self.initial_gap + ufl.dot(displacement, self.normal)

    def penetration(self, displacement):
        """Return the non-negative normal penetration."""

        return ufl.max_value(-self.gap(displacement), 0.0)

    def pressure(self, displacement):
        """Return the non-negative penalty contact pressure."""

        return self.penalty * self.penetration(displacement)

    def energy_density(self, displacement):
        penetration = self.penetration(displacement)
        return 0.5 * self.penalty * penetration * penetration

    def energy_form(self, displacement):
        return self.energy_density(displacement) * self.location.measure

    def form(self, displacement, test):
        """Return the exact first variation of the contact potential."""

        return (
            -self.pressure(displacement)
            * ufl.dot(test, self.normal)
            * self.location.measure
        )

    def tangent(self, displacement, trial, test):
        """Return the consistent semismooth tangent of :meth:`form`."""

        return ufl.derivative(self.form(displacement, test), displacement, trial)

    def capabilities(self):
        from agentfem.constraints import ConstraintCapabilities

        return ConstraintCapabilities(
            kind="contact_constraint",
            enforcement="one_sided_rigid_plane_penalty",
            analyses=("nonlinear_static",),
            procedures=("incremental_newton",),
            strict=False,
            supports_parallel=True,
            reaction_evidence="provider_dual_required",
            work_evidence=(
                "provider_dual_path_required"
                if self.has_prescribed_motion
                else "internal_energy_operator"
            ),
        )

    def dual_evidence(self, problem):
        """Recover the converged contact force, distribution, and energy."""

        solution = getattr(problem, "solution", None)
        dual_state = getattr(problem, "_constraint_dual_state", None)
        if (
            solution is None
            or (
                getattr(problem, "last_solve_info", None) is None
                and dual_state != "accepted"
            )
        ):
            raise RuntimeError(
                "Rigid obstacle contact evidence requires its converged "
                "incremental nonlinear problem."
            )
        consumed = tuple(getattr(problem, "constraint_assets", ()))
        if not any(item is self for item in consumed):
            raise ValueError(
                "Rigid obstacle contact evidence requires the problem that "
                "consumed this contact provider instance."
            )

        from dolfinx import fem
        import dolfinx.fem.petsc as fem_petsc
        from petsc4py import PETSc

        from agentfem.constraints import constraint_dual

        test = ufl.TestFunction(solution.function_space)
        residual = fem_petsc.assemble_vector(fem.form(self.form(solution, test)))
        try:
            residual.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE,
            )
            distribution = fem.Function(
                solution.function_space,
                name=f"{self.name}_reaction",
            )
            owned = int(
                solution.function_space.dofmap.index_map.size_local
                * solution.function_space.dofmap.index_map_bs
            )
            distribution.x.array[:owned] = -np.asarray(residual.array_r[:owned])
            distribution.x.scatter_forward()
        finally:
            residual.destroy()

        values = np.asarray(distribution.x.array[:owned], dtype=float)
        shape = tuple(getattr(solution, "ufl_shape", ()))
        if len(shape) != 1:
            raise NotImplementedError(
                "Rigid obstacle contact evidence requires a vector field."
            )
        components = int(shape[0])
        if values.size % components:
            raise RuntimeError(
                "Contact reaction storage is incompatible with the field."
            )
        local_resultant = np.sum(values.reshape((-1, components)), axis=0)
        resultant = np.empty(components, dtype=float)
        comm = solution.function_space.mesh.comm
        comm.Allreduce(local_resultant, resultant, op=MPI.SUM)

        energy_local = fem.assemble_scalar(fem.form(self.energy_form(solution)))
        energy = float(comm.allreduce(float(energy_local), op=MPI.SUM))
        penetration = self.penetration(solution)
        penetration_l2_local = fem.assemble_scalar(
            fem.form(penetration * penetration * self.location.measure)
        )
        contact_area_local = fem.assemble_scalar(
            fem.form(
                ufl.conditional(ufl.gt(penetration, 0.0), 1.0, 0.0)
                * self.location.measure
            )
        )
        rigid_state = self._rigid_state()
        generalized_force = resultant
        generalized_coordinate = np.zeros_like(resultant)
        contact_moment = None
        projection_diagnostics = None
        if rigid_state is not None:
            function_space = solution.function_space
            block_size = int(function_space.dofmap.index_map_bs)
            block_count = int(function_space.dofmap.index_map.size_local)
            if block_size != components:
                raise NotImplementedError(
                    "Rigid contact moment recovery requires one blocked vector "
                    "space with one displacement block per node."
                )
            coordinates = np.asarray(
                function_space.tabulate_dof_coordinates()[:block_count, :components],
                dtype=float,
            )
            displacement_values = np.asarray(
                solution.x.array[:owned], dtype=float
            ).reshape((-1, components))
            force_values = values.reshape((-1, components))
            current_positions = coordinates + displacement_values
            force_norms = np.linalg.norm(force_values, axis=1)
            global_force_scale = max(
                float(
                    comm.allreduce(
                        float(np.max(force_norms)) if force_norms.size else 0.0,
                        op=MPI.MAX,
                    )
                ),
                1.0,
            )
            active = force_norms > 256.0 * np.finfo(float).eps * global_force_scale
            projection = self.surface.project(
                current_positions[active],
                motion=self.motion,
                factor=rigid_state["factor"],
            )
            local_count = projection.point_count
            local_invalid = int(np.count_nonzero(~projection.valid))
            global_count = int(comm.allreduce(local_count, op=MPI.SUM))
            global_invalid = int(comm.allreduce(local_invalid, op=MPI.SUM))
            local_minimum = (
                float(np.min(projection.signed_gaps))
                if local_count
                else float("inf")
            )
            local_maximum = (
                float(np.max(projection.signed_gaps))
                if local_count
                else float("-inf")
            )
            projection_diagnostics = {
                "method": projection.method,
                "sample": "nonzero_reaction_nodes",
                "point_count": global_count,
                "invalid_count": global_invalid,
                "all_valid": global_invalid == 0,
                "signed_gap_convention": (
                    "positive_admissible_negative_penetration"
                ),
                "minimum_signed_gap": (
                    None
                    if not global_count
                    else float(comm.allreduce(local_minimum, op=MPI.MIN))
                ),
                "maximum_signed_gap": (
                    None
                    if not global_count
                    else float(comm.allreduce(local_maximum, op=MPI.MAX))
                ),
            }
            arm = current_positions - np.asarray(
                rigid_state["reference_point"], dtype=float
            )
            if components == 2:
                contact_moment = np.asarray(
                    [
                        np.sum(
                            arm[:, 0] * force_values[:, 1]
                            - arm[:, 1] * force_values[:, 0]
                        )
                    ],
                    dtype=float,
                )
            else:
                contact_moment = np.sum(np.cross(arm, force_values), axis=0)
            global_moment = np.empty_like(contact_moment)
            comm.Allreduce(contact_moment, global_moment, op=MPI.SUM)
            contact_moment = global_moment
            generalized_force = np.concatenate((resultant, contact_moment))
            generalized_coordinate = (
                np.zeros_like(generalized_force)
                if self.motion is None
                else self.motion.generalized_coordinate(rigid_state["factor"])
            )

        diagnostics = {
            "status": "complete",
            "contact_energy": energy,
            "penetration_l2_norm": float(
                np.sqrt(comm.allreduce(float(penetration_l2_local), op=MPI.SUM))
            ),
            "active_contact_measure": float(
                comm.allreduce(float(contact_area_local), op=MPI.SUM)
            ),
            "resultant_norm": float(np.linalg.norm(resultant)),
            "reaction_distribution": distribution.name,
            "distribution_location": "nodes",
            "energy_accounting": "conservative_internal_contact_potential",
            "comm_size": int(comm.size),
        }
        if rigid_state is not None:
            diagnostics.update(
                {
                    "surface": self.surface.summary(),
                    "rigid_motion": (
                        None if self.motion is None else self.motion.summary()
                    ),
                    "rigid_translation": np.asarray(
                        rigid_state["translation"], dtype=float
                    ).tolist(),
                    "rigid_rotation": np.asarray(
                        rigid_state["rotation"], dtype=float
                    ).tolist(),
                    "rigid_reference_point": np.asarray(
                        rigid_state["reference_point"], dtype=float
                    ).tolist(),
                    "contact_moment": contact_moment.tolist(),
                    "surface_projection": projection_diagnostics,
                    "generalized_force_convention": (
                        "translation_resultant_then_rotation_moment"
                    ),
                }
            )
        return constraint_dual(
            self,
            force=generalized_force,
            # A fixed obstacle has zero generalized translation at every
            # station.  Its external work is therefore zero; the penalty
            # potential remains a separate internal-energy contribution.
            coordinate=generalized_coordinate,
            resultant=resultant,
            distribution=distribution,
            diagnostics=diagnostics,
            role="contact_constraint",
            source="rigid_obstacle_penalty_potential",
            complete=True,
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "rigid_obstacle_penalty_contact",
            "location": getattr(self.location, "name", None),
            "penalty": describe_value(self.penalty),
            "normal": describe_value(self.normal),
            "initial_gap": describe_value(self.initial_gap),
            "friction": "none",
            "obstacle": (
                "prescribed_rigid_plane"
                if self.has_prescribed_motion
                else "fixed_plane"
            ),
            "surface": None if self.surface is None else self.surface.summary(),
            "motion": None if self.motion is None else self.motion.summary(),
        }


def elastic_foundation(
    *, on=None, location=None, stiffness, mode: str = "isotropic", normal=None,
    name: str = "elastic_foundation",
) -> ElasticFoundation:
    selected = location if location is not None else on
    if on is not None and location is not None:
        raise ValueError("Pass either on=... or location=..., not both.")
    if selected is None or not hasattr(selected, "domain"):
        raise ValueError("elastic_foundation requires a boundary region.")
    return ElasticFoundation(
        stiffness=constants.constant(selected.domain, stiffness),
        location=selected,
        mode=mode,
        normal=normal,
        name=name,
    )


def rigid_obstacle_contact(
    *,
    on=None,
    location=None,
    penalty,
    normal=None,
    initial_gap=0.0,
    surface: RigidPlaneSurface | None = None,
    motion: PrescribedRigidMotion | None = None,
    name: str = "rigid_obstacle_contact",
) -> RigidObstaclePenaltyContact:
    """Create conservative frictionless contact with one analytical plane.

    ``normal`` and ``initial_gap`` preserve the original fixed-plane route.
    ``surface`` makes the plane geometry explicit and may be paired with one
    normalized prescribed rigid-body ``motion``.
    """

    selected = location if location is not None else on
    if on is not None and location is not None:
        raise ValueError("Pass either on=... or location=..., not both.")
    if selected is None or not hasattr(selected, "domain"):
        raise ValueError("rigid_obstacle_contact requires a boundary region.")
    if isinstance(penalty, Real) and (
        not np.isfinite(float(penalty)) or float(penalty) <= 0.0
    ):
        raise ValueError("Contact penalty must be finite and positive.")
    if isinstance(initial_gap, Real) and not np.isfinite(float(initial_gap)):
        raise ValueError("Contact initial_gap must be finite.")
    dimension = int(selected.domain.geometry.dim)
    if surface is not None and not isinstance(surface, RigidPlaneSurface):
        raise TypeError("surface must be a RigidPlaneSurface.")
    if motion is not None and not isinstance(motion, PrescribedRigidMotion):
        raise TypeError("motion must be a PrescribedRigidMotion.")
    if motion is not None and surface is None:
        raise ValueError("Prescribed rigid motion requires an explicit surface.")
    if surface is not None and normal is not None:
        raise ValueError("Pass either normal=... or surface=..., not both.")
    if surface is not None and isinstance(initial_gap, Real) and float(initial_gap) != 0.0:
        raise ValueError(
            "Explicit rigid-plane geometry defines the gap; initial_gap must be zero."
        )
    if surface is not None:
        if surface.dimension != dimension:
            raise ValueError("Rigid-plane surface must match the mesh dimension.")
        if motion is not None and motion.dimension != dimension:
            raise ValueError("Rigid motion must match the mesh dimension.")
        state = surface.transformed(motion, 0.0)
        contact_normal = np.asarray(state["normal"], dtype=float)
        contact_point = constants.constant(selected.domain, state["point"])
        motion_factor = constants.constant(selected.domain, 0.0)
    else:
        if normal is None:
            raise ValueError("Fixed rigid-plane contact requires normal=....")
        contact_normal = np.asarray(normal, dtype=float)
        contact_point = None
        motion_factor = None
    raw_normal = np.asarray(contact_normal, dtype=float)
    if raw_normal.shape != (dimension,) or not np.all(np.isfinite(raw_normal)):
        raise ValueError(
            "Contact normal must be one finite vector matching the mesh "
            f"dimension {dimension}."
        )
    normal_norm = float(np.linalg.norm(raw_normal))
    if not np.isclose(normal_norm, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "Contact normal must be a unit vector so gap and penalty units "
            f"remain explicit; norm={normal_norm:.16g}."
        )
    return RigidObstaclePenaltyContact(
        penalty=constants.constant(selected.domain, penalty),
        location=selected,
        normal=constants.constant(selected.domain, contact_normal),
        initial_gap=constants.constant(selected.domain, initial_gap),
        surface=surface,
        motion=motion,
        surface_point=contact_point,
        motion_factor=motion_factor,
        name=name,
    )


__all__ = [
    "ElasticFoundation",
    "RigidObstaclePenaltyContact",
    "elastic_foundation",
    "rigid_obstacle_contact",
]
