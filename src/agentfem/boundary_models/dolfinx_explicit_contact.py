# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit-dynamics residual adapter for the reviewed contact stack.

This module is deliberately the first, narrow Procedure consumer of the
backend-neutral surface, projection, response, and trace contracts.  It adds a
memoryless frictionless penalty contribution to an already assembled DOLFINx
residual.  Moving tools, friction, and an implicit tangent remain separate
capability gates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from agentfem import operators

from .contact_lifecycle import ContactProjectionLifecycle
from .contact_response import (
    FrictionlessPenaltyContactLaw,
    frictionless_penalty_contact_law,
)
from .contact_trace import ContactTraceAssembly
from .dolfinx_contact_trace import DolfinxContactTraceAdapter
from .rigid import _readonly_array


@dataclass(frozen=True, eq=False)
class ExplicitContactEvidence:
    """MPI-global evidence from one contact residual evaluation."""

    potential_energy: float
    contact_force_on_structure: object
    contact_force_on_surface: object
    surface_generalized_moment: object | None
    active_point_count: int
    invalid_point_count: int

    def __post_init__(self) -> None:
        potential = float(self.potential_energy)
        if not np.isfinite(potential) or potential < 0.0:
            raise ValueError("Contact potential energy must be finite and non-negative.")
        structure = np.asarray(self.contact_force_on_structure, dtype=float)
        surface = np.asarray(self.contact_force_on_surface, dtype=float)
        if structure.shape != (3,) or surface.shape != (3,):
            raise ValueError("Explicit contact resultants must be three-dimensional.")
        if not np.all(np.isfinite(structure)) or not np.all(np.isfinite(surface)):
            raise ValueError("Explicit contact resultants must be finite.")
        if not np.allclose(
            structure + surface,
            0.0,
            rtol=256.0 * np.finfo(float).eps,
            atol=256.0 * np.finfo(float).eps,
        ):
            raise ValueError("Explicit contact action and reaction do not balance.")
        moment = self.surface_generalized_moment
        if moment is not None:
            moment = np.asarray(moment, dtype=float)
            if moment.shape != (3,) or not np.all(np.isfinite(moment)):
                raise ValueError("Explicit contact moment must be one finite 3-vector.")
            object.__setattr__(self, "surface_generalized_moment", _readonly_array(moment))
        active = int(self.active_point_count)
        invalid = int(self.invalid_point_count)
        if active < 0 or invalid < 0:
            raise ValueError("Explicit contact point counts must be non-negative.")
        object.__setattr__(self, "potential_energy", potential)
        object.__setattr__(
            self,
            "contact_force_on_structure",
            _readonly_array(structure),
        )
        object.__setattr__(
            self,
            "contact_force_on_surface",
            _readonly_array(surface),
        )
        object.__setattr__(self, "active_point_count", active)
        object.__setattr__(self, "invalid_point_count", invalid)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "explicit_contact_evidence",
            "potential_energy": self.potential_energy,
            "contact_force_on_structure": self.contact_force_on_structure.tolist(),
            "contact_force_on_surface": self.contact_force_on_surface.tolist(),
            "surface_generalized_moment": (
                None
                if self.surface_generalized_moment is None
                else self.surface_generalized_moment.tolist()
            ),
            "active_point_count": self.active_point_count,
            "invalid_point_count": self.invalid_point_count,
        }


class DolfinxExplicitContactResidual:
    """Add reviewed trace contact to a DOLFINx explicit residual.

    The adapter owns no time integration.  ``ExplicitDynamicsStep`` evaluates
    it after displacement prediction and kinematic projection, then accepts or
    rejects its projection trial with the surrounding increment.  A declared
    contact stability limit is mandatory because a penalty spring can reduce
    the central-difference critical time step below the body-wave estimate.
    """

    def __init__(
        self,
        base,
        *,
        adapter: DolfinxContactTraceAdapter,
        displacement,
        lifecycle: ContactProjectionLifecycle,
        law: FrictionlessPenaltyContactLaw,
        maximum_stable_time_increment: float,
        surface_reference_point=None,
        projection_options=None,
        name: str = "dolfinx_explicit_contact_residual",
    ) -> None:
        if not isinstance(adapter, DolfinxContactTraceAdapter):
            raise TypeError("Explicit contact requires DolfinxContactTraceAdapter.")
        if displacement.function_space is not adapter.function_space:
            raise ValueError("Explicit contact displacement differs from its trace space.")
        if not isinstance(lifecycle, ContactProjectionLifecycle):
            raise TypeError("Explicit contact requires ContactProjectionLifecycle.")
        if not np.array_equal(lifecycle.point_ids, adapter.trace.point_ids):
            raise ValueError("Explicit contact lifecycle and trace identities differ.")
        if not isinstance(law, FrictionlessPenaltyContactLaw):
            raise TypeError("Explicit contact requires FrictionlessPenaltyContactLaw.")
        limit = float(maximum_stable_time_increment)
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError(
                "maximum_stable_time_increment must be finite and positive."
            )
        options = {} if projection_options is None else dict(projection_options)
        if any(not isinstance(key, str) or not key for key in options):
            raise ValueError("Contact projection option names must be non-empty strings.")
        reference = (
            None
            if surface_reference_point is None
            else np.asarray(surface_reference_point, dtype=float).reshape(-1)
        )
        if reference is not None and (
            reference.shape != (3,) or not np.all(np.isfinite(reference))
        ):
            raise ValueError("Surface reference point must be one finite 3-vector.")
        if not str(name).strip():
            raise ValueError("Explicit contact residual requires a name.")
        self.base = base
        self.adapter = adapter
        self.displacement = displacement
        self.lifecycle = lifecycle
        self.law = law
        self.maximum_stable_time_increment = limit
        self.surface_reference_point = reference
        self.projection_options = options
        self.name = str(name)
        self.trial_evidence: ExplicitContactEvidence | None = None
        self.accepted_evidence: ExplicitContactEvidence | None = None
        self.accepted_evaluations = 0

    @property
    def communicator(self):
        return self.adapter.communicator

    def validate_time_increment(self, dt: float) -> None:
        """Reject a step exceeding the separately screened contact limit."""

        selected = float(dt)
        if not np.isfinite(selected) or selected <= 0.0:
            raise ValueError("Explicit contact requires a finite positive dt.")
        tolerance = 64.0 * np.finfo(float).eps * self.maximum_stable_time_increment
        if selected > self.maximum_stable_time_increment + tolerance:
            raise ValueError(
                "Explicit contact dt exceeds its declared stability limit "
                f"({selected:.6g} > {self.maximum_stable_time_increment:.6g})."
            )

    def _evaluate_local_contact(self) -> tuple[ContactTraceAssembly, object]:
        trace_evaluation = self.adapter.evaluate(self.displacement)
        projection = self.lifecycle.evaluate(
            trace_evaluation.query_points,
            **self.projection_options,
        )
        response = self.law.evaluate(projection.record.projection)
        assembly = trace_evaluation.assemble(
            projection.record,
            response,
            surface_reference_point=self.surface_reference_point,
        )
        return assembly, response

    def _collective_local_contact(self) -> tuple[ContactTraceAssembly, object]:
        local_error = None
        outcome = None
        try:
            outcome = self._evaluate_local_contact()
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        messages = tuple(self.communicator.allgather(local_error))
        failures = tuple(
            f"rank {rank}: {message}"
            for rank, message in enumerate(messages)
            if message is not None
        )
        if failures:
            self.lifecycle.rollback_increment()
            self.trial_evidence = None
            raise ValueError(
                "Explicit contact evaluation rejected collectively; "
                + "; ".join(failures)
            )
        if outcome is None:  # pragma: no cover - collective consensus guards this
            raise RuntimeError("Explicit contact consensus produced no local assembly.")
        return outcome

    def _global_evidence(self, assembly, response) -> ExplicitContactEvidence:
        moment = assembly.surface_generalized_moment
        local = np.concatenate(
            (
                np.asarray((assembly.potential_energy,), dtype=float),
                np.asarray(assembly.contact_force_on_structure, dtype=float),
                np.asarray(assembly.contact_force_on_surface, dtype=float),
                np.zeros(3, dtype=float) if moment is None else np.asarray(moment),
                np.asarray(
                    (
                        np.count_nonzero(response.active),
                        np.count_nonzero(~response.projection.valid),
                    ),
                    dtype=float,
                ),
            )
        )
        global_values = np.empty_like(local)
        self.communicator.Allreduce(local, global_values, op=MPI.SUM)
        return ExplicitContactEvidence(
            potential_energy=global_values[0],
            contact_force_on_structure=global_values[1:4],
            contact_force_on_surface=global_values[4:7],
            surface_generalized_moment=(
                None if moment is None else global_values[7:10]
            ),
            active_point_count=int(round(global_values[10])),
            invalid_point_count=int(round(global_values[11])),
        )

    def assemble_vector(self):
        """Assemble base and contact residuals without double-counting ghosts."""

        vector = operators.assemble_vector(self.base)
        try:
            assembly, response = self._collective_local_contact()
            contact = vector.duplicate()
            try:
                values = np.asarray(
                    assembly.nodal_structural_residual,
                    dtype=float,
                ).reshape(-1)
                with contact.localForm() as local:
                    local.set(0.0)
                    if local.array.shape != values.shape:
                        raise ValueError(
                            "Contact trace residual and PETSc local layouts differ: "
                            f"trace={values.shape}, vector={local.array.shape}."
                        )
                    local.array[:] = values
                contact.ghostUpdate(
                    addv=PETSc.InsertMode.ADD_VALUES,
                    mode=PETSc.ScatterMode.REVERSE,
                )
                if vector.array.shape != contact.array.shape:
                    raise ValueError("Contact and base owned vector layouts differ.")
                vector.array[:] += contact.array
            finally:
                contact.destroy()
            self.trial_evidence = self._global_evidence(assembly, response)
            return vector
        except Exception:
            vector.destroy()
            self.rollback()
            raise

    def commit(self) -> None:
        if self.trial_evidence is None or self.lifecycle.state.trial is None:
            raise RuntimeError("No explicit contact trial is available to commit.")
        if hasattr(self.base, "commit"):
            self.base.commit()
        self.lifecycle.commit_increment()
        self.accepted_evidence = self.trial_evidence
        self.trial_evidence = None
        self.accepted_evaluations += 1

    def rollback(self) -> None:
        if hasattr(self.base, "rollback"):
            self.base.rollback()
        if self.lifecycle.state.trial is not None:
            self.lifecycle.rollback_increment()
        self.trial_evidence = None

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "dolfinx_explicit_contact_residual",
            "procedure": "central_difference",
            "contact": "frictionless_penalty",
            "surface_motion": "fixed_only",
            "maximum_stable_time_increment": self.maximum_stable_time_increment,
            "stability": "caller_supplied_screening_limit_enforced",
            "parallel_assembly": "ghost_reverse_add_then_owned_accumulation",
            "projection_update": "every_residual_evaluation",
            "restart": "memoryless_projection_recomputed",
            "accepted_evaluations": self.accepted_evaluations,
            "accepted_evidence": (
                None
                if self.accepted_evidence is None
                else self.accepted_evidence.summary()
            ),
            "adapter": self.adapter.summary(),
            "lifecycle": self.lifecycle.summary(),
            "law": self.law.summary(),
            "base": self.base.summary() if hasattr(self.base, "summary") else repr(self.base),
        }


def dolfinx_explicit_contact_residual(
    base,
    *,
    adapter,
    displacement,
    projector,
    penalty,
    maximum_stable_time_increment,
    invalid_policy: str = "reject",
    surface_reference_point=None,
    projection_options=None,
    name: str = "dolfinx_explicit_contact_residual",
) -> DolfinxExplicitContactResidual:
    """Build the reviewed first explicit contact residual consumer."""

    lifecycle = ContactProjectionLifecycle(projector, adapter.trace.point_ids)
    law = frictionless_penalty_contact_law(
        penalty,
        invalid_policy=invalid_policy,
        name=f"{name}_law",
    )
    return DolfinxExplicitContactResidual(
        base,
        adapter=adapter,
        displacement=displacement,
        lifecycle=lifecycle,
        law=law,
        maximum_stable_time_increment=maximum_stable_time_increment,
        surface_reference_point=surface_reference_point,
        projection_options=projection_options,
        name=name,
    )


__all__ = [
    "DolfinxExplicitContactResidual",
    "ExplicitContactEvidence",
    "dolfinx_explicit_contact_residual",
]
