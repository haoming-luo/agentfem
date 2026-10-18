# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Private DOLFINx trial bridge for completely certified P1 contact faces.

This is deliberately *not* a Step provider. It owns neither accepted contact
State nor path work, rollback, checkpoint, MPI assembly, or Result publication.
It tests the boundary between a Model pair, pure face Operator and PETSc rows
without disguising face-integrated evidence as quadrature-point evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._petsc_contact_rows import add_contact_coo_tangent, add_contact_nodal_residual
from .closed_polyhedron import ClosedTriangulatedRigidSolid
from .contact_face_operator_research import (
    CertifiedFaceContactOperator,
    evaluate_certified_closed_p1_contact_operator,
    evaluate_moving_certified_closed_p1_contact_operator,
    extract_complete_p1_face_layout,
)
from .contact_pair import RigidContactPair
from .dolfinx_contact_trace import DolfinxContactTraceAdapter


@dataclass(frozen=True, eq=False)
class CertifiedFaceContactTrial:
    """One immutable operator with the exact displacement/pose it certifies."""

    operator: CertifiedFaceContactOperator
    pair_identity: str
    current_nodal_positions: np.ndarray
    motion_factor: float

    @property
    def nodal_structural_residual(self) -> np.ndarray:
        return self.operator.nodal_residual

    @property
    def potential_energy(self) -> float:
        return self.operator.potential_energy

    def tangent_action(self, direction) -> np.ndarray:
        return self.operator.tangent_action(direction)


@dataclass(eq=False)
class DolfinxCertifiedFaceContactTrialProvider:
    """Serial research adapter with fail-before-mutation trial checks."""

    pair: RigidContactPair
    adapter: DolfinxContactTraceAdapter
    displacement: object
    maximum_slave_faces: int = 128
    maximum_master_facets: int = 64
    order: int = 24
    current_factor: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.pair, RigidContactPair) or not isinstance(
            self.adapter, DolfinxContactTraceAdapter
        ):
            raise TypeError(
                "Certified face trial requires a rigid pair and DOLFINx trace."
            )
        if (
            not isinstance(self.pair.rigid_body.surface, ClosedTriangulatedRigidSolid)
            or self.pair.friction is not None
            or self.pair.law.invalid_policy != "reject"
            or self.pair.initial_penetration_limit is not None
            or self.adapter.facet_topology != "triangle"
            or self.adapter.interpolation_degree != 1
            or self.adapter.geometry_degree != 1
            or self.adapter.quadrature_rule != "triangle_degree_two_three_point"
        ):
            raise NotImplementedError(
                "Research face trial admits only frictionless reviewed closed P1 "
                "triangles with scalar penalty; initial-overlap preflight is absent."
            )
        if (
            self.pair.slave_boundary.domain is not self.displacement.function_space.mesh
            or self.adapter.function_space is not self.displacement.function_space
            or self.adapter.communicator.size != 1
        ):
            raise NotImplementedError(
                "Research face trial requires one serial mesh and displacement space."
            )
        extract_complete_p1_face_layout(
            self.adapter.trace, self.adapter.reference_nodal_positions
        )
        schedule = self.pair.rigid_body.motion_schedule
        if schedule is not None and (
            schedule.start_time != 0.0
            or schedule.end_time != 1.0
            or schedule.amplitude is not None
        ):
            raise NotImplementedError(
                "Research face trial requires normalized rigid motion without amplitude."
            )

    def update_motion(self, factor: float) -> None:
        selected = float(factor)
        if not np.isfinite(selected) or not 0.0 <= selected <= 1.0 + 1e-12:
            raise ValueError("Research face trial factor must lie in [0, 1].")
        schedule = self.pair.rigid_body.motion_schedule
        if schedule is not None:
            schedule.motion.state(selected)
        self.current_factor = selected

    def evaluate(self) -> CertifiedFaceContactTrial:
        current = self.adapter.current_nodal_positions(self.displacement)
        reference = self.adapter.reference_nodal_positions
        surface = self.pair.rigid_body.surface
        arguments = dict(
            penalty=float(np.asarray(self.pair.law.penalty)),
            order=self.order,
            maximum_slave_faces=self.maximum_slave_faces,
            maximum_master_facets=self.maximum_master_facets,
        )
        schedule = self.pair.rigid_body.motion_schedule
        if schedule is None:
            operator = evaluate_certified_closed_p1_contact_operator(
                self.adapter.trace, reference, current, surface, **arguments
            )
        else:
            operator = evaluate_moving_certified_closed_p1_contact_operator(
                self.adapter.trace,
                reference,
                current,
                surface,
                motion=schedule.motion,
                factor=self.current_factor,
                **arguments,
            )
        saved = np.asarray(current, dtype=float).copy()
        saved.setflags(write=False)
        return CertifiedFaceContactTrial(
            operator=operator,
            pair_identity=self.pair.scientific_identity,
            current_nodal_positions=saved,
            motion_factor=self.current_factor,
        )

    def _check_current(self, trial: CertifiedFaceContactTrial) -> None:
        if (
            not isinstance(trial, CertifiedFaceContactTrial)
            or trial.pair_identity != self.pair.scientific_identity
            or trial.operator.tool_fingerprint
            != self.pair.rigid_body.surface.geometry_fingerprint
            or trial.motion_factor != self.current_factor
            or not np.array_equal(
                trial.current_nodal_positions,
                self.adapter.current_nodal_positions(self.displacement),
            )
        ):
            raise ValueError(
                "Certified face contribution belongs to another Newton trial or pair."
            )

    def add_residual(self, vector, trial: CertifiedFaceContactTrial) -> None:
        self._check_current(trial)
        add_contact_nodal_residual(vector, trial.nodal_structural_residual)

    def add_tangent(
        self, matrix, trial: CertifiedFaceContactTrial, *, constrained_dofs=()
    ) -> None:
        self._check_current(trial)
        operator = trial.operator
        add_contact_coo_tangent(
            matrix,
            operator.tangent_rows,
            operator.tangent_columns,
            operator.tangent_values,
            constrained_dofs=constrained_dofs,
        )
