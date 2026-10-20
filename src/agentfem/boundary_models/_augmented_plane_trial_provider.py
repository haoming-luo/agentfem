# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded fixed-plane AL Operator/State bridge, not a new Newton solver.

This private admission uses a coalesced nodal reference trace on serial P1
tetrahedra. It does not claim a stable general mortar multiplier space, admit
MPC/moving tools, or reinterpret the signed AL functional as stored energy.
The existing standard-J2 Procedure owns every inner Newton solve and joint
accepted-increment transaction.
"""

from __future__ import annotations

import numpy as np

from ..steps import NormalAugmentationOptions
from .contact_state import ContactProjectionRecord
from .contact_work import _fixed_normal_work_increment
from .dolfinx_contact_trace import dolfinx_boundary_region_contact_trace
from .dolfinx_implicit_facet_contact import DolfinxImplicitRigidContact
from .normal_augmentation import (
    FixedTraceNormalMultiplierState,
    integrate_fixed_trace_augmented_normal,
)
from .rigid import RigidPlaneSurface, _readonly_array
from .trace_policy import RigidContactTracePolicy


def validate_free_plane_normal_rows(trace, normal, constrained_dofs):
    """Refuse a multiplier row whose normal motion is fixed by strong BCs.

    Coalesced nodal support makes the remaining rows disjoint. This tests
    algebraic independence in this specific fixed-plane space, not a general
    mesh-uniform inf-sup condition or a contact infeasibility theorem.
    """

    nodes = trace._direct_nodes
    if nodes is None or len(np.unique(nodes)) != len(nodes):
        raise ValueError("AL requires distinct coalesced nodal pressure support.")
    rows = 3 * nodes[:, None] + np.arange(3)
    free_normal = np.broadcast_to(np.asarray(normal, dtype=float), rows.shape).copy()
    free_normal[np.isin(rows, np.asarray(constrained_dofs, dtype=np.int64))] = 0.
    blocked = np.flatnonzero(np.linalg.norm(free_normal, axis=1) <= 128 * np.finfo(float).eps)
    if blocked.size:
        identity = int(trace.point_ids[blocked[0]])
        raise NotImplementedError(
            "AFM-CONTACT-AL-REDUNDANT: the fixed plane multiplier has no free "
            f"normal displacement at contact_point_id={identity}; this bounded "
            "admission does not split reactions between coincident strong/contact constraints."
        )


class DolfinxAugmentedPlaneTrialProvider(DolfinxImplicitRigidContact):
    """Fixed nodal multiplier State composed with ordinary contact row assembly."""

    physical_contact_potential_available = False
    requires_accepted_state = True

    def __init__(self, penalty_provider, *, policy: NormalAugmentationOptions):
        if type(penalty_provider) is not DolfinxImplicitRigidContact:
            raise TypeError("AL research admission needs one ordinary rigid-pair provider.")
        if not isinstance(policy, NormalAugmentationOptions):
            raise TypeError("AL research admission requires explicit acceptance policy.")
        base = penalty_provider
        if (
            not isinstance(base.surface, RigidPlaneSurface)
            or base.surface.dimension != 3
            or base.has_prescribed_motion
            or base.pair.friction is not None
            or base.contact_integration != "gauss"
            or base.adapter.communicator.size != 1
            or base.adapter.facet_topology != "triangle"
            or base.adapter.interpolation_degree != 1
            or base.adapter.geometry_degree != 1
            or np.asarray(base.pair.law.penalty).ndim != 0
        ):
            raise NotImplementedError(
                "AFM-CONTACT-AL-SCOPE: one fixed frictionless 3D plane and serial "
                "P1 tetrahedral displacement only; no MPC or changing trace."
            )
        adapter = dolfinx_boundary_region_contact_trace(
            base.pair.slave_boundary, base.displacement.function_space,
            integration="nodal_lumped",
        )
        adapter = RigidContactTracePolicy(
            "nodal_lumped", coalesce_nodes=True,
        ).apply(adapter, base.pair)
        super().__init__(base.pair, adapter, base.displacement, None)
        self.policy = policy
        self.normal_multiplier_state = FixedTraceNormalMultiplierState(
            adapter.trace, pair_identity=self.name,
            reference_evaluation=adapter.trace.evaluate(adapter.reference_nodal_positions),
            surface_fingerprint=self.surface.geometry_fingerprint,
            penalty=float(base.pair.law.penalty),
            penetration_tolerance=policy.penetration_tolerance,
            complementarity_tolerance=policy.complementarity_tolerance,
            maximum_augmentations=policy.maximum_augmentations,
            length_unit=policy.length_unit, pressure_unit=policy.pressure_unit,
        )
        self._pending_acceptance = None
        self._accepted_positions = adapter.evaluate(self.displacement).query_points.copy()
        self._last_normal_evaluation = None
        self._outer_in_progress = False

    def normal_response(self):
        evaluation = self.adapter.evaluate(self.displacement)
        record = ContactProjectionRecord(
            evaluation.trace.point_ids, self.surface.project(evaluation.query_points),
        )
        response = self.normal_multiplier_state.evaluate(evaluation, record)
        self._last_normal_evaluation = evaluation, record
        return response

    def evaluate(self):
        response = self.normal_response()
        return integrate_fixed_trace_augmented_normal(
            response.evaluation, response.record, response,
            np.zeros((response.record.projection.point_count, 3, 3)),
            surface_reference_point=self.pair.rigid_body.reference_point,
        )

    def finish_frozen_equilibrium(self, *, residual_norm, residual_tolerance):
        if self._last_normal_evaluation is None:
            raise RuntimeError("AL acceptance requires the current total residual evaluation.")
        evaluation, record = self._last_normal_evaluation
        if not np.array_equal(
            evaluation.query_points, self.adapter.evaluate(self.displacement).query_points,
        ):
            raise RuntimeError("AL equilibrium evidence refers to another displacement.")
        state = self.normal_multiplier_state
        state.mark_equilibrium_converged(
            evaluation, record, total_residual_norm=residual_norm,
            total_residual_tolerance=residual_tolerance,
        )
        diagnostics = state.acceptance_diagnostics(evaluation, record)
        ready = (
            diagnostics["maximum_penetration"] <= state.penetration_tolerance
            and diagnostics["maximum_complementarity"] <= state.complementarity_tolerance
        )
        self._pending_acceptance = (evaluation, record) if ready else None
        return ready, diagnostics

    def augment(self):
        if self._last_normal_evaluation is None:
            raise RuntimeError("AL update needs converged equilibrium evidence.")
        self.normal_multiplier_state.augment_after_converged(*self._last_normal_evaluation)
        self._last_normal_evaluation = None
        self._pending_acceptance = None

    def commit_accepted(self):
        if self._pending_acceptance is None:
            raise RuntimeError("AL cannot commit an unconverged outer iteration.")
        evaluation, record = self._pending_acceptance
        if not np.array_equal(
            evaluation.query_points, self.adapter.evaluate(self.displacement).query_points,
        ):
            raise RuntimeError("AL converged displacement changed before joint commit.")
        self.normal_multiplier_state.commit(evaluation, record)
        self._accepted_positions = evaluation.query_points.copy()
        self._pending_acceptance = None
        self._last_normal_evaluation = None
        self._outer_in_progress = False

    def normal_work_increment(self):
        """Observe a converged outer endpoint before the joint State commit.

        Only the enclosing Procedure publishes this increment. It never
        sums Newton iterates or multiplier augmentations as physical motion.
        """

        if self._pending_acceptance is None:
            raise RuntimeError("AL normal work requires a converged outer endpoint.")
        response = self.normal_response()
        if not np.array_equal(
            self._pending_acceptance[0].query_points, response.evaluation.query_points,
        ):
            raise RuntimeError("AL normal-work endpoint changed after convergence.")
        old_gap = self.surface.project(self._accepted_positions).signed_gaps
        old_pressure = np.maximum(
            0., self.normal_multiplier_state.accepted_multiplier
            - self.normal_multiplier_state.penalty * old_gap,
        )
        return _fixed_normal_work_increment(
            weights=self.adapter.trace.weights,
            old_gap=old_gap, new_gap=response.record.projection.signed_gaps,
            old_pressure=old_pressure, new_pressure=response.pressure,
        )

    def _require_accepted_boundary(self):
        if (self._outer_in_progress or self._pending_acceptance is not None
                or self.normal_multiplier_state.augmentation_count):
            raise RuntimeError("AL checkpoint cannot contain an inner/outer trial.")

    def snapshot_accepted_state(self):
        self._require_accepted_boundary()
        return {
            "schema": "agentfem.augmented-plane-state.v1",
            "multiplier": self.normal_multiplier_state.accepted_snapshot(),
            "query_points": self._accepted_positions.tolist(),
        }

    def restore_accepted_state(self, saved):
        if (not isinstance(saved, dict)
                or set(saved) != {"schema", "multiplier", "query_points"}
                or saved["schema"] != "agentfem.augmented-plane-state.v1"):
            raise ValueError("AL accepted checkpoint schema differs.")
        positions = np.asarray(saved["query_points"], dtype=float)
        if positions.shape != self._accepted_positions.shape or not np.all(np.isfinite(positions)):
            raise ValueError("AL checkpoint query geometry differs.")
        self.normal_multiplier_state.restore_accepted(saved["multiplier"])
        self._accepted_positions = positions.copy()
        self._pending_acceptance = None
        self._last_normal_evaluation = None
        self._outer_in_progress = False

    def snapshot_runtime_state(self):
        self._require_accepted_boundary()
        return (self.normal_multiplier_state.snapshot_runtime_state(),
                _readonly_array(self._accepted_positions))

    def restore_runtime_state(self, saved):
        if not isinstance(saved, tuple) or len(saved) != 2:
            raise ValueError("AL runtime snapshot shape differs.")
        positions = np.asarray(saved[1], dtype=float)
        if positions.shape != self._accepted_positions.shape or not np.all(np.isfinite(positions)):
            raise ValueError("AL runtime query geometry differs.")
        self.normal_multiplier_state.restore_runtime_state(saved[0])
        self._accepted_positions = _readonly_array(positions)
        self._pending_acceptance = None
        self._last_normal_evaluation = None
        self._outer_in_progress = False

    def validate_accepted_configuration(self):
        if self._outer_in_progress or self._pending_acceptance is not None:
            raise RuntimeError("AL output requires a fully accepted outer increment.")
        if not np.allclose(
            self._accepted_positions, self.adapter.evaluate(self.displacement).query_points,
            rtol=1e-12, atol=1e-12,
        ):
            raise ValueError("AL accepted multiplier belongs to another displacement.")

    def _current_point_response(self, *, _include_partition=False):
        raise NotImplementedError(
            "AFM-CONTACT-AL-DIAGNOSTIC: penalty point/partition diagnostics do not "
            "represent augmented contact; use normal_response or accepted_point_samples."
        )

    def quadrature_sensitivity(self, *args, **kwargs):
        raise NotImplementedError("AFM-CONTACT-AL-TRANSFER: changing the multiplier trace needs transfer.")

    def reference_subdivision_sensitivity(self, *args, **kwargs):
        raise NotImplementedError("AFM-CONTACT-AL-TRANSFER: changing the multiplier trace needs transfer.")

    def closed_penalty_energy_enclosure(self, *args, **kwargs):
        raise NotImplementedError("AFM-CONTACT-AL-ENERGY: AL has no physical penalty-potential bound.")

    def accepted_point_samples(self):
        """Export the actual accepted augmented pressure, never the old penalty law."""

        self.validate_accepted_configuration()
        response = self.normal_response()
        projection = response.record.projection
        trace = response.evaluation.trace
        count = trace.point_count
        return {
            "constraint_name": self.name, "surface_name": self.surface.name,
            "geometry_fingerprint": self.surface.geometry_fingerprint,
            "trace_source": trace.source, "load_factor": 0.,
            "location": "coalesced_reference_boundary_nodes",
            "pressure_measure": "augmented_force_per_reference_area_not_cauchy_pressure",
            "enforcement": "fixed_trace_normal_augmented_lagrangian_research",
            "point_ids": trace.point_ids.copy(),
            "slave_positions": projection.query_points.copy(),
            "master_closest_points": projection.closest_points.copy(),
            "surface_normals": projection.normals.copy(),
            "signed_gaps": projection.signed_gaps.copy(),
            "reference_weights": trace.weights.copy(),
            "normal_pressures": response.pressure.copy(),
            "penetrations": np.maximum(-projection.signed_gaps, 0.),
            "active": response.active.copy(),
            "entity_ids": np.full(count, -1, dtype=np.int64),
            "surface_friction_tractions": np.zeros((count, 3)),
            "accepted_elastic_tangential_slips": np.zeros((count, 3)),
            "cumulative_friction_dissipation_densities": np.zeros(count),
        }

    def current_point_diagnostics(self):
        response = self.normal_response()
        return {
            "location": "coalesced_reference_boundary_nodes",
            "pressure_measure": "augmented_force_per_reference_area_not_cauchy_pressure",
            "active_point_count": int(np.count_nonzero(response.active)),
            "active_reference_area": float(np.sum(self.adapter.trace.weights[response.active])),
            "maximum_pressure": float(np.max(response.pressure, initial=0.0)),
            "maximum_penetration": float(np.max(
                np.maximum(-response.record.projection.signed_gaps, 0.0), initial=0.0,
            )),
            "initial_contact": dict(self._initial_contact_diagnostics),
            "normal_augmentation": self.normal_multiplier_state.acceptance_diagnostics(
                response.evaluation, response.record,
            ),
            "physical_contact_energy_available": False,
        }

    def dual_evidence(self, problem):
        from dolfinx import fem
        from ..constraints import constraint_dual

        if getattr(problem, "solution", None) is not self.displacement:
            raise ValueError("AL dual belongs to another problem.")
        self.validate_accepted_configuration()
        if (getattr(problem, "last_solve_info", None) is None
                and getattr(problem, "_constraint_dual_state", None) != "accepted"):
            raise RuntimeError("AL dual requires an accepted solution.")
        contribution = self.evaluate()
        distribution = fem.Function(self.displacement.function_space, name=f"{self.name}_reaction")
        distribution.x.array[:] = -contribution.nodal_structural_residual.reshape(-1)
        distribution.x.scatter_forward()
        return constraint_dual(
            self.pair, force=contribution.contact_force_on_structure,
            resultant=contribution.contact_force_on_structure, distribution=distribution,
            role="contact_constraint", source="bounded_fixed_plane_normal_augmentation",
            diagnostics=self.current_point_diagnostics(),
        )

    def summary(self):
        return {
            **super().summary(),
            "kind": "bounded_fixed_plane_normal_augmentation",
            "enforcement": "fixed_trace_normal_augmented_lagrangian_research",
            "contact_integration": "coalesced_reference_nodal",
            "policy": self.policy.summary(),
            "physical_contact_energy_available": False,
            "multiplier_space": "coalesced_reference_nodal_pressure",
            "spatial_stability_certified": False,
        }

    def _linearization_name(self):
        return "fixed_multiplier_nodal_normal_semismooth"

    def capabilities(self):
        from dataclasses import replace

        return replace(
            super().capabilities(),
            enforcement="fixed_trace_normal_augmented_lagrangian_research",
            work_evidence="provider_dual_required",
        )
