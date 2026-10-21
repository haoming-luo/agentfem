# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private serial deformable contact composition for existing central difference."""

from dataclasses import replace
from contextlib import contextmanager
from time import monotonic

import numpy as np

from .._deformable_q1_contact import DeformableQ1Penalty, DeformableContactResponse
from .._contact_path_guard import unresolved_outside_paths, outside_gap_lower_bounds
from ..fields import unwrap
from ..provenance import content_fingerprint


class DeformableQ1ContactResidual:
    @contextmanager
    def overlap_execution(self):
        """Run-scoped speculative geometry; PETSc remains on the calling thread.

        The geometry operator/cache belongs exclusively to this residual. Join
        its worker before rollback or scope exit. This changes scheduling only,
        not force assembly order, accepted history or failure precedence.
        """
        from concurrent.futures import ThreadPoolExecutor
        from threading import get_ident
        if getattr(self, '_geometry_executor', None) is not None:
            raise RuntimeError('Contact overlap execution is already active')
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='agentfem-contact') as pool:
            self._geometry_executor = pool
            self._geometry_owner = get_ident()
            try:
                yield
            finally:
                self._geometry_executor = None

    def __init__(
        self,
        base,
        *,
        operator,
        displacement,
        secondary_dofs,
        primary_dofs,
        time_increment_ceiling,
        name="deformable_contact",
    ):
        if not isinstance(operator, DeformableQ1Penalty):
            raise TypeError("Expected a DeformableQ1Penalty operator.")
        self.base, self.operator = base, operator
        self.displacement = unwrap(displacement)
        space = self.displacement.function_space
        if space.mesh.comm.size != 1:
            raise NotImplementedError("Deformable Q1 contact execution is serial only.")
        if (
            space.dofmap.index_map_bs != 3
            or space.element.basix_element.degree != 1
            or space.element.basix_element.cell_type.name != "hexahedron"
            or space.element.basix_element.discontinuous
        ):
            raise ValueError(
                "Deformable contact requires a continuous 3D Hex8 displacement."
            )
        if (
            not np.isfinite(time_increment_ceiling)
            or time_increment_ceiling <= 0
            or not str(name).strip()
        ):
            raise ValueError(
                "A positive finite increment ceiling and a name are required."
            )
        self.ceiling, self.name = float(time_increment_ceiling), str(name)
        xyz = space.tabulate_dof_coordinates()
        maps = []
        for values, trace in (
            (secondary_dofs, operator.secondary),
            (primary_dofs, operator.primary),
        ):
            indices = np.asarray(values)
            if (
                indices.dtype.kind not in "iu"
                or indices.shape != (len(trace.vertices),)
                or np.any(indices < 0)
                or np.any(indices >= len(xyz))
            ):
                raise ValueError(
                    "Contact dof mapping requires valid trace node indices."
                )
            if not np.allclose(xyz[indices], trace.vertices, rtol=0, atol=1e-12):
                raise ValueError("Contact trace geometry differs from mapped dofs.")
            indices = indices.astype(np.int64, copy=True)
            indices.flags.writeable = False
            maps.append(indices)
        if len(np.unique(np.concatenate(maps))) != sum(map(len, maps)):
            raise ValueError(
                "Deformable contact requires disjoint independent trace dofs."
            )
        self.secondary_dofs, self.primary_dofs = maps
        self.identity = content_fingerprint(
            {
                "operator": operator.identity,
                "name": self.name,
                "ceiling": self.ceiling,
                "secondary_dofs": maps[0].tolist(),
                "primary_dofs": maps[1].tolist(),
            }
        )
        self.accepted = self.trial = None
        self._separation_anchor = None
        self._reset_separation_anchor = False
        self.time = self.accepted_time = 0.0
        self.checkpoint_state_required = True

    def _values(self):
        values = self.displacement.x.array.reshape(-1, 3)
        return values[self.secondary_dofs].copy(), values[self.primary_dofs].copy()

    def initialize_accepted_state(self, *, time=0.0):
        if not np.isfinite(time) or time != self.accepted_time:
            raise ValueError(
                "Contact initialization time differs from its accepted state."
            )
        if hasattr(self.base, "initialize_accepted_state"):
            self.base.initialize_accepted_state(time=time)
        if self.accepted is None:
            values = self._values()
            self.accepted = (*values, self.operator.evaluate(*values))

    def update_time(self, time):
        if not np.isfinite(time) or time < self.accepted_time:
            raise ValueError("Contact time cannot precede its accepted state.")
        self.base.update_time(time)
        self.time = float(time)

    def validate_time_increment(self, dt):
        if not np.isfinite(dt) or dt <= 0 or dt > self.ceiling * (1 + 1e-12):
            raise ValueError("Deformable contact caller increment ceiling exceeded.")
        if hasattr(self.base, "validate_time_increment"):
            self.base.validate_time_increment(dt)

    def _add(self, vector, response):
        values = vector.array.reshape(-1, 3)
        values[self.secondary_dofs] += response.secondary_residual
        values[self.primary_dofs] += response.primary_residual

    def assemble_vector(self):
        if getattr(self, '_geometry_executor', None) is not None:
            return self._assemble_overlapped()
        vector = None
        try:
            values = self._values()
            try:
                response = self._evaluate_contact_timed(values)
            finally:
                self._record_contact_time()
            vector = self.base.assemble_vector()
            self._add(vector, response)
            self.trial = (*values, response)
            return vector
        except BaseException:
            if vector is not None:
                vector.destroy()
            self.rollback()
            raise

    def _assemble_overlapped(self):
        from threading import get_ident
        if get_ident() != self._geometry_owner or getattr(self, '_geometry_busy', False):
            raise RuntimeError('Contact overlap requires its exclusive owner thread')
        self._geometry_busy = True
        vector = future = None
        try:
            values = self._values()
            for value in values:
                value.flags.writeable = False
            self._last_contact_seconds = None
            future = self._geometry_executor.submit(self._evaluate_contact_timed, values)
            body_error = None
            try:
                vector = self.base.assemble_vector()
            except BaseException as error:
                body_error = error
            # Preserve contact-before-body failure precedence of serial assembly.
            started = monotonic()
            try:
                response = future.result()
            finally:
                ledger = getattr(self, 'performance', None)
                if ledger is not None:
                    ledger.add('contact_owner_wait', monotonic() - started)
            if body_error is not None:
                raise body_error
            self._add(vector, response)
            self.trial = (*values, response)
            return vector
        except BaseException:
            if future is not None:
                while not future.done():
                    try:
                        future.result()
                    except BaseException:
                        # An owner interrupt must not race geometry rollback.
                        pass
            if vector is not None:
                vector.destroy()
            self.rollback()
            raise
        finally:
            if future is not None:
                # Success and error paths have both joined the geometry worker.
                # Never publish into the shared ledger from the worker itself.
                self._record_contact_time()
            self._geometry_busy = False

    def _evaluate_contact_timed(self, values):
        started = monotonic()
        try:
            return self._evaluate_contact(values)
        finally:
            self._last_contact_seconds = monotonic() - started

    def _record_contact_time(self):
        ledger = getattr(self, 'performance', None)
        elapsed = getattr(self, '_last_contact_seconds', None)
        if ledger is not None and elapsed is not None:
            ledger.add('deformable_contact_inclusive', elapsed)

    def _evaluate_contact(self, values):
        """Geometry-only evaluation from copied displacement arrays.

        Does not access PETSc vectors or commit a physical trial. The operator's
        geometry cache is mutable, so callers must retain exclusive ownership.
        """
        response = (
            self.accepted[2]
            if self.accepted is not None
            and all(
                np.array_equal(a, b)
                for a, b in zip(values, self.accepted[:2], strict=True)
            )
            else self._evaluate_current(values)
        )
        if (
            self.accepted is not None
            and self.operator.geometry_policy == "reviewed_closed_solid"
        ):
            uncertain = unresolved_outside_paths(
                self.accepted[2].path_gaps,
                response.path_gaps,
                (values[0] - self.accepted[0])[self.operator.nodes],
                values[1] - self.accepted[1],
            )
            if np.any(uncertain):
                from .._q1_swept_separation import separated_q1_paths

                execution = self.operator.execution
                separate = (separated_q1_paths if execution is None else
                            lambda *args: execution.separated(*args, reference=separated_q1_paths))

                indices = self.operator.nodes[uncertain]
                separated = separate(
                    self.operator.primary.vertices + self.accepted[1],
                    self.operator.primary.vertices + values[1],
                    self.operator.primary.quadrilaterals,
                    self.operator.secondary.vertices[indices]
                    + self.accepted[0][indices],
                    self.operator.secondary.vertices[indices] + values[0][indices],
                )
                uncertain[np.flatnonzero(uncertain)[separated]] = False
            if np.any(uncertain):
                from .._contact_path_guard import ContactPathResolutionError

                raise ContactPathResolutionError(np.flatnonzero(uncertain))
        return response

    def _evaluate_current(self, values):
        # Only a closed solid supplies signed-distance separation evidence.
        # Keep bounds distinct from fabricated closest points or normals.
        if (
            self.accepted is not None
            and self.operator.geometry_policy == "reviewed_closed_solid"
        ):
            lower = outside_gap_lower_bounds(
                self.accepted[2].gap_lower_bounds,
                (values[0] - self.accepted[0])[self.operator.nodes],
                values[1] - self.accepted[1],
            )
            execution = self.operator.execution
            if execution is not None and execution.separation_reuse == "anchored":
                from .._contact_path_guard import AnchoredSeparationBounds
                nodes = self.operator.nodes
                if self._separation_anchor is None and np.any(self.accepted[2].gap_lower_bounds > 0):
                    self._separation_anchor = AnchoredSeparationBounds(
                        self.accepted[2].gap_lower_bounds, self.accepted[0][nodes], self.accepted[1])
                if self._separation_anchor is not None:
                    anchored = self._separation_anchor.bounds(
                        self.accepted[0][nodes], self.accepted[1], values[0][nodes], values[1])
                    lower = np.maximum(lower, anchored)
                    # Reset only at acceptance, never by committing a speculative
                    # geometry query. An exhausted certificate is not an error.
                    self._reset_separation_anchor = not np.any(anchored > 0)
            if np.all(lower > 0):
                return DeformableContactResponse(
                    np.zeros_like(values[0]),
                    np.zeros_like(values[1]),
                    0.0,
                    None,
                    None,
                    None,
                    lower,
                )
            if np.any(lower > 0):
                return self.operator.evaluate(*values, _path_lower_bounds=lower)
        return self.operator.evaluate(*values)

    def assemble_accepted_vector(self):
        if (
            self.accepted is None
            or self.trial is not None
            or any(
                not np.array_equal(a, b)
                for a, b in zip(self._values(), self.accepted[:2], strict=True)
            )
        ):
            raise RuntimeError("Contact force requires its accepted configuration.")
        vector = self.base.assemble_accepted_vector()
        self._add(vector, self.accepted[2])
        return vector

    def frozen_projection_row_sums(self, nodal_mass):
        """Accepted-contact PSD contribution only, NOT a nonlinear step bound.

        Freeze projection weights/normals and the active set at the accepted
        state. Moving-projection curvature, body stiffness, other contacts and
        damping must not be inferred from this diagnostic contribution.
        """
        from .._penalty_stability import penalty_stencil_row_sums

        if self.accepted is None or self.trial is not None:
            raise RuntimeError("Contact stiffness evidence requires accepted state.")
        mass = np.asarray(nodal_mass)
        expected = self.displacement.x.array.reshape(-1, 3).shape
        if mass.shape != expected or np.iscomplexobj(mass):
            raise ValueError("Contact inertia must match the nodal vector layout.")
        response = self.accepted[2]
        projection = response.projection
        if projection is None:
            nodes = np.empty((0, 5), dtype=np.int64)
            shapes = np.empty((0, 5))
            weights = np.empty(0)
        else:
            indices = (
                np.arange(len(self.operator.nodes))
                if response.projection_indices is None
                else response.projection_indices
            )
            active = projection.signed_gaps < 0
            nodes = np.column_stack(
                (
                    self.secondary_dofs[self.operator.nodes[indices[active]]],
                    self.primary_dofs[response.primary_nodes[active]],
                )
            )
            shapes = np.column_stack(
                (np.ones(active.sum()), -response.primary_weights[active])
            )
            weights = self.operator.weights[indices[active]]
        return penalty_stencil_row_sums(
            nodes, shapes, weights, self.operator.law.penalty, mass
        )

    def commit(self):
        if self.trial is None or any(
            not np.array_equal(a, b)
            for a, b in zip(self._values(), self.trial[:2], strict=True)
        ):
            raise RuntimeError("Contact trial is absent or its configuration changed.")
        self.base.commit()
        self.accepted, self.trial = self.trial, None
        self.accepted_time = self.time
        if self._reset_separation_anchor:
            self._separation_anchor = None
            self._reset_separation_anchor = False

    def rollback(self):
        self.base.rollback()
        self.trial = None
        self.time = self.accepted_time
        self._separation_anchor = None
        self._reset_separation_anchor = False

    def _snapshot(self, method):
        if self.trial is not None:
            raise RuntimeError("Contact checkpoint requires an accepted boundary.")
        return {
            "schema": "agentfem.private-deformable-q1-contact.v1",
            "identity": self.identity,
            "time": self.accepted_time,
            "accepted": None
            if self.accepted is None
            else tuple(v.copy() for v in self.accepted[:2]),
            "base": getattr(self.base, method, self.base.snapshot)(),
        }

    def transaction_snapshot(self):
        return self._snapshot("transaction_snapshot")

    def _increment_field_scope(self, record):
        from .._field_rollback_scope import _increment_field_scope

        return _increment_field_scope(self.base, record["base"])

    def checkpoint_snapshot(self):
        return self._snapshot("checkpoint_snapshot")

    def snapshot(self):
        record = self._snapshot("snapshot")
        if record["accepted"] is not None:
            record["accepted"] = [v.tolist() for v in record["accepted"]]
        return record

    def restore(self, record):
        if (
            not isinstance(record, dict)
            or set(record) != {"schema", "identity", "accepted", "base", "time"}
            or record["schema"] != "agentfem.private-deformable-q1-contact.v1"
            or record["identity"] != self.identity
        ):
            raise ValueError("Deformable contact checkpoint identity differs.")
        stamp = record["time"]
        if (
            not np.isscalar(stamp)
            or not np.isreal(stamp)
            or not np.isfinite(stamp)
            or stamp < 0
            or (stamp > 0 and record["accepted"] is None)
        ):
            raise ValueError(
                "Contact checkpoint lacks its accepted time/configuration."
            )
        restored = None
        if record["accepted"] is not None:
            if (
                not isinstance(record["accepted"], (tuple, list))
                or len(record["accepted"]) != 2
            ):
                raise ValueError("Contact checkpoint requires two trace arrays.")
            values = tuple(np.asarray(v) for v in record["accepted"])
            response = self.operator.evaluate(*values)
            restored = (*(v.copy() for v in values), response)
        self.base.restore(record["base"])
        self.accepted, self.trial = restored, None
        self.time = self.accepted_time = float(stamp)
        # Derived geometry is rebuilt from restored accepted coordinates.
        self._separation_anchor = None
        self._reset_separation_anchor = False

    def checkpoint_capabilities(self):
        return replace(
            self.base.checkpoint_capabilities(),
            rank_count_portability="unsupported",
            state_components=self.base.checkpoint_capabilities().state_components
            + ("accepted deformable contact configuration",),
        )

    def contact_energy_evidence(self):
        if self.accepted is None:
            raise RuntimeError("Deformable contact has no accepted energy.")
        nested = getattr(self.base, "contact_energy_evidence", lambda: ())()
        return (
            *nested,
            {
                "name": self.name,
                "contact_potential_energy": self.accepted[2].potential_energy,
                "contact_friction_dissipation": 0.0,
                "contact_separation_release": 0.0,
                "contact_motion_work": 0.0,
                "moving": False,
            },
        )

    def accepted_contact_observation(self):
        """Selected secondary-node evidence, not unrestricted exterior coverage.

        Residuals are potential gradients; physical contact forces have the
        opposite sign. Moments use current coordinates and a common origin.
        Certified-separated and unresolved zero-gap points have no invented
        closest-point projection or active area.
        """
        if (self.accepted is None or self.trial is not None
                or self.time != self.accepted_time
                or any(not np.array_equal(a, b) for a, b in
                       zip(self._values(), self.accepted[:2], strict=True))):
            raise RuntimeError("Deformable contact observation requires an accepted station")
        secondary, primary, response = self.accepted
        forces = (-response.secondary_residual, -response.primary_residual)
        resultants = [force.sum(axis=0) for force in forces]
        moments = [np.cross(x, force).sum(axis=0)
                   for x, force in zip((secondary, primary), forces, strict=True)]
        projection = response.projection
        projected = 0 if projection is None else len(projection.signed_gaps)
        active = np.zeros(0, dtype=bool) if projection is None else projection.signed_gaps < 0
        indices = (np.arange(projected) if response.projection_indices is None else
                   response.projection_indices)
        area = float(self.operator.weights[indices[active]].sum())
        return {"schema": "agentfem.accepted_contact_observation.v1",
                "name": self.name, "kind": "deformable_contact", "time": self.accepted_time,
                "secondary_force": resultants[0].tolist(), "primary_force": resultants[1].tolist(),
                "secondary_moment": moments[0].tolist(), "primary_moment": moments[1].tolist(),
                "moment_reference_point": [0., 0., 0.],
                "force_balance_residual": (resultants[0] + resultants[1]).tolist(),
                "moment_balance_residual": (moments[0] + moments[1]).tolist(),
                "potential_energy": response.potential_energy,
                "maximum_penetration": (0.0 if projection is None else
                                        float(np.maximum(-projection.signed_gaps, 0).max(initial=0))),
                "candidate_point_count": len(self.operator.nodes),
                "projected_point_count": projected,
                "zero_gap_resolution_count": len(response.zero_gap_indices),
                "certified_separated_point_count": (len(self.operator.nodes)-projected
                                                    -len(response.zero_gap_indices)),
                "active_point_count": int(active.sum()), "active_reference_area": area,
                "integration_measure": "secondary_reference_nodal_area",
                "coverage_scope": "declared_secondary_candidates_only",
                "scientifically_validated": False}

    def summary(self):
        return {
            **self.operator.summary(),
            "name": self.name,
            "identity": self.identity,
            "accepted_potential": None
            if self.accepted is None
            else self.accepted[2].potential_energy,
            "restart": "accepted configuration; deterministic closest-point reconstruction",
            "increment_ceiling_is_certificate": False,
            "outside_crossing_guard": self.operator.geometry_policy
            == "reviewed_closed_solid",
            "continuous_collision_detection": False,
            "base": self.base.summary(),
        }
