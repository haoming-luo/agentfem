# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private serial deformable contact composition for existing central difference."""

from dataclasses import replace

import numpy as np

from .._deformable_q1_contact import DeformableQ1Penalty, DeformableContactResponse
from .._contact_path_guard import unresolved_outside_paths, outside_gap_lower_bounds
from ..fields import unwrap
from ..provenance import content_fingerprint


class DeformableQ1ContactResidual:
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
        vector = None
        try:
            values = self._values()
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
                    raise ValueError(
                        "Potential missed contact crossing: outside endpoints do not "
                        "resolve the swept path; reduce the increment or use a "
                        "validated continuous-contact procedure."
                    )
            vector = self.base.assemble_vector()
            self._add(vector, response)
            self.trial = (*values, response)
            return vector
        except BaseException:
            if vector is not None:
                vector.destroy()
            self.rollback()
            raise

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

    def commit(self):
        if self.trial is None or any(
            not np.array_equal(a, b)
            for a, b in zip(self._values(), self.trial[:2], strict=True)
        ):
            raise RuntimeError("Contact trial is absent or its configuration changed.")
        self.base.commit()
        self.accepted, self.trial = self.trial, None
        self.accepted_time = self.time

    def rollback(self):
        self.base.rollback()
        self.trial = None
        self.time = self.accepted_time

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
