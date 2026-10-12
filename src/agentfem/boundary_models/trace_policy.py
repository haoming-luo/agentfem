# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Conservative serial trace reductions for uniform frictionless rigid tools."""
from dataclasses import dataclass, fields, replace
import numpy as np
from .contact_trace import ContactTrace
from .dolfinx_contact_trace import DolfinxContactTraceAdapter


@dataclass(frozen=True, eq=False)
class _BoundedTraceAdapter(DolfinxContactTraceAdapter):
    bound_direction: tuple = (1., 0., 0.)
    displacement_budget: float = 1.

    def current_nodal_positions(self, displacement):
        values = np.asarray(displacement.x.array).reshape(-1, 3)
        projected = values @ np.asarray(self.bound_direction)
        if not np.isfinite(values).all() or np.max(abs(projected), initial=0) > self.displacement_budget:
            raise ValueError("Rigid-tool trace displacement budget exceeded; rebuild the trace")
        return super().current_nodal_positions(displacement)

    def summary(self):
        return {**super().summary(), "bound_direction": self.bound_direction,
                "displacement_budget": self.displacement_budget,
                "broad_phase": "guarded_cylinder_transverse_strip"}


@dataclass(frozen=True)
class RigidContactTracePolicy:
    integration: str = "gauss"
    coalesce_nodes: bool = False
    bound_direction: tuple | None = None
    displacement_budget: float | None = None

    def __post_init__(self):
        if self.integration not in ("gauss", "nodal_lumped") or not isinstance(self.coalesce_nodes, bool):
            raise ValueError("Invalid contact trace integration policy")
        if (self.bound_direction is None) != (self.displacement_budget is None):
            raise ValueError("Trace bound requires both direction and budget")
        if self.coalesce_nodes or self.bound_direction is not None:
            if self.integration != "nodal_lumped":
                raise ValueError("Trace reduction requires nodal_lumped integration")
        if self.bound_direction is not None:
            direction = np.asarray(self.bound_direction, dtype=float)
            budget = self.displacement_budget
            if (direction.shape != (3,) or not np.isfinite(direction).all()
                    or not np.isclose(np.linalg.norm(direction), 1., rtol=0, atol=1e-14)
                    or isinstance(budget, (bool, complex)) or not np.isscalar(budget)
                    or not np.isfinite(budget) or budget <= 0):
                raise ValueError("Trace bound needs a unit direction and positive finite budget")
            object.__setattr__(self, "bound_direction", tuple(float(v) for v in direction))

    def summary(self):
        return {"integration": self.integration, "coalesce_nodes": self.coalesce_nodes,
                "bound_direction": self.bound_direction, "displacement_budget": self.displacement_budget}

    def apply(self, adapter, pair):
        from .rigid import RigidCylinderSurface
        if pair.friction is not None:
            raise NotImplementedError("Trace reductions require uniform frictionless contact")
        if not self.coalesce_nodes and self.bound_direction is None:
            return adapter
        trace = adapter.trace
        if (adapter.communicator.size != 1 or trace._direct_nodes is None
                or trace.measure_configuration != "reference"):
            raise ValueError("Trace reduction requires serial nodal reference-area integration")
        if self.coalesce_nodes:
            nodes, inverse = np.unique(trace._direct_nodes, return_inverse=True)
            weights = np.bincount(inverse, weights=trace.weights, minlength=len(nodes))
            identities = np.full(len(nodes), np.iinfo(np.int64).max, dtype=np.int64)
            np.minimum.at(identities, inverse, trace.point_ids)
            trace = ContactTrace(identities, nodes[:, None], np.ones((len(nodes), 1)), weights,
                measure_configuration="reference", source=trace.source + "_uniform_frictionless_coalesced")
            adapter = replace(adapter, trace=trace, quadrature_rule=adapter.quadrature_rule + "_coalesced")
        if self.bound_direction is None:
            return adapter
        surface, schedule = pair.rigid_body.surface, pair.rigid_body.motion_schedule
        direction = np.asarray(self.bound_direction)
        if not isinstance(surface, RigidCylinderSurface) or surface.admissible_side != "exterior":
            raise NotImplementedError("Transverse trace bound requires an exterior cylinder")
        if np.dot(surface.axis_direction, direction) != 0:
            raise ValueError("Trace bound direction must be perpendicular to cylinder axis")
        if schedule is not None and (np.any(np.asarray(schedule.motion.rotation) != 0)
                or np.dot(schedule.motion.translation, direction) != 0):
            raise ValueError("Tool motion invalidates the transverse trace bound")
        coordinate = adapter.reference_nodal_positions[trace._direct_nodes] @ direction
        center = np.dot(surface.axis_point, direction)
        margin = 256 * np.finfo(float).eps * max(1., float(abs(coordinate).max(initial=0)),
                                                 abs(center), self.displacement_budget)
        selected = abs(coordinate - center) <= surface.radius + self.displacement_budget + margin
        trace = ContactTrace(trace.point_ids[selected], trace.node_ids[selected],
            trace.shape_values[selected], trace.weights[selected],
            measure_configuration="reference", source=trace.source + "_guarded_transverse_budget")
        values = {field.name: getattr(adapter, field.name) for field in fields(DolfinxContactTraceAdapter)}
        return _BoundedTraceAdapter(**{**values, "trace": trace},
            bound_direction=self.bound_direction, displacement_budget=self.displacement_budget)
