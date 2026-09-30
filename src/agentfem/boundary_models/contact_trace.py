# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral trace interpolation and contact contribution assembly.

The trace contract carries only stable point identity, interpolation, and
integration measure.  It neither searches a master surface nor owns nonlinear
state.  A backend adapter may construct this contract from its native element
and quadrature machinery without exposing backend objects to the contact law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact_friction import PenaltyCoulombFrictionResponse
from .contact_response import FrictionlessPenaltyContactResponse
from .contact_state import ContactProjectionRecord
from .rigid import _readonly_array


def _integer_array(value, *, name: str, dimensions: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != dimensions:
        raise ValueError(f"{name} must be {dimensions}-dimensional.")
    if raw.size and raw.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must contain explicit integers.")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must fit signed 64-bit identity.")
    selected = raw.astype(np.int64, copy=False)
    if np.any(selected < 0):
        raise ValueError(f"{name} must be non-negative.")
    return selected


@dataclass(frozen=True, eq=False)
class ContactTrace:
    """Stable quadrature/interpolation contract for one slave boundary shard.

    ``weights`` already include the backend-selected boundary Jacobian and
    quadrature weight. ``measure_configuration`` states whether that measure is
    reference or current. No hidden geometry update is performed here.
    """

    point_ids: object
    node_ids: object
    shape_values: object
    weights: object
    measure_configuration: str = "reference"
    source: str = "backend_contact_trace"

    def __post_init__(self) -> None:
        point_ids = _integer_array(
            self.point_ids, name="Contact point IDs", dimensions=1
        )
        if np.unique(point_ids).size != point_ids.size:
            raise ValueError("Contact point IDs must be unique.")
        node_ids = _integer_array(self.node_ids, name="Contact node IDs", dimensions=2)
        if node_ids.shape[0] != point_ids.size or node_ids.shape[1] < 1:
            raise ValueError(
                "Contact node IDs must provide at least one node per point."
            )
        if any(np.unique(row).size != row.size for row in node_ids):
            raise ValueError("Contact node IDs must be unique within each point row.")
        shape_values = np.asarray(self.shape_values, dtype=float)
        if shape_values.shape != node_ids.shape or not np.all(
            np.isfinite(shape_values)
        ):
            raise ValueError(
                "Contact shape values must be finite and match the node-ID shape."
            )
        if not np.allclose(
            np.sum(shape_values, axis=1),
            1.0,
            rtol=256.0 * np.finfo(float).eps,
            atol=256.0 * np.finfo(float).eps,
        ):
            raise ValueError("Contact shape values must form a partition of unity.")
        weights = np.asarray(self.weights, dtype=float)
        if weights.shape != (point_ids.size,) or not np.all(np.isfinite(weights)):
            raise ValueError("Contact weights must contain one finite value per point.")
        if np.any(weights <= 0.0):
            raise ValueError("Contact integration weights must be positive.")
        if self.measure_configuration not in {"reference", "current"}:
            raise ValueError(
                "Contact measure_configuration must be 'reference' or 'current'."
            )
        if not str(self.source).strip():
            raise ValueError("Contact trace requires a source description.")
        order = np.argsort(point_ids, kind="stable")
        point_ids = point_ids[order]
        node_ids = node_ids[order]
        shape_values = shape_values[order]
        weights = weights[order]
        object.__setattr__(
            self, "point_ids", _readonly_array(point_ids, dtype=np.int64)
        )
        object.__setattr__(self, "node_ids", _readonly_array(node_ids, dtype=np.int64))
        object.__setattr__(self, "shape_values", _readonly_array(shape_values))
        object.__setattr__(self, "weights", _readonly_array(weights))
        object.__setattr__(self, "source", str(self.source))

    @property
    def point_count(self) -> int:
        return int(self.point_ids.size)

    @property
    def nodes_per_point(self) -> int:
        return int(self.node_ids.shape[1])

    def evaluate(self, nodal_positions) -> ContactTraceEvaluation:
        """Interpolate current slave-point positions from backend nodal values."""

        positions = np.asarray(nodal_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] not in {2, 3}:
            raise ValueError("Contact nodal positions must have shape (nodes, 2 or 3).")
        if not np.all(np.isfinite(positions)):
            raise ValueError("Contact nodal positions must be finite.")
        if self.node_ids.size and int(np.max(self.node_ids)) >= positions.shape[0]:
            raise ValueError("Contact trace references a node outside nodal positions.")
        query_points = np.einsum(
            "qi,qid->qd",
            self.shape_values,
            positions[self.node_ids],
        )
        return ContactTraceEvaluation(
            trace=self,
            query_points=query_points,
            number_of_nodes=int(positions.shape[0]),
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "contact_trace",
            "point_count": self.point_count,
            "nodes_per_point": self.nodes_per_point,
            "measure_configuration": self.measure_configuration,
            "source": self.source,
            "identity": "stable_int64_point_ids",
        }


@dataclass(frozen=True, eq=False)
class ContactTraceAssembly:
    """Integrated residual, potential, and generalized rigid-surface evidence."""

    point_ids: object
    nodal_structural_residual: object
    point_structural_residual_contributions: object
    point_potential_contributions: object
    potential_energy: float
    contact_force_on_structure: object
    contact_force_on_surface: object
    surface_generalized_force: object
    surface_generalized_moment: object | None
    measure_configuration: str

    def __post_init__(self) -> None:
        point_ids = _integer_array(
            self.point_ids, name="Contact point IDs", dimensions=1
        )
        if np.unique(point_ids).size != point_ids.size:
            raise ValueError("Contact point IDs must be unique.")
        nodal = np.asarray(self.nodal_structural_residual, dtype=float)
        point_residual = np.asarray(
            self.point_structural_residual_contributions,
            dtype=float,
        )
        point_potential = np.asarray(self.point_potential_contributions, dtype=float)
        if nodal.ndim != 2 or nodal.shape[1] not in {2, 3}:
            raise ValueError("Contact nodal residual must have shape (nodes, 2 or 3).")
        dimension = int(nodal.shape[1])
        if point_residual.shape != (point_ids.size, dimension):
            raise ValueError("Contact point residual contributions have wrong shape.")
        if point_potential.shape != (point_ids.size,):
            raise ValueError("Contact point potential contributions have wrong shape.")
        if not (
            np.all(np.isfinite(nodal))
            and np.all(np.isfinite(point_residual))
            and np.all(np.isfinite(point_potential))
        ):
            raise ValueError("Contact trace assembly quantities must be finite.")
        potential = float(self.potential_energy)
        if not np.isfinite(potential) or potential < 0.0:
            raise ValueError(
                "Contact potential energy must be finite and non-negative."
            )
        if self.measure_configuration not in {"reference", "current"}:
            raise ValueError(
                "Contact measure_configuration must be 'reference' or 'current'."
            )
        vectors = {}
        for name in (
            "contact_force_on_structure",
            "contact_force_on_surface",
            "surface_generalized_force",
        ):
            selected = np.asarray(getattr(self, name), dtype=float)
            if selected.shape != (dimension,) or not np.all(np.isfinite(selected)):
                raise ValueError(f"{name} must be one finite spatial vector.")
            vectors[name] = selected
        moment = self.surface_generalized_moment
        if moment is not None:
            selected_moment = np.asarray(moment, dtype=float)
            expected = (1,) if dimension == 2 else (3,)
            if selected_moment.shape != expected or not np.all(
                np.isfinite(selected_moment)
            ):
                raise ValueError(
                    "Surface generalized moment has the wrong spatial shape."
                )
            object.__setattr__(
                self,
                "surface_generalized_moment",
                _readonly_array(selected_moment),
            )
        if not np.allclose(
            vectors["contact_force_on_structure"] + vectors["contact_force_on_surface"],
            0.0,
            rtol=256.0 * np.finfo(float).eps,
            atol=256.0 * np.finfo(float).eps,
        ):
            raise ValueError("Integrated contact action and reaction do not balance.")
        if not np.allclose(
            vectors["surface_generalized_force"],
            vectors["contact_force_on_structure"],
            rtol=256.0 * np.finfo(float).eps,
            atol=256.0 * np.finfo(float).eps,
        ):
            raise ValueError("Surface generalized force violates the sign contract.")
        if not np.allclose(
            potential,
            np.sum(point_potential),
            rtol=256.0 * np.finfo(float).eps,
            atol=256.0 * np.finfo(float).eps,
        ):
            raise ValueError("Contact potential does not equal point contributions.")
        if not np.allclose(
            vectors["contact_force_on_structure"],
            -np.sum(point_residual, axis=0),
            rtol=256.0 * np.finfo(float).eps,
            atol=256.0 * np.finfo(float).eps,
        ):
            raise ValueError("Contact force does not equal the integrated residual.")
        if not np.allclose(
            np.sum(nodal, axis=0),
            np.sum(point_residual, axis=0),
            rtol=256.0 * np.finfo(float).eps,
            atol=256.0 * np.finfo(float).eps,
        ):
            raise ValueError(
                "Nodal residual does not preserve the integrated residual."
            )
        order = np.argsort(point_ids, kind="stable")
        point_ids = point_ids[order]
        point_residual = point_residual[order]
        point_potential = point_potential[order]
        object.__setattr__(
            self, "point_ids", _readonly_array(point_ids, dtype=np.int64)
        )
        object.__setattr__(self, "nodal_structural_residual", _readonly_array(nodal))
        object.__setattr__(
            self,
            "point_structural_residual_contributions",
            _readonly_array(point_residual),
        )
        object.__setattr__(
            self,
            "point_potential_contributions",
            _readonly_array(point_potential),
        )
        object.__setattr__(self, "potential_energy", potential)
        for name, value in vectors.items():
            object.__setattr__(self, name, _readonly_array(value))

    def summary(self) -> dict[str, object]:
        return {
            "kind": "contact_trace_assembly",
            "point_count": int(self.point_ids.size),
            "potential_energy": self.potential_energy,
            "contact_force_norm": float(
                np.linalg.norm(self.contact_force_on_structure)
            ),
            "moment_available": self.surface_generalized_moment is not None,
            "measure_configuration": self.measure_configuration,
            "linearization": "not_provided",
        }


@dataclass(frozen=True, eq=False)
class FrictionContactTraceAssembly:
    """Integrated tangential response and its distinct energy channels."""

    mechanical: ContactTraceAssembly
    point_dissipation_increment_contributions: object
    point_cumulative_dissipation_contributions: object
    point_separation_release_increment_contributions: object
    point_cumulative_separation_release_contributions: object
    dissipation_increment: float
    cumulative_dissipation: float
    separation_release_increment: float
    cumulative_separation_release: float

    def __post_init__(self) -> None:
        if not isinstance(self.mechanical, ContactTraceAssembly):
            raise TypeError(
                "Friction trace assembly requires ContactTraceAssembly mechanics."
            )
        count = int(self.mechanical.point_ids.size)
        pairs = (
            (
                "point_dissipation_increment_contributions",
                "dissipation_increment",
            ),
            (
                "point_cumulative_dissipation_contributions",
                "cumulative_dissipation",
            ),
            (
                "point_separation_release_increment_contributions",
                "separation_release_increment",
            ),
            (
                "point_cumulative_separation_release_contributions",
                "cumulative_separation_release",
            ),
        )
        for point_name, total_name in pairs:
            point_values = np.asarray(getattr(self, point_name), dtype=float)
            if point_values.shape != (count,) or not np.all(np.isfinite(point_values)):
                raise ValueError(
                    f"{point_name} must contain one finite value per trace point."
                )
            if np.any(point_values < 0.0):
                raise ValueError(f"{point_name} cannot be negative.")
            total = float(getattr(self, total_name))
            if not np.isfinite(total) or total < 0.0:
                raise ValueError(f"{total_name} must be finite and non-negative.")
            if not np.isclose(
                total,
                float(np.sum(point_values)),
                rtol=256.0 * np.finfo(float).eps,
                atol=256.0 * np.finfo(float).eps,
            ):
                raise ValueError(f"{total_name} does not equal its point contributions.")
            object.__setattr__(self, point_name, _readonly_array(point_values))
            object.__setattr__(self, total_name, total)

    @property
    def point_ids(self) -> np.ndarray:
        return self.mechanical.point_ids

    @property
    def nodal_structural_residual(self) -> np.ndarray:
        return self.mechanical.nodal_structural_residual

    @property
    def contact_force_on_structure(self) -> np.ndarray:
        return self.mechanical.contact_force_on_structure

    @property
    def contact_force_on_surface(self) -> np.ndarray:
        return self.mechanical.contact_force_on_surface

    @property
    def surface_generalized_moment(self) -> np.ndarray | None:
        return self.mechanical.surface_generalized_moment

    @property
    def recoverable_penalty_energy(self) -> float:
        return self.mechanical.potential_energy

    def summary(self) -> dict[str, object]:
        return {
            "kind": "friction_contact_trace_assembly",
            "mechanical": self.mechanical.summary(),
            "recoverable_penalty_energy": self.recoverable_penalty_energy,
            "dissipation_increment": self.dissipation_increment,
            "cumulative_dissipation": self.cumulative_dissipation,
            "separation_release_increment": self.separation_release_increment,
            "cumulative_separation_release": self.cumulative_separation_release,
            "energy_semantics": "recoverable_dissipated_release_separate",
        }


@dataclass(frozen=True, eq=False)
class ContactTraceEvaluation:
    """Current slave-point positions bound to one immutable trace contract."""

    trace: ContactTrace
    query_points: object
    number_of_nodes: int

    def __post_init__(self) -> None:
        if not isinstance(self.trace, ContactTrace):
            raise TypeError("Contact trace evaluation requires ContactTrace.")
        query = np.asarray(self.query_points, dtype=float)
        if (
            query.ndim != 2
            or query.shape[1] not in {2, 3}
            or not np.all(np.isfinite(query))
        ):
            raise ValueError("Contact query points must be finite 2D or 3D vectors.")
        if query.shape[0] != self.trace.point_count:
            raise ValueError("Contact query points have the wrong trace point count.")
        count = int(self.number_of_nodes)
        if count < 0:
            raise ValueError("Contact trace node count must be non-negative.")
        if self.trace.node_ids.size and int(np.max(self.trace.node_ids)) >= count:
            raise ValueError("Contact trace evaluation has an insufficient node count.")
        object.__setattr__(self, "query_points", _readonly_array(query))
        object.__setattr__(self, "number_of_nodes", count)

    def assemble(
        self,
        record: ContactProjectionRecord,
        response: FrictionlessPenaltyContactResponse,
        *,
        surface_reference_point=None,
    ) -> ContactTraceAssembly:
        """Integrate one matching local response through the trace contract."""

        if not isinstance(record, ContactProjectionRecord):
            raise TypeError("Contact trace assembly requires a projection record.")
        if not isinstance(response, FrictionlessPenaltyContactResponse):
            raise TypeError("Contact trace assembly requires a penalty response.")
        if not np.array_equal(record.point_ids, self.trace.point_ids):
            raise ValueError("Contact projection identity differs from this trace.")
        projection = record.projection
        if response.projection is not projection:
            raise ValueError("Contact response and projection record differ.")
        if projection.point_count != self.trace.point_count or not np.array_equal(
            projection.query_points,
            self.query_points,
        ):
            raise ValueError(
                "Contact response was not evaluated at this trace evaluation."
            )
        return self._assemble_fields(
            record,
            structural_residual_tractions=response.structural_residual_tractions,
            surface_generalized_tractions=response.surface_generalized_tractions,
            potential_densities=response.potential_densities,
            dimension=response.dimension,
            valid=response.projection.valid,
            surface_reference_point=surface_reference_point,
        )

    def assemble_friction(
        self,
        record: ContactProjectionRecord,
        response: PenaltyCoulombFrictionResponse,
        *,
        surface_reference_point=None,
    ) -> FrictionContactTraceAssembly:
        """Integrate one matching tangential return-map response."""

        if not isinstance(record, ContactProjectionRecord):
            raise TypeError("Friction trace assembly requires a projection record.")
        if not isinstance(response, PenaltyCoulombFrictionResponse):
            raise TypeError("Friction trace assembly requires a friction response.")
        if not np.array_equal(
            record.point_ids,
            self.trace.point_ids,
        ) or not np.array_equal(response.record.point_ids, self.trace.point_ids):
            raise ValueError("Friction projection or state identity differs from this trace.")
        projection = record.projection
        if projection.point_count != self.trace.point_count or not np.array_equal(
            projection.query_points,
            self.query_points,
        ):
            raise ValueError("Friction response was not evaluated at this trace evaluation.")
        if not np.allclose(
            response.record.normals,
            projection.normals,
            rtol=1.0e-12,
            atol=1.0e-14,
        ):
            raise ValueError("Friction state normals differ from the current projection.")
        mechanical = self._assemble_fields(
            record,
            structural_residual_tractions=response.structural_residual_tractions,
            surface_generalized_tractions=response.surface_generalized_tractions,
            potential_densities=response.recoverable_penalty_energy_densities,
            dimension=response.record.dimension,
            valid=projection.valid,
            surface_reference_point=surface_reference_point,
        )
        weights = self.trace.weights
        point_dissipation = response.dissipation_increment_densities * weights
        point_cumulative_dissipation = (
            response.record.cumulative_dissipation_densities * weights
        )
        point_release = response.separation_release_densities * weights
        point_cumulative_release = (
            response.record.cumulative_separation_release_densities * weights
        )
        return FrictionContactTraceAssembly(
            mechanical=mechanical,
            point_dissipation_increment_contributions=point_dissipation,
            point_cumulative_dissipation_contributions=(
                point_cumulative_dissipation
            ),
            point_separation_release_increment_contributions=point_release,
            point_cumulative_separation_release_contributions=(
                point_cumulative_release
            ),
            dissipation_increment=float(np.sum(point_dissipation)),
            cumulative_dissipation=float(np.sum(point_cumulative_dissipation)),
            separation_release_increment=float(np.sum(point_release)),
            cumulative_separation_release=float(np.sum(point_cumulative_release)),
        )

    def _assemble_fields(
        self,
        record: ContactProjectionRecord,
        *,
        structural_residual_tractions,
        surface_generalized_tractions,
        potential_densities,
        dimension: int,
        valid,
        surface_reference_point,
    ) -> ContactTraceAssembly:
        """Integrate reviewed point fields without taking law ownership."""

        weights = self.trace.weights
        structural = np.asarray(structural_residual_tractions, dtype=float)
        surface = np.asarray(surface_generalized_tractions, dtype=float)
        potential = np.asarray(potential_densities, dtype=float)
        expected_vectors = (self.trace.point_count, int(dimension))
        if structural.shape != expected_vectors or surface.shape != expected_vectors:
            raise ValueError("Contact point traction fields have the wrong shape.")
        if potential.shape != (self.trace.point_count,):
            raise ValueError("Contact point potential field has the wrong shape.")
        selected_valid = np.asarray(valid, dtype=bool)
        if selected_valid.shape != (self.trace.point_count,):
            raise ValueError("Contact validity field has the wrong shape.")
        if not (
            np.all(np.isfinite(structural))
            and np.all(np.isfinite(surface))
            and np.all(np.isfinite(potential))
        ):
            raise ValueError("Contact point fields must be finite before assembly.")
        point_residual = structural * weights[:, None]
        point_potential = potential * weights
        nodal_residual = np.zeros(
            (self.number_of_nodes, int(dimension)),
            dtype=float,
        )
        for local_node in range(self.trace.nodes_per_point):
            contribution = self.trace.shape_values[:, local_node, None] * point_residual
            np.add.at(
                nodal_residual,
                self.trace.node_ids[:, local_node],
                contribution,
            )
        surface_generalized = np.sum(
            surface * weights[:, None],
            axis=0,
        )
        force_on_structure = -np.sum(point_residual, axis=0)
        force_on_surface = -surface_generalized
        moment = None
        if surface_reference_point is not None:
            reference = np.asarray(surface_reference_point, dtype=float).reshape(-1)
            if reference.shape != (int(dimension),) or not np.all(
                np.isfinite(reference)
            ):
                raise ValueError(
                    "Surface reference point must match the spatial dimension."
                )
            point_surface_generalized = surface * weights[:, None]
            arms = record.projection.closest_points[selected_valid] - reference
            if int(dimension) == 2:
                vectors = point_surface_generalized[selected_valid]
                moment = np.asarray(
                    [np.sum(arms[:, 0] * vectors[:, 1] - arms[:, 1] * vectors[:, 0])]
                )
            else:
                moment = np.sum(
                    np.cross(arms, point_surface_generalized[selected_valid]),
                    axis=0,
                )
        return ContactTraceAssembly(
            point_ids=self.trace.point_ids,
            nodal_structural_residual=nodal_residual,
            point_structural_residual_contributions=point_residual,
            point_potential_contributions=point_potential,
            potential_energy=float(np.sum(point_potential)),
            contact_force_on_structure=force_on_structure,
            contact_force_on_surface=force_on_surface,
            surface_generalized_force=surface_generalized,
            surface_generalized_moment=moment,
            measure_configuration=self.trace.measure_configuration,
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "contact_trace_evaluation",
            "point_count": self.trace.point_count,
            "dimension": int(self.query_points.shape[1]),
            "number_of_nodes": self.number_of_nodes,
            "measure_configuration": self.trace.measure_configuration,
        }


__all__ = [
    "ContactTrace",
    "ContactTraceAssembly",
    "ContactTraceEvaluation",
    "FrictionContactTraceAssembly",
]
