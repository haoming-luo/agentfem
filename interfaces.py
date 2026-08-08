"""Experimental constitutive contracts for material interfaces.

The first public slice is deliberately independent of DOLFINx assembly.  It
defines the local, irreversible traction--separation response that a paired
facet element consumes.  Keeping this material-point contract separate from
mesh topology and time integration makes its energy, rollback, and restart
semantics testable before it is used in a dynamic fracture calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class CohesiveResponse:
    """One Mode-I traction--separation update.

    ``traction`` is positive in opening and negative in compression.
    ``stored_energy`` and ``dissipated_energy`` are energies per undeformed
    interface area.  The latter is cumulative and irreversible.
    """

    opening: np.ndarray
    traction: np.ndarray
    tangent: np.ndarray
    maximum_opening: np.ndarray
    damage: np.ndarray
    stored_energy: np.ndarray
    dissipated_energy: np.ndarray


@dataclass(frozen=True)
class BilinearCohesiveLaw:
    """Irreversible bilinear Mode-I cohesive law.

    The virgin response reaches ``strength`` at ``peak_opening`` and then
    softens linearly to zero at ``failure_opening``.  Consequently, the exact
    area under the monotonic envelope is ``fracture_energy``.

    Unloading and reloading are secant-linear through the origin.  Compressive
    closure uses a separate penalty stiffness and does not heal or advance
    tensile damage.
    """

    strength: float
    fracture_energy: float
    initial_stiffness: float
    compression_stiffness: float | None = None
    name: str = "bilinear Mode-I cohesive law"

    def __post_init__(self) -> None:
        for field_name in ("strength", "fracture_energy", "initial_stiffness"):
            value = float(getattr(self, field_name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive.")
        if self.compression_stiffness is not None and (
            not isfinite(float(self.compression_stiffness))
            or float(self.compression_stiffness) <= 0.0
        ):
            raise ValueError("compression_stiffness must be finite and positive.")
        if self.failure_opening <= self.peak_opening:
            minimum = self.strength**2 / (2.0 * self.initial_stiffness)
            raise ValueError(
                "The requested fracture energy is too small for the declared "
                "strength and initial stiffness. It must exceed "
                f"strength**2/(2*initial_stiffness) = {minimum:.16g}."
            )

    @property
    def peak_opening(self) -> float:
        """Opening at peak traction."""

        return float(self.strength / self.initial_stiffness)

    @property
    def failure_opening(self) -> float:
        """Opening at complete tensile decohesion."""

        return float(2.0 * self.fracture_energy / self.strength)

    @property
    def closure_stiffness(self) -> float:
        return float(
            self.initial_stiffness
            if self.compression_stiffness is None
            else self.compression_stiffness
        )

    def characteristic_length(self, elastic_modulus: float) -> float:
        """Return the declared cohesive length ``E*Gamma/strength^2``."""

        modulus = float(elastic_modulus)
        if not isfinite(modulus) or modulus <= 0.0:
            raise ValueError("elastic_modulus must be finite and positive.")
        return modulus * self.fracture_energy / self.strength**2

    def envelope_traction(self, opening) -> np.ndarray:
        """Return the monotonic tensile envelope traction."""

        value = _finite_array(opening, name="opening")
        positive = np.maximum(value, 0.0)
        d0 = self.peak_opening
        df = self.failure_opening
        traction = np.where(
            positive <= d0,
            self.initial_stiffness * positive,
            self.strength * np.maximum(df - positive, 0.0) / (df - d0),
        )
        return np.where(positive >= df, 0.0, traction)

    def envelope_work(self, opening) -> np.ndarray:
        """Integrate the monotonic envelope exactly up to ``opening``."""

        value = _finite_array(opening, name="opening")
        selected = np.clip(value, 0.0, self.failure_opening)
        d0 = self.peak_opening
        df = self.failure_opening
        elastic = 0.5 * self.initial_stiffness * selected**2
        softening = (
            0.5 * self.strength * d0
            + self.strength
            / (df - d0)
            * (df * (selected - d0) - 0.5 * (selected**2 - d0**2))
        )
        work = np.where(selected <= d0, elastic, softening)
        return np.where(value >= df, self.fracture_energy, work)

    def damage_from_maximum(self, maximum_opening) -> np.ndarray:
        """Return the secant damage associated with a maximum opening."""

        maximum = np.maximum(
            _finite_array(maximum_opening, name="maximum_opening"), 0.0
        )
        d0 = self.peak_opening
        df = self.failure_opening
        denominator = np.maximum(maximum * (df - d0), np.finfo(float).tiny)
        softening = df * (maximum - d0) / denominator
        return np.where(
            maximum <= d0,
            0.0,
            np.where(maximum >= df, 1.0, np.clip(softening, 0.0, 1.0)),
        )

    def update(self, opening, committed_maximum=0.0) -> CohesiveResponse:
        """Evaluate a trial state without mutating committed history."""

        value = _finite_array(opening, name="opening")
        committed = _finite_array(committed_maximum, name="committed_maximum")
        value, committed = np.broadcast_arrays(value, committed)
        if np.any(committed < 0.0):
            raise ValueError("committed_maximum cannot be negative.")

        maximum = np.maximum(committed, np.maximum(value, 0.0))
        damage = self.damage_from_maximum(maximum)
        tensile_stiffness = (1.0 - damage) * self.initial_stiffness
        is_compression = value < 0.0
        is_new_loading = value > committed
        traction = np.where(
            is_compression,
            self.closure_stiffness * value,
            tensile_stiffness * value,
        )
        envelope_tangent = np.where(
            value <= self.peak_opening,
            self.initial_stiffness,
            np.where(
                value < self.failure_opening,
                -self.strength / (self.failure_opening - self.peak_opening),
                0.0,
            ),
        )
        tangent = np.where(
            is_compression,
            self.closure_stiffness,
            np.where(is_new_loading, envelope_tangent, tensile_stiffness),
        )
        tensile_opening = np.maximum(value, 0.0)
        stored = np.where(
            is_compression,
            0.5 * self.closure_stiffness * value**2,
            0.5 * tensile_stiffness * tensile_opening**2,
        )

        envelope = self.envelope_traction(maximum)
        maximum_stored = 0.5 * envelope * maximum
        dissipated = np.maximum(
            self.envelope_work(maximum) - maximum_stored,
            0.0,
        )
        return CohesiveResponse(
            opening=np.array(value, dtype=float, copy=True),
            traction=np.asarray(traction, dtype=float),
            tangent=np.asarray(tangent, dtype=float),
            maximum_opening=np.asarray(maximum, dtype=float),
            damage=np.asarray(damage, dtype=float),
            stored_energy=np.asarray(stored, dtype=float),
            dissipated_energy=np.asarray(dissipated, dtype=float),
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": "cohesive_traction_separation",
            "mode": "normal",
            "envelope": "bilinear",
            "strength": self.strength,
            "fracture_energy": self.fracture_energy,
            "initial_stiffness": self.initial_stiffness,
            "compression_stiffness": self.closure_stiffness,
            "peak_opening": self.peak_opening,
            "failure_opening": self.failure_opening,
            "characteristic_length_definition": "E*Gamma/strength^2",
            "state": ["maximum_opening", "damage", "dissipated_energy"],
            "maturity": "experimental_material_point",
        }


class CohesiveTransaction:
    """Trial/commit/rollback state for a batch of cohesive points."""

    def __init__(self, law: BilinearCohesiveLaw, size: int):
        if int(size) <= 0:
            raise ValueError("CohesiveTransaction.size must be positive.")
        self.law = law
        self._committed_maximum = np.zeros(int(size), dtype=float)
        self._trial: CohesiveResponse | None = None

    @property
    def size(self) -> int:
        return int(self._committed_maximum.size)

    @property
    def committed_maximum(self) -> np.ndarray:
        return self._committed_maximum.copy()

    @property
    def trial(self) -> CohesiveResponse | None:
        return self._trial

    def begin(self, opening) -> CohesiveResponse:
        """Create a replaceable trial state from committed history."""

        values = _finite_array(opening, name="opening")
        if values.shape != self._committed_maximum.shape:
            raise ValueError(
                f"opening must have shape {self._committed_maximum.shape}, "
                f"got {values.shape}."
            )
        self._trial = self.law.update(values, self._committed_maximum)
        return self._trial

    def commit(self) -> None:
        if self._trial is None:
            raise RuntimeError("No cohesive trial state is available to commit.")
        self._committed_maximum[:] = self._trial.maximum_opening
        self._trial = None

    def rollback(self) -> None:
        """Discard the trial state without changing committed history."""

        self._trial = None

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": "agentfem.cohesive-state.v1",
            "law": self.law.summary(),
            "maximum_opening": self._committed_maximum.tolist(),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        if snapshot.get("schema") != "agentfem.cohesive-state.v1":
            raise ValueError("Unsupported cohesive-state schema.")
        values = _finite_array(snapshot.get("maximum_opening"), name="maximum_opening")
        if values.shape != self._committed_maximum.shape:
            raise ValueError("Cohesive-state size does not match this transaction.")
        if np.any(values < 0.0):
            raise ValueError("Cohesive maximum opening cannot be negative.")
        self._committed_maximum[:] = values
        self._trial = None

    def initialize(self, maximum_opening) -> None:
        """Set an initial intact or pre-debonded state before execution."""

        if self._trial is not None:
            raise RuntimeError("Rollback the cohesive trial state before initialization.")
        values = _finite_array(maximum_opening, name="maximum_opening")
        values = np.broadcast_to(values, self._committed_maximum.shape)
        if np.any(values < 0.0):
            raise ValueError("Initial cohesive maximum opening cannot be negative.")
        self._committed_maximum[:] = values


@dataclass(frozen=True)
class PairedLineFacets:
    """Deterministically paired zero-thickness line facets for a 2D mesh.

    Positive-side node order is permuted to match the negative-side geometry.
    ``normals`` point in the caller-declared direction from the negative side
    toward the positive side.  Coincident geometry alone cannot infer that
    direction, hence ``normal_hint`` is mandatory in the constructor helper.
    """

    negative_nodes: np.ndarray
    positive_nodes: np.ndarray
    normals: np.ndarray
    lengths: np.ndarray
    tolerance: float
    facet_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        negative = np.asarray(self.negative_nodes, dtype=np.int64)
        positive = np.asarray(self.positive_nodes, dtype=np.int64)
        normals = np.asarray(self.normals, dtype=float)
        lengths = np.asarray(self.lengths, dtype=float).reshape(-1)
        tolerance = float(self.tolerance)
        if negative.ndim != 2 or negative.shape[1] != 2:
            raise ValueError("PairedLineFacets negative_nodes must have shape (facets, 2).")
        if positive.shape != negative.shape:
            raise ValueError("PairedLineFacets positive_nodes must match negative_nodes.")
        if normals.shape[0] != negative.shape[0] or normals.ndim != 2:
            raise ValueError("PairedLineFacets normals must provide one vector per facet.")
        if (
            lengths.shape != (negative.shape[0],)
            or np.any(~np.isfinite(lengths))
            or np.any(lengths <= 0.0)
        ):
            raise ValueError("PairedLineFacets lengths must be positive per facet.")
        if np.any(~np.isfinite(normals)):
            raise ValueError("PairedLineFacets normals must be finite.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("PairedLineFacets tolerance must be finite and positive.")
        keys = tuple(str(value) for value in self.facet_keys)
        if keys and (len(keys) != negative.shape[0] or len(set(keys)) != len(keys)):
            raise ValueError("PairedLineFacets facet_keys must be unique per facet.")
        object.__setattr__(self, "negative_nodes", negative.copy())
        object.__setattr__(self, "positive_nodes", positive.copy())
        object.__setattr__(self, "normals", normals.copy())
        object.__setattr__(self, "lengths", lengths.copy())
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "facet_keys", keys)

    @property
    def number_of_facets(self) -> int:
        return int(self.negative_nodes.shape[0])

    @property
    def number_of_points(self) -> int:
        return 2 * self.number_of_facets

    def summary(self) -> dict[str, object]:
        return {
            "kind": "paired_line_facets",
            "number_of_facets": self.number_of_facets,
            "quadrature_points_per_facet": 2,
            "reference_length": float(np.sum(self.lengths)),
            "pairing_tolerance": self.tolerance,
            "dof_sides": "independent",
            "state_identity": self.identity(),
        }

    def identity(self) -> dict[str, object]:
        """Return the durable identity used by irreversible interface state.

        Physical keys created by :func:`pair_coincident_line_facets` are
        independent of DOLFINx dof numbering.  Directly constructed legacy
        topologies retain an explicit node-order-scoped fallback.
        """

        if self.facet_keys:
            keys = self.facet_keys
            scope = "ordered_reference_facet_geometry"
        else:
            keys = tuple(
                ":".join(str(int(value)) for value in (*negative, *positive))
                for negative, positive in zip(
                    self.negative_nodes,
                    self.positive_nodes,
                    strict=True,
                )
            )
            scope = "legacy_node_order"
        digest = sha256()
        for key in keys:
            digest.update(key.encode("utf-8"))
            digest.update(b"\0")
        digest.update(
            np.rint(self.normals / self.tolerance).astype("<i8").tobytes()
        )
        digest.update(
            np.rint(self.lengths / self.tolerance).astype("<i8").tobytes()
        )
        return {
            "schema": "agentfem.cohesive-interface-identity.v1",
            "sha256": digest.hexdigest(),
            "scope": scope,
            "number_of_facets": self.number_of_facets,
            "quadrature_points_per_facet": 2,
            "pairing_tolerance": self.tolerance,
            "facet_keys": list(keys),
            "orientation_sensitive": True,
        }


@dataclass(frozen=True)
class SplitInterfaceMesh:
    """Array-level result of splitting one conforming 2D interface path."""

    coordinates: np.ndarray
    cells: np.ndarray
    negative_facets: np.ndarray
    positive_facets: np.ndarray
    original_to_duplicate: dict[int, int]
    positive_cells: np.ndarray

    def summary(self) -> dict[str, object]:
        return {
            "kind": "split_zero_thickness_interface_mesh",
            "number_of_cells": int(self.cells.shape[0]),
            "number_of_original_interface_nodes": len(self.original_to_duplicate),
            "number_of_interface_facets": int(self.negative_facets.shape[0]),
            "independent_sides": True,
        }


def create_dolfinx_split_mesh(
    split: SplitInterfaceMesh,
    *,
    comm=None,
    cell_type: str | None = None,
    input_order: str = "counterclockwise",
):
    """Create the first executable DOLFINx mesh for a split 2D interface.

    ``split_conforming_line_interface`` deliberately operates on plain arrays
    so imported meshes can be audited before a solver owns them.  This adapter
    is the corresponding execution boundary.  ``input_order`` is explicit
    because conventional CAE quadrilaterals enumerate vertices around the
    perimeter whereas the Basix reference quadrilateral uses tensor-product
    vertex order.

    The first cohesive global consumer is serial by design.  Distributed
    interface ownership needs a separate, tested identity contract; silently
    partitioning coincident interface nodes here would make restart and
    irreversible state ambiguous.
    """

    import basix.ufl
    import ufl
    from dolfinx import mesh as dolfinx_mesh
    from mpi4py import MPI

    if not isinstance(split, SplitInterfaceMesh):
        raise TypeError("create_dolfinx_split_mesh requires SplitInterfaceMesh.")
    selected_comm = MPI.COMM_SELF if comm is None else comm
    if selected_comm.size != 1:
        raise NotImplementedError(
            "Split cohesive mesh execution is serial-only until deterministic "
            "distributed facet ownership and state identity are verified."
        )
    nodes_per_cell = int(split.cells.shape[1])
    inferred = {3: "triangle", 4: "quadrilateral"}.get(nodes_per_cell)
    selected_cell = inferred if cell_type is None else str(cell_type).strip().lower()
    if selected_cell not in {"triangle", "quadrilateral"}:
        raise ValueError(
            "The first split-interface adapter supports triangle or quadrilateral cells."
        )
    expected_nodes = 3 if selected_cell == "triangle" else 4
    if nodes_per_cell != expected_nodes:
        raise ValueError(
            f"cell_type={selected_cell!r} requires {expected_nodes} nodes per cell."
        )
    ordering = str(input_order).strip().lower().replace("-", "_")
    if ordering not in {"counterclockwise", "dolfinx"}:
        raise ValueError("input_order must be 'counterclockwise' or 'dolfinx'.")
    cells = np.asarray(split.cells, dtype=np.int64).copy()
    if selected_cell == "quadrilateral" and ordering == "counterclockwise":
        cells = cells[:, [0, 1, 3, 2]]
    coordinate_element = basix.ufl.element(
        "Lagrange",
        selected_cell,
        1,
        shape=(2,),
    )
    domain = dolfinx_mesh.create_mesh(
        selected_comm,
        cells,
        ufl.Mesh(coordinate_element),
        np.asarray(split.coordinates, dtype=float),
    )
    # The input indices are the durable bridge from the audited array mesh to
    # the solver mesh, including two distinct ids at coincident coordinates.
    expected = set(range(split.coordinates.shape[0]))
    retained = set(np.asarray(domain.geometry.input_global_indices, dtype=int))
    if retained != expected:
        raise RuntimeError(
            "DOLFINx did not retain every split input-node identity; cohesive "
            "DOF recovery would be unsafe."
        )
    return domain


@dataclass(frozen=True)
class CohesiveFacetResponse:
    """Trial force and energy from a batch of paired 2D facets."""

    internal_force: np.ndarray
    opening: np.ndarray
    traction: np.ndarray
    damage: np.ndarray
    stored_energy: float
    dissipated_energy: float


class ModeICohesiveFacetAssembler:
    """Two-point line integration for a fixed-path 2D Mode-I interface.

    This is an assembly kernel, not a mesh adapter.  Node numbers refer to a
    coordinate/displacement array in which the two coincident sides already
    own independent rows.  A future DOLFINx adapter maps these nodal forces to
    distributed vector degrees of freedom and assigns every facet pair one
    deterministic MPI owner.
    """

    _GAUSS = np.array(
        [
            [0.5 * (1.0 + 1.0 / np.sqrt(3.0)), 0.5 * (1.0 - 1.0 / np.sqrt(3.0))],
            [0.5 * (1.0 - 1.0 / np.sqrt(3.0)), 0.5 * (1.0 + 1.0 / np.sqrt(3.0))],
        ],
        dtype=float,
    )

    def __init__(
        self,
        topology: PairedLineFacets,
        law: BilinearCohesiveLaw,
        *,
        number_of_nodes: int,
        thickness: float = 1.0,
    ):
        if int(number_of_nodes) <= 0:
            raise ValueError("number_of_nodes must be positive.")
        if not isfinite(float(thickness)) or float(thickness) <= 0.0:
            raise ValueError("thickness must be finite and positive.")
        largest = int(
            max(np.max(topology.negative_nodes), np.max(topology.positive_nodes))
        )
        if largest >= int(number_of_nodes):
            raise ValueError("Paired facet node number exceeds number_of_nodes.")
        self.topology = topology
        self.law = law
        self.number_of_nodes = int(number_of_nodes)
        self.thickness = float(thickness)
        self.state = CohesiveTransaction(law, topology.number_of_points)
        self._trial: CohesiveFacetResponse | None = None
        self.last_committed_response: CohesiveFacetResponse | None = None

    def initialize_precrack(self, facets) -> None:
        """Mark selected facet indices as fully separated before execution."""

        selected = np.asarray(facets)
        if selected.dtype == bool:
            if selected.shape != (self.topology.number_of_facets,):
                raise ValueError("Boolean precrack mask has the wrong facet shape.")
            mask = selected
        else:
            mask = np.zeros(self.topology.number_of_facets, dtype=bool)
            indices = np.asarray(selected, dtype=int)
            if np.any(indices < 0) or np.any(indices >= mask.size):
                raise ValueError("Precrack facet index is out of range.")
            mask[indices] = True
        maximum = np.zeros((self.topology.number_of_facets, 2), dtype=float)
        maximum[mask, :] = self.law.failure_opening
        self.state.initialize(maximum.reshape(-1))

    def begin(self, displacement) -> CohesiveFacetResponse:
        """Assemble a replaceable trial force from nodal displacement."""

        values = _finite_array(displacement, name="displacement")
        if values.ndim != 2 or values.shape[0] != self.number_of_nodes:
            raise ValueError(
                "displacement must have shape (number_of_nodes, geometric_dimension)."
            )
        if values.shape[1] != self.topology.normals.shape[1]:
            raise ValueError("Displacement and interface normal dimensions differ.")

        negative = values[self.topology.negative_nodes]
        positive = values[self.topology.positive_nodes]
        nodal_jump = positive - negative
        jump_at_points = np.einsum("qi,fid->fqd", self._GAUSS, nodal_jump)
        opening = np.einsum("fqd,fd->fq", jump_at_points, self.topology.normals)
        material = self.state.begin(opening.reshape(-1))
        traction = material.traction.reshape((-1, 2))
        damage = material.damage.reshape((-1, 2))

        force = np.zeros_like(values)
        point_scale = 0.5 * self.topology.lengths * self.thickness
        for local_node in range(2):
            scalar = np.sum(
                self._GAUSS[:, local_node][None, :] * traction,
                axis=1,
            ) * point_scale
            vector = scalar[:, None] * self.topology.normals
            np.add.at(force, self.topology.positive_nodes[:, local_node], vector)
            np.add.at(force, self.topology.negative_nodes[:, local_node], -vector)

        stored = float(
            np.sum(
                material.stored_energy.reshape((-1, 2))
                * point_scale[:, None]
            )
        )
        dissipated = float(
            np.sum(
                material.dissipated_energy.reshape((-1, 2))
                * point_scale[:, None]
            )
        )
        self._trial = CohesiveFacetResponse(
            internal_force=force,
            opening=opening,
            traction=traction,
            damage=damage,
            stored_energy=stored,
            dissipated_energy=dissipated,
        )
        return self._trial

    def commit(self) -> None:
        if self._trial is None:
            raise RuntimeError("No cohesive facet trial response is available to commit.")
        self.last_committed_response = self._trial
        self.state.commit()
        self._trial = None

    def material_point_response(
        self,
        response: CohesiveFacetResponse | None = None,
    ) -> CohesiveResponse:
        """Recover current per-quadrature-point constitutive quantities.

        ``CohesiveFacetResponse`` intentionally stores interface-integrated
        energies for global balance.  Research field output needs the local
        energy densities instead; this method preserves that distinction.
        """

        selected = self.last_committed_response if response is None else response
        if selected is None:
            raise RuntimeError("No cohesive facet response is available.")
        return self.law.update(
            np.asarray(selected.opening, dtype=float).reshape(-1),
            self.state.committed_maximum,
        )

    def rollback(self) -> None:
        self.state.rollback()
        self._trial = None


def pair_coincident_line_facets(
    coordinates,
    negative_facets,
    positive_facets,
    *,
    normal_hint,
    tolerance: float = 1.0e-10,
) -> PairedLineFacets:
    """Pair coincident two-node line facets with a declared normal direction."""

    points = _finite_array(coordinates, name="coordinates")
    negative = np.asarray(negative_facets, dtype=int)
    positive = np.asarray(positive_facets, dtype=int)
    hint = _finite_array(normal_hint, name="normal_hint")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("The first paired-facet kernel requires 2D coordinates.")
    if negative.ndim != 2 or negative.shape[1] != 2:
        raise ValueError("negative_facets must have shape (facets, 2).")
    if positive.ndim != 2 or positive.shape[1] != 2:
        raise ValueError("positive_facets must have shape (facets, 2).")
    if negative.shape[0] != positive.shape[0]:
        raise ValueError("The two interface sides must contain the same facet count.")
    if hint.shape != (2,) or np.linalg.norm(hint) == 0.0:
        raise ValueError("normal_hint must be one nonzero 2D vector.")
    if not isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    if np.any(negative < 0) or np.any(positive < 0):
        raise ValueError("Facet node numbers cannot be negative.")
    if np.any(negative >= points.shape[0]) or np.any(positive >= points.shape[0]):
        raise ValueError("Facet node number exceeds the coordinate array.")
    if np.intersect1d(negative.reshape(-1), positive.reshape(-1)).size:
        raise ValueError(
            "Cohesive interface sides must use independent node identities; "
            "split or duplicate the shared interface nodes first."
        )

    negative_centroids = np.mean(points[negative], axis=1)
    positive_centroids = np.mean(points[positive], axis=1)
    distances = np.linalg.norm(
        negative_centroids[:, None, :] - positive_centroids[None, :, :], axis=2
    )
    ordered_positive = np.empty_like(negative)
    used: set[int] = set()
    for index in range(negative.shape[0]):
        candidates = np.flatnonzero(distances[index] <= tolerance)
        candidates = np.asarray([item for item in candidates if int(item) not in used])
        matches = []
        for candidate in candidates:
            source = points[negative[index]]
            target = points[positive[candidate]]
            direct = np.max(np.linalg.norm(source - target, axis=1))
            reverse = np.max(np.linalg.norm(source - target[::-1], axis=1))
            if min(direct, reverse) <= tolerance:
                matches.append((int(candidate), direct <= reverse))
        if len(matches) != 1:
            raise ValueError(
                "Every negative interface facet must have exactly one coincident "
                f"positive partner; facet {index} has {len(matches)}."
            )
        candidate, direct = matches[0]
        used.add(candidate)
        ordered_positive[index] = (
            positive[candidate] if direct else positive[candidate, ::-1]
        )

    segment = points[negative[:, 1]] - points[negative[:, 0]]
    lengths = np.linalg.norm(segment, axis=1)
    if np.any(lengths <= tolerance):
        raise ValueError("Interface facets must have positive reference length.")
    normals = np.column_stack((-segment[:, 1], segment[:, 0])) / lengths[:, None]
    hint = hint / np.linalg.norm(hint)
    signs = np.where(np.einsum("fd,d->f", normals, hint) >= 0.0, 1.0, -1.0)
    normals *= signs[:, None]
    quantized = np.rint(points[negative] / float(tolerance)).astype("<i8")
    facet_keys = tuple(
        sha256(np.ascontiguousarray(item).tobytes()).hexdigest()
        for item in quantized
    )
    return PairedLineFacets(
        negative_nodes=negative.copy(),
        positive_nodes=ordered_positive,
        normals=normals,
        lengths=lengths,
        tolerance=float(tolerance),
        facet_keys=facet_keys,
    )


def split_conforming_line_interface(
    coordinates,
    cells,
    interface_facets,
    *,
    positive_cells,
) -> SplitInterfaceMesh:
    """Duplicate nodes on a declared conforming 2D cell interface.

    ``positive_cells`` is an explicit set of cell indices whose interface
    nodes are replaced by duplicates.  Requiring this side identity avoids a
    fragile geometric guess and lets an Abaqus/Gmsh adapter preserve source
    surface semantics.  Every interface facet must separate exactly one
    selected cell from one unselected cell.
    """

    points = _finite_array(coordinates, name="coordinates")
    connectivity = np.asarray(cells, dtype=int)
    facets = np.asarray(interface_facets, dtype=int)
    selected = np.asarray(positive_cells)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Interface splitting currently requires 2D coordinates.")
    if connectivity.ndim != 2 or connectivity.shape[0] == 0:
        raise ValueError("cells must be one nonempty connectivity block.")
    if facets.ndim != 2 or facets.shape[1] != 2 or facets.shape[0] == 0:
        raise ValueError("interface_facets must have shape (facets, 2).")
    if np.any(connectivity < 0) or np.any(connectivity >= points.shape[0]):
        raise ValueError("Cell connectivity contains an invalid node number.")
    if np.any(facets < 0) or np.any(facets >= points.shape[0]):
        raise ValueError("Interface connectivity contains an invalid node number.")
    if selected.dtype == bool:
        if selected.shape != (connectivity.shape[0],):
            raise ValueError("Boolean positive_cells mask has the wrong shape.")
        positive_mask = selected.copy()
    else:
        positive_mask = np.zeros(connectivity.shape[0], dtype=bool)
        indices = np.asarray(selected, dtype=int)
        if np.any(indices < 0) or np.any(indices >= connectivity.shape[0]):
            raise ValueError("positive_cells contains an out-of-range cell index.")
        positive_mask[indices] = True
    if not np.any(positive_mask) or np.all(positive_mask):
        raise ValueError("Interface splitting requires cells on both declared sides.")

    for facet_index, facet in enumerate(facets):
        incident = np.flatnonzero(
            np.sum(np.isin(connectivity, facet), axis=1) == 2
        )
        if incident.size != 2:
            raise ValueError(
                f"Interface facet {facet_index} must have exactly two incident cells; "
                f"found {incident.size}."
            )
        sides = positive_mask[incident]
        if int(np.sum(sides)) != 1:
            raise ValueError(
                f"Interface facet {facet_index} does not separate one positive and "
                "one negative cell."
            )

    interface_nodes = np.unique(facets)
    duplicates = np.arange(
        points.shape[0], points.shape[0] + interface_nodes.size, dtype=int
    )
    mapping = {
        int(source): int(target)
        for source, target in zip(interface_nodes, duplicates, strict=True)
    }
    split_points = np.vstack((points, points[interface_nodes]))
    split_cells = connectivity.copy()
    for source, target in mapping.items():
        rows, columns = np.nonzero(
            positive_mask[:, None] & (split_cells == source)
        )
        split_cells[rows, columns] = target
    positive_facets_array = np.vectorize(mapping.__getitem__, otypes=[int])(facets)
    return SplitInterfaceMesh(
        coordinates=split_points,
        cells=split_cells,
        negative_facets=facets.copy(),
        positive_facets=np.asarray(positive_facets_array, dtype=int),
        original_to_duplicate=mapping,
        positive_cells=np.flatnonzero(positive_mask),
    )


@dataclass(frozen=True)
class CohesiveSurface:
    """Public description of a fixed-path zero-thickness interface."""

    law: BilinearCohesiveLaw
    mode: str = "normal"
    name: str = "cohesive surface"
    maturity: str = "experimental"

    def __post_init__(self) -> None:
        if self.mode != "normal":
            raise NotImplementedError(
                "The first cohesive-surface contract supports mode='normal' only."
            )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "zero_thickness_cohesive_surface",
            "mode": self.mode,
            "law": self.law.summary(),
            "topology_requirement": "paired coincident facets with independent dofs",
            "maturity": self.maturity,
        }


def bilinear_cohesive(
    *,
    strength: float,
    fracture_energy: float,
    initial_stiffness: float,
    compression_stiffness: float | None = None,
    name: str = "bilinear Mode-I cohesive law",
) -> BilinearCohesiveLaw:
    """Create a bilinear Mode-I cohesive law."""

    return BilinearCohesiveLaw(
        strength=strength,
        fracture_energy=fracture_energy,
        initial_stiffness=initial_stiffness,
        compression_stiffness=compression_stiffness,
        name=name,
    )


def cohesive_surface(
    *,
    law: BilinearCohesiveLaw,
    mode: str = "normal",
    name: str = "cohesive surface",
) -> CohesiveSurface:
    """Declare a fixed-path zero-thickness cohesive interface."""

    return CohesiveSurface(law=law, mode=mode, name=name)


def cohesive_characteristic_length(
    *, young: float, fracture_energy: float, strength: float
) -> float:
    """Return the declared scale ``E * Gamma / strength**2``.

    Different cohesive-zone conventions introduce order-one factors.  This
    helper intentionally reports the unscaled definition and leaves the
    convention visible in benchmark metadata.
    """

    values = (float(young), float(fracture_energy), float(strength))
    if any(not isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("young, fracture_energy, and strength must be positive.")
    return float(young * fracture_energy / strength**2)


def _finite_array(value, *, name: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must contain only finite values.")
    return selected


__all__ = [
    "BilinearCohesiveLaw",
    "CohesiveResponse",
    "CohesiveSurface",
    "CohesiveTransaction",
    "CohesiveFacetResponse",
    "ModeICohesiveFacetAssembler",
    "PairedLineFacets",
    "SplitInterfaceMesh",
    "bilinear_cohesive",
    "cohesive_characteristic_length",
    "cohesive_surface",
    "create_dolfinx_split_mesh",
    "pair_coincident_line_facets",
    "split_conforming_line_interface",
]
