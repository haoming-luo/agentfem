"""Real-geometry periodic cubes with deterministic spherical void assets.

This fixture deliberately keeps geometry generation outside AgentFEM's core.
Gmsh creates a first-order tetrahedral mesh with exactly matching opposite
faces; the returned source-node equations exercise the same public
``AbaqusPeriodicConstraint`` used for imported engineering meshes.

The pure-Python realization objects below intentionally precede any public
random-RVE API.  They freeze the scientific identity of a small, strictly
interior, non-overlapping spherical-void population without making claims
about statistical representativeness or periodic boundary-crossing pores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from agentfem import constraints, mesh
from agentfem.mesh import abaqus


PERIODIC_VOID_GMSH_OPTIONS = {
    "General.NumThreads": 1.0,
    "Mesh.Algorithm3D": 1.0,
    "Mesh.RandomFactor": 1.0e-9,
    "Mesh.RandomFactor3D": 1.0e-12,
    "Mesh.RandomSeed": 1.0,
}

SPHERICAL_VOID_REALIZATION_SCHEMA = "agentfem.test-spherical-void-realization.v2"
SPHERICAL_VOID_SAMPLER = "hard-core-periodic-cubic-pcg64-v2"


@dataclass(frozen=True)
class SphericalVoid:
    """One spherical pore in source coordinates."""

    center: tuple[float, float, float]
    radius: float

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center)
        radius = float(self.radius)
        if len(center) != 3 or not np.all(np.isfinite(center)):
            raise ValueError(
                "A spherical void center must contain three finite values."
            )
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("A spherical void radius must be finite and positive.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)

    def canonical_key(self) -> tuple[float, float, float, float]:
        return (*self.center, self.radius)


@dataclass(frozen=True)
class SphericalVoidRealization:
    """Canonical identity for one fixed, non-overlapping pore population.

    The first realization contract is deliberately conservative: every sphere
    must remain strictly inside a cubic cell and every pair must retain a
    positive minimum-image surface-to-surface clearance.  Pores crossing the
    cell boundary are a separate geometry capability and are rejected here;
    interactions with periodic images are nevertheless enforced.
    """

    side_length: float
    spheres: tuple[SphericalVoid, ...]
    seed: int
    minimum_inter_void_clearance: float
    minimum_boundary_clearance: float
    attempts: int
    sampler: str = SPHERICAL_VOID_SAMPLER

    def __post_init__(self) -> None:
        side_length = float(self.side_length)
        spheres = tuple(
            sphere if isinstance(sphere, SphericalVoid) else SphericalVoid(**sphere)
            for sphere in self.spheres
        )
        spheres = tuple(sorted(spheres, key=SphericalVoid.canonical_key))
        if (
            isinstance(self.seed, (bool, np.bool_))
            or int(self.seed) != self.seed
            or int(self.seed) < 0
        ):
            raise ValueError("seed must be a non-negative integer.")
        seed = int(self.seed)
        inter_clearance = float(self.minimum_inter_void_clearance)
        boundary_clearance = float(self.minimum_boundary_clearance)
        if (
            isinstance(self.attempts, (bool, np.bool_))
            or int(self.attempts) != self.attempts
        ):
            raise ValueError("attempts must be an integer.")
        attempts = int(self.attempts)
        sampler = str(self.sampler).strip()

        if not np.isfinite(side_length) or side_length <= 0.0:
            raise ValueError("side_length must be finite and positive.")
        if not spheres:
            raise ValueError("A spherical-void realization requires at least one void.")
        if not np.isfinite(inter_clearance) or inter_clearance <= 0.0:
            raise ValueError(
                "minimum_inter_void_clearance must be finite and positive."
            )
        if not np.isfinite(boundary_clearance) or boundary_clearance <= 0.0:
            raise ValueError("minimum_boundary_clearance must be finite and positive.")
        if attempts < len(spheres):
            raise ValueError("attempts cannot be smaller than the accepted void count.")
        if not sampler:
            raise ValueError("sampler must be a non-empty versioned name.")

        tolerance = 64.0 * np.finfo(float).eps * max(1.0, side_length)
        for sphere in spheres:
            observed = _sphere_boundary_clearance(sphere, side_length)
            if observed + tolerance < boundary_clearance:
                raise ValueError(
                    "A spherical void violates the declared boundary clearance: "
                    f"observed={observed:.16g}, required={boundary_clearance:.16g}."
                )
        for index, first in enumerate(spheres):
            for second in spheres[index + 1 :]:
                observed = _sphere_pair_clearance(
                    first,
                    second,
                    side_length=side_length,
                )
                if observed + tolerance < inter_clearance:
                    raise ValueError(
                        "Spherical voids violate the declared periodic inter-void "
                        "clearance: "
                        f"observed={observed:.16g}, required={inter_clearance:.16g}."
                    )

        object.__setattr__(self, "side_length", side_length)
        object.__setattr__(self, "spheres", spheres)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "minimum_inter_void_clearance", inter_clearance)
        object.__setattr__(self, "minimum_boundary_clearance", boundary_clearance)
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "sampler", sampler)

    @property
    def cell_reference_volume(self) -> float:
        return float(self.side_length**3)

    @property
    def exact_void_volume(self) -> float:
        return float(
            sum(4.0 * np.pi * sphere.radius**3 / 3.0 for sphere in self.spheres)
        )

    @property
    def actual_void_fraction(self) -> float:
        return float(self.exact_void_volume / self.cell_reference_volume)

    @property
    def observed_boundary_clearance(self) -> float:
        return float(
            min(
                _sphere_boundary_clearance(sphere, self.side_length)
                for sphere in self.spheres
            )
        )

    @property
    def observed_periodic_inter_void_clearance(self) -> float | None:
        clearances = [
            _sphere_pair_clearance(first, second, side_length=self.side_length)
            for index, first in enumerate(self.spheres)
            for second in self.spheres[index + 1 :]
        ]
        return None if not clearances else float(min(clearances))

    @property
    def observed_inter_void_clearance(self) -> float | None:
        """Backward-readable alias for the periodic minimum-image clearance."""

        return self.observed_periodic_inter_void_clearance

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": SPHERICAL_VOID_REALIZATION_SCHEMA,
            "sampler": self.sampler,
            "seed": self.seed,
            "cell": {
                "origin": [0.0, 0.0, 0.0],
                "side_length": self.side_length,
            },
            "constraints": {
                "minimum_inter_void_clearance": self.minimum_inter_void_clearance,
                "minimum_boundary_clearance": self.minimum_boundary_clearance,
                "periodic_boundary_crossing": False,
            },
            "attempts": self.attempts,
            "voids": [
                {
                    "id": f"void-{index:04d}",
                    "center": list(sphere.center),
                    "radius": sphere.radius,
                }
                for index, sphere in enumerate(self.spheres, start=1)
            ],
            "actual_void_fraction": self.actual_void_fraction,
            "observed_periodic_inter_void_clearance": (
                self.observed_periodic_inter_void_clearance
            ),
            "observed_boundary_clearance": self.observed_boundary_clearance,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_payload(),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def scientific_identity(self) -> dict[str, object]:
        payload = self.canonical_payload()
        return {
            **payload,
            "fingerprint": hashlib.sha256(
                self.canonical_json().encode("utf-8")
            ).hexdigest(),
        }


def sample_hard_core_spherical_voids(
    *,
    side_length: float,
    count: int,
    radius: float,
    seed: int,
    minimum_inter_void_clearance: float,
    minimum_boundary_clearance: float,
    maximum_attempts: int = 10_000,
) -> SphericalVoidRealization:
    """Sample one bounded, reproducible hard-core sphere realization.

    Sampling uses NumPy's explicitly named PCG64 generator and tests every
    pair with the cubic minimum-image convention.  A failed packing raises
    instead of returning a smaller population.  Spheres intersecting a
    periodic boundary are outside this first contract and cannot be sampled.
    """

    side_length = float(side_length)
    radius = float(radius)
    inter_clearance = float(minimum_inter_void_clearance)
    boundary_clearance = float(minimum_boundary_clearance)
    if isinstance(count, (bool, np.bool_)) or int(count) != count or int(count) <= 0:
        raise ValueError("count must be a positive integer.")
    count = int(count)
    if isinstance(seed, (bool, np.bool_)) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer.")
    seed = int(seed)
    if (
        isinstance(maximum_attempts, (bool, np.bool_))
        or int(maximum_attempts) != maximum_attempts
        or int(maximum_attempts) < count
    ):
        raise ValueError(
            "maximum_attempts must be an integer at least as large as count."
        )
    maximum_attempts = int(maximum_attempts)
    if not np.isfinite(side_length) or side_length <= 0.0:
        raise ValueError("side_length must be finite and positive.")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive.")
    if not np.isfinite(inter_clearance) or inter_clearance <= 0.0:
        raise ValueError("minimum_inter_void_clearance must be finite and positive.")
    if not np.isfinite(boundary_clearance) or boundary_clearance <= 0.0:
        raise ValueError("minimum_boundary_clearance must be finite and positive.")

    lower = radius + boundary_clearance
    upper = side_length - lower
    if lower > upper:
        raise ValueError(
            "Impossible spherical-void packing: radius plus boundary clearance "
            "does not fit inside the cell."
        )

    generator = np.random.Generator(np.random.PCG64(seed))
    accepted: list[SphericalVoid] = []
    attempts = 0
    required_distance = 2.0 * radius + inter_clearance
    while len(accepted) < count and attempts < maximum_attempts:
        attempts += 1
        candidate = SphericalVoid(
            center=tuple(float(value) for value in generator.uniform(lower, upper, 3)),
            radius=radius,
        )
        if all(
            _periodic_center_distance(candidate, other, side_length=side_length)
            >= required_distance
            for other in accepted
        ):
            accepted.append(candidate)

    if len(accepted) != count:
        raise ValueError(
            "Impossible spherical-void packing under the bounded hard-core "
            f"sampler: accepted {len(accepted)} of {count} voids after "
            f"{attempts} attempts (seed={seed})."
        )
    return SphericalVoidRealization(
        side_length=side_length,
        spheres=tuple(accepted),
        seed=seed,
        minimum_inter_void_clearance=inter_clearance,
        minimum_boundary_clearance=boundary_clearance,
        attempts=attempts,
    )


def _periodic_center_distance(
    first: SphericalVoid,
    second: SphericalVoid,
    *,
    side_length: float,
) -> float:
    difference = np.abs(
        np.asarray(first.center, dtype=float) - np.asarray(second.center, dtype=float)
    )
    minimum_image = np.minimum(difference, float(side_length) - difference)
    return float(np.linalg.norm(minimum_image))


def _sphere_pair_clearance(
    first: SphericalVoid,
    second: SphericalVoid,
    *,
    side_length: float,
) -> float:
    distance = _periodic_center_distance(
        first,
        second,
        side_length=side_length,
    )
    return float(distance - first.radius - second.radius)


def _sphere_boundary_clearance(sphere: SphericalVoid, side_length: float) -> float:
    center = np.asarray(sphere.center, dtype=float)
    return float(
        min(
            np.min(center - sphere.radius),
            np.min(float(side_length) - center - sphere.radius),
        )
    )


@dataclass(frozen=True)
class PeriodicVoidFixture:
    """Gmsh void cell plus exact source-coordinate periodic semantics."""

    domain: object
    cell_tags: object
    facet_tags: object
    nodes: abaqus.AbaqusNodeTable
    equations: abaqus.AbaqusEquationSet
    deformation_gradient: np.ndarray
    anchor_node: int
    reference_nodes: tuple[int, int, int]
    side_length: float
    void_radius: float
    periodic_pairing_error: float

    @property
    def cell_reference_volume(self) -> float:
        return float(self.side_length**3)

    @property
    def exact_solid_volume(self) -> float:
        return float(self.side_length**3 - 4.0 * np.pi * self.void_radius**3 / 3.0)

    def constraint(self, displacement):
        return constraints.abaqus_periodic_cell(
            displacement,
            nodes=self.nodes,
            equations=self.equations,
            deformation_gradient=self.deformation_gradient,
            anchor_node=self.anchor_node,
            reference_nodes=self.reference_nodes,
            tolerance=2.0e-9 * self.side_length,
            name="gmsh_spherical_void_periodic_cell",
        )


@dataclass(frozen=True)
class PeriodicMultiVoidFixture:
    """Periodic Gmsh cell carrying one canonical multi-void realization."""

    domain: object
    cell_tags: object
    facet_tags: object
    nodes: abaqus.AbaqusNodeTable
    equations: abaqus.AbaqusEquationSet
    deformation_gradient: np.ndarray
    anchor_node: int
    reference_nodes: tuple[int, int, int]
    realization: SphericalVoidRealization
    periodic_pairing_error: float
    void_surface_count: int

    @property
    def side_length(self) -> float:
        return self.realization.side_length

    @property
    def cell_reference_volume(self) -> float:
        return self.realization.cell_reference_volume

    @property
    def exact_solid_volume(self) -> float:
        return float(
            self.realization.cell_reference_volume - self.realization.exact_void_volume
        )

    @property
    def actual_void_fraction(self) -> float:
        return self.realization.actual_void_fraction

    @property
    def realization_fingerprint(self) -> str:
        return str(self.realization.scientific_identity()["fingerprint"])

    def constraint(self, displacement):
        return constraints.abaqus_periodic_cell(
            displacement,
            nodes=self.nodes,
            equations=self.equations,
            deformation_gradient=self.deformation_gradient,
            anchor_node=self.anchor_node,
            reference_nodes=self.reference_nodes,
            tolerance=2.0e-9 * self.side_length,
            name="gmsh_multi_spherical_void_periodic_cell",
        )


def periodic_spherical_void_cell(
    comm,
    *,
    side_length: float = 1.0,
    void_radius: float = 0.2,
    mesh_size: float = 0.18,
    stretch: float = 1.01,
    model_rank: int = 0,
) -> PeriodicVoidFixture:
    """Build a periodic tetrahedral cube containing a true spherical void.

    The void is geometric: no low-stiffness surrogate phase is present.  The
    macroscopic loading is isochoric uniaxial tension by default.
    """

    side_length = float(side_length)
    void_radius = float(void_radius)
    mesh_size = float(mesh_size)
    stretch = float(stretch)
    if not np.isfinite(side_length) or side_length <= 0.0:
        raise ValueError("side_length must be finite and positive.")
    if not np.isfinite(void_radius) or not 0.0 < void_radius < 0.5 * side_length:
        raise ValueError("void_radius must lie strictly inside half the cell size.")
    if not np.isfinite(mesh_size) or mesh_size <= 0.0:
        raise ValueError("mesh_size must be finite and positive.")
    if not np.isfinite(stretch) or stretch <= 0.0:
        raise ValueError("stretch must be finite and positive.")

    imported, semantics = _lower_periodic_spherical_void_mesh(
        comm,
        side_length=side_length,
        spheres=(
            SphericalVoid(
                center=(0.5 * side_length,) * 3,
                radius=void_radius,
            ),
        ),
        mesh_size=mesh_size,
        model_rank=model_rank,
        model_name="agentfem_periodic_spherical_void",
    )

    lateral = 1.0 / np.sqrt(stretch)
    return PeriodicVoidFixture(
        domain=imported.domain,
        cell_tags=imported.cell_tags,
        facet_tags=imported.facet_tags,
        nodes=abaqus.AbaqusNodeTable(
            labels=np.asarray(semantics["labels"], dtype=np.int64),
            coordinates=np.asarray(semantics["coordinates"], dtype=float),
        ),
        equations=abaqus.AbaqusEquationSet(
            tuple(
                abaqus.LinearEquation(
                    tuple(
                        abaqus.EquationTerm(
                            int(node),
                            int(component),
                            float(coefficient),
                        )
                        for node, component, coefficient in terms
                    )
                )
                for terms in semantics["equations"]
            )
        ),
        deformation_gradient=np.diag((stretch, lateral, lateral)),
        anchor_node=int(semantics["anchor_node"]),
        reference_nodes=tuple(int(value) for value in semantics["reference_nodes"]),
        side_length=side_length,
        void_radius=void_radius,
        periodic_pairing_error=float(semantics["periodic_pairing_error"]),
    )


def periodic_multi_spherical_void_cell(
    comm,
    *,
    realization: SphericalVoidRealization,
    mesh_size: float = 0.18,
    stretch: float = 1.01,
    model_rank: int = 0,
) -> PeriodicMultiVoidFixture:
    """Lower one fixed multi-sphere realization to an exact periodic mesh.

    All pore surfaces share physical tag 20.  Stable per-pore identity remains
    in ``realization``; Gmsh entity numbers are deliberately not promoted to a
    scientific identifier.  Periodic boundary-crossing pores remain outside
    this fixture's supported geometry.
    """

    if not isinstance(realization, SphericalVoidRealization):
        raise TypeError("realization must be a SphericalVoidRealization.")
    mesh_size = float(mesh_size)
    stretch = float(stretch)
    if not np.isfinite(mesh_size) or mesh_size <= 0.0:
        raise ValueError("mesh_size must be finite and positive.")
    if not np.isfinite(stretch) or stretch <= 0.0:
        raise ValueError("stretch must be finite and positive.")

    imported, semantics = _lower_periodic_spherical_void_mesh(
        comm,
        side_length=realization.side_length,
        spheres=realization.spheres,
        mesh_size=mesh_size,
        model_rank=model_rank,
        model_name="agentfem_periodic_multi_spherical_void",
    )
    lateral = 1.0 / np.sqrt(stretch)
    nodes, equations = _abaqus_source_semantics(semantics)
    return PeriodicMultiVoidFixture(
        domain=imported.domain,
        cell_tags=imported.cell_tags,
        facet_tags=imported.facet_tags,
        nodes=nodes,
        equations=equations,
        deformation_gradient=np.diag((stretch, lateral, lateral)),
        anchor_node=int(semantics["anchor_node"]),
        reference_nodes=tuple(int(value) for value in semantics["reference_nodes"]),
        realization=realization,
        periodic_pairing_error=float(semantics["periodic_pairing_error"]),
        void_surface_count=int(semantics["void_surface_count"]),
    )


def _lower_periodic_spherical_void_mesh(
    comm,
    *,
    side_length: float,
    spheres: tuple[SphericalVoid, ...],
    mesh_size: float,
    model_rank: int,
    model_name: str,
):
    """Build and import one cube cut by strictly interior spheres."""

    spheres = tuple(sorted(spheres, key=SphericalVoid.canonical_key))
    if not spheres:
        raise ValueError("At least one spherical void is required.")
    gmsh = mesh.require_gmsh()
    initialized_here = not gmsh.isInitialized()
    if initialized_here:
        gmsh.initialize()
    semantics = None
    previous_options = {}
    try:
        if comm.rank == model_rank:
            gmsh.clear()
            selected_options = {
                "General.Verbosity": 0.0,
                "Mesh.MeshSizeMin": mesh_size,
                "Mesh.MeshSizeMax": mesh_size,
                **PERIODIC_VOID_GMSH_OPTIONS,
            }
            previous_options = {
                name: float(gmsh.option.getNumber(name)) for name in selected_options
            }
            for name, value in selected_options.items():
                gmsh.option.setNumber(name, float(value))
            gmsh.model.add(model_name)
            box = gmsh.model.occ.addBox(
                0.0,
                0.0,
                0.0,
                side_length,
                side_length,
                side_length,
            )
            sphere_entities = [
                (
                    3,
                    gmsh.model.occ.addSphere(
                        sphere.center[0],
                        sphere.center[1],
                        sphere.center[2],
                        sphere.radius,
                    ),
                )
                for sphere in spheres
            ]
            volumes, _ = gmsh.model.occ.cut(
                [(3, box)],
                sphere_entities,
                removeObject=True,
                removeTool=True,
            )
            gmsh.model.occ.synchronize()
            if len(volumes) != 1:
                raise RuntimeError("The spherical-void Boolean cut is not unique.")

            outer_faces, void_faces = _classify_faces(gmsh, side_length)
            if len(void_faces) != len(spheres):
                raise RuntimeError(
                    "The spherical-void Boolean cut did not retain one distinct "
                    f"surface per void: expected {len(spheres)}, observed "
                    f"{len(void_faces)}."
                )
            for axis in range(3):
                transform = np.eye(4)
                transform[axis, 3] = side_length
                gmsh.model.mesh.setPeriodic(
                    2,
                    [outer_faces[(axis, 1)]],
                    [outer_faces[(axis, 0)]],
                    transform.reshape(-1).tolist(),
                )

            volume_tag = int(volumes[0][1])
            gmsh.model.addPhysicalGroup(3, [volume_tag], 1)
            gmsh.model.setPhysicalName(3, 1, "matrix")
            gmsh.model.addPhysicalGroup(2, list(outer_faces.values()), 10)
            gmsh.model.setPhysicalName(2, 10, "periodic_boundary")
            gmsh.model.addPhysicalGroup(2, void_faces, 20)
            gmsh.model.setPhysicalName(2, 20, "void_surface")
            gmsh.model.mesh.generate(3)
            semantics = _periodic_semantics(
                gmsh,
                outer_faces,
                side_length=side_length,
                tolerance=1.0e-9 * side_length,
            )
            semantics["void_surface_count"] = len(void_faces)

        semantics = comm.bcast(semantics, root=model_rank)
        imported = mesh.import_gmsh_model(
            gmsh.model,
            comm,
            model_rank=model_rank,
            gdim=3,
        )
    finally:
        if comm.rank == model_rank and gmsh.isInitialized():
            for name, value in previous_options.items():
                gmsh.option.setNumber(name, value)
        if initialized_here:
            gmsh.finalize()
    return imported, semantics


def _abaqus_source_semantics(semantics):
    nodes = abaqus.AbaqusNodeTable(
        labels=np.asarray(semantics["labels"], dtype=np.int64),
        coordinates=np.asarray(semantics["coordinates"], dtype=float),
    )
    equations = abaqus.AbaqusEquationSet(
        tuple(
            abaqus.LinearEquation(
                tuple(
                    abaqus.EquationTerm(
                        int(node),
                        int(component),
                        float(coefficient),
                    )
                    for node, component, coefficient in terms
                )
            )
            for terms in semantics["equations"]
        )
    )
    return nodes, equations


def _classify_faces(gmsh, side_length: float):
    tolerance = 1.0e-8 * side_length
    outer = {}
    void = []
    for _dimension, tag in gmsh.model.getEntities(2):
        center = np.asarray(gmsh.model.occ.getCenterOfMass(2, tag), dtype=float)
        selected = None
        for axis in range(3):
            if abs(center[axis]) <= tolerance:
                selected = (axis, 0)
                break
            if abs(center[axis] - side_length) <= tolerance:
                selected = (axis, 1)
                break
        if selected is None:
            void.append(int(tag))
        elif selected in outer:
            raise RuntimeError(f"Multiple faces were classified as {selected}.")
        else:
            outer[selected] = int(tag)
    expected = {(axis, side) for axis in range(3) for side in (0, 1)}
    if set(outer) != expected or not void:
        raise RuntimeError("Could not identify six outer faces and the void surface.")
    return outer, void


def _periodic_semantics(
    gmsh,
    faces,
    *,
    side_length: float,
    tolerance: float,
    origin=(0.0, 0.0, 0.0),
    periods=None,
):
    """Return source-node equations for one rectangular periodic lattice.

    ``side_length`` remains the backward-compatible cubic default.  ``origin``
    and ``periods`` let external benchmark fixtures reuse the exact same node
    graph for shifted or thin extruded cells.
    """

    origin = np.asarray(origin, dtype=float)
    periods = (
        np.full(3, float(side_length), dtype=float)
        if periods is None
        else np.asarray(periods, dtype=float)
    )
    if origin.shape != (3,) or periods.shape != (3,):
        raise ValueError("origin and periods must contain three coordinates.")
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(periods)):
        raise ValueError("origin and periods must be finite.")
    if np.any(periods <= 0.0):
        raise ValueError("periods must be positive.")
    node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes()
    coordinates = np.asarray(node_coordinates, dtype=float).reshape(-1, 3)
    labels = np.asarray(node_tags, dtype=np.int64)
    coordinate_to_label = {}
    coordinate_by_label = {}
    for label, coordinate in zip(labels, coordinates):
        key = _coordinate_key(coordinate, tolerance)
        if key in coordinate_to_label:
            raise RuntimeError(
                "Gmsh produced duplicate coordinates for a first-order mesh."
            )
        coordinate_to_label[key] = int(label)
        coordinate_by_label[int(label)] = coordinate

    def label_at(coordinate):
        key = _coordinate_key(coordinate, tolerance)
        try:
            return coordinate_to_label[key]
        except KeyError as exc:
            raise RuntimeError(
                f"The periodic mesh has no node at {np.asarray(coordinate).tolist()}."
            ) from exc

    anchor = label_at(origin)
    references = tuple(
        label_at(origin + np.eye(3)[axis] * periods[axis]) for axis in range(3)
    )
    control_nodes = {anchor, *references}
    boundary_labels = sorted(
        int(label)
        for label, coordinate in zip(labels, coordinates)
        if np.any(np.isclose(coordinate, origin, rtol=0.0, atol=tolerance))
        or np.any(np.isclose(coordinate, origin + periods, rtol=0.0, atol=tolerance))
    )
    equations = []
    for slave in boundary_labels:
        coordinate = coordinate_by_label[slave]
        active_axes = tuple(
            axis
            for axis in range(3)
            if abs(coordinate[axis] - (origin[axis] + periods[axis])) <= tolerance
        )
        if not active_axes or slave in control_nodes:
            continue
        wrapped = coordinate.copy()
        wrapped[list(active_axes)] = origin[list(active_axes)]
        base = label_at(wrapped)
        for component in (1, 2, 3):
            terms = [(slave, component, 1.0), (base, component, -1.0)]
            terms.extend((references[axis], component, -1.0) for axis in active_axes)
            terms.append((anchor, component, float(len(active_axes))))
            equations.append(terms)

    pairing_error = 0.0
    for axis in range(3):
        selected_faces = faces[(axis, 1)]
        if np.isscalar(selected_faces):
            selected_faces = (int(selected_faces),)
        found_pair = False
        for selected_face in selected_faces:
            _master, slaves, masters, _transform = gmsh.model.mesh.getPeriodicNodes(
                2,
                int(selected_face),
                True,
            )
            if len(slaves) == 0:
                continue
            found_pair = True
            translation = np.eye(3)[axis] * periods[axis]
            pairing_error = max(
                pairing_error,
                max(
                    float(
                        np.linalg.norm(
                            coordinate_by_label[int(slave)]
                            - coordinate_by_label[int(master)]
                            - translation
                        )
                    )
                    for slave, master in zip(slaves, masters)
                ),
            )
        if not found_pair:
            raise RuntimeError("Gmsh did not retain periodic surface-node pairs.")

    selected_coordinates = [
        coordinate_by_label[label].tolist() for label in boundary_labels
    ]
    return {
        "labels": boundary_labels,
        "coordinates": selected_coordinates,
        "equations": equations,
        "anchor_node": anchor,
        "reference_nodes": references,
        "periodic_pairing_error": pairing_error,
    }


def _coordinate_key(coordinate, tolerance: float):
    return tuple(
        np.rint(np.asarray(coordinate, dtype=float) / tolerance).astype(np.int64)
    )
