"""Integration-point storage for stateful constitutive models."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import basix
import basix.ufl
import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI

from .plasticity import J2LinearIsotropicHardening, J2PlasticState
from .creep import ImplicitCreepState, IsotropicPowerLawCreepMaterial


def _material_record(material) -> dict[str, object]:
    if hasattr(material, "as_dict"):
        return material.as_dict()
    if hasattr(material, "summary"):
        return material.summary()
    raise TypeError(
        f"Material {type(material).__name__} needs as_dict() or summary() "
        "for a restartable quadrature contract."
    )


@dataclass(frozen=True)
class QuadratureMaterialMap:
    """Cell-region material dispatch shared by stateful solid procedures.

    ``cell_regions`` contains one stable region id for every locally visible
    cell, including ghosts.  It is deliberately separate from a UFL measure:
    local constitutive integration needs an unambiguous material at every
    quadrature point, while the resulting stress/tangent fields can still be
    assembled over the ordinary whole-domain measure.
    """

    domain: object
    materials: Mapping[int, object]
    cell_regions: np.ndarray
    region_names: Mapping[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        selected = {int(key): value for key, value in self.materials.items()}
        if not selected:
            raise ValueError("QuadratureMaterialMap requires at least one material.")
        regions = np.asarray(self.cell_regions, dtype=np.int64).reshape(-1)
        cell_map = self.domain.topology.index_map(self.domain.topology.dim)
        expected = int(cell_map.size_local + cell_map.num_ghosts)
        if regions.size != expected:
            raise ValueError(
                f"Quadrature material map requires {expected} visible cells, "
                f"got {regions.size}."
            )
        unknown = sorted(set(regions.tolist()) - set(selected))
        if unknown:
            raise ValueError(f"Quadrature material map has unknown regions: {unknown}.")
        object.__setattr__(self, "materials", selected)
        object.__setattr__(self, "cell_regions", regions)
        object.__setattr__(
            self,
            "region_names",
            {int(key): str(value) for key, value in self.region_names.items()},
        )

    @classmethod
    def from_assignments(cls, domain, assignments, *, material_type):
        """Build a complete owned/ghost map from Model material assignments."""

        records = tuple(assignments)
        if not records:
            raise ValueError("At least one material assignment is required.")
        if len(records) == 1 and records[0].region is None:
            material = records[0].item
            if not isinstance(material, material_type):
                raise TypeError(
                    f"Expected {material_type.__name__}, got "
                    f"{type(material).__name__}."
                )
            cell_map = domain.topology.index_map(domain.topology.dim)
            return cls(
                domain,
                {0: material},
                np.zeros(cell_map.size_local + cell_map.num_ghosts, dtype=np.int64),
                {0: "whole_domain"},
            )
        if any(record.region is None for record in records):
            raise ValueError(
                "Multiple stateful materials require a CellRegion for every "
                "material assignment."
            )
        cell_map = domain.topology.index_map(domain.topology.dim)
        owned = int(cell_map.size_local)
        original = _original_cell_keys(domain)
        owned_regions = np.full(owned, -1, dtype=np.int64)
        materials = {}
        names = {}
        local_problem = None
        for region_id, record in enumerate(records, start=1):
            if not isinstance(record.item, material_type):
                raise TypeError(
                    f"Expected {material_type.__name__}, got "
                    f"{type(record.item).__name__}."
                )
            region = record.region
            if region.domain is not domain:
                raise ValueError("Material CellRegion belongs to a different mesh.")
            cells = np.asarray(region.cell_tags.find(region.tag), dtype=np.int64)
            cells = cells[cells < owned]
            if np.any(owned_regions[cells] >= 0):
                local_problem = (
                    f"Stateful material regions overlap in {region.name!r}."
                )
            owned_regions[cells] = region_id
            materials[region_id] = record.item
            names[region_id] = region.name
        missing = np.flatnonzero(owned_regions < 0)
        if missing.size and local_problem is None:
            local_problem = (
                "Stateful material regions do not cover every owned cell; "
                f"sample local cells={missing[:8].tolist()}."
            )
        problems = domain.comm.allgather(local_problem)
        if any(problem is not None for problem in problems):
            rank = next(
                index for index, problem in enumerate(problems) if problem is not None
            )
            raise ValueError(f"Rank {rank}: {problems[rank]}")
        gathered = domain.comm.allgather(
            (original[:owned].copy(), owned_regions.copy())
        )
        lookup = {}
        for keys, values in gathered:
            for key, value in zip(keys, values, strict=True):
                previous = lookup.setdefault(int(key), int(value))
                if previous != int(value):
                    raise ValueError(
                        f"Physical cell {int(key)} has inconsistent material regions."
                    )
        visible = np.asarray([lookup[int(key)] for key in original], dtype=np.int64)
        return cls(domain, materials, visible, names)

    def material_for_point(self, point: int, *, points_per_cell: int):
        cell = int(point) // int(points_per_cell)
        return self.materials[int(self.cell_regions[cell])]

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "quadrature_material_map",
            "regions": [
                {
                    "id": region,
                    "name": self.region_names.get(region, f"region_{region}"),
                    "material": _material_record(self.materials[region]),
                }
                for region in sorted(self.materials)
            ],
        }


def _original_cell_keys(domain) -> np.ndarray:
    """Return partition-stable original input cell ids for visible cells."""

    cell_map = domain.topology.index_map(domain.topology.dim)
    count = int(cell_map.size_local + cell_map.num_ghosts)
    keys = np.asarray(domain.topology.original_cell_index, dtype=np.int64)
    if keys.size != count:
        raise RuntimeError(
            "DOLFINx original_cell_index does not cover every visible cell."
        )
    owned = keys[: int(cell_map.size_local)]
    gathered = domain.comm.allgather(owned.copy())
    global_keys = np.concatenate(gathered) if gathered else np.empty(0, np.int64)
    if global_keys.size != int(cell_map.size_global):
        raise RuntimeError("Owned physical-cell identity count is inconsistent.")
    if np.unique(global_keys).size != global_keys.size:
        raise RuntimeError("Original physical-cell identities are not globally unique.")
    return keys


def _quadrature_rule_identity(field, *, degree: int, scheme: str) -> dict[str, object]:
    points = np.ascontiguousarray(field.points, dtype=np.float64)
    weights = np.ascontiguousarray(field.weights, dtype=np.float64)
    return {
        "degree": int(degree),
        "scheme": str(scheme),
        "points_per_cell": int(len(points)),
        "reference_points_sha256": sha256(points.tobytes()).hexdigest(),
        "weights_sha256": sha256(weights.tobytes()).hexdigest(),
    }


def _portable_state_identity(state, material=None) -> dict[str, object]:
    from ..checkpointing import mesh_portable_identity

    transaction = state.transaction
    fields = transaction.committed
    return {
        "schema": transaction.schema,
        "schema_version": transaction.schema_version,
        "mesh": mesh_portable_identity(state.domain),
        "quadrature": _quadrature_rule_identity(
            next(iter(fields.values())), degree=state.degree, scheme=state.scheme
        ),
        "state": {
            name: {
                "value_shape": list(selected.value_shape),
                "dtype": str(selected.function.x.array.dtype),
            }
            for name, selected in fields.items()
        },
        "materials": (
            None if material is None else _material_record(material)
        ),
    }


def save_portable_quadrature_state(path, state, *, material=None) -> Path:
    """Collectively save committed state by physical cell and point identity."""

    selected = Path(path)
    if selected.suffix != ".npz":
        selected = selected.with_suffix(".npz")
    comm = state.domain.comm
    identity = _portable_state_identity(state, material)
    cell_map = state.domain.topology.index_map(state.domain.topology.dim)
    owned = int(cell_map.size_local)
    keys = _original_cell_keys(state.domain)[:owned]
    points_per_cell = len(state.stress.points)
    snapshot = state.snapshot()
    local = {
        name: np.asarray(values).reshape((
            cell_map.size_local + cell_map.num_ghosts,
            points_per_cell,
            *np.asarray(values).shape[1:],
        ))[:owned].copy()
        for name, values in snapshot.items()
    }
    regions = (
        np.zeros(owned, dtype=np.int64)
        if not isinstance(material, QuadratureMaterialMap)
        else material.cell_regions[:owned].copy()
    )
    gathered = comm.gather((keys, regions, local), root=0)
    error = None
    if comm.rank == 0:
        try:
            selected.parent.mkdir(parents=True, exist_ok=True)
            all_keys = np.concatenate([item[0] for item in gathered])
            all_regions = np.concatenate([item[1] for item in gathered])
            order = np.argsort(all_keys, kind="stable")
            all_keys = all_keys[order]
            all_regions = all_regions[order]
            if np.unique(all_keys).size != all_keys.size:
                raise ValueError("Portable quadrature archive has duplicate cells.")
            arrays = {
                "schema": "agentfem.portable-quadrature-state.v1",
                "identity": json.dumps(identity, sort_keys=True),
                "cell_keys": all_keys,
                "material_regions": all_regions,
            }
            for name in state.transaction.names:
                arrays[name] = np.concatenate(
                    [item[2][name] for item in gathered], axis=0
                )[order]
            from ..checkpointing import atomic_savez

            atomic_savez(selected, **arrays)
        except Exception as exc:  # pragma: no cover - filesystem failure
            error = f"{type(exc).__name__}: {exc}"
    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(f"Portable quadrature checkpoint write failed: {error}")
    comm.barrier()
    return selected


def load_portable_quadrature_state(path, state, *, material=None) -> None:
    """Collectively restore committed state under a changed MPI partition."""

    selected = Path(path)
    comm = state.domain.comm
    archive = None
    error = None
    if comm.rank == 0:
        try:
            with np.load(selected, allow_pickle=False) as data:
                if str(data["schema"]) != "agentfem.portable-quadrature-state.v1":
                    raise ValueError("Unsupported portable quadrature state schema.")
                archive = {name: np.asarray(data[name]).copy() for name in data.files}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(f"Portable quadrature checkpoint read failed: {error}")
    archive = comm.bcast(archive, root=0)
    archive.pop("schema", None)
    stored_identity = json.loads(str(archive.pop("identity")))
    current_identity = json.loads(
        json.dumps(_portable_state_identity(state, material), sort_keys=True)
    )
    if stored_identity != current_identity:
        raise ValueError(
            "Portable quadrature mesh, rule, material regions, state schema, "
            "or value layout differs from the current analysis."
        )
    stored_keys = np.asarray(archive["cell_keys"], dtype=np.int64)
    lookup = {int(key): index for index, key in enumerate(stored_keys)}
    visible_keys = _original_cell_keys(state.domain)
    local_problem = None
    try:
        rows = np.asarray([lookup[int(key)] for key in visible_keys], dtype=np.int64)
    except KeyError as exc:
        rows = np.empty(0, dtype=np.int64)
        local_problem = (
            f"Portable quadrature state lacks physical cell {int(exc.args[0])}."
        )
    expected_regions = (
        np.zeros(len(visible_keys), dtype=np.int64)
        if not isinstance(material, QuadratureMaterialMap)
        else material.cell_regions
    )
    if local_problem is None and not np.array_equal(
        archive["material_regions"][rows], expected_regions
    ):
        local_problem = "Portable quadrature material-region assignment differs."
    problems = comm.allgather(local_problem)
    if any(problem is not None for problem in problems):
        rank = next(
            index for index, problem in enumerate(problems) if problem is not None
        )
        raise ValueError(f"Rank {rank}: {problems[rank]}")
    snapshot = {}
    for name in state.transaction.names:
        values = np.asarray(archive[name])[rows]
        snapshot[name] = values.reshape((-1, *values.shape[2:]))
    state.restore(snapshot)


@dataclass
class QuadratureTransaction:
    """Shared trial/commit/rollback contract for integration-point state.

    The transaction owns state transitions only. A material consumer remains
    responsible for local integration, stress, consistent tangent, and error
    estimates. J2, creep, viscoplasticity, and damage can therefore share one
    failure-safe state mechanism without sharing a constitutive algorithm.
    """

    committed: Mapping[str, object]
    trial: Mapping[str, object]
    schema: str
    schema_version: str = "0.1.0"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        committed = dict(self.committed)
        trial = dict(self.trial)
        if not committed:
            raise ValueError("QuadratureTransaction requires committed state.")
        if set(committed) != set(trial):
            raise ValueError(
                "Committed and trial quadrature state names must match."
            )
        if not str(self.schema).strip():
            raise ValueError("QuadratureTransaction.schema must not be empty.")
        for name, selected in committed.items():
            candidate = trial[name]
            for label, field_value in (
                ("Committed", selected),
                ("Trial", candidate),
            ):
                if not hasattr(field_value, "values") or not hasattr(
                    field_value, "assign"
                ):
                    raise TypeError(
                        f"{label} state {name!r} must provide values and assign()."
                    )
            if np.asarray(selected.values).shape != np.asarray(candidate.values).shape:
                raise ValueError(
                    f"Committed/trial state {name!r} shapes do not match."
                )
        self.committed = committed
        self.trial = trial
        self.metadata = dict(self.metadata)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.committed)

    def begin(self) -> None:
        """Start an attempt from the last committed state."""

        self.rollback()

    def commit(self) -> None:
        """Atomically accept every trial state variable."""

        snapshots = {
            name: np.asarray(selected.values).copy()
            for name, selected in self.trial.items()
        }
        for name, selected in self.committed.items():
            selected.assign(snapshots[name])

    def rollback(self) -> None:
        """Discard all trial variables and restore the committed state."""

        snapshots = {
            name: np.asarray(selected.values).copy()
            for name, selected in self.committed.items()
        }
        for name, selected in self.trial.items():
            selected.assign(snapshots[name])

    def snapshot(self) -> dict[str, np.ndarray]:
        """Return a restart-safe copy of committed state values."""

        return {
            name: np.asarray(selected.values).copy()
            for name, selected in self.committed.items()
        }

    def restore(self, snapshot: Mapping[str, object]) -> None:
        if set(snapshot) != set(self.committed):
            raise ValueError(
                "Quadrature snapshot variables differ from this transaction; "
                f"expected={self.names}, received={tuple(snapshot)}."
            )
        for name, selected in self.committed.items():
            selected.assign(snapshot[name])
        self.rollback()

    def summary(self) -> dict[str, object]:
        return {
            "kind": "quadrature_transaction",
            "schema": self.schema,
            "schema_version": self.schema_version,
            "state_variables": self.names,
            "metadata": dict(self.metadata),
        }


@dataclass
class QuadratureField:
    """A DOLFINx quadrature function with an explicit NumPy point view."""

    function: object
    points: np.ndarray
    weights: np.ndarray
    value_shape: tuple[int, ...]

    @classmethod
    def create(
        cls,
        domain,
        *,
        name: str,
        degree: int,
        value_shape=(),
        scheme: str = "default",
    ):
        shape = tuple(value_shape)
        points, weights = basix.make_quadrature(
            domain.basix_cell(),
            int(degree),
            rule=getattr(basix.QuadratureType, scheme),
        )
        element = basix.ufl.quadrature_element(
            domain.basix_cell(),
            value_shape=shape,
            points=points,
            weights=weights,
        )
        space = fem.functionspace(domain, element)
        return cls(
            function=fem.Function(space, name=name),
            points=np.asarray(points),
            weights=np.asarray(weights),
            value_shape=shape,
        )

    @property
    def component_count(self) -> int:
        return int(np.prod(self.value_shape)) if self.value_shape else 1

    @property
    def values(self) -> np.ndarray:
        """Return values in deterministic ``cell, point, component`` order.

        Local quadrature dof numbering is not guaranteed to equal local cell
        numbering on a distributed mesh. Going through the cell dofmap is
        therefore required; a direct reshape is only accidentally correct on
        many serial meshes.
        """

        V = self.function.function_space
        cell_count = self._cell_count()
        points_per_cell = len(self.points)
        block_size = int(V.dofmap.bs)
        flat = np.empty((cell_count * points_per_cell, block_size))
        for cell in range(cell_count):
            dofs = V.dofmap.cell_dofs(cell)
            if len(dofs) != points_per_cell:
                raise RuntimeError(
                    "Quadrature dof count does not match the selected rule."
                )
            for point, dof in enumerate(dofs):
                start = int(dof) * block_size
                flat[cell * points_per_cell + point] = (
                    self.function.x.array[start : start + block_size]
                )
        trailing = self.value_shape if self.value_shape else ()
        return flat.reshape((-1, *trailing))

    @property
    def owned_values(self) -> np.ndarray:
        """Return integration points owned by this rank, excluding ghosts."""

        domain = self.function.function_space.mesh
        cell_map = domain.topology.index_map(domain.topology.dim)
        count = int(cell_map.size_local) * len(self.points)
        return self.values[:count]

    def global_max(self) -> float:
        """Return the MPI-global maximum without double-counting ghost cells."""

        owned = np.asarray(self.owned_values)
        local = float(np.max(owned, initial=-np.inf))
        return float(
            self.function.function_space.mesh.comm.allreduce(local, op=MPI.MAX)
        )

    def global_count_nonzero(self, *, tolerance: float = 0.0) -> int:
        """Count owned integration-point entries above a scalar tolerance."""

        owned = np.asarray(self.owned_values).reshape(-1)
        local = int(np.count_nonzero(np.abs(owned) > float(tolerance)))
        return int(
            self.function.function_space.mesh.comm.allreduce(local, op=MPI.SUM)
        )

    def assign(self, values) -> None:
        selected = np.asarray(values, dtype=self.function.x.array.dtype)
        expected = self._cell_count() * len(self.points) * self.component_count
        if selected.size != expected:
            raise ValueError(
                f"{self.function.name} requires {expected} cell-point values, "
                f"got {selected.size}."
            )
        V = self.function.function_space
        block_size = int(V.dofmap.bs)
        flat = selected.reshape((-1, block_size))
        points_per_cell = len(self.points)
        for cell in range(self._cell_count()):
            dofs = V.dofmap.cell_dofs(cell)
            for point, dof in enumerate(dofs):
                start = int(dof) * block_size
                self.function.x.array[start : start + block_size] = flat[
                    cell * points_per_cell + point
                ]
        self.function.x.scatter_forward()

    def cell_average(self, *, name: str | None = None):
        """Recover quadrature values as weighted DG0 cell averages.

        This is a discontinuous scientific field, not nodal extrapolation or
        contour smoothing. The reference-cell quadrature weights are retained
        explicitly instead of assuming that every rule has equal weights.
        """

        domain = self.function.function_space.mesh
        element = (
            ("DG", 0)
            if not self.value_shape
            else ("DG", 0, self.value_shape)
        )
        output = fem.Function(
            fem.functionspace(domain, element),
            name=name or self.function.name,
        )
        point_values = self.values.reshape(
            (self._cell_count(), len(self.points), self.component_count)
        )
        normalized_weights = self.weights / np.sum(self.weights)
        averages = np.einsum("p,cpi->ci", normalized_weights, point_values)
        block_size = int(output.function_space.dofmap.bs)
        for cell in range(self._cell_count()):
            dofs = output.function_space.dofmap.cell_dofs(cell)
            if len(dofs) != 1:
                raise RuntimeError("DG0 output requires one block dof per cell.")
            start = int(dofs[0]) * block_size
            output.x.array[start : start + block_size] = averages[cell]
        output.x.scatter_forward()
        return output

    def _cell_count(self) -> int:
        domain = self.function.function_space.mesh
        index_map = domain.topology.index_map(domain.topology.dim)
        return int(index_map.size_local + index_map.num_ghosts)


@dataclass
class J2QuadratureState:
    """Committed/trial integration-point state for 3D small-strain J2."""

    plastic_strain: QuadratureField
    equivalent_plastic_strain: QuadratureField
    trial_plastic_strain: QuadratureField
    trial_equivalent_plastic_strain: QuadratureField
    stress: QuadratureField
    tangent: QuadratureField
    degree: int
    scheme: str = "default"
    transaction: QuadratureTransaction = field(init=False)

    def __post_init__(self) -> None:
        self.transaction = QuadratureTransaction(
            committed={
                "plastic_strain": self.plastic_strain,
                "equivalent_plastic_strain": self.equivalent_plastic_strain,
            },
            trial={
                "plastic_strain": self.trial_plastic_strain,
                "equivalent_plastic_strain": (
                    self.trial_equivalent_plastic_strain
                ),
            },
            schema="agentfem.j2-small-strain-state",
            metadata={
                "integration": "radial_return",
                "tangent": "algorithmic_consistent",
            },
        )

    @classmethod
    def create(cls, domain, *, degree: int = 2, scheme: str = "default"):
        common = {"degree": degree, "scheme": scheme}
        return cls(
            plastic_strain=QuadratureField.create(
                domain, name="PE", value_shape=(3, 3), **common
            ),
            equivalent_plastic_strain=QuadratureField.create(
                domain, name="PEEQ", **common
            ),
            trial_plastic_strain=QuadratureField.create(
                domain, name="PE_trial", value_shape=(3, 3), **common
            ),
            trial_equivalent_plastic_strain=QuadratureField.create(
                domain, name="PEEQ_trial", **common
            ),
            stress=QuadratureField.create(
                domain, name="S", value_shape=(3, 3), **common
            ),
            tangent=QuadratureField.create(
                domain, name="DDSDDE", value_shape=(3, 3, 3, 3), **common
            ),
            degree=int(degree),
            scheme=scheme,
        )

    @property
    def domain(self):
        return self.stress.function.function_space.mesh

    @property
    def measure(self):
        return ufl.Measure(
            "dx",
            domain=self.domain,
            metadata={
                "quadrature_degree": self.degree,
                "quadrature_scheme": self.scheme,
            },
        )

    def evaluate_strain(self, strain_expression) -> np.ndarray:
        """Evaluate a 3x3 strain expression at this state's quadrature points."""

        topology = self.domain.topology
        cell_dim = topology.dim
        topology.create_connectivity(cell_dim, cell_dim)
        index_map = topology.index_map(cell_dim)
        cells = np.arange(
            index_map.size_local + index_map.num_ghosts,
            dtype=np.int32,
        )
        evaluated = fem.Expression(
            strain_expression,
            self.stress.points,
        ).eval(self.domain, cells)
        return np.asarray(evaluated).reshape((-1, 3, 3))

    def update(
        self,
        strain_values,
        material: J2LinearIsotropicHardening | QuadratureMaterialMap,
    ) -> dict[str, float | int]:
        """Update trial stress/state from the last committed state."""

        strains = np.asarray(strain_values, dtype=float).reshape((-1, 3, 3))
        committed_pe = self.plastic_strain.values
        committed_peeq = self.equivalent_plastic_strain.values.reshape(-1)
        if len(strains) != len(committed_pe):
            raise ValueError(
                "Strain evaluation and quadrature-state layouts do not match."
            )
        stresses = np.empty_like(self.stress.values)
        tangents = np.empty_like(self.tangent.values)
        trial_pe = np.empty_like(self.trial_plastic_strain.values)
        trial_peeq = np.empty_like(
            self.trial_equivalent_plastic_strain.values.reshape(-1)
        )
        plastic_points = 0
        maximum_increment = 0.0
        local_problem = None
        points_per_cell = len(self.stress.points)
        cell_map = self.domain.topology.index_map(self.domain.topology.dim)
        owned_points = int(cell_map.size_local) * points_per_cell
        for index, strain in enumerate(strains):
            old = J2PlasticState(committed_pe[index], committed_peeq[index])
            selected_material = (
                material.material_for_point(index, points_per_cell=points_per_cell)
                if isinstance(material, QuadratureMaterialMap)
                else material
            )
            try:
                update = selected_material.update(strain, old)
            except Exception as exc:
                local_problem = (
                    f"J2 material update failed at local quadrature point {index}: "
                    f"{type(exc).__name__}: {exc}"
                )
                break
            stresses[index] = update.stress
            tangents[index] = update.algorithmic_tangent
            trial_pe[index] = update.state.plastic_strain
            trial_peeq[index] = update.state.equivalent_plastic_strain
            if index < owned_points:
                plastic_points += int(not update.elastic)
                maximum_increment = max(
                    maximum_increment,
                    update.plastic_multiplier_increment,
                )
        problems = self.domain.comm.allgather(local_problem)
        if any(problem is not None for problem in problems):
            rank = next(
                index for index, problem in enumerate(problems) if problem is not None
            )
            raise RuntimeError(f"Rank {rank}: {problems[rank]}")
        self.stress.assign(stresses)
        self.tangent.assign(tangents)
        self.trial_plastic_strain.assign(trial_pe)
        self.trial_equivalent_plastic_strain.assign(trial_peeq)
        return {
            "points": int(self.domain.comm.allreduce(owned_points, op=MPI.SUM)),
            "plastic_points": int(
                self.domain.comm.allreduce(plastic_points, op=MPI.SUM)
            ),
            "maximum_plastic_increment": float(
                self.domain.comm.allreduce(maximum_increment, op=MPI.MAX)
            ),
        }

    def commit(self) -> None:
        self.transaction.commit()

    def rollback(self) -> None:
        self.transaction.rollback()

    def snapshot(self) -> dict[str, np.ndarray]:
        return self.transaction.snapshot()

    def restore(self, snapshot: dict[str, np.ndarray]) -> None:
        self.transaction.restore(snapshot)

    def save(self, path, *, material=None) -> Path:
        """Collectively save state by original physical cell and point."""

        return save_portable_quadrature_state(path, self, material=material)

    def load(self, path, *, material=None) -> None:
        load_portable_quadrature_state(path, self, material=material)

    def summary(self) -> dict[str, object]:
        cell_map = self.domain.topology.index_map(self.domain.topology.dim)
        points_per_cell = len(self.stress.points)
        return {
            "kind": "j2_quadrature_state",
            "degree": self.degree,
            "scheme": self.scheme,
            "points_local": int(len(self.equivalent_plastic_strain.values)),
            "points_owned": int(cell_map.size_local) * points_per_cell,
            "points_global": int(cell_map.size_global) * points_per_cell,
            "portable_cell_identity": "dolfinx_original_cell_index",
            "state_variables": ("plastic_strain", "equivalent_plastic_strain"),
            "trial_fields": ("stress", "algorithmic_tangent"),
            "transaction": self.transaction.summary(),
        }

    def output_fields(self) -> tuple[object, ...]:
        """Return presentation-ready cell averages of committed J2 fields."""

        return (
            self.stress.cell_average(name="S"),
            self.plastic_strain.cell_average(name="PE"),
            self.equivalent_plastic_strain.cell_average(name="PEEQ"),
            self.equivalent_stress().cell_average(name="MISES"),
        )

    def equivalent_stress(self) -> QuadratureField:
        """Return pointwise von Mises stress on the constitutive quadrature."""

        stress = self.stress.values
        trace = np.trace(stress, axis1=-2, axis2=-1)
        identity = np.eye(3, dtype=stress.dtype)
        deviator = stress - trace[:, None, None] * identity / 3.0
        mises = np.sqrt(1.5 * np.sum(deviator * deviator, axis=(-2, -1)))
        output = QuadratureField.create(
            self.domain,
            name="MISES",
            degree=self.degree,
            scheme=self.scheme,
        )
        output.assign(mises)
        return output


@dataclass
class CreepQuadratureState:
    """Committed/trial integration-point state for implicit 3D creep."""

    creep_strain: QuadratureField
    equivalent_creep_strain: QuadratureField
    trial_creep_strain: QuadratureField
    trial_equivalent_creep_strain: QuadratureField
    stress: QuadratureField
    tangent: QuadratureField
    degree: int
    scheme: str = "default"
    transaction: QuadratureTransaction = field(init=False)

    def __post_init__(self) -> None:
        self.transaction = QuadratureTransaction(
            committed={
                "creep_strain": self.creep_strain,
                "equivalent_creep_strain": self.equivalent_creep_strain,
            },
            trial={
                "creep_strain": self.trial_creep_strain,
                "equivalent_creep_strain": self.trial_equivalent_creep_strain,
            },
            schema="agentfem.power-law-creep-small-strain-state",
            metadata={
                "integration": "backward_euler",
                "tangent": "algorithmic_consistent",
            },
        )

    @classmethod
    def create(cls, domain, *, degree: int = 2, scheme: str = "default"):
        common = {"degree": degree, "scheme": scheme}
        return cls(
            creep_strain=QuadratureField.create(
                domain, name="CE", value_shape=(3, 3), **common
            ),
            equivalent_creep_strain=QuadratureField.create(
                domain, name="CEEQ", **common
            ),
            trial_creep_strain=QuadratureField.create(
                domain, name="CE_trial", value_shape=(3, 3), **common
            ),
            trial_equivalent_creep_strain=QuadratureField.create(
                domain, name="CEEQ_trial", **common
            ),
            stress=QuadratureField.create(
                domain, name="S", value_shape=(3, 3), **common
            ),
            tangent=QuadratureField.create(
                domain, name="DDSDDE", value_shape=(3, 3, 3, 3), **common
            ),
            degree=int(degree),
            scheme=scheme,
        )

    @property
    def domain(self):
        return self.stress.function.function_space.mesh

    @property
    def measure(self):
        return ufl.Measure(
            "dx",
            domain=self.domain,
            metadata={
                "quadrature_degree": self.degree,
                "quadrature_scheme": self.scheme,
            },
        )

    def evaluate_strain(self, strain_expression) -> np.ndarray:
        topology = self.domain.topology
        cell_dim = topology.dim
        topology.create_connectivity(cell_dim, cell_dim)
        index_map = topology.index_map(cell_dim)
        cells = np.arange(
            index_map.size_local + index_map.num_ghosts,
            dtype=np.int32,
        )
        evaluated = fem.Expression(strain_expression, self.stress.points).eval(
            self.domain,
            cells,
        )
        return np.asarray(evaluated).reshape((-1, 3, 3))

    def evaluate_scalar(self, expression) -> np.ndarray:
        """Evaluate a scalar field at the creep quadrature identity."""

        selected = getattr(expression, "value", expression)
        topology = self.domain.topology
        cell_dim = topology.dim
        topology.create_connectivity(cell_dim, cell_dim)
        index_map = topology.index_map(cell_dim)
        cells = np.arange(
            index_map.size_local + index_map.num_ghosts,
            dtype=np.int32,
        )
        evaluated = fem.Expression(selected, self.stress.points).eval(
            self.domain,
            cells,
        )
        return np.asarray(evaluated, dtype=float).reshape(-1)

    def update(
        self,
        strain_values,
        material: IsotropicPowerLawCreepMaterial | QuadratureMaterialMap,
        *,
        time_start: float,
        time_end: float,
        temperature_values=None,
    ) -> dict[str, float | int]:
        """Update trial creep state, stress, and tangent from committed state."""

        strains = np.asarray(strain_values, dtype=float).reshape((-1, 3, 3))
        committed_ce = self.creep_strain.values
        committed_ceeq = self.equivalent_creep_strain.values.reshape(-1)
        if len(strains) != len(committed_ce):
            raise ValueError(
                "Strain evaluation and creep quadrature layouts do not match."
            )
        stresses = np.empty_like(self.stress.values)
        tangents = np.empty_like(self.tangent.values)
        trial_ce = np.empty_like(self.trial_creep_strain.values)
        trial_ceeq = np.empty_like(
            self.trial_equivalent_creep_strain.values.reshape(-1)
        )
        maximum_increment = 0.0
        maximum_local_iterations = 0
        active_points = 0
        local_problem = None
        temperatures = None
        if temperature_values is not None:
            temperatures = np.asarray(temperature_values, dtype=float).reshape(-1)
            if len(temperatures) != len(strains):
                raise ValueError(
                    "Temperature and creep quadrature layouts do not match."
                )
            if np.any(~np.isfinite(temperatures)) or np.any(temperatures <= 0.0):
                raise ValueError(
                    "Arrhenius quadrature temperatures must be positive kelvin values."
                )
        points_per_cell = len(self.stress.points)
        cell_map = self.domain.topology.index_map(self.domain.topology.dim)
        owned_points = int(cell_map.size_local) * points_per_cell
        local_temperatures = (
            None if temperatures is None else temperatures[:owned_points]
        )
        for index, strain in enumerate(strains):
            old = ImplicitCreepState(committed_ce[index], committed_ceeq[index])
            selected_material = (
                material.material_for_point(index, points_per_cell=points_per_cell)
                if isinstance(material, QuadratureMaterialMap)
                else material
            )
            try:
                update = selected_material.update(
                    strain,
                    time_start=time_start,
                    time_end=time_end,
                    state=old,
                    temperature=(
                        None if temperatures is None else float(temperatures[index])
                    ),
                )
            except Exception as exc:
                local_problem = (
                    "Creep material update failed at local quadrature point "
                    f"{index}: {type(exc).__name__}: {exc}"
                )
                break
            stresses[index] = update.stress
            tangents[index] = update.algorithmic_tangent
            trial_ce[index] = update.state.creep_strain
            trial_ceeq[index] = update.state.equivalent_creep_strain
            if index < owned_points:
                maximum_increment = max(
                    maximum_increment, update.equivalent_increment
                )
                maximum_local_iterations = max(
                    maximum_local_iterations,
                    update.local_iterations,
                )
                active_points += int(update.equivalent_increment > 0.0)
        problems = self.domain.comm.allgather(local_problem)
        if any(problem is not None for problem in problems):
            rank = next(
                index for index, problem in enumerate(problems) if problem is not None
            )
            raise RuntimeError(f"Rank {rank}: {problems[rank]}")
        self.stress.assign(stresses)
        self.tangent.assign(tangents)
        self.trial_creep_strain.assign(trial_ce)
        self.trial_equivalent_creep_strain.assign(trial_ceeq)
        return {
            "points": int(self.domain.comm.allreduce(owned_points, op=MPI.SUM)),
            "creeping_points": int(
                self.domain.comm.allreduce(active_points, op=MPI.SUM)
            ),
            "maximum_creep_increment": float(
                self.domain.comm.allreduce(maximum_increment, op=MPI.MAX)
            ),
            "maximum_local_iterations": int(
                self.domain.comm.allreduce(maximum_local_iterations, op=MPI.MAX)
            ),
            "minimum_temperature": (
                None
                if local_temperatures is None
                else float(
                    self.domain.comm.allreduce(
                        np.min(local_temperatures, initial=np.inf), op=MPI.MIN
                    )
                )
            ),
            "maximum_temperature": (
                None
                if local_temperatures is None
                else float(
                    self.domain.comm.allreduce(
                        np.max(local_temperatures, initial=-np.inf), op=MPI.MAX
                    )
                )
            ),
        }

    def refresh_response(
        self,
        strain_values,
        material: IsotropicPowerLawCreepMaterial | QuadratureMaterialMap,
    ) -> None:
        """Recover accepted stress without changing committed/trial state."""

        strains = np.asarray(strain_values, dtype=float).reshape((-1, 3, 3))
        committed_ce = self.creep_strain.values
        committed_ceeq = self.equivalent_creep_strain.values.reshape(-1)
        if len(strains) != len(committed_ce):
            raise ValueError(
                "Strain evaluation and creep quadrature layouts do not match."
            )
        stresses = np.empty_like(self.stress.values)
        points_per_cell = len(self.stress.points)
        tangents = np.empty_like(self.tangent.values)
        for index, strain in enumerate(strains):
            selected_material = (
                material.material_for_point(index, points_per_cell=points_per_cell)
                if isinstance(material, QuadratureMaterialMap)
                else material
            )
            stresses[index] = selected_material.stress_from_state(
                strain,
                ImplicitCreepState(committed_ce[index], committed_ceeq[index]),
            )
            tangents[index] = selected_material.elastic_tangent()
        self.stress.assign(stresses)
        self.tangent.assign(tangents)
        self.rollback()

    def save(self, path, *, material=None) -> Path:
        """Collectively save state by original physical cell and point."""

        return save_portable_quadrature_state(path, self, material=material)

    def load(self, path, *, material=None) -> None:
        load_portable_quadrature_state(path, self, material=material)

    def commit(self) -> None:
        self.transaction.commit()

    def rollback(self) -> None:
        self.transaction.rollback()

    def snapshot(self) -> dict[str, np.ndarray]:
        return self.transaction.snapshot()

    def restore(self, snapshot: Mapping[str, object]) -> None:
        self.transaction.restore(snapshot)

    def equivalent_stress(self) -> QuadratureField:
        stress = self.stress.values
        trace = np.trace(stress, axis1=-2, axis2=-1)
        identity = np.eye(3, dtype=stress.dtype)
        deviator = stress - trace[:, None, None] * identity / 3.0
        mises = np.sqrt(1.5 * np.sum(deviator * deviator, axis=(-2, -1)))
        output = QuadratureField.create(
            self.domain,
            name="MISES",
            degree=self.degree,
            scheme=self.scheme,
        )
        output.assign(mises)
        return output

    def output_fields(self) -> tuple[object, ...]:
        return (
            self.stress.cell_average(name="S"),
            self.creep_strain.cell_average(name="CE"),
            self.equivalent_creep_strain.cell_average(name="CEEQ"),
            self.equivalent_stress().cell_average(name="MISES"),
        )

    def summary(self) -> dict[str, object]:
        cell_map = self.domain.topology.index_map(self.domain.topology.dim)
        points_per_cell = len(self.stress.points)
        return {
            "kind": "creep_quadrature_state",
            "degree": self.degree,
            "scheme": self.scheme,
            "points_local": int(len(self.equivalent_creep_strain.values)),
            "points_owned": int(cell_map.size_local) * points_per_cell,
            "points_global": int(cell_map.size_global) * points_per_cell,
            "portable_cell_identity": "dolfinx_original_cell_index",
            "state_variables": ("creep_strain", "equivalent_creep_strain"),
            "trial_fields": ("stress", "algorithmic_tangent"),
            "transaction": self.transaction.summary(),
        }


__all__ = [
    "CreepQuadratureState",
    "J2QuadratureState",
    "QuadratureField",
    "QuadratureMaterialMap",
    "QuadratureTransaction",
    "load_portable_quadrature_state",
    "save_portable_quadrature_state",
]
