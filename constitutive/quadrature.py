"""Integration-point storage for stateful constitutive models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import basix
import basix.ufl
import numpy as np
import ufl
from dolfinx import fem

from .plasticity import J2LinearIsotropicHardening, J2PlasticState
from .creep import ImplicitCreepState, IsotropicPowerLawCreepMaterial


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
        material: J2LinearIsotropicHardening,
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
        for index, strain in enumerate(strains):
            old = J2PlasticState(committed_pe[index], committed_peeq[index])
            update = material.update(strain, old)
            stresses[index] = update.stress
            tangents[index] = update.algorithmic_tangent
            trial_pe[index] = update.state.plastic_strain
            trial_peeq[index] = update.state.equivalent_plastic_strain
            plastic_points += int(not update.elastic)
            maximum_increment = max(
                maximum_increment,
                update.plastic_multiplier_increment,
            )
        self.stress.assign(stresses)
        self.tangent.assign(tangents)
        self.trial_plastic_strain.assign(trial_pe)
        self.trial_equivalent_plastic_strain.assign(trial_peeq)
        return {
            "points": len(strains),
            "plastic_points": plastic_points,
            "maximum_plastic_increment": maximum_increment,
        }

    def commit(self) -> None:
        self.transaction.commit()

    def rollback(self) -> None:
        self.transaction.rollback()

    def snapshot(self) -> dict[str, np.ndarray]:
        return self.transaction.snapshot()

    def restore(self, snapshot: dict[str, np.ndarray]) -> None:
        self.transaction.restore(snapshot)

    def save(self, path) -> Path:
        """Write a serial checkpoint; distributed checkpointing stays explicit."""

        if self.domain.comm.size != 1:
            raise NotImplementedError(
                "Portable distributed quadrature checkpoints require a global "
                "cell identity map and are not implemented yet."
            )
        selected = Path(path)
        selected.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            selected,
            schema="agentfem.j2-quadrature-state.v1",
            degree=self.degree,
            plastic_strain=self.plastic_strain.values,
            equivalent_plastic_strain=(
                self.equivalent_plastic_strain.values
            ),
        )
        return selected

    def load(self, path) -> None:
        if self.domain.comm.size != 1:
            raise NotImplementedError(
                "Portable distributed quadrature checkpoints are not implemented yet."
            )
        with np.load(path, allow_pickle=False) as data:
            if str(data["schema"]) != "agentfem.j2-quadrature-state.v1":
                raise ValueError("Unsupported J2 quadrature checkpoint schema.")
            if int(data["degree"]) != self.degree:
                raise ValueError("Checkpoint quadrature degree does not match the state.")
            self.restore(
                {
                    "plastic_strain": data["plastic_strain"],
                    "equivalent_plastic_strain": data[
                        "equivalent_plastic_strain"
                    ],
                }
            )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "j2_quadrature_state",
            "degree": self.degree,
            "scheme": self.scheme,
            "points_local": int(len(self.equivalent_plastic_strain.values)),
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
        material: IsotropicPowerLawCreepMaterial,
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
        for index, strain in enumerate(strains):
            old = ImplicitCreepState(committed_ce[index], committed_ceeq[index])
            update = material.update(
                strain,
                time_start=time_start,
                time_end=time_end,
                state=old,
                temperature=(
                    None if temperatures is None else float(temperatures[index])
                ),
            )
            stresses[index] = update.stress
            tangents[index] = update.algorithmic_tangent
            trial_ce[index] = update.state.creep_strain
            trial_ceeq[index] = update.state.equivalent_creep_strain
            maximum_increment = max(maximum_increment, update.equivalent_increment)
            maximum_local_iterations = max(
                maximum_local_iterations,
                update.local_iterations,
            )
            active_points += int(update.equivalent_increment > 0.0)
        self.stress.assign(stresses)
        self.tangent.assign(tangents)
        self.trial_creep_strain.assign(trial_ce)
        self.trial_equivalent_creep_strain.assign(trial_ceeq)
        return {
            "points": len(strains),
            "creeping_points": active_points,
            "maximum_creep_increment": maximum_increment,
            "maximum_local_iterations": maximum_local_iterations,
            "minimum_temperature": (
                None if temperatures is None else float(np.min(temperatures))
            ),
            "maximum_temperature": (
                None if temperatures is None else float(np.max(temperatures))
            ),
        }

    def refresh_response(
        self,
        strain_values,
        material: IsotropicPowerLawCreepMaterial,
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
        for index, strain in enumerate(strains):
            stresses[index] = material.stress_from_state(
                strain,
                ImplicitCreepState(committed_ce[index], committed_ceeq[index]),
            )
        self.stress.assign(stresses)
        elastic_tangent = material.elastic_tangent()
        self.tangent.assign(
            np.broadcast_to(
                elastic_tangent,
                (len(strains), *elastic_tangent.shape),
            )
        )
        self.rollback()

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
        return {
            "kind": "creep_quadrature_state",
            "degree": self.degree,
            "scheme": self.scheme,
            "points_local": int(len(self.equivalent_creep_strain.values)),
            "state_variables": ("creep_strain", "equivalent_creep_strain"),
            "trial_fields": ("stress", "algorithmic_tangent"),
            "transaction": self.transaction.summary(),
        }


__all__ = [
    "CreepQuadratureState",
    "J2QuadratureState",
    "QuadratureField",
    "QuadratureTransaction",
]
