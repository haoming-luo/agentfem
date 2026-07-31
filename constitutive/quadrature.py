"""Integration-point storage for stateful constitutive models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import basix
import basix.ufl
import numpy as np
import ufl
from dolfinx import fem

from .plasticity import J2LinearIsotropicHardening, J2PlasticState


@dataclass
class QuadratureField:
    """A DOLFINx quadrature function with an explicit NumPy point view."""

    function: object
    points: np.ndarray
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
        points, _ = basix.make_quadrature(
            domain.basix_cell(),
            int(degree),
            rule=getattr(basix.QuadratureType, scheme),
        )
        element = basix.ufl.quadrature_element(
            domain.basix_cell(),
            value_shape=shape,
            points=points,
            weights=basix.make_quadrature(
                domain.basix_cell(),
                int(degree),
                rule=getattr(basix.QuadratureType, scheme),
            )[1],
        )
        space = fem.functionspace(domain, element)
        return cls(
            function=fem.Function(space, name=name),
            points=np.asarray(points),
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
        """Project quadrature values to an XDMF-friendly DG0 cell field."""

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
        averages = self.values.reshape(
            (self._cell_count(), len(self.points), self.component_count)
        ).mean(axis=1)
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
        self.plastic_strain.assign(self.trial_plastic_strain.values)
        self.equivalent_plastic_strain.assign(
            self.trial_equivalent_plastic_strain.values
        )

    def rollback(self) -> None:
        self.trial_plastic_strain.assign(self.plastic_strain.values)
        self.trial_equivalent_plastic_strain.assign(
            self.equivalent_plastic_strain.values
        )

    def snapshot(self) -> dict[str, np.ndarray]:
        return {
            "plastic_strain": self.plastic_strain.values.copy(),
            "equivalent_plastic_strain": (
                self.equivalent_plastic_strain.values.copy()
            ),
        }

    def restore(self, snapshot: dict[str, np.ndarray]) -> None:
        self.plastic_strain.assign(snapshot["plastic_strain"])
        self.equivalent_plastic_strain.assign(
            snapshot["equivalent_plastic_strain"]
        )
        self.rollback()

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
        }

    def output_fields(self) -> tuple[object, ...]:
        """Return cell-averaged stress, plastic strain, and PEEQ fields."""

        return (
            self.stress.cell_average(name="S"),
            self.plastic_strain.cell_average(name="PE"),
            self.equivalent_plastic_strain.cell_average(name="PEEQ"),
        )


__all__ = ["J2QuadratureState", "QuadratureField"]
