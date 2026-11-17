# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Owned-cell composite assembly using DOLFINx DOFs and PETSc ghost exchange."""

from dataclasses import dataclass
import numpy as np
from .composite_tet import (
    CompositeTet10,
    composite_tet10,
    composite_tet11,
    _EDGES,
    _SUBTETS,
    _QUAD,
)
from ..provenance import collective_call


def is_composite_space(V):
    return getattr(V, "_agentfem_composite_tet10", False)


def local_order(V):
    import basix

    el = V.element.basix_element
    from ._composite_storage import scalar_element

    independent = (
        el.family == basix.ElementFamily.custom
        and basix.finite_element.FiniteElement(el) == scalar_element()
    )
    if (
        V.mesh.geometry.dim != 3
        or V.mesh.topology.dim != 3
        or V.mesh.topology.cell_type.name != "tetrahedron"
        or not (independent or (el.family == basix.ElementFamily.P and el.degree == 2))
        or el.discontinuous
        or V.dofmap.bs != 3
        or V.dofmap.index_map_bs != 3
    ):
        raise ValueError(
            "AFM-COMPOSITE-TET: blocked composite-compatible tetrahedral xyz storage required."
        )
    vertices = np.vstack((np.zeros(3), np.eye(3)))
    expected = np.vstack((vertices, vertices[_EDGES].mean(axis=1)))
    if independent:
        expected = np.vstack((expected, np.full((1, 3), 0.25)))
    points = el.points
    order = np.argmin(np.linalg.norm(expected[:, None] - points[None], axis=2), axis=1)
    if not np.allclose(expected, points[order], rtol=0, atol=1e-13):
        raise ValueError("AFM-COMPOSITE-TET: vertex/mid-edge nodal storage required.")
    return order


def interpolation_weights(V, cell, point):
    """Physical piecewise-linear weights in DOLFINx cell DOF order."""
    order = local_order(V)
    ids = V.dofmap.cell_dofs(cell)[order]
    x = V.tabulate_dof_coordinates()[ids]
    q = np.linalg.solve((x[1:4] - x[0]).T, np.asarray(point) - x[0])
    vertices = np.vstack((np.zeros(3), np.eye(3)))
    ref = np.vstack((vertices, vertices[_EDGES].mean(axis=1), np.full(3, 0.25)))
    for sub in _SUBTETS:
        p = ref[sub]
        tail = np.linalg.solve((p[1:] - p[0]).T, q - p[0])
        weights = np.r_[1 - tail.sum(), tail]
        if weights.min() >= -1e-10:
            full = np.zeros(11)
            full[sub] = weights
            if len(order) == 10:
                full[4:10] += full[10] / 6
            result = np.zeros(len(order))
            result[order] = full[: len(order)]
            return result
    raise ValueError("AFM-COMPOSITE-TET: point outside composite cell.")


@dataclass(frozen=True)
class _ProjectedCell:
    """Compact recovery data: do not retain 48-point stabilization scratch."""

    volume: float
    gradient_coefficients: np.ndarray
    scalar_mass: np.ndarray
    gradients = CompositeTet10.gradients
    strain_matrices = CompositeTet10.strain_matrices


class CompositeTetResidual:
    def __init__(self, displacement, matrices, density, policy):
        from ..fields import unwrap
        from petsc4py import PETSc
        from .. import assembly

        self.displacement = unwrap(displacement)
        V = self.displacement.function_space
        self.comm = V.mesh.comm
        self.policy = policy
        self.material_batch = None
        self.trial_time = 0.0
        self.cells = None  # compatibility with shared explicit-body ledger

        def prepare():
            order = local_order(V)
            if len(order) != (11 if policy.independent_center else 10):
                raise ValueError(
                    "Composite policy and nodal storage disagree; declare independent_center on the field."
                )
            self.nodes_per_cell = len(order)
            self.dofs_per_cell = 3 * len(order)
            count = V.mesh.topology.index_map(3).size_local
            self.cell_nodes = np.array(
                [V.dofmap.cell_dofs(i)[order] for i in range(count)], dtype=np.int32
            ).reshape(-1, self.nodes_per_cell)
            xyz = V.tabulate_dof_coordinates()
            self.materials = np.broadcast_to(matrices, (count, 6, 6)).copy()
            reorder = [0, 1, 2, 5, 3, 4]
            self.materials = self.materials[:, reorder][:, :, reorder]
            self.stiffness = np.empty((count, self.dofs_per_cell, self.dofs_per_cell))
            self.cell_mass = np.empty((count, self.dofs_per_cell))
            self.elements, self.cell_dt = [], []
            for c, ids in enumerate(self.cell_nodes):
                e = (
                    composite_tet11(xyz[ids[:10]])
                    if policy.independent_center
                    else composite_tet10(xyz[ids])
                )
                K = e.stiffness(
                    self.materials[c], mean_dilatation=policy.mean_dilatation
                )
                mass = e.mass(density[c])
                invsqrt = 1 / np.sqrt(mass)
                omega2 = np.linalg.eigvalsh(invsqrt[:, None] * K * invsqrt[None, :])[-1]
                self.stiffness[c] = K
                self.cell_mass[c] = mass
                self.cell_dt.append(float(2 / np.sqrt(omega2)))
                self.elements.append(
                    _ProjectedCell(e.volume, e.gradient_coefficients, e.scalar_mass)
                )

        collective_call(prepare, comm=self.comm, label="Composite Tet10 preparation")
        vector = self.displacement.x.petsc_vec.duplicate()
        try:
            with vector.localForm() as local:
                local.set(0)
                np.add.at(
                    local.array.reshape(-1, 3),
                    self.cell_nodes.ravel(),
                    self.cell_mass.reshape(-1, 3),
                )
            vector.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )
            self.mass_diagonal = vector.array.copy()
        finally:
            vector.destroy()
        self.inv_mass = assembly.inverse_diagonal(self.mass_diagonal, comm=self.comm)
        self.identity_arrays = (
            self.cell_nodes,
            self.mass_diagonal,
            self.stiffness,
            V.mesh.geometry.x,
            self.materials,
        )
        self.schema = "agentfem.composite-tet10-elastic.v1"
        V._agentfem_composite_tet10 = True
        V._agentfem_element_policy = policy

    def assemble_vector(self):
        from petsc4py import PETSc

        self.displacement.x.scatter_forward()
        vector = self.displacement.x.petsc_vec.duplicate()
        try:

            def local_assembly():
                u = self.displacement.x.array.reshape(-1, 3)[self.cell_nodes].reshape(
                    -1, self.dofs_per_cell
                )
                # Translation is an exact null mode; remove it before arithmetic
                # to avoid cancellation when rigid motion dwarfs deformation.
                u = (
                    u.reshape(-1, self.nodes_per_cell, 3)
                    - u.reshape(-1, self.nodes_per_cell, 3).mean(axis=1)[:, None, :]
                ).reshape(-1, self.dofs_per_cell)
                force = (
                    np.einsum("cij,cj->ci", self.stiffness, u)
                    if self.material_batch is None
                    else self.material_batch.evaluate(u, self.trial_time)
                )
                with vector.localForm() as local:
                    local.set(0)
                    np.add.at(
                        local.array.reshape(-1, 3),
                        self.cell_nodes.ravel(),
                        force.reshape(-1, 3),
                    )

            collective_call(
                local_assembly, comm=self.comm, label="Composite Tet10 force"
            )
            vector.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )
            return vector
        except Exception:
            vector.destroy()
            raise

    def enable_material(self, material):
        if self.policy.mean_dilatation or self.policy.independent_center:
            raise NotImplementedError(
                "Nonlinear composite materials currently require mean_dilatation=False and independent_center=False; elastic stabilization cannot stand in for plastic constitutive response."
            )
        from ._composite_material import CompositeMaterialBatch

        self.material_batch = CompositeMaterialBatch(self.elements, material)
        import json

        self.identity_arrays = (
            *self.identity_arrays,
            np.frombuffer(
                json.dumps(material.summary(), sort_keys=True).encode(), dtype=np.uint8
            ),
        )
        self.schema = "agentfem.composite-tet10-material.v1"

    def energies(self):
        from mpi4py import MPI

        if self.material_batch is not None:
            b = self.material_batch
            stored = np.sum(b.volume[:, None] / 4 * b.stored.reshape(-1, 4))
            dissipated = np.sum(b.volume[:, None] / 4 * b.dissipation.reshape(-1, 4))
            values = self.comm.allreduce(np.array([stored, dissipated]), op=MPI.SUM)
            return dict(
                strain_energy=float(values[0]),
                hourglass_energy=0.0,
                material_dissipation=float(values[1]),
            )

        self.displacement.x.scatter_forward()
        u = self.displacement.x.array.reshape(-1, 3)[self.cell_nodes].reshape(
            -1, self.dofs_per_cell
        )
        u = (
            u.reshape(-1, self.nodes_per_cell, 3)
            - u.reshape(-1, self.nodes_per_cell, 3).mean(axis=1)[:, None, :]
        ).reshape(-1, self.dofs_per_cell)
        total = 0.5 * np.einsum("ci,cij,cj->", u, self.stiffness, u)
        arithmetic_scale = 0.5 * np.einsum(
            "ci,cij,cj->", np.abs(u), np.abs(self.stiffness), np.abs(u)
        )
        physical = total
        if self.policy.mean_dilatation or self.policy.independent_center:
            # Contract displacement first: no per-cell B-matrix reconstruction.
            coefficients = np.asarray(
                [e.gradient_coefficients for e in self.elements]
            ).reshape(-1, 4, self.nodes_per_cell, 3)
            grad = np.einsum(
                "qa,caij,cik->cqkj",
                _QUAD,
                coefficients,
                u.reshape(-1, self.nodes_per_cell, 3),
                optimize=True,
            )
            strain = np.stack(
                (
                    grad[:, :, 0, 0],
                    grad[:, :, 1, 1],
                    grad[:, :, 2, 2],
                    grad[:, :, 0, 1] + grad[:, :, 1, 0],
                    grad[:, :, 1, 2] + grad[:, :, 2, 1],
                    grad[:, :, 0, 2] + grad[:, :, 2, 0],
                ),
                axis=-1,
            )
            trace = strain[:, :, :3].sum(axis=-1)
            if self.policy.mean_dilatation:
                strain[:, :, :3] += (trace.mean(axis=1)[:, None] - trace)[
                    :, :, None
                ] / 3
            volume = np.asarray([e.volume for e in self.elements])
            physical = np.einsum(
                "c,cqi,cij,cqj->",
                volume / 8,
                strain,
                self.materials,
                strain,
                optimize=True,
            )
        physical, total, arithmetic_scale = self.comm.allreduce(
            np.array([physical, total, arithmetic_scale]), op=MPI.SUM
        )
        tolerance = (
            64 * np.finfo(float).eps * max(arithmetic_scale, np.finfo(float).tiny)
        )
        if total < -tolerance:
            raise ValueError(
                "Composite total strain energy is negative beyond roundoff."
            )
        total = max(total, 0.0)
        correction = (
            total - physical
            if (self.policy.mean_dilatation or self.policy.independent_center)
            else 0.0
        )
        if correction < -tolerance:
            raise ValueError(
                "Composite stabilization energy is negative beyond roundoff."
            )
        correction = max(correction, 0.0)
        return {
            "strain_energy": float(total - correction),
            "hourglass_energy": float(correction),
        }

    def stable_dt(self, safety=0.8):
        from mpi4py import MPI

        return safety * self.comm.allreduce(
            min(self.cell_dt, default=np.inf), op=MPI.MIN
        )
