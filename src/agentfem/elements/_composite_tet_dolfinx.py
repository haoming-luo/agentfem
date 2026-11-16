# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Owned-cell composite assembly using DOLFINx DOFs and PETSc ghost exchange."""

from dataclasses import dataclass
import numpy as np
from .composite_tet import CompositeTet10, composite_tet10, _EDGES, _SUBTETS
from ..provenance import collective_call


def is_composite_space(V):
    return getattr(V, "_agentfem_composite_tet10", False)


def local_order(V):
    import basix

    el = V.element.basix_element
    if (
        V.mesh.geometry.dim != 3
        or V.mesh.topology.dim != 3
        or V.mesh.topology.cell_type.name != "tetrahedron"
        or el.family != basix.ElementFamily.P
        or el.degree != 2
        or el.discontinuous
        or V.dofmap.bs != 3
        or V.dofmap.index_map_bs != 3
    ):
        raise ValueError(
            "AFM-COMPOSITE-TET: blocked continuous P2 tetrahedral xyz storage required."
        )
    vertices = np.vstack((np.zeros(3), np.eye(3)))
    expected = np.vstack((vertices, vertices[_EDGES].mean(axis=1)))
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
            full[4:10] += full[10] / 6
            result = np.zeros(10)
            result[order] = full[:10]
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
        self.cells = None  # compatibility with shared explicit-body ledger

        def prepare():
            order = local_order(V)
            count = V.mesh.topology.index_map(3).size_local
            self.cell_nodes = np.array(
                [V.dofmap.cell_dofs(i)[order] for i in range(count)], dtype=np.int32
            ).reshape(-1, 10)
            xyz = V.tabulate_dof_coordinates()
            self.materials = np.broadcast_to(matrices, (count, 6, 6)).copy()
            reorder = [0, 1, 2, 5, 3, 4]
            self.materials = self.materials[:, reorder][:, :, reorder]
            self.stiffness = np.empty((count, 30, 30))
            self.cell_mass = np.empty((count, 30))
            self.elements, self.cell_dt = [], []
            for c, ids in enumerate(self.cell_nodes):
                e = composite_tet10(xyz[ids])
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
                    -1, 30
                )
                # Translation is an exact null mode; remove it before arithmetic
                # to avoid cancellation when rigid motion dwarfs deformation.
                u = (
                    u.reshape(-1, 10, 3) - u.reshape(-1, 10, 3).mean(axis=1)[:, None, :]
                ).reshape(-1, 30)
                force = np.einsum("cij,cj->ci", self.stiffness, u)
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

    def energies(self):
        from mpi4py import MPI

        self.displacement.x.scatter_forward()
        u = self.displacement.x.array.reshape(-1, 3)[self.cell_nodes].reshape(-1, 30)
        u = (
            u.reshape(-1, 10, 3) - u.reshape(-1, 10, 3).mean(axis=1)[:, None, :]
        ).reshape(-1, 30)
        total = 0.5 * np.einsum("ci,cij,cj->", u, self.stiffness, u)
        arithmetic_scale = 0.5 * np.einsum(
            "ci,cij,cj->", np.abs(u), np.abs(self.stiffness), np.abs(u)
        )
        physical = 0.0
        for e, C, uc in zip(self.elements, self.materials, u):
            strain = e.strain_matrices(mean_dilatation=self.policy.mean_dilatation) @ uc
            physical += e.volume / 8 * np.einsum("qi,ij,qj->", strain, C, strain)
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
        correction = total - physical if self.policy.mean_dilatation else 0.0
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
