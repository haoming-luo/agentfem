# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private lowering to DOLFINx owned cells/ghost DOFs; no alternative mesh owner."""

import numpy as np

from ._uniform_hex import UniformHexBatch
from ..provenance import collective_call


class UniformHexResidual:
    """Elastic internal residual, positive assembled mass and separate energies.

    Restricted to blocked vector Q1. DOLFINx owns cells, DOFs and ghost exchange.
    This operator does not define constraints, results or checkpoint policy.
    """

    def __init__(
        self,
        displacement,
        stiffness,
        *,
        density,
        hourglass_modulus,
        hourglass_scale,
        chunk_size=1024,
    ):
        from ..fields import unwrap

        self.displacement = unwrap(displacement)
        space = self.displacement.function_space
        self.comm = space.mesh.comm
        collective_call(
            lambda: self._prepare_cells(
                stiffness, density, hourglass_modulus, hourglass_scale, chunk_size
            ),
            comm=self.comm,
            label="Uniform Hex8 local preparation",
        )
        vector = self.displacement.x.petsc_vec.duplicate()
        try:
            from petsc4py import PETSc

            with vector.localForm() as local:
                local.set(0)
                if self.cells is not None:
                    values = local.array.reshape(-1, 3)
                    for component in range(3):
                        np.add.at(
                            values[:, component],
                            self.cell_nodes.ravel(),
                            self.cells.lumped_mass.ravel(),
                        )
            vector.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )
            self.mass_diagonal = vector.array.copy()
        finally:
            vector.destroy()
        from .. import assembly

        self.inv_mass = assembly.inverse_diagonal(self.mass_diagonal, comm=self.comm)
        self.mass_diagonal.setflags(write=False)
        self.inv_mass.setflags(write=False)

    def _prepare_cells(
        self, stiffness, density, hourglass_modulus, hourglass_scale, chunk_size
    ):
        """Rank-local validation, guarded collectively before ghost assembly."""
        space = self.displacement.function_space
        element = space.element.basix_element
        if (
            space.dofmap.index_map_bs != 3
            or element.degree != 1
            or element.discontinuous
            or element.cell_type.name != "hexahedron"
        ):
            raise ValueError(
                "Uniform Hex8 requires continuous blocked vector Q1 hexahedra."
            )
        maps = getattr(space.mesh.geometry, "cmaps", None)
        coordinate_map = maps[0] if maps is not None else space.mesh.geometry.cmap
        if coordinate_map.degree != 1:
            raise ValueError("Uniform Hex8 requires trilinear geometry.")
        count = space.mesh.topology.index_map(3).size_local
        self.cell_nodes = np.array(
            [space.dofmap.cell_dofs(k) for k in range(count)], dtype=np.int32
        ).reshape(-1, 8)
        self.cell_nodes.setflags(write=False)
        # Reuse compact chunk-local scatter schedules. Bincount over the whole
        # mesh per chunk would allocate O(global_nodes) repeatedly; these maps
        # contain only nodes touched by the bounded cell chunk.
        self._scatter_plans = []
        for start in range(0, count, chunk_size):
            nodes, inverse = np.unique(
                self.cell_nodes[start : start + chunk_size].ravel(), return_inverse=True
            )
            self._scatter_plans.append((nodes, inverse.astype(np.int32)))
        if count == 0:
            self.cells = None
            return
        coordinates = space.tabulate_dof_coordinates()[self.cell_nodes]
        self.cells = UniformHexBatch(
            coordinates,
            stiffness,
            density=density,
            hourglass_modulus=hourglass_modulus,
            hourglass_scale=hourglass_scale,
            chunk_size=chunk_size,
        )

    def _responses(self):
        if self.cells is None:
            return iter(())
        values = self.displacement.x.array.reshape(-1, 3)
        return self.cells.iter_responses(values, node_map=self.cell_nodes)

    def assemble_vector(self):
        from petsc4py import PETSc

        self.displacement.x.scatter_forward()
        vector = self.displacement.x.petsc_vec.duplicate()
        try:

            def assemble_local():
                with vector.localForm() as local:
                    local.set(0)
                    values = local.array.reshape(-1, 3)
                    for (_, response), (nodes, inverse) in zip(
                        self._responses(), self._scatter_plans
                    ):
                        force = response.internal_force.reshape(-1, 3)
                        for component in range(3):
                            values[nodes, component] += np.bincount(
                                inverse,
                                weights=force[:, component],
                                minlength=len(nodes),
                            )

            collective_call(
                assemble_local, comm=self.comm, label="Uniform Hex8 residual"
            )
            vector.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )
        except Exception:
            vector.destroy()
            raise
        return vector

    def energies(self):
        from mpi4py import MPI

        self.displacement.x.scatter_forward()

        def local_energy():
            values = np.zeros(2)
            if self.cells is not None:
                nodal = self.displacement.x.array.reshape(-1, 3)
                for _, physical, artificial in self.cells.iter_energies(
                    nodal, node_map=self.cell_nodes
                ):
                    values += [physical.sum(), artificial.sum()]
            return values

        local = collective_call(
            local_energy, comm=self.comm, label="Uniform Hex8 energy"
        )
        physical, artificial = self.comm.allreduce(local, op=MPI.SUM)
        return {"strain_energy": physical, "hourglass_energy": artificial}

    def stable_dt(self, *, safety=0.8):
        from mpi4py import MPI

        def local_bound():
            selected = float(safety)
            if not np.isfinite(selected) or not 0 < selected < 1:
                raise ValueError("Stability safety must be between zero and one.")
            return 0.0 if self.cells is None else self.cells.stability_bound()

        bound = collective_call(
            local_bound, comm=self.comm, label="Uniform Hex8 stability"
        )
        safeties = self.comm.allgather(float(safety))
        if any(value != safeties[0] for value in safeties):
            raise ValueError("Uniform Hex8 stability safety differs across ranks.")
        bound = self.comm.allreduce(bound, op=MPI.MAX)
        if not np.isfinite(bound) or bound <= 0:
            raise ValueError(
                "Uniform Hex8 requires a positive finite global spectral bound."
            )
        return float(safety) * 2 / np.sqrt(bound)
