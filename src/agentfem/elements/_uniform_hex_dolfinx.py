# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private serial lowering to DOLFINx DOFs; no alternative mesh or time solver."""

import numpy as np

from ._uniform_hex import UniformHexBatch


class UniformHexResidual:
    """Elastic internal residual, positive assembled mass and separate energies.

    Restricted to blocked vector Q1 on a serial hexahedral mesh. The mesh and
    function space own the cell node order. No MPI, constraints, loads, result
    classification or checkpoint policy is invented by this operator.
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
        if space.mesh.comm.size != 1:
            raise NotImplementedError(
                "Uniform Hex8 MPI assembly has not been verified."
            )
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
        )
        self.cell_nodes.setflags(write=False)
        coordinates = space.tabulate_dof_coordinates()[self.cell_nodes]
        self.cells = UniformHexBatch(
            coordinates,
            stiffness,
            density=density,
            hourglass_modulus=hourglass_modulus,
            hourglass_scale=hourglass_scale,
            chunk_size=chunk_size,
        )
        mass = np.zeros(space.dofmap.index_map.size_local)
        np.add.at(mass, self.cell_nodes.ravel(), self.cells.lumped_mass.ravel())
        if np.any(mass <= 0) or not np.all(np.isfinite(mass)):
            raise ValueError("Uniform Hex8 assembled mass must be positive finite.")
        self.mass_diagonal = np.repeat(mass, 3)
        self.mass_diagonal.setflags(write=False)
        self.inv_mass = 1 / self.mass_diagonal
        if not np.all(np.isfinite(self.inv_mass)):
            raise ValueError("Uniform Hex8 inverse mass overflowed.")
        self.inv_mass.setflags(write=False)

    def _responses(self):
        values = self.displacement.x.array.reshape(-1, 3)
        return self.cells.iter_responses(values, node_map=self.cell_nodes)

    def assemble_vector(self):
        from petsc4py import PETSc

        vector = PETSc.Vec().createSeq(
            len(self.mass_diagonal), comm=self.displacement.function_space.mesh.comm
        )
        try:
            values = vector.array.reshape(-1, 3)
            values[:] = 0
            for region, response in self._responses():
                np.add.at(
                    values,
                    self.cell_nodes[region].ravel(),
                    response.internal_force.reshape(-1, 3),
                )
        except Exception:
            vector.destroy()
            raise
        return vector

    def energies(self):
        physical = artificial = 0.0
        for _, response in self._responses():
            physical += float(response.physical_energy.sum())
            artificial += float(response.hourglass_energy.sum())
        return {"strain_energy": physical, "hourglass_energy": artificial}

    def stable_dt(self, *, safety=0.8):
        safety = float(safety)
        if not np.isfinite(safety) or not 0 < safety < 1:
            raise ValueError("Stability safety must be between zero and one.")
        return safety * 2 / np.sqrt(self.cells.stability_bound())
