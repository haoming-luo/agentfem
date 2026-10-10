# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Serial P1/Q1 lowering of reference interface residuals to PETSc assembly."""

import numpy as np

from ._interface_pairing import FixedReferenceCohesiveAssembler


class NonmatchingCohesiveForce:
    """A force contribution, not a solver or a finite-rotation formulation.

    The explicit dof maps must come from the mesh/space owner. Coincident
    positions do not identify independent interface degrees of freedom.
    """

    def __init__(self, assembler, displacement, *, negative_dofs, positive_dofs):
        from .fields import unwrap

        if not isinstance(assembler, FixedReferenceCohesiveAssembler):
            raise TypeError("Expected a fixed-reference cohesive assembler.")
        from ._elastic_cohesive import ElasticCohesiveLaw

        if isinstance(assembler.law, ElasticCohesiveLaw) and (
            assembler.law.second_tangential_stiffness
            != assembler.law.tangential_stiffness
        ):
            raise NotImplementedError(
                "Unequal tangential stiffnesses require an explicitly oriented interface material frame; this global adapter currently supports isotropic tangential response only."
            )
        self.displacement = unwrap(displacement)
        space = self.displacement.function_space
        if space.mesh.comm.size != 1:
            raise NotImplementedError(
                "Nonmatching force MPI ownership is not yet verified."
            )
        if (
            space.dofmap.index_map_bs != 3
            or space.element.basix_element.degree != 1
            or space.element.basix_element.discontinuous
        ):
            raise ValueError(
                "Nonmatching force requires a continuous three-component degree-one displacement."
            )
        topology = space.element.basix_element.cell_type.name
        width = assembler.pairing.negative_nodes.shape[1]
        other_width = assembler.pairing.positive_nodes.shape[1]
        if (topology, width, other_width) not in {
            ("tetrahedron", 3, 3),
            ("hexahedron", 4, 4),
        }:
            raise NotImplementedError(
                "Trace basis must match volume topology: tetrahedral P1 or hexahedral Q1; Q1 is not P1."
            )
        self.assembler = assembler
        count = self.displacement.x.array.size // 3
        maps = []
        coordinates = space.tabulate_dof_coordinates()
        for value, surface in (
            (negative_dofs, assembler.pairing.negative),
            (positive_dofs, assembler.pairing.positive),
        ):
            value = np.asarray(value)
            if value.shape != (len(surface.vertices),) or value.dtype.kind not in "iu":
                raise ValueError(
                    "Interface dof map must contain one integer per trace node."
                )
            if np.any(value < 0) or np.any(value >= count):
                raise ValueError("Interface dof map contains an invalid index.")
            if not np.allclose(
                coordinates[value],
                surface.vertices,
                rtol=0,
                atol=assembler.pairing.tolerance,
            ):
                raise ValueError(
                    "Interface dof coordinates differ from pairing geometry."
                )
            value = value.astype(np.int32, copy=True)
            value.setflags(write=False)
            maps.append(value)
        if len(np.unique(np.concatenate(maps))) != sum(map(len, maps)):
            raise ValueError(
                "Interface traces require independent, unique degrees of freedom."
            )
        self.negative_dofs, self.positive_dofs = maps

    def _values(self):
        values = self.displacement.x.array.reshape(-1, 3)
        return values[self.negative_dofs], values[self.positive_dofs]

    def begin(self):
        return self.assembler.begin(*self._values())

    def evaluate(self):
        return self.assembler.evaluate(*self._values())

    def elastic_stability_bound(self, mass_diagonal):
        """Conservative assembled mass-scaled eigenvalue bound for elastic laws."""
        from ._elastic_cohesive import ElasticCohesiveLaw

        law = self.assembler.law
        if not isinstance(law, ElasticCohesiveLaw):
            raise NotImplementedError(
                "Nonmatching stability currently supports elastic interfaces only."
            )
        mass = np.asarray(mass_diagonal, dtype=float)
        if (
            mass.shape != self.displacement.x.array.shape
            or not np.all(np.isfinite(mass))
            or np.any(mass <= 0)
        ):
            raise ValueError(
                "Interface stability requires positive finite global diagonal mass."
            )
        nodal_mass = mass.reshape(-1, 3).min(axis=1)
        pair = self.assembler.pairing
        negative = self.negative_dofs[pair.negative_nodes]
        positive = self.positive_dofs[pair.positive_nodes]
        wn = np.abs(pair.negative_weights) / np.sqrt(nodal_mass[negative])
        wp = np.abs(pair.positive_weights) / np.sqrt(nodal_mass[positive])
        stiffness = max(
            law.normal_stiffness,
            law.tangential_stiffness,
            law.second_tangential_stiffness,
        )
        weight = pair.weights * stiffness * (wn.sum(axis=1) + wp.sum(axis=1))
        rows = np.zeros_like(nodal_mass)
        np.add.at(rows, negative.ravel(), (weight[:, None] * wn).ravel())
        np.add.at(rows, positive.ravel(), (weight[:, None] * wp).ravel())
        bound = float(rows.max())
        if not np.isfinite(bound) or bound <= 0:
            raise ValueError("Interface spectral bound must be positive finite.")
        return bound

    def add_to_vector(self, vector):
        response = self.begin()
        values = vector.array.reshape(-1, 3)
        values[self.negative_dofs] += response.negative_residual
        values[self.positive_dofs] += response.positive_residual
        return response

    def add_to_matrix(self, matrix):
        from petsc4py import PETSc

        matrix.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        for negative, positive, values in self.assembler.tangent_blocks(
            *self._values()
        ):
            blocks = np.concatenate(
                (self.negative_dofs[negative], self.positive_dofs[positive])
            )
            dofs = (3 * blocks[:, None] + np.arange(3)).ravel().astype(PETSc.IntType)
            matrix.setValuesLocal(dofs, dofs, values, addv=PETSc.InsertMode.ADD_VALUES)

    def commit(self):
        self.assembler.commit()

    def rollback(self):
        self.assembler.rollback()

    def snapshot(self):
        return self.assembler.snapshot()

    def restore(self, snapshot):
        self.assembler.restore(snapshot)

    def summary(self):
        return {
            "kind": "nonmatching_cohesive_force",
            "pairing": self.assembler.pairing.summary(),
            "law": self.assembler.law.summary(),
            "execution_scope": "serial_fixed_reference_degree_one",
            "volume_topology": self.displacement.function_space.element.basix_element.cell_type.name,
            "finite_rotation": False,
            "portable_restart": False,
        }
