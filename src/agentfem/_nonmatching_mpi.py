# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded replicated reference geometry, distributed elastic quadrature State."""

import numpy as np
from mpi4py import MPI

from ._nonmatching_force import NonmatchingCohesiveForce
from .provenance import collective_call, collective_canonical_record


class DistributedNonmatchingForce(NonmatchingCohesiveForce):
    """Sparse nodal exchange; each reference quadrature point is evaluated once.

    Geometry is replicated at construction, not a scalable distributed search.
    Local maps use -1 for absent nodes; coincident sides retain distinct IDs.
    """

    def __init__(self, assembler, displacement, *, negative_dofs, positive_dofs):
        from .fields import unwrap
        from .fracture import _SparseCohesiveExchange
        from ._elastic_cohesive import ElasticCohesiveLaw
        from ._interface_pairing import FixedReferenceCohesiveAssembler, _make_pairing

        self.displacement = unwrap(displacement)
        space = self.displacement.function_space
        self.comm = comm = space.mesh.comm
        pair = assembler.pairing
        index = space.dofmap.index_map

        def validate():
            if not isinstance(assembler.law, ElasticCohesiveLaw):
                raise NotImplementedError("Distributed nonmatching force admits elastic interfaces only.")
            if assembler.law.tangential_stiffness != assembler.law.second_tangential_stiffness:
                raise NotImplementedError("Distributed interface requires equal tangential stiffnesses.")
            cell = space.element.basix_element.cell_type.name
            if (space.dofmap.index_map_bs != 3 or space.element.basix_element.degree != 1
                    or space.element.basix_element.discontinuous
                    or (cell, pair.negative_nodes.shape[1], pair.positive_nodes.shape[1])
                    not in {("hexahedron", 4, 4), ("tetrahedron", 3, 3)}):
                raise ValueError("Distributed trace requires matching P1/Q1 volume basis.")
            if not pair.method.startswith("coplanar-"):
                raise NotImplementedError("Distributed interface requires common-refinement coverage.")
            maps = []
            xyz = space.tabulate_dof_coordinates()
            for raw, surface in ((negative_dofs, pair.negative), (positive_dofs, pair.positive)):
                mapping = np.asarray(raw)
                if (mapping.shape != (len(surface.vertices),) or mapping.dtype.kind not in "iu"
                        or np.any(mapping < -1) or np.any(mapping >= len(xyz))):
                    raise ValueError("Distributed trace map requires valid local DOFs or -1.")
                present = mapping >= 0
                if not np.allclose(xyz[mapping[present]], surface.vertices[present],
                                   rtol=0, atol=pair.tolerance):
                    raise ValueError("Distributed trace coordinates differ from pairing.")
                maps.append(mapping.astype(np.int32, copy=True))
            present = np.concatenate(maps)
            present = present[present >= 0]
            if np.unique(present).size != present.size:
                raise ValueError("Distributed traces require independent DOFs.")
            return maps

        self.negative_dofs, self.positive_dofs = collective_call(
            validate, comm=comm, label="Distributed nonmatching admission")
        collective_canonical_record(assembler.snapshot(), comm=comm,
                                    label="Distributed nonmatching identity")
        self.global_pairing = pair
        selected = np.arange(len(pair.weights))[np.arange(len(pair.weights)) % comm.size == comm.rank]
        names = ("negative_nodes", "positive_nodes", "negative_weights", "positive_weights",
                 "weights", "normals", "reference_mismatch")
        local_pair = _make_pairing(pair.negative, pair.positive, pair.tolerance,
                                  {name: getattr(pair, name)[selected] for name in names}, pair.method)
        self.assembler = FixedReferenceCohesiveAssembler(
            local_pair, assembler.law, tangential=assembler.tangential,
            tangential_stiffness=assembler.tangential_stiffness)
        self.mapping = np.concatenate((self.negative_dofs, self.positive_dofs))
        self.offset = len(pair.negative.vertices)
        required = np.unique(np.concatenate((local_pair.negative_nodes.ravel(),
                                             local_pair.positive_nodes.ravel() + self.offset)))
        owned = (self.mapping >= 0) & (self.mapping < index.size_local)
        self.owned_nodes = np.flatnonzero(owned)
        self.exchange = _SparseCohesiveExchange(
            comm=comm, input_node_to_block_dof=self.mapping, input_node_owned=owned,
            required_nodes=required, owned_interface_nodes=self.owned_nodes, block_size=3)

    def _values(self):
        compact = self.exchange.gather_owned_dof_values(self.displacement.x.array.reshape(-1, 3))
        values = np.zeros((len(self.mapping), 3))
        values[self.exchange.required_nodes] = compact
        return values[:self.offset], values[self.offset:]

    def _response(self, *, begin):
        negative, positive = self._values()
        try:
            return collective_call(
                lambda: (self.assembler.begin if begin else self.assembler.evaluate)(negative, positive),
                comm=self.comm, label="Distributed interface response")
        except BaseException:
            self.assembler.rollback()
            raise

    def begin(self):
        return self._response(begin=True)

    def evaluate(self):
        return self._response(begin=False)

    def add_response_to_vector(self, vector, response):
        values = np.concatenate((response.negative_residual, response.positive_residual))
        owned = self.exchange.accumulate_to_owners(values[self.exchange.required_nodes])
        vector.array.reshape(-1, 3)[self.mapping[self.owned_nodes]] += owned

    def add_to_vector(self, vector):
        response = self.begin()
        self.add_response_to_vector(vector, response)
        return response

    def add_to_matrix(self, matrix):
        raise NotImplementedError("Distributed nonmatching implicit matrix assembly is not admitted.")

    def elastic_stability_bound(self, mass_diagonal):
        mass = self.exchange.gather_owned_dof_values(np.asarray(mass_diagonal).reshape(-1, 3))
        collective_call(lambda: self._positive_mass(mass), comm=self.comm,
                        label="Distributed interface mass")
        nodal = np.ones(len(self.mapping))
        nodal[self.exchange.required_nodes] = mass.min(axis=1)
        pair, law = self.assembler.pairing, self.assembler.law
        negative, positive = pair.negative_nodes, pair.positive_nodes + self.offset
        wn = abs(pair.negative_weights) / np.sqrt(nodal[negative])
        wp = abs(pair.positive_weights) / np.sqrt(nodal[positive])
        weight = pair.weights * max(law.normal_stiffness, law.tangential_stiffness) * (wn.sum(axis=1) + wp.sum(axis=1))
        rows = np.zeros(len(self.mapping))
        np.add.at(rows, negative.ravel(), (weight[:, None] * wn).ravel())
        np.add.at(rows, positive.ravel(), (weight[:, None] * wp).ravel())
        owned = self.exchange.accumulate_to_owners(rows[self.exchange.required_nodes, None])
        return float(self.comm.allreduce(float(owned.max(initial=0)), op=MPI.MAX))

    @staticmethod
    def _positive_mass(mass):
        if np.any(mass <= 0):
            raise ValueError("Interface mass must be positive.")

    def summary(self):
        return {**super().summary(), "execution_scope": "mpi_elastic_reference_explicit",
                "geometry_scope": "replicated_reference_trace",
                "global_pairing": self.global_pairing.fingerprint,
                "exchange": self.exchange.summary()}
