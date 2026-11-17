# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared fixed-material explicit-body partition, residual and energy ledger."""

from hashlib import sha256
import numpy as np
from .constitutive.elasticity import _constant_stiffness_matrix_3d
from .provenance import collective_call


def _material_partition(model, domain, *, label="Hex8"):
    count = domain.topology.index_map(3).size_local
    if count == 0 and domain.comm.size == 1:
        raise ValueError(f"{label} requires nonempty cells.")
    matrices = np.empty((count, 6, 6))
    density = np.empty(count)
    coverage = np.zeros(count, dtype=np.int32)
    for record in model.materials:
        region = record.region
        if region is None:
            indices = np.arange(count)
        else:
            if region.domain is not domain:
                raise ValueError("Material region belongs to another mesh.")
            indices = region.cell_tags.find(region.tag)
            indices = indices[indices < count]
        if not len(indices) and domain.comm.size == 1:
            raise ValueError(f"{label} material region is empty.")
        matrices[indices] = _constant_stiffness_matrix_3d(record.item)
        rho = getattr(record.item, "density", None)
        if rho is None or not np.isfinite(rho) or rho <= 0:
            raise ValueError(f"Every {label} material needs positive finite density.")
        density[indices] = rho
        coverage[indices] += 1
    if np.any(coverage != 1):
        raise ValueError(
            f"{label} material partition must cover each cell exactly once."
        )
    if count and np.all(matrices == matrices[0]):
        matrices = matrices[0].copy()
    return matrices, density


class _Residual:
    def __init__(self, internal, external, cohesive=None):
        self.internal, self.external = internal, external
        self.cohesive = cohesive
        digest = sha256()
        arrays = list(
            getattr(
                internal,
                "identity_arrays",
                (internal.cell_nodes, internal.mass_diagonal),
            )
        )
        if internal.cells is not None:
            arrays.extend(
                (
                    internal.cells.stiffness,
                    internal.cells.volume,
                    internal.cells.average_gradient,
                    internal.cells.hourglass_modes,
                    internal.cells.hourglass_coefficient,
                )
            )
        for array in arrays:
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        self.identity = digest.hexdigest()

    def bind_initial_inputs(self, bcs):
        from . import operators
        from dolfinx.fem import petsc
        from .time.explicit import _owned_dirichlet_dofs

        digest = sha256(self.identity.encode())
        if self.external is not None:
            vector = operators.assemble_vector(self.external)
            try:
                digest.update(vector.array.tobytes())
            finally:
                vector.destroy()
        vector = self.internal.displacement.x.petsc_vec.duplicate()
        try:
            vector.set(0)
            petsc.set_bc(vector, bcs)
            digest.update(_owned_dirichlet_dofs(bcs).tobytes())
            digest.update(vector.array.tobytes())
        finally:
            vector.destroy()
        # The checkpoint auxiliary record is shared by all ranks. Retain the
        # ordered partition identities; this is deliberately not portable.
        partitions = self.internal.comm.allgather(digest.hexdigest())
        self.identity = sha256("|".join(partitions).encode()).hexdigest()

    def assemble_vector(self):
        from . import operators

        vector = self.internal.assemble_vector()
        try:
            if self.cohesive is not None:
                self.cohesive.add_to_vector(vector)
            if self.external is not None:
                force = operators.assemble_vector(self.external)
                try:
                    vector.axpy(-1, force)
                finally:
                    force.destroy()

            def check_finite():
                if not np.all(np.isfinite(vector.array)):
                    raise ValueError(
                        "Non-finite explicit-body residual or external load."
                    )

            collective_call(
                check_finite,
                comm=self.internal.comm,
                label="Explicit-body total residual",
            )
            return vector
        except Exception:
            vector.destroy()
            raise

    def snapshot(self):
        return {
            "schema": getattr(
                self.internal, "schema", "agentfem.uniform-hex-elastic.v1"
            ),
            "operator_identity": self.identity,
            "cohesive": None if self.cohesive is None else self.cohesive.snapshot(),
            **(
                {"material": self.internal.material_batch.snapshot()}
                if getattr(self.internal, "material_batch", None) is not None
                else {}
            ),
        }

    def summary(self):
        return {
            "kind": getattr(
                self.internal, "schema", "uniform_strain_hex8_elastic_residual"
            ),
            "operator_identity": self.identity,
            "assembly": "owned_cells_reverse_ghost_accumulation",
            "interface": None if self.cohesive is None else self.cohesive.summary(),
        }

    def update_time(self, time):
        if getattr(self.internal, "material_batch", None) is not None:
            self.internal.trial_time = float(time)

    def restore(self, record):
        expected = self.snapshot()
        if {k: v for k, v in record.items() if k != "material"} != {
            k: v for k, v in expected.items() if k != "material"
        } or ("material" in record) != ("material" in expected):
            raise ValueError("Explicit-body checkpoint operator identity mismatch.")
        if "material" in expected:
            self.internal.material_batch.restore(record["material"])
            self.internal.trial_time = self.internal.material_batch.time
        if self.cohesive is not None:
            self.cohesive.rollback()
            self.cohesive.restore(record["cohesive"])

    def commit(self):
        if getattr(self.internal, "material_batch", None) is not None:
            self.internal.material_batch.commit()
        if self.cohesive is not None:
            self.cohesive.commit()

    def rollback(self):
        if getattr(self.internal, "material_batch", None) is not None:
            self.internal.material_batch.rollback()
        if self.cohesive is not None:
            self.cohesive.rollback()


class _Energy:
    def __init__(self, internal, cohesive=None):
        self.internal = internal
        self.cohesive = cohesive

    def evaluate(self, *, displacement, velocity):
        from .kernel.dofs import owned_array
        from mpi4py import MPI

        values = self.internal.energies()
        kinetic = float(
            0.5 * np.sum(self.internal.mass_diagonal * owned_array(velocity) ** 2)
        )
        kinetic = self.internal.comm.allreduce(kinetic, op=MPI.SUM)
        physical = values["strain_energy"] + kinetic
        if self.cohesive is not None:
            interface = self.cohesive.evaluate().stored_energy
            values["cohesive_stored_energy"] = interface
            physical += interface
        return {
            **values,
            "kinetic_energy": kinetic,
            "total_mechanical_energy": physical,
            "total_discrete_energy": physical + values["hourglass_energy"],
            "accounted_internal_kinetic_energy": physical
            + values["hourglass_energy"]
            + values.get("material_dissipation", 0.0),
        }
