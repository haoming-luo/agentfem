# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private accepted material-energy view for the shared dynamic work ledger."""

from dataclasses import dataclass

import numpy as np

from ..diagnostics import kinetic_energy


@dataclass(frozen=True)
class FiniteHexEnergyMonitor:
    residual: object

    def evaluate(self, *, displacement, velocity):
        residual = self.residual
        # Also rejects an unaccepted or changed configuration without re-evaluation.
        from ..fields import unwrap
        from ..provenance import collective_call
        from mpi4py import MPI

        def local_values():
            residual.require_accepted_configuration()
            if unwrap(displacement) is not residual.internal.displacement:
                raise ValueError(
                    "Energy must use the residual's accepted displacement."
                )
            return dict(residual._accepted_energy)

        values = collective_call(
            local_values, comm=residual.comm, label="Finite Hex8 accepted energy"
        )
        names = tuple(values)
        local = np.asarray([values[name] for name in names], dtype=float)
        global_values = np.empty_like(local)
        residual.comm.Allreduce(local, global_values, op=MPI.SUM)
        values = dict(zip(names, map(float, global_values)))
        kinetic = kinetic_energy(residual.internal.mass_diagonal, velocity)
        mechanical = (
            kinetic
            + values["bulk_stored_energy"]
            + values["hourglass_energy"]
            + values["interface_stored_energy"]
        )
        return {
            **values,
            "kinetic_energy": kinetic,
            "total_mechanical_energy": mechanical,
            "accounted_internal_kinetic_energy": mechanical
            + values["material_dissipation"],
        }
