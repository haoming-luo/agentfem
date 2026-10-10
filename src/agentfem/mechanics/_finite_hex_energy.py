# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private accepted material-energy view for the shared dynamic work ledger."""

from dataclasses import dataclass

from ..diagnostics import kinetic_energy


@dataclass(frozen=True)
class FiniteHexEnergyMonitor:
    residual: object

    def evaluate(self, *, displacement, velocity):
        residual = self.residual
        # Also rejects an unaccepted or changed configuration without re-evaluation.
        residual.require_accepted_configuration()
        from ..fields import unwrap

        if unwrap(displacement) is not residual.internal.displacement:
            raise ValueError("Energy must use the residual's accepted displacement.")
        values = dict(residual._accepted_energy)
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
