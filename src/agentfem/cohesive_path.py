# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Accepted-state bookkeeping for displacement-controlled cohesive paths."""

from math import isfinite


class MonotonicCohesivePath:
    """Accept equilibria and integrate work along a nonnegative loading coordinate.

    ``equilibrium`` supplies a converged response and handles failed-trial
    rollback; ``residual.commit`` commits the associated material state.
    ``observe(response)`` supplies specimen-specific observables, without
    changing state. The conjugate reaction is interpreted as a magnitude,
    matching positive displacement-controlled opening/bending paths. This is
    not a cyclic, signed-work or arc-length driver.

    Dissipation is measured relative to the first accepted state (including
    any initialized precrack). Cutbacks remain the responsibility of
    ``steps.advance_monotonic_targets``. Failed trials change no path totals.
    """

    def __init__(self, equilibrium, residual, cohesive, *, observe):
        self.equilibrium = equilibrium
        self.residual = residual
        self.cohesive = cohesive
        self.observe = observe
        self.coordinate = 0.0
        self.reaction = 0.0
        self.external_work = 0.0
        self._initial_dissipation = None

    def try_accept(self, coordinate):
        coordinate = float(coordinate)
        if not isfinite(coordinate) or coordinate < self.coordinate:
            raise ValueError("Loading coordinate must be finite and nondecreasing.")
        try:
            solved = self.equilibrium(load=coordinate, branch="monotonic", cycle=0)
        except RuntimeError:
            info = self.equilibrium.last_info
            if info is None or info.converged:
                raise
            return None
        if not solved["converged"]:
            return None
        response = self.cohesive.begin()
        reaction = abs(float(solved["reaction"]))
        work = self.external_work + 0.5 * (self.reaction + reaction) * (
            coordinate - self.coordinate
        )
        raw_dissipation = float(response.dissipated_energy)
        initial = (raw_dissipation if self._initial_dissipation is None
                   else self._initial_dissipation)
        observed = self.observe(response)
        accepted = {
            "solved": solved,
            "response": response,
            "reaction": reaction,
            "bulk_energy": float(solved["bulk_strain_energy"]),
            "stored": float(response.stored_energy),
            "dissipated": raw_dissipation - initial,
        }
        if accepted.keys() & observed.keys():
            raise ValueError("Observables must not overwrite equilibrium or energy fields.")
        accepted.update(observed)
        self.residual.commit()
        self.coordinate = coordinate
        self.reaction = reaction
        self.external_work = work
        self._initial_dissipation = initial
        return accepted
