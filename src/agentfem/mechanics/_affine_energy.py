# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Material-owned energy observations for the primal affine J2 Procedure."""

import numpy as np
from dolfinx import fem
from mpi4py import MPI

from ..results._accepted_energy_components import AcceptedEnergyComponents


def attach_affine_material_energy(problem, response, contacts):
    """Reuse accepted-observer transactions; do not integrate work a second time.

    This owner has no independent dead loads and a primal material energy.
    Augmented/multiplier contact has no physical penalty potential and retains
    its own explicit normal-work gate; it is not assigned a zero potential.
    Mixed saddle-point energies are not admitted by this constructor.
    """
    contacts = tuple(contacts)
    if any(not getattr(p, "physical_contact_potential_available", True) for p in contacts):
        problem.material_energy_scope = "unavailable_contact_enforcement_energy_requires_separate_normal_work"
        return
    forms = (
        fem.form(response.strain_energy_density.function * response.measure),
        fem.form(response.state.committed["plastic_dissipation"].function * response.measure),
    )
    comm = problem.solution.function_space.mesh.comm
    roles = {"stored_energy": "stored", "plastic_dissipation": "cumulative_dissipation"}
    if contacts:
        roles["contact_potential"] = "stored"
    friction = any(getattr(p.pair, "friction", None) is not None for p in contacts)
    if friction:
        roles.update(friction_dissipation="cumulative_dissipation",
                     friction_separation_release="cumulative_numerical_release")

    def evaluate(snapshot):
        # Integration uses the committed response already produced by the
        # owning transaction. No extra constitutive call or field copy.
        values = np.asarray([fem.assemble_scalar(f) for f in forms], dtype=float)
        if comm.size > 1:
            comm.Allreduce(MPI.IN_PLACE, values, op=MPI.SUM)
        result = {"stored_energy": float(values[0]), "plastic_dissipation": float(values[1])}
        if contacts:
            potential = dissipated = released = 0.
            for provider in contacts:
                scalar_energy = getattr(provider, "current_energy_components", None)
                if callable(scalar_energy):
                    components = scalar_energy()
                    potential += components["contact_potential"]
                    dissipated += components["friction_dissipation"]
                    released += components["friction_separation_release"]
                    continue
                contribution = provider.evaluate()
                potential += contribution.potential_energy
                tangential = getattr(contribution, "friction_assembly", None)
                if tangential is not None:
                    dissipated += tangential.cumulative_dissipation
                    released += tangential.cumulative_separation_release
            result["contact_potential"] = float(potential)
            if friction:
                result.update(friction_dissipation=float(dissipated),
                              friction_separation_release=float(released))
        return result

    recorder = AcceptedEnergyComponents(roles, evaluate)
    problem.accepted_observers = (*problem.accepted_observers, recorder)
    problem.accepted_history_recorders["material_energy"] = recorder
    problem.material_energy_scope = "primal_material_and_physical_penalty_energy"
