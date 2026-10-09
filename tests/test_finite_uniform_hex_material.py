# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from agentfem import constitutive
from agentfem.constitutive.material_driver import MaterialQuadratureResponse
from agentfem.elements._finite_uniform_hex import FiniteUniformHexBatch
from agentfem.elements._finite_uniform_hex_material import evaluate_material_trial


def setup():
    domain = mesh.create_box(
        MPI.COMM_SELF,
        [[0, 0, 0], [2, 1, 1]],
        [2, 1, 1],
        cell_type=mesh.CellType.hexahedron,
    )
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    nodes = np.array([space.dofmap.cell_dofs(k) for k in range(2)])
    op = FiniteUniformHexBatch(
        space.tabulate_dof_coordinates()[nodes],
        density=2,
        hourglass_modulus=30,
        hourglass_scale=0.1,
    )
    law = constitutive.finite_strain_j2_logarithmic(
        young=210000, poisson=0.3, yield_stress=250, hardening_modulus=1000
    )
    response = MaterialQuadratureResponse.create(
        domain,
        law.state_schema,
        degree=1,
        stored_energy_component_names=law.stored_energy_component_names,
    )
    return op, law, response


def update(op, law, response, stretch=1.06):
    f = np.diag([stretch, 1 / np.sqrt(stretch), 1 / np.sqrt(stretch)])
    return evaluate_material_trial(
        op,
        response,
        law,
        op.coordinates @ (f - np.eye(3)).T,
        deformation_gradient_old=np.tile(np.eye(3), (2, 1, 1)),
        time=0,
        time_increment=0.1,
    )


def test_finite_cell_material_trial_commit_and_rollback():
    op, law, response = setup()
    initial = response.state.committed_state_vectors().copy()
    trial = update(op, law, response)
    np.testing.assert_array_equal(response.state.committed_state_vectors(), initial)
    assert not np.array_equal(response.state.trial_state_vectors(), initial)
    assert trial.material_response.provider_batch_calls == 1
    assert trial.material_response.committed is False
    assert not trial.deformation_gradient.flags.writeable
    response.rollback()
    np.testing.assert_array_equal(response.state.trial_state_vectors(), initial)
    update(op, law, response)
    response.commit()
    assert not np.array_equal(response.state.committed_state_vectors(), initial)


def test_downstream_force_failure_restores_response_fields_and_trial_state(monkeypatch):
    op, law, response = setup()
    update(op, law, response)
    response.commit()
    accepted = response.state.committed_state_vectors().copy()
    stress = response.first_piola_stress.values.copy()
    tangent = response.tangent.values.copy()

    def fail(*args, **kwargs):
        raise RuntimeError("injected force mapping failure")

    monkeypatch.setattr(op, "response", fail)
    with pytest.raises(RuntimeError, match="injected"):
        update(op, law, response, stretch=1.12)
    np.testing.assert_array_equal(response.state.committed_state_vectors(), accepted)
    np.testing.assert_array_equal(response.state.trial_state_vectors(), accepted)
    np.testing.assert_array_equal(response.first_piola_stress.values, stress)
    np.testing.assert_array_equal(response.tangent.values, tangent)


def test_invalid_cell_count_discards_previous_unaccepted_trial():
    op, law, response = setup()
    accepted = response.state.committed_state_vectors().copy()
    update(op, law, response)
    op.coordinates = op.coordinates[:1]
    with pytest.raises(ValueError, match="counts differ"):
        update(op, law, response)
    np.testing.assert_array_equal(response.state.trial_state_vectors(), accepted)


def test_energy_free_provider_is_rejected_without_committing_material():
    from dataclasses import replace

    op, law, original = setup()

    class StressOnly:
        name = "stress only"
        state_schema = law.state_schema
        tangent_convention = law.tangent_convention

        def update(self, point):
            return replace(
                law.update(point),
                strain_energy_density=None,
                stored_energy_density_components={},
            )

    response = MaterialQuadratureResponse.create(
        original.domain, law.state_schema, degree=1
    )
    initial = response.state.committed_state_vectors().copy()
    with pytest.raises(ValueError, match="requires material stored energy"):
        update(op, StressOnly(), response)
    np.testing.assert_array_equal(response.state.committed_state_vectors(), initial)
    np.testing.assert_array_equal(response.state.trial_state_vectors(), initial)
