# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from agentfem import constitutive
from agentfem.constitutive.material_driver import MaterialQuadratureResponse
from agentfem.elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual


def setup():
    domain = mesh.create_box(
        MPI.COMM_SELF,
        [[0, 0, 0], [2, 1, 1]],
        [3, 2, 2],
        cell_type=mesh.CellType.hexahedron,
    )
    u = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
    law = constitutive.finite_strain_j2_logarithmic(
        young=210000, poisson=0.3, yield_stress=250, hardening_modulus=1000
    )
    response = MaterialQuadratureResponse.create(
        domain,
        law.state_schema,
        degree=1,
        stored_energy_component_names=law.stored_energy_component_names,
    )
    residual = FiniteUniformHexResidual(
        u,
        response,
        density=2,
        hourglass_modulus=80000,
        hourglass_scale=0.1,
        chunk_size=5,
    )
    return u, law, response, residual


def evaluate(residual, law):
    return residual.evaluate(
        law,
        deformation_gradient_old=np.tile(np.eye(3), (len(residual.cell_nodes), 1, 1)),
        time=0,
        time_increment=0.1,
    )


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_open_trial_scope_cleans_up_downstream_failure(exception):
    from agentfem.elements._finite_uniform_hex_material import response_fields

    u, law, response, residual = setup()
    before = {
        name: field.x.array.copy() for name, field in response_fields(response).items()
    }
    u.x.array[:] = (0.02 * u.function_space.tabulate_dof_coordinates()).ravel()
    with pytest.raises(exception):
        with residual.trial_evaluation(
            law,
            deformation_gradient_old=np.tile(
                np.eye(3), (len(residual.cell_nodes), 1, 1)
            ),
            time=0,
            time_increment=0.1,
        ) as (vector, trial):
            assert vector.norm() > 0
            raise exception("downstream rejection")
    for name, field in response_fields(response).items():
        np.testing.assert_array_equal(field.x.array, before[name])
    np.testing.assert_array_equal(
        response.state.trial_state_vectors(), response.state.committed_state_vectors()
    )
    with pytest.raises(RuntimeError, match="valid material trial"):
        residual.tangent_action(u.x.array)


def test_global_finite_patch_force_moment_and_tangent():
    u, law, response, residual = setup()
    x = u.function_space.tabulate_dof_coordinates()
    f = np.diag([1.06, 1 / np.sqrt(1.06), 1 / np.sqrt(1.06)])
    u.x.array[:] = (x @ (f - np.eye(3)).T).ravel()
    initial = response.state.committed_state_vectors().copy()
    vector, trial = evaluate(residual, law)
    try:
        force = vector.array.reshape(-1, 3).copy()
    finally:
        vector.destroy()
    np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(np.cross(x @ f.T, force).sum(axis=0), 0, atol=1e-9)
    interior = np.all((x > 1e-10) & (x < np.array([2, 1, 1]) - 1e-10), axis=1)
    assert interior.any()
    np.testing.assert_allclose(force[interior], 0, atol=1e-9)
    right = np.isclose(x[:, 0], 2)
    np.testing.assert_allclose(
        force[right].sum(axis=0), response.first_piola_stress.values[0, :, 0], atol=1e-9
    )
    np.testing.assert_array_equal(response.state.committed_state_vectors(), initial)
    assert trial.material_response.provider_batch_calls == 1
    assert np.isclose(residual.mass_diagonal[::3].sum(), 4)
    direction = np.random.default_rng(892).normal(size=u.x.array.shape)
    tangent = residual.tangent_action(direction)
    try:
        expected = tangent.array.copy()
    finally:
        tangent.destroy()
    base = u.x.array.copy()
    forces = []
    for sign in (1, -1):
        u.x.array[:] = base + sign * 1e-7 * direction
        vec, _ = evaluate(residual, law)
        forces.append(vec.array.copy())
        vec.destroy()
    np.testing.assert_allclose(
        expected, (forces[0] - forces[1]) / 2e-7, rtol=2e-5, atol=0.05
    )
    np.testing.assert_array_equal(response.state.committed_state_vectors(), initial)


def test_scatter_failure_restores_material_trial_and_response(monkeypatch):
    u, law, response, residual = setup()
    with pytest.raises(RuntimeError, match="valid material trial"):
        residual.tangent_action(u.x.array)
    original = response.first_piola_stress.values.copy()
    accepted = response.state.committed_state_vectors().copy()
    u.x.array[:] = (0.06 * u.function_space.tabulate_dof_coordinates()).ravel()

    def fail(*args):
        raise RuntimeError("injected scatter failure")

    monkeypatch.setattr(residual, "_scatter", fail)
    with pytest.raises(RuntimeError, match="injected"):
        evaluate(residual, law)
    np.testing.assert_array_equal(response.state.trial_state_vectors(), accepted)
    np.testing.assert_array_equal(response.first_piola_stress.values, original)
    with pytest.raises(RuntimeError, match="valid material trial"):
        residual.tangent_action(u.x.array)


def test_incremental_global_equilibrium_uses_fixed_committed_material_history():
    """Tiny dense Newton oracle, not a second production solver."""
    u, law, response, residual = setup()
    x = u.function_space.tabulate_dof_coordinates()
    interior = np.all((x > 1e-10) & (x < np.array([2, 1, 1]) - 1e-10), axis=1)
    free = (3 * np.flatnonzero(interior)[:, None] + np.arange(3)).ravel()
    old_f = np.tile(np.eye(3), (len(residual.cell_nodes), 1, 1))
    for step, stretch in enumerate((1.01, 1.03, 1.06, 1.02), start=1):
        f = np.diag([stretch, 1 / np.sqrt(stretch), 1 / np.sqrt(stretch)])
        expected = (x @ (f - np.eye(3)).T).ravel()
        u.x.array[:] = expected
        u.x.array[free] += np.random.default_rng(step).normal(
            scale=0.001, size=len(free)
        )
        committed = response.state.committed_state_vectors().copy()
        for iteration in range(15):
            force, trial = residual.evaluate(
                law,
                deformation_gradient_old=old_f,
                time=(step - 1) * 0.1,
                time_increment=0.1,
            )
            rhs = force.array[free].copy()
            force.destroy()
            np.testing.assert_array_equal(
                response.state.committed_state_vectors(), committed
            )
            if np.linalg.norm(rhs) < 1e-7:
                break
            columns = []
            for dof in free:
                direction = np.zeros_like(u.x.array)
                direction[dof] = 1
                column = residual.tangent_action(direction)
                columns.append(column.array[free].copy())
                column.destroy()
            u.x.array[free] -= np.linalg.solve(np.column_stack(columns), rhs)
        else:
            pytest.fail("Manufactured finite Hex8 equilibrium did not converge.")
        np.testing.assert_allclose(u.x.array, expected, atol=2e-9)
        response.commit()
        old_f = trial.deformation_gradient.copy()


def test_material_negative_curvature_is_not_global_instability_verdict():
    u, law, response, residual = setup()
    x = u.function_space.tabulate_dof_coordinates()
    f = np.diag([1.06, 1 / np.sqrt(1.06), 1 / np.sqrt(1.06)])
    u.x.array[:] = (x @ (f - np.eye(3)).T).ravel()
    vector, _ = evaluate(residual, law)
    vector.destroy()
    report = residual.cells.tangent_spectral_report(
        first_piola_tangent=response.tangent.values
    )
    assert report.negative_material_curvature_cells > 0
    interior = np.all((x > 1e-10) & (x < np.array([2, 1, 1]) - 1e-10), axis=1)
    free = (3 * np.flatnonzero(interior)[:, None] + np.arange(3)).ravel()
    columns = []
    for dof in free:
        direction = np.zeros_like(u.x.array)
        direction[dof] = 1
        column = residual.tangent_action(direction)
        columns.append(column.array[free].copy())
        column.destroy()
    # Same material, assembled and constrained equilibrium has positive curvature.
    assert np.linalg.eigvalsh(np.column_stack(columns))[0] > 0
