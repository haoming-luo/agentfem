# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from dolfinx import fem, mesh

from agentfem import constitutive, fields, fracture, interfaces, models, studies
from test_nonmatching_global import _blocks


def _trace(space, positive):
    domain = space.mesh
    domain.topology.create_connectivity(2, 3)
    facets = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[2], 0))
    xyz = space.tabulate_dof_coordinates()
    quads = []
    for facet in facets:
        cell = domain.topology.connectivity(2, 3).links(facet)[0]
        nodes = space.dofmap.cell_dofs(cell)
        if bool(xyz[nodes, 2].mean() > 0) != positive:
            continue
        nodes = fem.locate_dofs_topological(space, 2, np.array([facet], dtype=np.int32))
        points = xyz[nodes, :2]
        center = points.mean(axis=0)
        angle = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        nodes = nodes[np.argsort(angle)]
        quads.append(nodes[::-1] if positive else nodes)
    unique, inverse = np.unique(np.array(quads), return_inverse=True)
    return interfaces.reference_trace(
        xyz[unique], inverse.reshape(-1, 4), topology="quadrilateral", tolerance=1e-10
    ), unique


@pytest.mark.parametrize("n,m", [(1, 3), (2, 3)])
def test_q1_nonmatching_global_compliance_and_energy(n, m):
    domain = _blocks(n, m, cell_type="hexahedron")
    model = models.create(study=studies.static_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(constitutive.isotropic_elastic(young=100, poisson=0, density=2))
    model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(0, 1))
    model.fix(u, on=lambda x: np.isclose(x[2], -1), components=2)
    model.fix(u, on=lambda x: np.isclose(x[2], 1), components=2, value=0.02)
    a, na = _trace(u.value.function_space, False)
    b, nb = _trace(u.value.function_space, True)
    pair = interfaces.pair_reference_traces(a, b, tolerance=1e-10)
    law = interfaces.elastic_cohesive(normal_stiffness=1000, tangential_stiffness=500)
    force = fracture.nonmatching_cohesive_force(
        pair, u, law, negative_dofs=na, positive_dofs=nb
    )
    step = model.step(target=u, cohesive_force=force)
    result = step.solve_result()
    assert pair.summary()["negative_basis"] == "Q1"
    assert result.quantities["free_residual_norm"].value < 1e-10
    expected_force = 0.02 / (2 / 100 + 1 / 1000)
    assert result.quantities["interface_stored_energy"].value == pytest.approx(
        expected_force**2 / 2000, rel=1e-10
    )
    assert abs(result.quantities["energy_balance_residual"].value) < 1e-12


def _dynamic_interface(stiffness=1000):
    from agentfem import elements

    domain = _blocks(1, 2, cell_type="hexahedron")
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(constitutive.isotropic_elastic(young=100, poisson=0, density=2))
    a, na = _trace(u.value.function_space, False)
    b, nb = _trace(u.value.function_space, True)
    pair = interfaces.pair_reference_traces(a, b, tolerance=1e-10)
    law = interfaces.elastic_cohesive(
        normal_stiffness=stiffness, tangential_stiffness=stiffness / 2
    )
    force = fracture.nonmatching_cohesive_force(
        pair, u, law, negative_dofs=na, positive_dofs=nb
    )
    values = u.value.x.array.reshape(-1, 3)
    for index in range(domain.topology.index_map(3).size_local):
        nodes = u.value.function_space.dofmap.cell_dofs(index)
        side = u.value.function_space.tabulate_dof_coordinates()[nodes, 2].mean()
        values[nodes, 2] = 1e-4 if side > 0 else -1e-4
    step = model.step(
        target=u,
        element_policy=elements.uniform_strain_hex8(
            hourglass_modulus=50, hourglass_scale=0.1
        ),
        cohesive_force=force,
        dt=1e-5,
        steps=50,
        progress=False,
    )
    return step, force


def test_uniform_hex_nonmatching_interface_dynamics_restart_and_energy(tmp_path):
    reference, force = _dynamic_interface()
    result = reference.solve_result()
    history = reference.history_records
    initial = history[0]["total_discrete_energy"]
    assert initial == pytest.approx(0.5 * 1000 * (2e-4) ** 2)
    assert (
        max(abs(row["total_discrete_energy"] / initial - 1) for row in history) < 1e-4
    )
    assert "cohesive_stored_energy" in result.histories
    partial, _ = _dynamic_interface()
    partial.run(until_step=20)
    saved = partial.save_checkpoint(tmp_path / "combined")
    resumed, _ = _dynamic_interface()
    resumed.load_checkpoint(saved)
    resumed.run()
    np.testing.assert_allclose(
        resumed.state.u.value.x.array, reference.state.u.value.x.array, atol=1e-15
    )
    assert resumed.history_records == pytest.approx(reference.history_records)
    assert reference.stability["interface_omega_squared_bound"] > 0
    # Independently assembled dense interface-only spectrum verifies the bound.
    mass = reference.integrator.mass.mass_diagonal
    matrix = np.zeros((len(mass), len(mass)))
    for negative, positive, block in force.assembler.tangent_blocks(*force._values()):
        nodes = np.concatenate(
            (force.negative_dofs[negative], force.positive_dofs[positive])
        )
        dofs = (3 * nodes[:, None] + np.arange(3)).ravel()
        matrix[np.ix_(dofs, dofs)] += block
    scaled = matrix / np.sqrt(mass[:, None] * mass[None, :])
    assert np.linalg.eigvalsh(scaled).max() <= force.elastic_stability_bound(mass) * (
        1 + 1e-12
    )


def test_stiffer_interface_reduces_composed_stable_step():
    soft, _ = _dynamic_interface(1000)
    stiff, _ = _dynamic_interface(100000)
    assert stiff.stability["dt_limit"] < soft.stability["dt_limit"]


def test_global_interface_rejects_undeclared_anisotropic_tangent_frame():
    step, force = _dynamic_interface()
    law = interfaces.elastic_cohesive(
        normal_stiffness=1000, tangential_stiffness=500, second_tangential_stiffness=300
    )
    with pytest.raises(NotImplementedError, match="material frame"):
        fracture.nonmatching_cohesive_force(
            force.assembler.pairing,
            step.state.u,
            law,
            negative_dofs=force.negative_dofs,
            positive_dofs=force.positive_dofs,
        )


def test_combined_interface_commit_failure_restores_and_retries(monkeypatch):
    reference, _ = _dynamic_interface()
    reference.run()
    step, force = _dynamic_interface()
    step.run(until_step=3)
    accepted = step.state.snapshot()
    interface = force.snapshot()
    history = list(step.history_records)
    commit = force.commit

    def fail_after_commit():
        commit()
        raise RuntimeError("injected interface commit failure")

    monkeypatch.setattr(force, "commit", fail_after_commit)
    with pytest.raises(RuntimeError, match="injected interface"):
        step.run()
    assert step.completed_steps == 3
    assert step.history_records == history
    assert force.snapshot() == interface
    for name, values in accepted["fields"].items():
        np.testing.assert_array_equal(step.state.snapshot()["fields"][name], values)
    monkeypatch.setattr(force, "commit", commit)
    step.run()
    for name in ("u", "v", "a"):
        np.testing.assert_allclose(
            getattr(step.state, name).value.x.array,
            getattr(reference.state, name).value.x.array,
            atol=1e-15,
        )
