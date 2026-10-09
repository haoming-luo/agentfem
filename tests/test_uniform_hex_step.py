# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh

from agentfem import constitutive, elements, fields, models, studies


def _model(*, scale=0.1):
    domain = mesh.create_box(
        MPI.COMM_SELF,
        [[0, 0, 0], [1, 0.2, 0.2]],
        [4, 2, 2],
        cell_type=mesh.CellType.hexahedron,
    )
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(constitutive.isotropic_elastic(young=100, poisson=0, density=2))
    u.value.interpolate(
        lambda x: np.vstack(
            (1e-4 * np.cos(np.pi * x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]))
        )
    )
    policy = elements.uniform_strain_hex8(hourglass_modulus=50, hourglass_scale=scale)
    return model, u, policy


def _step(*, scale=0.1):
    model, u, policy = _model(scale=scale)
    return model.step(target=u, element_policy=policy, dt=1e-4, steps=4, progress=False)


def test_uniform_hex_ordinary_step_result_and_restart(tmp_path):
    reference = _step()
    result = reference.solve_result(output=tmp_path / "reference.xdmf")
    assert reference.completed_steps == 4
    assert "hourglass_energy" in result.histories
    assert "total_discrete_energy" in result.histories
    assert reference.summary()["element_policy"]["maturity"] == "experimental"
    partial = _step()
    partial.run(until_step=2)
    saved = partial.save_checkpoint(tmp_path / "restart")
    restarted = _step()
    restarted.load_checkpoint(saved)
    restarted.run()
    for name in ("u", "v", "a"):
        np.testing.assert_allclose(
            getattr(restarted.state, name).value.x.array,
            getattr(reference.state, name).value.x.array,
            atol=1e-15,
        )
    assert restarted.history_records == pytest.approx(reference.history_records)
    with pytest.raises(NotImplementedError, match="portable"):
        restarted.save_checkpoint(tmp_path / "portable", portable=True)
    changed = _step(scale=0.2)
    before = changed.state.snapshot()
    with pytest.raises(ValueError, match="identity"):
        changed.load_checkpoint(saved)
    for name, array in before["fields"].items():
        np.testing.assert_array_equal(changed.state.snapshot()["fields"][name], array)


def test_uniform_hex_step_checks_time_increment_and_policy():
    model, u, policy = _model()
    with pytest.raises(ValueError, match="conservative bound"):
        model.step(target=u, element_policy=policy, dt=100, steps=1)
    step = model.step(
        target=u, element_policy=policy, dt="auto", steps=1, progress=False
    )
    assert step.dt == step.stability["dt_limit"]
    step.run()
    with pytest.raises(ValueError, match="hourglass_scale"):
        elements.uniform_strain_hex8(hourglass_modulus=50, hourglass_scale=0)


@pytest.mark.parametrize("write", [False, True])
def test_uniform_hex_derived_fields_match_constitutive_cell_response(tmp_path, write):
    step = _step()
    result = step.solve_result(
        field_variables=("S", "E", "MISES", "SENER"),
        output=tmp_path / "cells.xdmf" if write else None,
    )
    for name in ("S", "E", "MISES", "SENER"):
        item = result.fields[name]
        assert item.location == "cells"
        assert (
            item.processing["method"] == "direct_uniform_gradient_constitutive_response"
        )
        assert item.processing["material_boundary_averaging"] is False
        assert (item.artifact is not None) == write
    internal = step.residual.internal
    energy = 0.0
    for region, response in internal._responses():
        for name, expected in (("S", response.stress), ("E", response.strain)):
            field = result.fields[name].field
            cell_dofs = [
                field.function_space.dofmap.cell_dofs(k)[0]
                for k in range(region.start, region.stop)
            ]
            values = field.x.array.reshape(-1, 3, 3)[cell_dofs]
            actual = values[:, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]].copy()
            if name == "E":
                actual[:, 3:] *= 2
            np.testing.assert_allclose(actual, expected, atol=1e-15)
        field = result.fields["SENER"].field
        dofs = [
            field.function_space.dofmap.cell_dofs(k)[0]
            for k in range(region.start, region.stop)
        ]
        energy += float(field.x.array[dofs] @ internal.cells.volume[region])
    assert energy == pytest.approx(step.history_records[-1]["strain_energy"])
    result.write_manifest(tmp_path / "result.json")


def test_uniform_hex_field_selection_rejects_before_advancing():
    step = _step()
    with pytest.raises(ValueError, match="support"):
        step.solve_result(field_variables=("DAMAGE",))
    assert step.completed_steps == 0
    with pytest.raises(ValueError, match="not both"):
        step.solve_result(field_variables=("S",), fields=(step.state.u,))
    assert step.completed_steps == 0
    result = step.solve_result(field_variables=())
    assert set(result.fields) == {step.state.u.value.name}


def test_uniform_hex_cell_fields_preserve_tensor_shear_and_material_jump():
    from agentfem import mesh as mesh_api
    from agentfem.results._uniform_hex import UniformHexCellFields

    original, _, policy = _model()
    domain = original.mesh
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    regions = mesh_api.partition_cells(
        domain, left=lambda x: x[0] < 0.5, right=lambda x: x[0] >= 0.5
    )
    for young, region in ((100, regions.left), (200, regions.right)):
        model.material(
            constitutive.isotropic_elastic(young=young, poisson=0, density=2),
            region=region,
        )
    # gamma_xy = 0.02, tensor epsilon_xy = 0.01; stress jumps with modulus.
    u.value.interpolate(lambda x: np.vstack((0.02 * x[1], 0 * x[0], 0 * x[0])))
    step = model.step(target=u, element_policy=policy, dt=1e-4, steps=1, progress=False)
    live = UniformHexCellFields(step.residual.internal, ("S", "E", "MISES", "SENER"))
    values = {field.name: field for field in live.update()}
    stress = values["S"].x.array.reshape(-1, 3, 3)
    strain = values["E"].x.array.reshape(-1, 3, 3)
    np.testing.assert_allclose(strain[:, 0, 1], 0.01, atol=1e-14)
    np.testing.assert_allclose(strain[:, 1, 0], 0.01, atol=1e-14)
    np.testing.assert_allclose(np.unique(np.round(stress[:, 0, 1], 12)), [1, 2])
    np.testing.assert_allclose(values["MISES"].x.array, np.sqrt(3) * stress[:, 0, 1])
    np.testing.assert_allclose(values["SENER"].x.array, 0.01 * stress[:, 0, 1])


def test_step_constraint_override_initializes_only_selected_motion():
    from agentfem import amplitudes, constraints

    model, u, policy = _model()
    calls = []
    unused = amplitudes.Amplitude("unused", lambda t: calls.append(t) or 123.0)
    model.fix(
        u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=0, value=unused
    )
    selected = constraints.time_dependent_component_dirichlet(
        u,
        0,
        marker=lambda x: np.ones(x.shape[1], dtype=bool),
        amplitude=amplitudes.Amplitude("selected", lambda t: 0.01 + 0.02 * t),
    )
    calls.clear()
    step = model.step(
        target=u,
        element_policy=policy,
        constraints=(selected,),
        dt=1e-4,
        steps=2,
        progress=False,
    )
    np.testing.assert_allclose(step.state.u.value.x.array[::3], 0.01)
    np.testing.assert_allclose(step.state.v.value.x.array[::3], 0.02)
    step.run()
    np.testing.assert_allclose(step.state.u.value.x.array[::3], 0.01 + 0.02 * 2e-4)
    assert calls == []


def test_initial_residual_failure_restores_user_displacement(monkeypatch):
    from agentfem.elements._uniform_hex_dolfinx import UniformHexResidual

    model, u, policy = _model()
    model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=0, value=0.02)
    before = u.value.x.array.copy()

    def fail(self):
        raise RuntimeError("injected initial assembly failure")

    monkeypatch.setattr(UniformHexResidual, "assemble_vector", fail)
    with pytest.raises(RuntimeError, match="injected"):
        model.step(target=u, element_policy=policy, dt=1e-4, steps=2, progress=False)
    np.testing.assert_array_equal(u.value.x.array, before)


def test_uniform_hex_rotated_anisotropic_material_matches_ufl():
    from agentfem import materials
    from agentfem.constitutive import elasticity
    from agentfem.constitutive.elasticity import _constant_stiffness_matrix_3d
    from agentfem.elements._uniform_hex_dolfinx import UniformHexResidual
    from dolfinx import fem
    from dolfinx.fem import petsc
    import ufl

    model, u, _ = _model()
    matrix = np.diag([200.0, 120.0, 80.0, 31.0, 27.0, 19.0])
    matrix[:3, :3] += 10
    law = elasticity.anisotropic_elastic_3d(stiffness_voigt=matrix, density=2)
    oriented = materials.oriented(
        law, materials.material_frame((1, 2, 1), secondary=(-1, 0, 1))
    )
    u.value.interpolate(
        lambda x: np.vstack(
            (0.1 * x[0] + 0.3 * x[1], -0.2 * x[0] + 0.2 * x[2], 0.4 * x[1] - 0.1 * x[2])
        )
    )
    internal = UniformHexResidual(
        u,
        _constant_stiffness_matrix_3d(oriented),
        density=2,
        hourglass_modulus=30,
        hourglass_scale=0.1,
    )
    test = ufl.TestFunction(u.value.function_space)
    reference = petsc.assemble_vector(
        fem.form(
            ufl.inner(elasticity.stress(u, oriented), ufl.sym(ufl.grad(test))) * ufl.dx
        )
    )
    actual = internal.assemble_vector()
    try:
        np.testing.assert_allclose(actual.array, reference.array, atol=1e-12)
    finally:
        actual.destroy()
        reference.destroy()


def test_uniform_hex_region_materials_preserve_mass_and_reject_overlap():
    from agentfem import mesh as mesh_api

    original, _, policy = _model()
    domain = original.mesh
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    regions = mesh_api.partition_cells(
        domain, left=lambda x: x[0] < 0.5, right=lambda x: x[0] >= 0.5
    )
    a = constitutive.isotropic_elastic(young=100, poisson=0, density=2)
    b = constitutive.isotropic_elastic(young=40, poisson=0.2, density=4)
    model.material(a, region=regions.left)
    model.material(b, region=regions.right)
    step = model.step(
        target=u, element_policy=policy, dt="auto", steps=2, progress=False
    )
    assert step.integrator.mass.mass_diagonal[::3].sum() == pytest.approx(0.04 * 3)
    step.run()
    model.material(a, region=regions.left)
    with pytest.raises(
        (ValueError, RuntimeError), match="(?i)(overlap|exactly once|coverage|invalid)"
    ):
        model.step(target=u, element_policy=policy, dt="auto", steps=2)
