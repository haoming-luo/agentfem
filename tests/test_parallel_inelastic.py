"""Distributed contracts for stateful inelastic solid mechanics."""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import constitutive, fields, mechanics, mesh, models, solvers, steps, studies


def _model_with_regions(*, creep: bool, yielding: bool = False):
    domain = dolfinx_mesh.create_box(
        MPI.COMM_WORLD,
        [np.zeros(3), np.asarray((1.0, 0.2, 0.2))],
        [2, 1, 1],
        cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    model = models.create(
        study=(studies.creep_solid() if creep else studies.nonlinear_static(
            physics="solid_mechanics", dimension=3
        )),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    regions = mesh.partition_cells(
        domain,
        left=lambda x: x[0] <= 0.5,
        right=lambda x: x[0] > 0.5,
    )
    if creep:
        factory = lambda young, coefficient: constitutive.isotropic_power_law(
            young=young,
            poisson=0.3,
            density=1.0,
            coefficient=coefficient,
            stress_exponent=2.0,
            reference_stress=1.0,
        )
        model.material(factory(1000.0, 2.0e-6), region=regions.left)
        model.material(factory(1500.0, 8.0e-6), region=regions.right)
    else:
        factory = lambda young: constitutive.J2LinearIsotropicHardening(
            young=young,
            poisson=0.3,
            yield_stress=(0.4 if yielding else 1.0e6),
            hardening_modulus=100.0,
        )
        model.material(factory(1000.0), region=regions.left)
        model.material(factory(1500.0), region=regions.right)
    model.fix(displacement, on=mesh.face(domain, axis="x", value=0.0), component=0)
    model.fix(displacement, on=mesh.face(domain, axis="y", value=0.0), component=1)
    model.fix(displacement, on=mesh.face(domain, axis="z", value=0.0), component=2)
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=1.0),
        component=0,
        value=0.002,
    )
    return model, displacement


def _distributed_step(model, displacement, *, creep: bool):
    material_type = (
        constitutive.IsotropicPowerLawCreepMaterial
        if creep
        else constitutive.J2LinearIsotropicHardening
    )
    material = constitutive.QuadratureMaterialMap.from_assignments(
        displacement.value.function_space.mesh,
        model.materials,
        material_type=material_type,
    )
    common = dict(
        displacement=displacement,
        material=material,
        external_force=None,
        constraints=model.constraints,
        study=model.study,
        incrementation=steps.fixed(2 if not creep else 1),
        solver_options=solvers.newton(
            maximum_iterations=15,
            line_search="basic",
        ),
        progress=False,
        _experimental_distributed=True,
    )
    return (
        mechanics.implicit_creep_step(duration=0.1, **common)
        if creep
        else mechanics.j2_plasticity_step(**common)
    )


def _owned_state_by_cell(step):
    state = step.state
    cell_map = state.domain.topology.index_map(state.domain.topology.dim)
    owned = int(cell_map.size_local)
    points = len(state.stress.points)
    keys = np.asarray(state.domain.topology.original_cell_index[:owned])
    values = (
        state.equivalent_creep_strain.values
        if hasattr(state, "equivalent_creep_strain")
        else state.equivalent_plastic_strain.values
    ).reshape((-1,))[: owned * points]
    return keys, values.reshape((owned, points))


def test_global_j2_step_consumes_complete_regional_material_map():
    model, displacement = _model_with_regions(creep=False)
    step = _distributed_step(model, displacement, creep=False)
    result = step.solve_result()

    assert step.last_solve_info.completed_step
    assert isinstance(step.material, constitutive.QuadratureMaterialMap)
    assert len(step.material.materials) == 2
    assert result.quantity("maximum_equivalent_plastic_strain") == pytest.approx(0.0)
    assert step.state.summary()["points_global"] > 0


def test_global_creep_step_consumes_complete_regional_material_map():
    model, displacement = _model_with_regions(creep=True)
    step = _distributed_step(model, displacement, creep=True)
    result = step.solve_result()

    assert step.last_solve_info.completed_step
    assert isinstance(step.material, constitutive.QuadratureMaterialMap)
    assert len(step.material.materials) == 2
    assert result.quantity("maximum_equivalent_creep_strain") > 0.0
    assert np.isfinite(result.quantity("maximum_equivalent_creep_strain"))


def test_distributed_j2_global_newton_balances_partition_interface():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("distributed J2 equilibrium acceptance requires two ranks")
    model, displacement = _model_with_regions(creep=False)
    step = _distributed_step(model, displacement, creep=False)
    result = step.solve_result()

    assert step.last_solve_info.completed_step
    assert step.last_solve_info.increments[-1].residual_norm < 1.0e-8
    assert result.quantity("maximum_equivalent_plastic_strain") == pytest.approx(0.0)


def test_public_model_step_dispatches_distributed_j2_without_private_switch():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("public distributed J2 dispatch acceptance requires two ranks")
    model, displacement = _model_with_regions(creep=False)
    step = model.step(
        target=displacement,
        incrementation=steps.fixed(2),
        solver_options=solvers.newton(maximum_iterations=15, line_search="basic"),
        progress=False,
    )

    result = step.solve_result()

    assert step.last_solve_info.completed_step
    assert result.quantity("maximum_equivalent_plastic_strain") == pytest.approx(0.0)


def test_distributed_creep_global_newton_evolves_regional_state():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("distributed creep equilibrium acceptance requires two ranks")
    model, displacement = _model_with_regions(creep=True)
    step = _distributed_step(model, displacement, creep=True)
    result = step.solve_result()

    assert step.last_solve_info.completed_step
    assert step.last_solve_info.increments[-1].residual_norm < 1.0e-7
    assert result.quantity("maximum_equivalent_creep_strain") > 0.0


def test_distributed_j2_cutback_rollback_matches_fixed_reference():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("distributed J2 rollback acceptance requires two ranks")
    cutback_model, cutback_u = _model_with_regions(creep=False, yielding=True)
    cutback = _distributed_step(cutback_model, cutback_u, creep=False)
    cutback.incrementation = steps.automatic(
        initial=1.0,
        minimum=1.0e-3,
        maximum=1.0,
        max_increments=40,
        max_cutbacks=10,
        cutback_factor=0.5,
        maximum_inelastic_increment=7.0e-4,
    )
    cutback.solve()

    assert cutback.last_solve_info.completed_step
    assert any(not item.converged for item in cutback.attempted_increments)
    accepted_path = tuple(
        item.load_factor for item in cutback.accepted_increments
    )
    reference_model, reference_u = _model_with_regions(
        creep=False, yielding=True
    )
    reference = _distributed_step(reference_model, reference_u, creep=False)
    reference.incrementation = steps.at(*accepted_path)
    reference.solve()

    reference_keys, reference_values = _owned_state_by_cell(reference)
    cutback_keys, cutback_values = _owned_state_by_cell(cutback)
    np.testing.assert_array_equal(cutback_keys, reference_keys)
    np.testing.assert_allclose(cutback_values, reference_values, rtol=1.0e-10, atol=1.0e-12)


def test_portable_quadrature_state_rejects_changed_material_contract(tmp_path):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("portable material identity rejection is a serial unit test")
    model, displacement = _model_with_regions(creep=False)
    original = constitutive.QuadratureMaterialMap.from_assignments(
        displacement.value.function_space.mesh,
        model.materials,
        material_type=constitutive.J2LinearIsotropicHardening,
    )
    state = constitutive.J2QuadratureState.create(original.domain, degree=2)
    state.equivalent_plastic_strain.assign(
        np.linspace(0.0, 0.1, len(state.equivalent_plastic_strain.values))
    )
    archive = state.save(tmp_path / "j2", material=original)

    altered_materials = dict(original.materials)
    first_region = min(altered_materials)
    selected = altered_materials[first_region]
    altered_materials[first_region] = constitutive.J2LinearIsotropicHardening(
        young=selected.young,
        poisson=selected.poisson,
        yield_stress=selected.yield_stress * 1.01,
        hardening_modulus=selected.hardening_modulus,
    )
    altered = constitutive.QuadratureMaterialMap(
        original.domain,
        altered_materials,
        original.cell_regions,
        original.region_names,
    )
    restored = constitutive.J2QuadratureState.create(original.domain, degree=2)
    with pytest.raises(ValueError, match="mesh, rule, material regions"):
        restored.load(archive, material=altered)


def test_distributed_quadrature_updates_dispatch_regional_j2_and_creep():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("distributed quadrature acceptance requires two ranks")
    j2_model, j2_displacement = _model_with_regions(creep=False)
    j2_map = constitutive.QuadratureMaterialMap.from_assignments(
        j2_displacement.value.function_space.mesh,
        j2_model.materials,
        material_type=constitutive.J2LinearIsotropicHardening,
    )
    j2 = constitutive.J2QuadratureState.create(j2_map.domain, degree=2)
    j2_strain = np.zeros((len(j2.stress.values), 3, 3))
    j2_strain[:, 0, 0] = 1.0e-3
    j2_update = j2.update(j2_strain, j2_map)

    creep_model, creep_displacement = _model_with_regions(creep=True)
    creep_map = constitutive.QuadratureMaterialMap.from_assignments(
        creep_displacement.value.function_space.mesh,
        creep_model.materials,
        material_type=constitutive.IsotropicPowerLawCreepMaterial,
    )
    creep = constitutive.CreepQuadratureState.create(creep_map.domain, degree=2)
    creep_strain = np.zeros((len(creep.stress.values), 3, 3))
    creep_strain[:, 0, 0] = 2.0e-3
    creep_strain[:, 1, 1] = -1.0e-3
    creep_strain[:, 2, 2] = -1.0e-3
    creep_update = creep.update(
        creep_strain,
        creep_map,
        time_start=0.0,
        time_end=0.1,
    )

    j2_cells = j2_map.domain.topology.index_map(j2_map.domain.topology.dim)
    creep_cells = creep_map.domain.topology.index_map(creep_map.domain.topology.dim)
    assert j2_update["points"] == j2_cells.size_global * len(j2.stress.points)
    assert creep_update["points"] == creep_cells.size_global * len(creep.stress.points)
    assert set(j2_map.cell_regions[: j2_cells.size_local]) <= {1, 2}
    assert set(creep_map.cell_regions[: creep_cells.size_local]) <= {1, 2}


def test_distributed_incomplete_material_coverage_fails_collectively():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("collective material-map failure requires two ranks")
    model, displacement = _model_with_regions(creep=False)
    with pytest.raises(ValueError, match="do not cover every owned cell"):
        constitutive.QuadratureMaterialMap.from_assignments(
            displacement.value.function_space.mesh,
            model.materials[:1],
            material_type=constitutive.J2LinearIsotropicHardening,
        )
