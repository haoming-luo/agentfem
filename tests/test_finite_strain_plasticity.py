from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI

from agentfem import constitutive, fields, mesh as agentfem_mesh, models, solvers, steps, studies
from agentfem.mechanics import experimental_finite_strain_j2_step


def _point(material, deformation_gradient, *, state=None, old=None):
    return constitutive.MaterialPointInput(
        deformation_gradient_old=(
            np.eye(3) if old is None else np.asarray(old, dtype=float)
        ),
        deformation_gradient_new=np.asarray(deformation_gradient, dtype=float),
        time=0.0,
        time_increment=0.1,
        properties=[],
        state_old=(
            material.state_schema.initial_state() if state is None else state
        ),
        state_schema=material.state_schema,
    )


def _rotation(angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )


def _isochoric_extension(stretch):
    lateral = 1.0 / np.sqrt(stretch)
    return np.diag((stretch, lateral, lateral))


def test_finite_strain_j2_declares_portable_state_and_rejects_bad_parameters():
    material = constitutive.finite_strain_j2_logarithmic(
        young=210_000.0,
        poisson=0.3,
        yield_stress=250.0,
        hardening_modulus=1_000.0,
    )

    state = material.state_schema.unpack(material.state_schema.initial_state())
    np.testing.assert_allclose(state["plastic_deformation_gradient"], np.eye(3))
    assert state["equivalent_plastic_strain"] == pytest.approx(0.0)
    assert state["plastic_dissipation"] == pytest.approx(0.0)
    assert material.tangent_convention.stress_measure == "first_piola"
    assert material.summary()["status"] == "experimental_material_point"
    with pytest.raises(ValueError, match="nu < 0.5"):
        constitutive.finite_strain_j2_logarithmic(
            young=210_000.0,
            poisson=0.5,
            yield_stress=250.0,
        )


def test_finite_strain_j2_rigid_rotation_is_stress_free_and_objective():
    material = constitutive.finite_strain_j2_logarithmic(
        young=210_000.0,
        poisson=0.3,
        yield_stress=250.0,
        hardening_modulus=1_000.0,
    )
    rotation = _rotation(0.61)
    rigid = material.update(_point(material, rotation))

    np.testing.assert_allclose(rigid.cauchy_stress, 0.0, atol=2.0e-10)
    np.testing.assert_allclose(
        rigid.state_new,
        material.state_schema.initial_state(),
        atol=2.0e-13,
    )

    deformation = _isochoric_extension(1.12)
    base = material.update(_point(material, deformation))
    rotated = material.update(_point(material, rotation @ deformation))
    np.testing.assert_allclose(
        rotated.cauchy_stress,
        rotation @ base.cauchy_stress @ rotation.T,
        rtol=2.0e-10,
        atol=2.0e-8,
    )
    base_state = material.state_schema.unpack(base.state_new)
    rotated_state = material.state_schema.unpack(rotated.state_new)
    assert rotated_state["equivalent_plastic_strain"] == pytest.approx(
        base_state["equivalent_plastic_strain"], rel=2.0e-10
    )
    np.testing.assert_allclose(
        rotated_state["plastic_deformation_gradient"],
        base_state["plastic_deformation_gradient"],
        rtol=2.0e-10,
        atol=2.0e-10,
    )


def test_finite_strain_j2_plastic_return_is_isochoric_and_yield_consistent():
    material = constitutive.finite_strain_j2_logarithmic(
        young=210_000.0,
        poisson=0.3,
        yield_stress=250.0,
        hardening_modulus=1_200.0,
    )
    deformation = _isochoric_extension(1.15)
    response = material.update(_point(material, deformation))
    state = material.state_schema.unpack(response.state_new)

    assert state["equivalent_plastic_strain"] > 0.0
    assert np.linalg.det(state["plastic_deformation_gradient"]) == pytest.approx(
        1.0, abs=2.0e-10
    )
    kirchhoff = np.linalg.det(deformation) * response.cauchy_stress
    deviator = kirchhoff - np.trace(kirchhoff) / 3.0 * np.eye(3)
    equivalent = np.sqrt(1.5 * np.tensordot(deviator, deviator))
    assert equivalent == pytest.approx(
        material.current_yield_stress(state["equivalent_plastic_strain"]),
        rel=2.0e-10,
    )


def test_finite_strain_j2_separates_recoverable_energy_components():
    material = constitutive.finite_strain_j2_logarithmic(
        young=210_000.0,
        poisson=0.3,
        yield_stress=250.0,
        hardening_modulus=1_200.0,
    )
    elastic = material.update(_point(material, _isochoric_extension(1.0002)))
    plastic = material.update(_point(material, _isochoric_extension(1.15)))
    plastic_state = material.state_schema.unpack(plastic.state_new)

    assert tuple(elastic.stored_energy_density_components) == (
        "ELENER",
        "HARDENER",
    )
    assert elastic.stored_energy_density_components["HARDENER"] == pytest.approx(
        0.0
    )
    assert elastic.strain_energy_density == pytest.approx(
        elastic.stored_energy_density_components["ELENER"]
    )
    assert plastic.stored_energy_density_components["HARDENER"] == pytest.approx(
        0.5
        * material.hardening_modulus
        * plastic_state["equivalent_plastic_strain"] ** 2,
        rel=2.0e-12,
    )
    assert plastic.strain_energy_density == pytest.approx(
        sum(plastic.stored_energy_density_components.values()),
        rel=2.0e-12,
    )
    assert plastic_state["plastic_dissipation"] == pytest.approx(
        material.yield_stress * plastic_state["equivalent_plastic_strain"],
        rel=2.0e-12,
    )
    assert "yield_stress_times_equivalent_plastic_strain" in (
        material.summary()["stored_energy_density"]["PDENER"]
    )


def test_finite_strain_j2_unloading_does_not_erase_plastic_history():
    material = constitutive.finite_strain_j2_logarithmic(
        young=210_000.0,
        poisson=0.3,
        yield_stress=250.0,
        hardening_modulus=1_000.0,
    )
    loaded_gradient = _isochoric_extension(1.12)
    loaded = material.update(_point(material, loaded_gradient))
    unloaded_gradient = _isochoric_extension(1.119)
    unloaded = material.update(
        _point(
            material,
            unloaded_gradient,
            state=loaded.state_new,
            old=loaded_gradient,
        )
    )
    loaded_state = material.state_schema.unpack(loaded.state_new)
    unloaded_state = material.state_schema.unpack(unloaded.state_new)

    assert unloaded_state["equivalent_plastic_strain"] == pytest.approx(
        loaded_state["equivalent_plastic_strain"], abs=2.0e-13
    )
    assert unloaded_state["plastic_dissipation"] == pytest.approx(
        loaded_state["plastic_dissipation"], abs=2.0e-10
    )
    np.testing.assert_allclose(
        unloaded_state["plastic_deformation_gradient"],
        loaded_state["plastic_deformation_gradient"],
        rtol=0.0,
        atol=2.0e-13,
    )

    reversed_gradient = _isochoric_extension(1.08)
    reversed_response = material.update(
        _point(
            material,
            reversed_gradient,
            state=loaded.state_new,
            old=loaded_gradient,
        )
    )
    reversed_state = material.state_schema.unpack(reversed_response.state_new)
    assert reversed_state["equivalent_plastic_strain"] > loaded_state[
        "equivalent_plastic_strain"
    ]
    assert reversed_state["plastic_dissipation"] > loaded_state[
        "plastic_dissipation"
    ]


def test_finite_strain_j2_reload_preserves_then_extends_history():
    material = constitutive.finite_strain_j2_logarithmic(
        young=210_000.0,
        poisson=0.3,
        yield_stress=250.0,
        hardening_modulus=1_000.0,
    )
    gradients = tuple(
        _isochoric_extension(stretch)
        for stretch in (1.004, 1.0035, 1.004, 1.005)
    )
    state = material.state_schema.initial_state()
    old = np.eye(3)
    responses = []
    states = []
    for gradient in gradients:
        response = material.update(
            _point(material, gradient, state=state, old=old)
        )
        responses.append(response)
        state = response.state_new.copy()
        states.append(material.state_schema.unpack(state))
        old = gradient

    loaded, unloaded, reloaded, extended = states
    assert loaded["equivalent_plastic_strain"] > 0.0
    assert unloaded["equivalent_plastic_strain"] == pytest.approx(
        loaded["equivalent_plastic_strain"], abs=2.0e-13
    )
    assert reloaded["equivalent_plastic_strain"] == pytest.approx(
        loaded["equivalent_plastic_strain"], abs=2.0e-13
    )
    assert unloaded["plastic_dissipation"] == pytest.approx(
        loaded["plastic_dissipation"], abs=2.0e-10
    )
    assert reloaded["plastic_dissipation"] == pytest.approx(
        loaded["plastic_dissipation"], abs=2.0e-10
    )
    np.testing.assert_allclose(
        reloaded["plastic_deformation_gradient"],
        loaded["plastic_deformation_gradient"],
        rtol=0.0,
        atol=2.0e-13,
    )
    assert extended["equivalent_plastic_strain"] > reloaded[
        "equivalent_plastic_strain"
    ]
    assert extended["plastic_dissipation"] > reloaded["plastic_dissipation"]
    for selected in states:
        assert np.linalg.det(
            selected["plastic_deformation_gradient"]
        ) == pytest.approx(1.0, abs=3.0e-10)


def test_finite_strain_j2_nonproportional_order_changes_history():
    material = constitutive.finite_strain_j2_logarithmic(
        young=210_000.0,
        poisson=0.3,
        yield_stress=250.0,
        hardening_modulus=1_000.0,
    )
    tension = _isochoric_extension(1.01)
    final = tension.copy()
    final[0, 1] = 0.015
    shear = np.eye(3)
    shear[0, 1] = 0.015

    def integrate(history):
        state = material.state_schema.initial_state()
        old = np.eye(3)
        response = None
        for gradient in history:
            response = material.update(
                _point(material, gradient, state=state, old=old)
            )
            state = response.state_new.copy()
            old = gradient
        return response, material.state_schema.unpack(state)

    tension_first, tension_state = integrate((tension, final))
    shear_first, shear_state = integrate((shear, final))

    assert abs(
        tension_state["equivalent_plastic_strain"]
        - shear_state["equivalent_plastic_strain"]
    ) > 1.0e-7
    assert np.linalg.norm(
        tension_state["plastic_deformation_gradient"]
        - shear_state["plastic_deformation_gradient"]
    ) > 1.0e-5
    assert np.linalg.norm(
        tension_first.cauchy_stress - shear_first.cauchy_stress
    ) > 1.0

    unloaded = final.copy()
    unloaded[0, 1] -= 1.0e-3
    unloaded_response = material.update(
        _point(
            material,
            unloaded,
            state=tension_first.state_new,
            old=final,
        )
    )
    assert material.state_schema.unpack(unloaded_response.state_new)[
        "equivalent_plastic_strain"
    ] == pytest.approx(
        tension_state["equivalent_plastic_strain"],
        abs=2.0e-13,
    )
    evidence = constitutive.check_material_tangent(
        material,
        constitutive.MaterialPointInput(
            deformation_gradient_old=unloaded,
            deformation_gradient_new=unloaded + 1.0e-5 * np.eye(3),
            time=1.0,
            time_increment=0.1,
            properties=[],
            state_old=unloaded_response.state_new,
            state_schema=material.state_schema,
        ),
        relative_step=2.5e-7,
        tolerance=3.0e-5,
    )
    assert evidence.accepted


@pytest.mark.parametrize("stretch", (1.0005, 1.12))
def test_finite_strain_j2_discrete_tangent_matches_independent_check(stretch):
    material = constitutive.finite_strain_j2_logarithmic(
        young=210_000.0,
        poisson=0.3,
        yield_stress=250.0,
        hardening_modulus=1_000.0,
        tangent_relative_step=2.0e-6,
    )
    point = _point(material, _isochoric_extension(stretch))
    evidence = constitutive.check_material_tangent(
        material,
        point,
        relative_step=2.5e-7,
        tolerance=2.0e-5,
    )

    assert evidence.accepted
    assert evidence.relative_error < 2.0e-5


def test_finite_strain_j2_rejects_nonisochoric_committed_plastic_state():
    material = constitutive.finite_strain_j2_logarithmic(
        young=210_000.0,
        poisson=0.3,
        yield_stress=250.0,
    )
    state = material.state_schema.initial_state()
    state[0] = 1.01
    point = _point(material, np.eye(3), state=state)

    with pytest.raises(ValueError, match="isochoric committed plastic state"):
        material.update(point)


def test_finite_strain_j2_point_properties_cannot_silently_override_provider():
    material = constitutive.finite_strain_j2_logarithmic(
        young=210_000.0,
        poisson=0.3,
        yield_stress=250.0,
    )
    point = replace(_point(material, np.eye(3)), properties=np.ones(4))

    with pytest.raises(ValueError, match="conflict with the provider"):
        material.update(point)


def test_finite_strain_j2_rejects_inconsistent_dissipation_state():
    material = constitutive.finite_strain_j2_logarithmic(
        young=210_000.0,
        poisson=0.3,
        yield_stress=250.0,
        hardening_modulus=1_000.0,
    )
    state = material.state_schema.initial_state()
    state[-2:] = (0.01, 0.0)
    with pytest.raises(ValueError, match="PDENER is inconsistent"):
        material.update(_point(material, np.eye(3), state=state))


def test_finite_strain_j2_quadrature_batch_respects_trial_commit_and_rollback():
    material = constitutive.finite_strain_j2_logarithmic(
        young=210_000.0,
        poisson=0.3,
        yield_stress=250.0,
        hardening_modulus=1_000.0,
    )
    domain = mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    state = constitutive.MaterialQuadratureState.create(
        domain,
        material.state_schema,
        degree=2,
    )
    point_count = len(state.reference_field.values)
    gradients = np.asarray(
        [_isochoric_extension(value) for value in np.linspace(1.06, 1.12, point_count)]
    )
    committed_before = state.committed_state_vectors()

    trial_result = constitutive.update_material_points(
        material,
        state,
        deformation_gradient_old=np.eye(3),
        deformation_gradient_new=gradients,
        time=0.0,
        time_increment=0.1,
    )
    assert not trial_result.committed
    np.testing.assert_allclose(state.committed_state_vectors(), committed_before)
    np.testing.assert_allclose(state.trial_state_vectors(), trial_result.state_new)
    assert np.all(
        state.trial_state_vectors()[:, -2]
        > state.committed_state_vectors()[:, -2]
    )
    assert np.all(
        state.trial_state_vectors()[:, -1]
        > state.committed_state_vectors()[:, -1]
    )

    state.rollback()
    np.testing.assert_allclose(state.trial_state_vectors(), committed_before)
    committed_result = constitutive.update_material_points(
        material,
        state,
        deformation_gradient_old=np.eye(3),
        deformation_gradient_new=gradients,
        time=0.0,
        time_increment=0.1,
        commit=True,
    )
    assert committed_result.committed
    np.testing.assert_allclose(
        state.committed_state_vectors(),
        committed_result.state_new,
    )
    np.testing.assert_allclose(
        committed_result.state_new[:, -1],
        material.yield_stress * committed_result.state_new[:, -2],
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert committed_result.summary()["point_count"] == point_count
    assert tuple(committed_result.stored_energy_density_components) == (
        "ELENER",
        "HARDENER",
    )
    np.testing.assert_allclose(
        committed_result.strain_energy_density,
        committed_result.stored_energy_density_components["ELENER"]
        + committed_result.stored_energy_density_components["HARDENER"],
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_quadrature_response_postprocessing_failure_does_not_commit(monkeypatch):
    material = constitutive.finite_strain_j2_logarithmic(
        young=210_000.0,
        poisson=0.3,
        yield_stress=250.0,
        hardening_modulus=1_000.0,
    )
    domain = mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    response = constitutive.MaterialQuadratureResponse.create(
        domain,
        material.state_schema,
        degree=2,
        stored_energy_component_names=material.stored_energy_component_names,
    )
    point_count = len(response.state.reference_field.values)
    gradients = np.asarray(
        [
            _isochoric_extension(value)
            for value in np.linspace(1.06, 1.12, point_count)
        ]
    )
    committed_before = response.state.committed_state_vectors().copy()

    def fail_assignment(_values):
        raise OSError("injected response-field failure")

    monkeypatch.setattr(response.first_piola_stress, "assign", fail_assignment)
    with pytest.raises(RuntimeError, match="response assignment failed"):
        response.update(
            material,
            deformation_gradient_old=np.eye(3),
            deformation_gradient_new=gradients,
            time=0.1,
            time_increment=0.1,
            commit=True,
        )

    np.testing.assert_array_equal(
        response.state.committed_state_vectors(), committed_before
    )
    np.testing.assert_array_equal(
        response.state.trial_state_vectors(), committed_before
    )


def test_material_batch_failure_restores_trial_state_atomically():
    material = constitutive.finite_strain_j2_logarithmic(
        young=210_000.0,
        poisson=0.3,
        yield_stress=250.0,
    )
    domain = mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    state = constitutive.MaterialQuadratureState.create(
        domain,
        material.state_schema,
        degree=2,
    )
    baseline = state.committed_state_vectors()

    class FailingProvider:
        name = "intentional batch failure"
        state_schema = material.state_schema
        tangent_convention = material.tangent_convention

        def __init__(self):
            self.calls = 0

        def update(self, point):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("intentional local failure")
            return material.update(point)

    with pytest.raises(RuntimeError, match="intentional local failure"):
        constitutive.update_material_points(
            FailingProvider(),
            state,
            deformation_gradient_old=np.eye(3),
            deformation_gradient_new=_isochoric_extension(1.08),
            time=0.0,
            time_increment=0.1,
        )

    np.testing.assert_allclose(state.committed_state_vectors(), baseline)
    np.testing.assert_allclose(state.trial_state_vectors(), baseline)


def _global_finite_strain_j2_patch(*, incrementation, cells=(1, 1, 1)):
    domain = mesh.create_unit_cube(MPI.COMM_SELF, *cells)
    study = studies.nonlinear_static(physics="solid_mechanics", dimension=3)
    model = models.create(study=study, mesh=domain, name="finite_strain_j2_patch")
    displacement = model.field(fields.displacement(domain))
    left = agentfem_mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="left", tag=1
    )
    right = agentfem_mesh.boundary(
        domain, lambda x: np.isclose(x[0], 1.0), name="right", tag=2
    )
    y_symmetry = agentfem_mesh.boundary(
        domain, lambda x: np.isclose(x[1], 0.0), name="y_symmetry", tag=3
    )
    z_symmetry = agentfem_mesh.boundary(
        domain, lambda x: np.isclose(x[2], 0.0), name="z_symmetry", tag=4
    )
    model.fix(displacement, on=left, component=0, value=0.0)
    model.fix(displacement, on=y_symmetry, component=1, value=0.0)
    model.fix(displacement, on=z_symmetry, component=2, value=0.0)
    model.fix(displacement, on=right, component=0, value=0.02)
    material = constitutive.finite_strain_j2_logarithmic(
        young=200_000.0,
        poisson=0.3,
        yield_stress=200.0,
        hardening_modulus=2_000.0,
    )
    return experimental_finite_strain_j2_step(
        displacement=displacement,
        material=material,
        constraints=model.constraints,
        incrementation=incrementation,
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
    )


def test_global_finite_strain_j2_patch_consumes_neutral_tangent_and_state():
    step = _global_finite_strain_j2_patch(incrementation=steps.fixed(4))
    solution = step.solve()
    state = step.response.state.committed_state_vectors()

    assert step.accepted_load_factor == pytest.approx(1.0)
    assert len(step.accepted_increments) == 4
    assert all(item.converged for item in step.accepted_increments)
    assert np.max(state[:, -2]) > 0.0
    assert np.max(state[:, -1]) > 0.0
    for plastic_gradient in state[:, :9].reshape((-1, 3, 3)):
        assert np.linalg.det(plastic_gradient) == pytest.approx(1.0, abs=5.0e-10)
    assert np.max(solution.x.array) == pytest.approx(0.02)
    assert step.summary()["maturity"] == "experimental_global_mpi_restart"
    assert step.summary()["evidence_level"] == "internal_serial_mpi_restart_verified"


def test_global_finite_strain_j2_inelastic_limit_forces_real_cutback():
    control = steps.automatic(
        initial=1.0,
        minimum=1.0e-4,
        maximum=1.0,
        max_increments=100,
        max_cutbacks=12,
        cutback_factor=0.5,
        maximum_inelastic_increment=0.006,
    )
    step = _global_finite_strain_j2_patch(incrementation=control)
    step.solve()
    rejected = [item for item in step.attempted_increments if not item.converged]

    assert step.accepted_load_factor == pytest.approx(1.0)
    assert rejected
    assert "maximum equivalent plastic-strain increment" in rejected[0].rejection_reason
    assert max(
        item.maximum_plastic_increment for item in step.accepted_increments
    ) <= control.maximum_inelastic_increment * (1.0 + 1.0e-10)


def test_global_finite_strain_j2_checkpoint_restart_matches_continuous_path(
    tmp_path,
):
    reference = _global_finite_strain_j2_patch(incrementation=steps.fixed(4))
    reference.solve()
    expected_solution = reference.solution.x.array.copy()
    expected_state = reference.response.state.committed_state_vectors()

    partial = _global_finite_strain_j2_patch(incrementation=steps.fixed(4))
    partial.solve(until=0.5)
    checkpoint = partial.save_checkpoint(tmp_path / "finite_strain_j2_restart")
    restarted = _global_finite_strain_j2_patch(incrementation=steps.fixed(4))
    restarted.load_checkpoint(checkpoint)
    restarted.solve()

    np.testing.assert_allclose(
        restarted.solution.x.array,
        expected_solution,
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        restarted.response.state.committed_state_vectors(),
        expected_state,
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        restarted.response.state.committed_state_vectors()[:, -1],
        reference.response.state.committed_state_vectors()[:, -1],
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    assert [
        item.load_factor for item in restarted.accepted_increments
    ] == pytest.approx([0.25, 0.5, 0.75, 1.0])

    incompatible = _global_finite_strain_j2_patch(incrementation=steps.fixed(5))
    with pytest.raises(ValueError, match="increment control"):
        incompatible.load_checkpoint(checkpoint)


def test_global_finite_strain_j2_multielement_patch_preserves_uniform_path():
    reference = _global_finite_strain_j2_patch(incrementation=steps.fixed(4))
    reference.solve()
    refined = _global_finite_strain_j2_patch(
        incrementation=steps.fixed(4),
        cells=(2, 2, 2),
    )
    refined.solve()
    reference_peeq = reference.response.state.committed_state_vectors()[:, -2]
    refined_peeq = refined.response.state.committed_state_vectors()[:, -2]

    assert len(refined_peeq) > len(reference_peeq)
    assert np.ptp(refined_peeq) < 2.0e-10
    assert np.mean(refined_peeq) == pytest.approx(
        np.mean(reference_peeq),
        rel=2.0e-8,
        abs=2.0e-10,
    )
    assert max(
        item.residual_norm for item in refined.accepted_increments
    ) < 1.0e-6
