import numpy as np
import pytest
from dolfinx import fem
from mpi4py import MPI
import ufl

from agentfem import (
    amplitudes,
    constitutive,
    constraints,
    diagnostics,
    fields,
    fracture,
    interfaces,
    mesh,
    models,
    operators,
    problems,
    studies,
    time,
)
from agentfem.step_providers import step_capability


def _dynamic_neo_hookean_model():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2,
            assumption="plane_strain",
            method="explicit",
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.neo_hookean(
            young=1.0e6,
            poisson=0.3,
            density=1000.0,
        )
    )
    return model, displacement, material


def test_neo_hookean_explicit_selects_finite_strain_provider_not_linear_elasticity():
    model, displacement, material = _dynamic_neo_hookean_model()
    capability = step_capability(
        model,
        target=displacement,
        options={"material": material, "dt": 1.0e-5, "steps": 1},
    )
    assert capability["supported"]
    assert capability["provider"]["name"] == (
        "neo_hookean_finite_strain_explicit_dynamics"
    )


def test_performance_ledger_reports_rank_local_nested_stages():
    ledger = diagnostics.PerformanceLedger()
    ledger.add("run_wall", 2.0)
    ledger.add("residual_assembly", 0.5)
    ledger.add("residual_assembly", 0.25)
    summary = ledger.summary()
    assert summary["run_wall_seconds"] == pytest.approx(2.0)
    residual = summary["stages"]["residual_assembly"]
    assert residual["calls"] == 2
    assert residual["seconds_per_call"] == pytest.approx(0.375)
    assert residual["fraction_of_run_wall"] == pytest.approx(0.375)


def test_finite_strain_explicit_reports_constitutive_not_quadratic_energy():
    model, displacement, material = _dynamic_neo_hookean_model()
    displacement.value.interpolate(
        lambda x: np.vstack((0.01 * x[0], np.zeros_like(x[1])))
    )
    step = model.step(
        target=displacement,
        material=material,
        dt=1.0e-7,
        steps=1,
        progress=False,
    )
    step.run()
    assert step.residual.summary()["family"] == "total_lagrangian_neo_hookean"
    assert "bulk_strain_energy" in step.history_records[0]
    assert "strain_energy" not in step.history_records[0]
    assert step.history_records[0]["bulk_strain_energy"] > 0.0
    assert "natural_load_work" in step.history_records[-1]
    assert "prescribed_motion_work" in step.history_records[-1]
    assert "energy_balance_error" in step.history_records[-1]
    performance = step.summary()["performance"]
    assert performance["stages"]["explicit_increment"]["calls"] == 1
    assert performance["stages"]["residual_assembly"]["seconds"] > 0.0


def test_explicit_prescribed_amplitude_sets_displacement_velocity_and_acceleration():
    domain = mesh.rectangle(
        (0.0, 0.0), (1.0, 1.0), (1, 1),
        comm=MPI.COMM_SELF, cell_type="quadrilateral",
    )
    displacement = fields.displacement(domain)
    state = problems.second_order_state(displacement)
    prescribed = constraints.time_dependent_component_dirichlet(
        displacement,
        0,
        on=lambda x: np.isclose(x[0], 0.0),
        amplitude=amplitudes.sine(amplitude=0.01, frequency=2.0),
    )
    mass = problems.LumpedMassOperator.assemble(displacement.space, density=1.0)
    zero = operators.OperatorForm(
        name="zero",
        expression=ufl.inner(
            fem.Constant(domain, np.array((0.0, 0.0))), displacement.test,
        ) * ufl.dx,
        kind="zero_force",
        role="vector",
        family="test",
    )
    integrator = time.explicit.central_difference(state=state, mass=mass)
    dt = 1.0e-4
    integrator.step(dt, time=dt, residual_operator=zero, prescribed=(prescribed,))
    dof_indices, first_ghost = prescribed.bc.dof_indices()
    selected = dof_indices[:first_ghost]
    expected_u = prescribed.amplitude(dt)
    expected_v = (
        prescribed.amplitude(dt + 0.5 * dt)
        - prescribed.amplitude(dt - 0.5 * dt)
    ) / dt
    expected_a = (
        prescribed.amplitude(dt + 0.5 * dt)
        - 2.0 * prescribed.amplitude(dt)
        + prescribed.amplitude(dt - 0.5 * dt)
    ) / (0.5 * dt) ** 2
    np.testing.assert_allclose(state.u.value.x.array[selected], expected_u)
    np.testing.assert_allclose(state.v.value.x.array[selected], expected_v)
    np.testing.assert_allclose(state.a.value.x.array[selected], expected_a)


def test_dynamic_energy_ledger_integrates_natural_and_prescribed_work():
    domain = mesh.rectangle(
        (0.0, 0.0), (1.0, 1.0), (1, 1),
        comm=MPI.COMM_SELF, cell_type="quadrilateral",
    )
    displacement = fields.displacement(domain)
    state = problems.second_order_state(displacement)
    fixed = constraints.component_dirichlet(
        displacement, 0, on=lambda x: np.isclose(x[0], 0.0), value=0.0,
    )
    force = operators.OperatorForm(
        name="constant_external",
        expression=ufl.inner(
            fem.Constant(domain, np.array((2.0, 0.0))), displacement.test,
        ) * ufl.dx,
        kind="external_force",
        role="vector",
        family="test",
    )
    zero = operators.OperatorForm(
        name="zero_residual",
        expression=ufl.inner(
            fem.Constant(domain, np.array((0.0, 0.0))), displacement.test,
        ) * ufl.dx,
        kind="zero_force",
        role="vector",
        family="test",
    )
    mass = problems.LumpedMassOperator.assemble(displacement.space, density=1.0)

    class ZeroEnergy:
        def evaluate(self, **_kwargs):
            return {"total_mechanical_energy": 0.0}

    ledger = fracture.DynamicEnergyLedger(
        energy=ZeroEnergy(),
        state=state,
        mass=mass,
        residual=zero,
        natural_force=force,
        prescribed=(fixed,),
    )
    ledger.evaluate(displacement=state.u, velocity=state.v)
    dofmap = state.u.function_space.dofmap
    owned = dofmap.index_map.size_local * dofmap.index_map_bs
    delta = np.zeros(owned)
    delta[0::2] = 0.1
    state.u.value.x.array[:owned] = delta
    state.a.value.x.array[:owned] = 2.0
    record = ledger.evaluate(displacement=state.u, velocity=state.v)
    assembled = operators.assemble_vector(force)
    try:
        expected_natural = float(np.dot(assembled.array[:owned], delta))
    finally:
        assembled.destroy()
    constrained, first_ghost = fixed.bc.dof_indices()
    expected_prescribed = 0.5 * float(
        np.dot(mass.mass[constrained[:first_ghost]] * 2.0, delta[constrained[:first_ghost]])
    )
    assert record["natural_load_work"] == pytest.approx(expected_natural)
    assert record["prescribed_motion_work"] == pytest.approx(expected_prescribed)
    assert record["external_work"] == pytest.approx(
        expected_natural + expected_prescribed
    )


def test_direct_explicit_convenience_cannot_silently_linearize_neo_hookean():
    model, displacement, _ = _dynamic_neo_hookean_model()
    step = model.explicit_dynamics_step(
        target=displacement,
        dt=1.0e-7,
        steps=1,
        progress=False,
    )
    assert step.residual.summary()["family"] == "total_lagrangian_neo_hookean"
    with pytest.raises(TypeError, match="deformation-dependent tangent"):
        model.stiffness(displacement)


def test_model_internal_force_dispatches_neo_hookean_to_first_piola_residual():
    model, displacement, _ = _dynamic_neo_hookean_model()
    internal = model.internal_force(displacement)
    summary = internal.summary()
    assert summary["kind"] == "finite_strain_internal_force"
    assert summary["metadata"]["stress_measure"] == "first_piola"


def test_reference_wave_speeds_and_stability_limits_are_explicitly_scoped():
    material = constitutive.neo_hookean(
        young=1.0e6,
        poisson=0.25,
        density=1000.0,
    )
    speeds = fracture.isotropic_reference_wave_speeds(material)
    assert speeds.pressure > speeds.shear > speeds.rayleigh > 0.0
    assert speeds.configuration == "unstretched_reference"

    estimate = fracture.estimate_stable_time_increment(
        characteristic_length=(0.1, 0.2),
        dilatational_speed=speeds.pressure,
        safety_factor=0.5,
        interface_stiffness=1.0e9,
        interface_area=0.01,
        negative_mass=0.1,
        positive_mass=0.1,
    )
    assert estimate.selected == pytest.approx(
        0.5 * min(estimate.body_limit, estimate.interface_limit)
    )
    assert estimate.controller in {"body", "interface"}


def test_incremental_acoustic_tensor_recovers_unstretched_bulk_modes():
    material = constitutive.neo_hookean(
        young=1.0e6,
        poisson=0.25,
        density=1000.0,
    )
    reference = fracture.isotropic_reference_wave_speeds(material)
    modes = fracture.incremental_wave_speeds(
        np.eye(2),
        (1.0, 0.0),
        material,
        direction_configuration="current",
    )
    np.testing.assert_allclose(
        modes.speeds,
        [reference.shear, reference.pressure],
    )
    np.testing.assert_allclose(modes.current_direction, [1.0, 0.0])
    np.testing.assert_allclose(modes.reference_direction, [1.0, 0.0])
    assert modes.summary()["rayleigh_speed"] is None


def test_principal_surface_wave_solver_recovers_classical_rayleigh_speed():
    material = constitutive.neo_hookean(
        young=1.0e6,
        poisson=0.25,
        density=1000.0,
    )
    classical = fracture.isotropic_reference_wave_speeds(material)
    surface = fracture.principal_surface_wave_speed(np.eye(2), material)
    assert surface.speed == pytest.approx(classical.rayleigh, rel=1.0e-8)
    assert surface.limiting_bulk_speed == pytest.approx(classical.shear)
    assert surface.secular_residual < 1.0e-8
    assert np.all(np.imag(surface.attenuation_roots) > 0.0)


def test_principal_surface_wave_solver_tracks_plane_stress_prestrain():
    material = constitutive.neo_hookean_plane_stress(
        young=1.0e6,
        poisson=0.49,
        density=1000.0,
    )
    transverse_then_axial = (
        constitutive.plane_stress_uniaxial_deformation_gradient(1.12, material)
    )
    F = np.diag(np.diag(transverse_then_axial)[::-1])
    undeformed = fracture.principal_surface_wave_speed(np.eye(2), material)
    prestrained = fracture.principal_surface_wave_speed(F, material)
    assert 0.0 < prestrained.speed < prestrained.limiting_bulk_speed
    assert prestrained.speed != pytest.approx(undeformed.speed)
    assert prestrained.secular_residual < 1.0e-8

    with pytest.raises(ValueError, match="diagonal deformation"):
        fracture.principal_surface_wave_speed(
            np.array(((1.0, 0.1), (0.0, 1.0))),
            material,
        )
    with pytest.raises(ValueError, match="not traction-free"):
        fracture.principal_surface_wave_speed(
            transverse_then_axial,
            material,
        )


def test_prestrained_wave_direction_is_explicitly_pulled_back_and_pushed_forward():
    material = constitutive.neo_hookean(
        young=1.0e6,
        poisson=0.3,
        density=1000.0,
    )
    F = np.array([[1.25, 0.15], [0.0, 0.9]])
    current = fracture.incremental_wave_speeds(
        F, (1.0, 0.0), material, direction_configuration="current",
    )
    reference = fracture.incremental_wave_speeds(
        F,
        current.reference_direction,
        material,
        direction_configuration="reference",
    )
    np.testing.assert_allclose(reference.speeds, current.speeds)
    np.testing.assert_allclose(
        reference.current_direction, current.current_direction, atol=1.0e-14
    )
    assert not np.allclose(
        current.speeds,
        fracture.incremental_wave_speeds(
            np.eye(2), (1.0, 0.0), material
        ).speeds,
    )


def test_neo_hookean_material_tangent_matches_first_piola_finite_difference():
    material = constitutive.neo_hookean(
        young=2.0e6,
        poisson=0.28,
        density=1100.0,
    )
    F = np.array([[1.1, 0.08], [0.03, 0.92]])
    tangent = fracture.neo_hookean_material_tangent(F, material)

    def first_piola(gradient):
        J = np.linalg.det(gradient)
        return (
            material.mu * gradient
            + (material.lambda_ * np.log(J) - material.mu)
            * np.linalg.inv(gradient).T
        )

    epsilon = 1.0e-7
    for k in range(2):
        for L in range(2):
            perturbation = np.zeros_like(F)
            perturbation[k, L] = epsilon
            numerical = (
                first_piola(F + perturbation) - first_piola(F - perturbation)
            ) / (2.0 * epsilon)
            np.testing.assert_allclose(
                tangent[:, :, k, L], numerical, rtol=2.0e-8, atol=1.0e-3
            )


def test_plane_stress_thickness_condensation_and_tangent_are_consistent():
    material = constitutive.neo_hookean_plane_stress(
        young=1.0e6,
        poisson=0.49,
        density=1000.0,
    )
    F = np.array(((1.12, 0.03), (0.01, 1.07)))
    thickness = constitutive.plane_stress_thickness_stretch_value(F, material)
    jacobian = np.linalg.det(F) * thickness
    p33 = (
        material.mu * (thickness - 1.0 / thickness)
        + material.lambda_ * np.log(jacobian) / thickness
    )
    assert thickness > 0.0
    assert abs(p33) < 1.0e-7 * material.young
    tangent = fracture.neo_hookean_material_tangent(F, material)
    numerical = np.empty_like(tangent)
    perturbation = 1.0e-6
    for k in range(2):
        for L in range(2):
            plus = F.copy()
            minus = F.copy()
            plus[k, L] += perturbation
            minus[k, L] -= perturbation
            numerical[:, :, k, L] = (
                constitutive.plane_stress_first_piola_value(plus, material)
                - constitutive.plane_stress_first_piola_value(minus, material)
            ) / (2.0 * perturbation)
    np.testing.assert_allclose(tangent, numerical, rtol=2.0e-6, atol=1.0e-3)


def test_plane_stress_ufl_condensation_closes_p33_for_affine_finite_strain():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    displacement = fields.displacement(domain)
    displacement.value.interpolate(
        lambda x: np.vstack(
            (0.12 * x[0] + 0.03 * x[1], 0.01 * x[0] + 0.07 * x[1])
        )
    )
    material = constitutive.neo_hookean_plane_stress(
        young=1.0e6,
        poisson=0.49,
        density=1000.0,
    )
    gradient = ufl.Identity(2) + ufl.grad(displacement.value)
    p33 = constitutive.plane_stress_out_of_plane_first_piola_from_gradient(
        gradient,
        material,
    )
    squared = fem.assemble_scalar(fem.form(p33**2 * ufl.dx))
    assert np.sqrt(squared) < 1.0e-8 * material.young


def test_plane_stress_neo_hookean_runs_through_public_explicit_provider():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2,
            assumption="plane_stress",
            method="explicit",
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.neo_hookean_plane_stress(
            young=1.0e6,
            poisson=0.49,
            density=1000.0,
        )
    )
    capability = step_capability(
        model,
        target=displacement,
        options={"material": material, "dt": 1.0e-6, "steps": 1},
    )
    assert capability["supported"]
    assert capability["provider"]["name"] == (
        "neo_hookean_finite_strain_explicit_dynamics"
    )
    step = model.step(
        target=displacement,
        material=material,
        dt=1.0e-6,
        steps=1,
        progress=False,
    )
    step.run()
    assert step.history_records[-1]["bulk_strain_energy"] == pytest.approx(0.0)


def test_wang_fineberg_needleman_mooney_rivlin_matches_equation_17():
    material = constitutive.mooney_rivlin_plane_stress(
        shear_modulus=500.5e3,
        first_invariant_fraction=0.2829,
        density=1020.0,
    )
    F = np.array(((1.12, 0.04), (0.01, 0.96)))
    invariant = np.trace(F @ F.T)
    jacobian = np.linalg.det(F)
    expected = 0.5 * material.mu * (
        0.2829 * (invariant + jacobian**-2 - 3.0)
        + (1.0 - 0.2829)
        * (jacobian**2 + invariant * jacobian**-2 - 3.0)
    )

    assert constitutive.hyperelasticity.mooney_rivlin_energy_value(
        F, material
    ) == pytest.approx(expected)
    assert material.c10 == pytest.approx(0.5 * 500.5e3 * 0.2829)
    assert material.c01 == pytest.approx(0.5 * 500.5e3 * (1.0 - 0.2829))
    assert material.as_dict()["source_equation"] == (
        "Wang-Fineberg-Needleman Eq. 17"
    )
    np.testing.assert_allclose(
        constitutive.hyperelasticity.mooney_rivlin_first_piola_value(F, material),
        np.array(
            (
                (163078.9242560367, 21669.180985113762),
                (18834.376194589076, 33467.54124463248),
            )
        ),
        rtol=2.0e-6,
        atol=0.1,
    )
    uniaxial = constitutive.plane_stress_uniaxial_deformation_gradient(
        1.12,
        material,
    )
    assert np.linalg.det(uniaxial) * 1.12**-0.5 == pytest.approx(1.0)
    assert constitutive.hyperelasticity.mooney_rivlin_first_piola_value(
        uniaxial,
        material,
    )[0, 0] == pytest.approx(0.0, abs=1.0e-3)


def test_mooney_rivlin_tangent_matches_first_piola_and_reference_shear_speed():
    material = constitutive.mooney_rivlin_plane_stress(
        shear_modulus=500.5e3,
        first_invariant_fraction=0.2829,
        density=1020.0,
    )
    F = np.array(((1.08, 0.02), (0.01, 0.94)))
    tangent = fracture.neo_hookean_material_tangent(F, material)
    numerical = np.empty_like(tangent)
    step = 2.0e-5
    for k in range(2):
        for L in range(2):
            perturbation = np.zeros_like(F)
            perturbation[k, L] = step
            numerical[:, :, k, L] = (
                constitutive.hyperelasticity.mooney_rivlin_first_piola_value(
                    F + perturbation, material
                )
                - constitutive.hyperelasticity.mooney_rivlin_first_piola_value(
                    F - perturbation, material
                )
            ) / (2.0 * step)
    np.testing.assert_allclose(tangent, numerical, rtol=3.0e-4, atol=20.0)
    reference = fracture.incremental_wave_speeds(
        np.eye(2), (1.0, 0.0), material, direction_configuration="reference"
    )
    assert reference.slowest == pytest.approx(
        np.sqrt(material.mu / material.density), rel=2.0e-4
    )


def test_plane_stress_mooney_rivlin_runs_through_public_explicit_provider():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2, assumption="plane_stress", method="explicit"
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.mooney_rivlin_plane_stress(
            shear_modulus=500.5e3,
            first_invariant_fraction=0.2829,
            density=1020.0,
        )
    )
    capability = step_capability(
        model,
        target=displacement,
        options={"material": material, "dt": 1.0e-6, "steps": 1},
    )
    assert capability["supported"]
    step = model.step(
        target=displacement,
        material=material,
        dt=1.0e-6,
        steps=1,
        progress=False,
    )
    step.run()
    assert step.history_records[-1]["bulk_strain_energy"] == pytest.approx(0.0)


def test_compressible_mooney_rivlin_rejects_unsupported_two_dimensional_study():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2,
            assumption="plane_strain",
            method="explicit",
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.mooney_rivlin(
            shear_modulus=1.0e6,
            first_invariant_fraction=0.25,
            bulk_modulus=10.0e6,
            density=1000.0,
        )
    )

    capability = step_capability(
        model,
        target=displacement,
        options={"material": material, "dt": 1.0e-6, "steps": 1},
    )
    assert not capability["supported"]


def test_three_dimensional_split_surface_reaches_dolfinx_force_and_restart(tmp_path):
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    split = interfaces.split_conforming_cell_interface(
        coordinates,
        np.array([[0, 1, 2, 3], [0, 2, 1, 4]]),
        positive_cells=[1],
    )
    domain = interfaces.create_dolfinx_split_mesh(
        split, comm=MPI.COMM_SELF, cell_type="tetrahedron"
    )
    displacement = fields.displacement(domain)
    law = interfaces.bilinear_cohesive(
        strength=10.0, fracture_energy=2.0, initial_stiffness=1000.0
    )
    cohesive = fracture.mode_i_cohesive_force(
        split,
        displacement,
        law,
        normal_hint=(0.0, 0.0, 1.0),
    )
    values = displacement.value.x.array.reshape((-1, 3))
    positive_nodes = np.unique(split.positive_facets)
    values[cohesive.node_to_block_dof[positive_nodes], 2] = 0.02
    trial = cohesive.begin()
    cohesive.commit()
    checkpoint = cohesive.save_portable_state(tmp_path / "surface")

    assert trial.opening.shape == (1, 3)
    assert checkpoint.exists()
    cohesive.assembler.state.initialize(0.0)
    cohesive.load_portable_state(checkpoint)
    assert np.all(cohesive.assembler.state.committed_maximum > 0.0)


def test_finite_strain_explicit_can_select_a_visible_automatic_time_increment():
    model, displacement, material = _dynamic_neo_hookean_model()
    step = model.step(
        target=displacement,
        material=material,
        steps=1,
        progress=False,
    )
    stability = step.summary()["stability"]
    assert step.dt == pytest.approx(stability["selected"])
    assert stability["controller"] == "body"
    assert stability["maturity"] == "screening_estimate"


def test_explicit_history_cadence_keeps_initial_periodic_and_final_records():
    model, displacement, material = _dynamic_neo_hookean_model()

    class CountingMonitor:
        calls = 0

        def evaluate(self, **_kwargs):
            self.calls += 1
            return {"evaluations": float(self.calls)}

    step = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        dt=1.0e-7,
        steps=5,
        history_every=2,
        progress=False,
    )
    monitor = CountingMonitor()
    step.history_monitor = monitor
    step.run()
    np.testing.assert_allclose(
        [record["time"] for record in step.history_records],
        [0.0, 2.0e-7, 4.0e-7, 5.0e-7],
    )
    assert monitor.calls == 6
    assert [record["evaluations"] for record in step.history_records] == [
        1.0, 3.0, 5.0, 6.0,
    ]
    assert step.summary()["history_every"] == 2
    assert step.summary()["history_evaluation_every"] == 1


def test_explicit_history_uses_lightweight_advance_between_energy_snapshots():
    model, displacement, material = _dynamic_neo_hookean_model()

    class StatefulMonitor:
        advances = 0
        evaluations = 0

        def advance(self, **_kwargs):
            self.advances += 1
            return {"path_state": float(self.advances)}

        def evaluate(self, **kwargs):
            self.evaluations += 1
            values = self.advance(**kwargs)
            return {
                "path_state": values["path_state"],
                "energy_snapshots": float(self.evaluations),
            }

    step = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        dt=1.0e-7,
        steps=5,
        history_every=2,
        progress=False,
    )
    monitor = StatefulMonitor()
    step.history_monitor = monitor
    step.run()
    assert monitor.advances == 6
    assert monitor.evaluations == 4
    assert [record["path_state"] for record in step.history_records] == [
        1.0,
        3.0,
        5.0,
        6.0,
    ]


def test_finite_strain_explicit_rejects_unimplemented_plane_stress():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2,
            assumption="plane_stress",
            method="explicit",
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.neo_hookean(
            young=1.0e6,
            poisson=0.3,
            density=1000.0,
        )
    )
    capability = step_capability(
        model,
        target=displacement,
        options={"material": material, "dt": 1.0e-5, "steps": 1},
    )
    assert not capability["supported"]


def test_neo_hookean_total_lagrangian_residual_is_objective_under_rigid_rotation():
    model, displacement, material = _dynamic_neo_hookean_model()
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    gradient = rotation - np.eye(2)
    displacement.value.interpolate(lambda x: gradient @ x[:2])

    density = constitutive.hyperelasticity.strain_energy_density(
        displacement.value,
        material,
    )
    energy = fem.assemble_scalar(fem.form(density * ufl.dx))
    residual = model.internal_force(displacement).assemble_vector()
    try:
        assert abs(energy) < 1.0e-10
        assert residual.norm() < 1.0e-8
    finally:
        residual.destroy()


def test_crack_tip_history_interpolates_damage_and_fits_speed_over_a_window():
    coordinate = np.linspace(0.0, 1.0, 101)
    times = np.linspace(0.0, 0.4, 9)
    expected_speed = 1.25
    fronts = 0.2 + expected_speed * times
    width = 0.01
    damage = np.asarray(
        [1.0 / (1.0 + np.exp((coordinate - front) / width)) for front in fronts]
    )
    history = fracture.crack_tip_history(
        times,
        coordinate,
        damage,
        threshold=0.5,
        fit_window=5,
    )
    np.testing.assert_allclose(history.position, fronts, atol=2.0e-3)
    np.testing.assert_allclose(history.speed[1:-1], expected_speed, atol=1.0e-2)
    assert history.summary()["method"].startswith("threshold_interpolation")


def test_representative_crack_speed_uses_declared_physical_path_interval():
    time = np.linspace(0.0, 1.0, 11)
    history = fracture.CohesiveCrackHistory(
        time=time,
        position=0.2 + 2.5 * time,
        speed=np.full(11, 2.5),
        damage_threshold=0.95,
        fit_window=5,
        direction="increasing",
    )
    fitted = fracture.fit_crack_propagation_speed(
        history,
        start_position=0.7,
        end_position=2.2,
    )

    assert fitted is not None
    assert fitted.speed == pytest.approx(2.5)
    assert fitted.r_squared == pytest.approx(1.0)
    assert fitted.samples >= 3
    assert fitted.summary()["method"] == (
        "least_squares_position_over_declared_path_interval"
    )
    assert fracture.fit_crack_propagation_speed(
        history,
        start_position=2.65,
        end_position=2.75,
    ) is None


def test_separation_classification_requires_independent_spall_evidence():
    assert fracture.separation_regime(
        crack_speed=1.2,
        rayleigh_wave_speed=0.9,
        shear_wave_speed=1.0,
        failed_fraction=0.4,
        simultaneous_failed_fraction=0.2,
    ) == "supershear"
    assert fracture.separation_regime(
        crack_speed=100.0,
        rayleigh_wave_speed=0.9,
        shear_wave_speed=1.0,
        failed_fraction=0.95,
        simultaneous_failed_fraction=0.9,
    ) == "spall_like"
    assert fracture.mach_cone_angle(
        crack_speed=2.0,
        shear_wave_speed=1.0,
    ) == pytest.approx(np.pi / 6.0)
    assert fracture.separation_regime(
        crack_speed=3.0,
        rayleigh_wave_speed=0.9,
        shear_wave_speed=1.0,
        pressure_wave_speed=2.0,
        failed_fraction=0.95,
        simultaneous_failed_fraction=0.2,
        rapid_failed_fraction=0.5,
        ligament_traction_ratio=1.0,
    ) == "spall_like"


def test_zero_preload_transfers_to_equilibrated_explicit_state():
    model, displacement, material = _dynamic_neo_hookean_model()
    state = problems.second_order_state(displacement)
    mass = problems.LumpedMassOperator.assemble(
        displacement.space,
        density=material.density,
    )
    residual = model.force_balance(
        internal=fracture.finite_strain_internal_force(
            state.u,
            displacement.test,
            material,
        )
    )
    report = fracture.transfer_preload_to_explicit(
        displacement,
        state=state,
        mass=mass,
        residual=residual,
        force_tolerance=1.0e-10,
    )
    assert report.equilibrium_accepted
    assert report.acceleration_norm == pytest.approx(0.0)
    np.testing.assert_allclose(state.u.value.x.array, displacement.value.x.array)


def test_explicit_step_transfers_held_prestrain_without_a_false_release():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 2),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2,
            assumption="plane_strain",
            method="explicit",
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    strain = 0.05
    displacement.value.interpolate(
        lambda x: np.vstack((np.zeros_like(x[0]), strain * x[1]))
    )
    material = model.material(
        constitutive.neo_hookean(
            young=1000.0,
            poisson=0.25,
            density=1.0,
        )
    )
    model.constraint(
        constraints.component_dirichlet(
            displacement,
            0,
            on=lambda x: np.ones(x.shape[1], dtype=bool),
            value=0.0,
            name="held_lateral_kinematics",
        )
    )
    model.constraint(
        constraints.component_dirichlet(
            displacement,
            1,
            on=lambda x: np.isclose(x[1], 0.0),
            value=0.0,
            name="held_bottom",
        )
    )
    model.constraint(
        constraints.component_dirichlet(
            displacement,
            1,
            on=lambda x: np.isclose(x[1], 0.2),
            value=strain * 0.2,
            name="held_top",
        )
    )
    step = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        dt=1.0e-4,
        steps=2,
        progress=False,
        name="held_prestrain_release_check",
    )
    source_values = step.history_monitor.energy.evaluate(
        displacement=displacement,
        velocity=step.state.v,
    )
    source_energy = source_values["total_mechanical_energy"]
    initial = displacement.value.x.array.copy()
    report = step.initialize_from_preload(
        displacement,
        source_step="quasi_static_preload",
        source_energy=source_energy,
        force_tolerance=1.0e-9,
    )
    assert report.equilibrium_accepted
    assert report.source_step == "quasi_static_preload"
    assert report.destination_step == step.name
    assert report.total_force_norm > report.residual_force_norm
    assert report.relative_energy_jump == pytest.approx(0.0, abs=1.0e-14)
    step.run()
    np.testing.assert_allclose(step.state.u.value.x.array, initial, atol=1.0e-12)


def test_preload_transfer_rejects_an_undeclared_force_release():
    model, displacement, material = _dynamic_neo_hookean_model()
    displacement.value.interpolate(
        lambda x: np.vstack((0.01 * x[0], np.zeros_like(x[1])))
    )
    state = problems.second_order_state(displacement)
    mass = problems.LumpedMassOperator.assemble(
        displacement.space,
        density=material.density,
    )
    residual = model.force_balance(
        internal=fracture.finite_strain_internal_force(
            state.u,
            displacement.test,
            material,
        )
    )
    with pytest.raises(RuntimeError, match="not in equilibrium"):
        fracture.transfer_preload_to_explicit(
            displacement,
            state=state,
            mass=mass,
            residual=residual,
            force_tolerance=1.0e-12,
        )
    released = fracture.transfer_preload_to_explicit(
        displacement,
        state=state,
        mass=mass,
        residual=residual,
        mode="release",
    )
    assert not released.equilibrium_accepted
    assert released.acceleration_norm > 0.0


def test_mass_damping_dissipation_is_typed_and_restart_equivalent(tmp_path):
    def make_step():
        model, displacement, material = _dynamic_neo_hookean_model()
        state = problems.second_order_state(displacement)
        state.v.value.interpolate(
            lambda x: np.vstack((0.1 * np.ones_like(x[0]), np.zeros_like(x[1])))
        )
        step = model.finite_strain_explicit_dynamics_step(
            target=displacement,
            material=material,
            state=state,
            dt=1.0e-5,
            steps=3,
            mass_damping=2.0,
            progress=False,
        )
        return step

    reference = make_step()
    reference.run()

    partial = make_step()
    partial.run(until_step=1)
    checkpoint = partial.save_checkpoint(tmp_path / "damped")
    restarted = make_step()
    restarted.load_checkpoint(checkpoint)
    restarted.run()

    np.testing.assert_allclose(
        restarted.state.u.value.x.array,
        reference.state.u.value.x.array,
    )
    np.testing.assert_allclose(
        restarted.state.v.value.x.array,
        reference.state.v.value.x.array,
    )
    assert restarted.history_records[-1][
        "numerical_damping_dissipation"
    ] == pytest.approx(
        reference.history_records[-1]["numerical_damping_dissipation"]
    )
    assert restarted.history_records[-1]["numerical_damping_dissipation"] > 0.0
