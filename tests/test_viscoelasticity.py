from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import (
    amplitudes,
    benchmarks,
    checkpointing,
    constraints,
    fields,
    mesh,
    models,
    procedures,
    results,
    solvers,
    steps,
    studies,
)
from agentfem.constitutive import (
    ArrheniusShift,
    GeneralizedMaxwell,
    IsotropicGeneralizedMaxwell,
    MaxwellState,
    QuadratureMaterialMap,
    WLFShift,
    fit_relaxation_prony,
    isotropic_generalized_maxwell,
    standard_linear_solid,
)


def test_viscoelastic_study_resolves_its_declared_procedure():
    study = studies.viscoelastic_solid(dimension=3)
    procedure = procedures.for_step(
        analysis=study.analysis,
        method=study.preferred_procedure,
        stateful=True,
    )

    assert study.physics == "solid_mechanics"
    assert study.is_transient
    assert procedure.algorithm == "exact_generalized_maxwell_equilibrium"
    assert procedure.stateful

    harmonic_study = studies.harmonic_solid(dimension=3)
    harmonic_procedure = procedures.for_step(
        analysis=harmonic_study.analysis,
        method=harmonic_study.preferred_procedure,
    )
    assert harmonic_study.is_frequency_domain
    assert not harmonic_study.is_transient
    assert harmonic_procedure.algorithm == "real_block_complex_harmonic"
    assert not harmonic_procedure.stateful


def test_standard_linear_solid_has_correct_time_and_frequency_limits():
    material = standard_linear_solid(
        equilibrium_modulus=2.0,
        relaxing_modulus=8.0,
        relaxation_time=3.0,
    )

    np.testing.assert_allclose(
        material.relaxation_modulus([0.0, 1.0e6]),
        [10.0, 2.0],
        atol=1.0e-10,
    )
    assert material.storage_modulus(0.0) == pytest.approx(2.0)
    assert material.loss_modulus(0.0) == pytest.approx(0.0)
    assert material.storage_modulus(1.0e9) == pytest.approx(10.0)


def test_isotropic_generalized_maxwell_exposes_bulk_shear_harmonic_contract():
    material = IsotropicGeneralizedMaxwell.from_prony(
        instantaneous_young_modulus=1000.0,
        instantaneous_poisson_ratio=0.0,
        shear_relaxation_ratios=[0.4],
        bulk_relaxation_ratios=[0.4],
        relaxation_times=[2.0],
    )
    harmonic = material.harmonic_moduli(0.5)
    transfer = 1j / (1.0 + 1j)
    expected_young = 600.0 + 400.0 * transfer

    assert harmonic.young == pytest.approx(expected_young)
    assert harmonic.bulk.imag > 0.0
    assert harmonic.shear.imag > 0.0
    assert harmonic.summary()["phasor_convention"] == "exp(+i*omega*t)"
    bulk, shear = material.complex_moduli([0.0, 1.0e12])
    np.testing.assert_allclose(
        bulk.real,
        [material.equilibrium_bulk_modulus, material.instantaneous_bulk_modulus],
        rtol=1.0e-11,
    )
    np.testing.assert_allclose(
        shear.real,
        [material.equilibrium_shear_modulus, material.instantaneous_shear_modulus],
        rtol=1.0e-11,
    )


def test_generalized_maxwell_frequency_limit_remains_finite_without_overflow():
    material = IsotropicGeneralizedMaxwell.from_prony(
        instantaneous_young_modulus=1000.0,
        instantaneous_poisson_ratio=0.2,
        shear_relaxation_ratios=[0.4],
        bulk_relaxation_ratios=[0.2],
        relaxation_times=[2.0],
    )

    bulk, shear = material.complex_moduli(np.finfo(float).max)

    assert np.isfinite(bulk)
    assert np.isfinite(shear)
    assert bulk == pytest.approx(material.instantaneous_bulk_modulus)
    assert shear == pytest.approx(material.instantaneous_shear_modulus)


def test_direct_harmonic_viscoelastic_bar_matches_complex_modulus(tmp_path):
    length = 2.0
    traction = 3.0
    instantaneous_young = 1000.0
    frequency = 1.0 / (4.0 * np.pi)
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (length, 1.0, 1.0),
        (4, 1, 1),
        comm=MPI.COMM_SELF,
        cell_type="hexahedron",
    )
    model = models.create(
        study=studies.harmonic_solid(dimension=3),
        mesh=domain,
        name="harmonic_viscoelastic_bar",
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        IsotropicGeneralizedMaxwell.from_prony(
            instantaneous_young_modulus=instantaneous_young,
            instantaneous_poisson_ratio=0.0,
            shear_relaxation_ratios=[0.4],
            bulk_relaxation_ratios=[0.4],
            relaxation_times=[2.0],
        )
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=0.0),
        component=0,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="y", value=0.0),
        component=1,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="z", value=0.0),
        component=2,
    )
    model.traction(
        (traction, 0.0, 0.0),
        on=mesh.face(domain, axis="x", value=length),
    )
    step = model.step(target=displacement, frequency=frequency)
    output = tmp_path / "fields.xdmf"
    result = step.solve_result(output=output, strict_output=True)
    manifest = result.write_manifest(tmp_path / "result.json")
    expected = traction * length / material.harmonic_moduli(0.5).young
    right = mesh.face(domain, axis="x", value=length)
    real = results.average(step.solution_real[0], measure=right.measure)
    imaginary = results.average(step.solution_imaginary[0], measure=right.measure)
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.global_viscoelastic_harmonic_bar"
    )
    normalized = (real + 1j * imaginary) * (
        instantaneous_young / (traction * length)
    )

    assert real + 1j * imaginary == pytest.approx(expected, rel=2.0e-10)
    golden.quantity("normalized_complex_end_displacement").assert_accepts(
        [normalized.real, normalized.imag]
    )
    assert step.procedure.algorithm == "real_block_complex_harmonic"
    assert result.quantity("frequency") == pytest.approx(frequency)
    assert result.quantity("dissipated_energy_per_cycle") > 0.0
    assert result.quantity("input_energy_per_cycle") > 0.0
    assert result.quantity("mean_kinetic_energy") == pytest.approx(0.0)
    assert result.quantity("relative_cycle_energy_balance_error") < 1.0e-10
    assert result.quantity("relative_residual_norm") < 1.0e-10
    assert result.quantity("relative_real_block_residual_norm") < 1.0e-10
    assert result.quantity("relative_imaginary_block_residual_norm") < 1.0e-10
    assert result.quantity("mean_stored_energy") > 0.0
    assert set(result.fields) == {"U_REAL", "U_IMAG", "U_AMPLITUDE", "U_PHASE"}
    assert output.exists()
    assert (tmp_path / "fields.h5").exists()
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert {item["name"] for item in saved["field_records"]} == set(result.fields)
    assert result.metadata["step"]["includes_inertia"] is False
    excitation = result.scientific_inputs["harmonic_excitation"]
    assert excitation["frequency"] == pytest.approx(frequency)
    assert excitation["phasor_convention"] == "exp(+i*omega*t)"
    assert result.metadata["step"]["solve"]["converged"]


def test_harmonic_viscoelastic_bar_with_inertia_converges_to_wave_solution():
    length = 2.0
    traction = 3.0
    density = 1.0
    omega = 8.0
    material = IsotropicGeneralizedMaxwell.from_prony(
        instantaneous_young_modulus=1000.0,
        instantaneous_poisson_ratio=0.0,
        shear_relaxation_ratios=[0.4],
        bulk_relaxation_ratios=[0.4],
        relaxation_times=[0.125],
    )
    complex_young = material.harmonic_moduli(omega).young
    wave_number = omega * np.sqrt(density / complex_young)
    expected = traction * np.tan(wave_number * length) / (
        complex_young * wave_number
    )
    errors = []

    finest_result = None
    for longitudinal_cells in (4, 8, 16):
        domain = mesh.cuboid(
            (0.0, 0.0, 0.0),
            (length, 1.0, 1.0),
            (longitudinal_cells, 1, 1),
            comm=MPI.COMM_SELF,
            cell_type="hexahedron",
        )
        model = models.create(
            study=studies.harmonic_solid(dimension=3),
            mesh=domain,
            name=f"harmonic_inertial_bar_{longitudinal_cells}",
        )
        displacement = model.field(fields.displacement(domain))
        model.material(material)
        model.fix(
            displacement,
            on=mesh.face(domain, axis="x", value=0.0),
            component=0,
        )
        model.fix(
            displacement,
            on=mesh.face(domain, axis="y", value=0.0),
            component=1,
        )
        model.fix(
            displacement,
            on=mesh.face(domain, axis="z", value=0.0),
            component=2,
        )
        loaded_end = mesh.face(domain, axis="x", value=length)
        model.traction((traction, 0.0, 0.0), on=loaded_end)

        step = model.step(
            target=displacement,
            angular_frequency=omega,
            density=density,
        )
        result = step.solve_result()
        finest_result = result
        tip = results.average(
            step.solution_real[0], measure=loaded_end.measure
        ) + 1j * results.average(
            step.solution_imaginary[0], measure=loaded_end.measure
        )
        errors.append(abs(tip - expected) / abs(expected))

        assert result.quantity("mean_kinetic_energy") > 0.0
        assert result.quantity("relative_cycle_energy_balance_error") < 1.0e-9

    assert errors[2] < errors[1] < errors[0]
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.global_viscoelastic_harmonic_bar"
    )
    golden.quantity("inertial_tip_relative_error").assert_accepts(errors[-1])
    golden.quantity("relative_cycle_energy_balance_error").assert_accepts(
        finest_result.quantity("relative_cycle_energy_balance_error")
    )


def test_harmonic_bar_resolves_independent_bulk_and_shear_relaxation():
    length = 2.0
    width = 1.0
    thickness = 0.5
    traction = 3.0
    omega = 2.5
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (length, width, thickness),
        (4, 2, 1),
        comm=MPI.COMM_SELF,
        cell_type="hexahedron",
    )
    model = models.create(
        study=studies.harmonic_solid(dimension=3),
        mesh=domain,
        name="independent_bulk_shear_harmonic_bar",
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        IsotropicGeneralizedMaxwell.from_prony(
            instantaneous_young_modulus=1200.0,
            instantaneous_poisson_ratio=0.25,
            shear_relaxation_ratios=[0.2],
            bulk_relaxation_ratios=[0.55],
            relaxation_times=[0.4],
        )
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=0.0),
        component=0,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="y", value=0.0),
        component=1,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="z", value=0.0),
        component=2,
    )
    loaded_end = mesh.face(domain, axis="x", value=length)
    lateral_y = mesh.face(domain, axis="y", value=width)
    lateral_z = mesh.face(domain, axis="z", value=thickness)
    model.traction((traction, 0.0, 0.0), on=loaded_end)

    step = model.step(target=displacement, angular_frequency=omega)
    step.solve()
    harmonic = material.harmonic_moduli(omega)

    def phasor(component, surface):
        return results.average(
            step.solution_real[component], measure=surface.measure
        ) + 1j * results.average(
            step.solution_imaginary[component], measure=surface.measure
        )

    assert phasor(0, loaded_end) == pytest.approx(
        traction * length / harmonic.young,
        rel=5.0e-10,
    )
    assert phasor(1, lateral_y) == pytest.approx(
        -harmonic.poisson * traction * width / harmonic.young,
        rel=5.0e-10,
    )
    assert phasor(2, lateral_z) == pytest.approx(
        -harmonic.poisson * traction * thickness / harmonic.young,
        rel=5.0e-10,
    )
    assert abs(harmonic.poisson.imag) > 1.0e-3


def test_harmonic_step_resolves_and_enforces_the_central_procedure_contract():
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1, 1, 1),
        comm=MPI.COMM_SELF,
        cell_type="hexahedron",
    )
    model = models.create(
        study=studies.harmonic_solid(dimension=3),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        IsotropicGeneralizedMaxwell.from_prony(
            instantaneous_young_modulus=1000.0,
            instantaneous_poisson_ratio=0.0,
            shear_relaxation_ratios=[0.2],
            bulk_relaxation_ratios=[0.2],
            relaxation_times=[1.0],
        )
    )

    capability = models.step_capability(
        model,
        target=displacement,
        options={"frequency": 1.0},
    )
    step = model.step(target=displacement, frequency=1.0)

    assert capability["procedure"]["algorithm"] == "real_block_complex_harmonic"
    assert capability["supported"] is True
    assert capability["ready"] is True
    assert step.procedure.algorithm == "real_block_complex_harmonic"
    with pytest.raises(ValueError, match="Unknown numerical method"):
        model.step(target=displacement, frequency=1.0, method="newmark")

    incomplete = models.step_capability(model, target=displacement)
    assert incomplete["supported"] is True
    assert incomplete["ready"] is False
    assert incomplete["readiness_issues"][0]["code"] == "AFM-STEP-OPTION-003"


def test_harmonic_provider_requires_one_unambiguous_material():
    domain = dolfinx_mesh.create_box(
        MPI.COMM_SELF,
        [np.zeros(3), np.ones(3)],
        [1, 2, 1],
        cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    model = models.create(
        study=studies.harmonic_solid(dimension=3),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    regions = mesh.partition_cells(
        domain,
        lower=lambda x: x[1] <= 0.5,
        upper=lambda x: x[1] > 0.5,
    )
    registered = []
    for young, region in ((1000.0, regions.lower), (2000.0, regions.upper)):
        registered.append(
            model.material(
                IsotropicGeneralizedMaxwell.from_prony(
                    instantaneous_young_modulus=young,
                    instantaneous_poisson_ratio=0.0,
                    shear_relaxation_ratios=[0.2],
                    bulk_relaxation_ratios=[0.2],
                    relaxation_times=[1.0],
                ),
                region=region,
            )
        )

    ambiguous = models.step_capability(
        model,
        target=displacement,
        options={"frequency": 1.0},
    )
    selected = models.step_capability(
        model,
        target=displacement,
        options={"frequency": 1.0, "material": registered[0]},
    )

    assert ambiguous["supported"] is False
    assert ambiguous["provider"] is None
    assert "harmonic_generalized_maxwell" in ambiguous["candidate_providers"]
    with pytest.raises(NotImplementedError, match="No step provider accepted"):
        model.step(target=displacement, frequency=1.0)
    assert selected["supported"] is True
    assert selected["ready"] is True
    step = model.step(
        target=displacement,
        material=registered[0],
        frequency=1.0,
    )
    assert step.material is registered[0]


def test_harmonic_step_rejects_ambiguous_frequency_and_time_domain_loading():
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1, 1, 1),
        comm=MPI.COMM_SELF,
        cell_type="hexahedron",
    )
    model = models.create(
        study=studies.harmonic_solid(dimension=3),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        IsotropicGeneralizedMaxwell.from_prony(
            instantaneous_young_modulus=1000.0,
            instantaneous_poisson_ratio=0.0,
            shear_relaxation_ratios=[0.2],
            bulk_relaxation_ratios=[0.2],
            relaxation_times=[1.0],
        )
    )
    model.traction(
        (1.0, 0.0, 0.0),
        on=mesh.face(domain, axis="x", value=1.0),
        amplitude=amplitudes.sine(amplitude=1.0, frequency=1.0),
    )

    with pytest.raises(ValueError, match="AmplitudeLoad"):
        model.step(target=displacement, frequency=1.0)

    clean = models.create(
        study=studies.harmonic_solid(dimension=3),
        mesh=domain,
    )
    clean_displacement = clean.field(fields.displacement(domain))
    clean.material(model.materials[0].item)
    with pytest.raises(TypeError, match="exactly one"):
        clean.step(
            target=clean_displacement,
            frequency=1.0,
            angular_frequency=2.0 * np.pi,
        )


def test_harmonic_step_rejects_nonhomogeneous_constraints_and_nested_lu():
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1, 1, 1),
        comm=MPI.COMM_SELF,
        cell_type="hexahedron",
    )
    material = IsotropicGeneralizedMaxwell.from_prony(
        instantaneous_young_modulus=1000.0,
        instantaneous_poisson_ratio=0.0,
        shear_relaxation_ratios=[0.2],
        bulk_relaxation_ratios=[0.2],
        relaxation_times=[1.0],
    )

    prescribed = models.create(
        study=studies.harmonic_solid(dimension=3),
        mesh=domain,
    )
    prescribed_u = prescribed.field(fields.displacement(domain))
    prescribed.material(material)
    prescribed.fix(
        prescribed_u,
        on=mesh.face(domain, axis="x", value=0.0),
        component=0,
        value=0.1,
    )
    with pytest.raises(NotImplementedError, match="homogeneous strong constraints"):
        prescribed.step(target=prescribed_u, frequency=1.0)

    factored = models.create(
        study=studies.harmonic_solid(dimension=3),
        mesh=domain,
    )
    factored_u = factored.field(fields.displacement(domain))
    factored.material(material)
    with pytest.raises(NotImplementedError, match="pc_type='fieldsplit'"):
        factored.step(
            target=factored_u,
            frequency=1.0,
            solver_options=solvers.direct_solver(),
        )

    unsupported = models.create(
        study=studies.harmonic_solid(dimension=3),
        mesh=domain,
    )
    unsupported_u = unsupported.field(fields.displacement(domain))
    unsupported.material(material)
    unsupported.add_constraint(object())
    with pytest.raises(NotImplementedError, match="strong Dirichlet constraints"):
        unsupported.step(target=unsupported_u, frequency=1.0)

    overflow = models.create(
        study=studies.harmonic_solid(dimension=3),
        mesh=domain,
    )
    overflow_u = overflow.field(fields.displacement(domain))
    overflow.material(material)
    with pytest.raises(ValueError, match="inertia coefficient"):
        overflow.step(
            target=overflow_u,
            angular_frequency=1.0e100,
            density=1.0e200,
        )


def test_generalized_maxwell_exact_update_commits_and_restores_state():
    material = GeneralizedMaxwell(2.0, [8.0], [1.0])
    state = MaxwellState.zero(1)
    snapshot = state.snapshot()
    update = material.update(state, np.asarray(0.1), 0.5)

    expected_tangent = 2.0 + 8.0 * (1.0 - np.exp(-0.5)) / 0.5
    assert update.algorithmic_modulus == pytest.approx(expected_tangent)
    assert float(update.stress) == pytest.approx(expected_tangent * 0.1)
    expected_dissipation = (
        1.6**2 / 8.0 * (0.5 - 2.0 * (1.0 - np.exp(-0.5)) + 0.5 * (1.0 - np.exp(-1.0)))
    )
    assert update.dissipated_energy_increment == pytest.approx(expected_dissipation)
    assert update.dissipated_energy_increment > 0.0
    assert float(state.strain) == 0.0
    state.commit(update)
    assert float(state.strain) == pytest.approx(0.1)
    state.restore(snapshot)
    assert float(state.strain) == 0.0
    assert state.dissipated_energy == 0.0


def test_generalized_maxwell_small_increment_retains_instantaneous_tangent():
    material = GeneralizedMaxwell(2.0, [8.0], [1.0])
    state = MaxwellState.zero(1)
    update = material.update(state, np.asarray(1.0e-12), 1.0e-16)

    assert update.algorithmic_modulus == pytest.approx(
        material.instantaneous_modulus,
        rel=1.0e-15,
    )
    assert np.isfinite(update.dissipated_energy_increment)
    assert update.dissipated_energy_increment >= 0.0


def test_prony_factory_and_temperature_shift_contracts():
    wlf = WLFShift(reference_temperature=293.15, c1=17.44, c2=51.6)
    arrhenius = ArrheniusShift(
        activation_energy=50.0e3,
        reference_temperature=293.15,
    )
    material = GeneralizedMaxwell.from_prony(
        10.0,
        [0.2, 0.3],
        [1.0, 100.0],
        shift=wlf,
    )

    assert material.equilibrium_modulus == pytest.approx(5.0)
    np.testing.assert_allclose(material.prony_ratios, [0.2, 0.3])
    assert float(wlf.factor(293.15)) == pytest.approx(1.0)
    assert float(arrhenius.factor(293.15)) == pytest.approx(1.0)
    assert float(wlf.factor(313.15)) < 1.0
    assert float(arrhenius.factor(313.15)) < 1.0


def test_temperature_shift_rejects_singular_or_nonfinite_inputs():
    wlf = WLFShift(reference_temperature=293.15, c1=17.44, c2=51.6)
    arrhenius = ArrheniusShift(
        activation_energy=50.0e3,
        reference_temperature=293.15,
    )

    with pytest.raises(ValueError, match="singularity"):
        wlf.factor(293.15 - 51.6)
    with pytest.raises(ValueError, match="finite"):
        arrhenius.factor(np.nan)


def test_maxwell_state_rejects_invalid_branch_count_snapshot_and_commit():
    with pytest.raises(ValueError, match="positive integer"):
        MaxwellState.zero(1.5)
    with pytest.raises(ValueError, match="positive integer"):
        MaxwellState.zero(True)

    state = MaxwellState.zero(1)
    invalid_snapshot = state.snapshot()
    invalid_snapshot["dissipated_energy"] = np.nan
    with pytest.raises(ValueError, match="finite state"):
        state.restore(invalid_snapshot)

    invalid_update = GeneralizedMaxwell(2.0, [8.0], [1.0]).update(
        state,
        np.asarray(0.1),
        0.5,
    )
    invalid_update = type(invalid_update)(
        strain=invalid_update.strain,
        overstress=invalid_update.overstress,
        stress=invalid_update.stress,
        algorithmic_modulus=invalid_update.algorithmic_modulus,
        dissipated_energy_increment=np.nan,
        mechanical_work_increment=invalid_update.mechanical_work_increment,
    )
    with pytest.raises(ValueError, match="nonnegative dissipation"):
        state.commit(invalid_update)


def test_maxwell_update_rejects_nonfinite_time_increment_and_shifted_times():
    material = GeneralizedMaxwell(2.0, [8.0], [1.0])
    state = MaxwellState.zero(1)
    with pytest.raises(ValueError, match="finite and positive"):
        material.update(state, np.asarray(0.1), np.nan)

    shifted = GeneralizedMaxwell(
        2.0,
        [8.0],
        [np.finfo(float).max],
        shift=WLFShift(reference_temperature=300.0, c1=1.0, c2=100.0),
    )
    with pytest.raises(ValueError, match="Shifted relaxation times"):
        shifted.shifted_relaxation_times(201.0)


def test_fixed_spectrum_prony_fit_recovers_positive_reference_model():
    reference = GeneralizedMaxwell(3.0, [5.0, 2.0], [0.2, 20.0])
    time = np.concatenate(([0.0], np.logspace(-3, 3, 120)))
    measured = reference.relaxation_modulus(time)
    fit = fit_relaxation_prony(time, measured, [0.2, 20.0])

    assert fit.relative_root_mean_square_error < 1.0e-12
    assert fit.model.equilibrium_modulus == pytest.approx(3.0)
    np.testing.assert_allclose(fit.model.branch_moduli, [5.0, 2.0])


def test_generalized_maxwell_history_uses_common_state_procedure_and_result():
    material = GeneralizedMaxwell(2.0, [8.0, 3.0], [0.2, 5.0])
    time = np.linspace(0.0, 2.0, 21)
    strain = np.minimum(time, 0.5) * 0.02

    step = material.history(time, strain, temperature=293.15)
    result = step.solve_result()

    assert step.procedure.algorithm == "exact_generalized_maxwell_update"
    assert not step.procedure.requires_global_solve
    assert result.metadata["procedure"] == step.procedure.summary()
    assert result.metadata["state"]["accepted_increments"] == 20
    assert result.metadata["state"]["restartable"]
    assert result.metadata["scope"]["level"] == "material_point"
    assert not result.metadata["scope"]["global_fem_provider"]
    assert set(result.histories) >= {
        "strain",
        "stress",
        "branch_overstress",
        "stored_energy",
        "dissipated_energy",
        "mechanical_work",
        "energy_balance_error",
        "temperature",
    }
    assert np.all(np.diff(result.histories["dissipated_energy"].values) >= 0.0)
    assert result.quantity("maximum_absolute_energy_balance_error") < 1.0e-14
    assert result.scientific_input_manifest()["complete"]


def test_generalized_maxwell_history_restart_matches_uninterrupted_path():
    material = GeneralizedMaxwell(2.0, [8.0, 3.0], [0.2, 5.0])
    time = np.linspace(0.0, 2.0, 21)
    strain = np.minimum(time, 0.5) * 0.02
    complete = material.history(time, strain).solve()

    first = material.history(time[:11], strain[:11]).solve()
    restarted = material.history(
        time[10:],
        strain[10:],
        initial_state=first.final_state,
    ).solve()

    np.testing.assert_allclose(restarted.stress, complete.stress[10:])
    np.testing.assert_allclose(
        restarted.branch_overstress,
        complete.branch_overstress[10:],
    )
    assert restarted.final_state.dissipated_energy == pytest.approx(
        complete.final_state.dissipated_energy
    )
    np.testing.assert_allclose(
        restarted.final_state.overstress,
        complete.final_state.overstress,
    )


def test_instantaneous_state_relaxes_to_the_closed_form_and_closes_energy():
    material = GeneralizedMaxwell(2.0, [8.0, 3.0], [0.2, 5.0])
    time = np.linspace(0.0, 10.0, 101)
    strain = np.full(time.size, 0.02)
    initial = material.initial_state(0.02, condition="instantaneous")

    response = material.history(time, strain, initial_state=initial).solve()

    np.testing.assert_allclose(
        response.stress,
        0.02 * material.relaxation_modulus(time),
        rtol=1.0e-13,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(response.mechanical_work, response.mechanical_work[0])
    assert np.max(np.abs(response.energy_balance_error)) < 1.0e-16
    assert response.dissipated_energy[-1] > 0.0


def test_generalized_maxwell_initial_state_has_explicit_history_semantics():
    material = GeneralizedMaxwell(2.0, [8.0], [1.0])
    equilibrated = material.initial_state([0.1, 0.2])
    instantaneous = material.initial_state([0.1, 0.2], condition="instantaneous")

    np.testing.assert_allclose(equilibrated.overstress, 0.0)
    np.testing.assert_allclose(instantaneous.overstress, [[0.8, 1.6]])
    with pytest.raises(ValueError, match="condition"):
        material.initial_state(0.0, condition="unknown")


def test_generalized_maxwell_history_rejects_incompatible_restart_and_time():
    material = GeneralizedMaxwell(2.0, [8.0], [1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        material.history([0.0, 0.0], [0.0, 0.1]).solve()

    state = MaxwellState.zero(1)
    state.strain[...] = 0.2
    with pytest.raises(ValueError, match="first strain sample"):
        material.history([0.0, 1.0], [0.0, 0.1], initial_state=state).solve()


def test_isotropic_tensor_prony_factory_preserves_instantaneous_elasticity():
    material = isotropic_generalized_maxwell(
        instantaneous_young_modulus=1200.0,
        instantaneous_poisson_ratio=0.3,
        shear_relaxation_ratios=[0.2, 0.3],
        bulk_relaxation_ratios=[0.1, 0.2],
        relaxation_times=[0.5, 5.0],
    )

    assert material.instantaneous_shear_modulus == pytest.approx(
        1200.0 / (2.0 * 1.3)
    )
    assert material.instantaneous_bulk_modulus == pytest.approx(
        1200.0 / (3.0 * 0.4)
    )
    bulk, shear = material.relaxation_moduli([0.0, 1.0e8])
    assert bulk[0] == pytest.approx(material.instantaneous_bulk_modulus)
    assert shear[0] == pytest.approx(material.instantaneous_shear_modulus)
    assert bulk[-1] == pytest.approx(material.equilibrium_bulk_modulus)
    assert shear[-1] == pytest.approx(material.equilibrium_shear_modulus)


def test_isotropic_tensor_maxwell_relaxation_and_energy_are_exact():
    material = IsotropicGeneralizedMaxwell.from_prony(
        instantaneous_young_modulus=1000.0,
        instantaneous_poisson_ratio=0.25,
        shear_relaxation_ratios=[0.4],
        bulk_relaxation_ratios=[0.2],
        relaxation_times=[2.0],
    )
    strain = np.diag([0.02, -0.005, 0.003])
    state = material.initial_state(strain, condition="instantaneous")
    unpacked = material.state_schema.unpack(state)
    initial_stress = (
        material.instantaneous_bulk_modulus * np.trace(strain) * np.eye(3)
        + 2.0
        * material.instantaneous_shear_modulus
        * (strain - np.trace(strain) * np.eye(3) / 3.0)
    )

    update = material.update(state, strain, 1.25)
    bulk, shear = material.relaxation_moduli(1.25)
    expected = (
        bulk * np.trace(strain) * np.eye(3)
        + 2.0 * shear * (strain - np.trace(strain) * np.eye(3) / 3.0)
    )

    np.testing.assert_allclose(
        material.equilibrium_bulk_modulus * np.trace(strain) * np.eye(3)
        + 2.0
        * material.equilibrium_shear_modulus
        * (strain - np.trace(strain) * np.eye(3) / 3.0)
        + np.sum(unpacked["shear_overstress"], axis=0)
        + np.sum(unpacked["bulk_overstress"]) * np.eye(3),
        initial_stress,
    )
    np.testing.assert_allclose(update.stress, expected, rtol=2.0e-14, atol=1.0e-13)
    assert update.mechanical_work_increment == pytest.approx(0.0, abs=1.0e-14)
    assert update.dissipated_energy_increment > 0.0
    previous_energy = (
        material.equilibrium_shear_modulus
        * np.sum((strain - np.trace(strain) * np.eye(3) / 3.0) ** 2)
        + 0.5 * material.equilibrium_bulk_modulus * np.trace(strain) ** 2
        + sum(
            np.sum(value**2) / (4.0 * modulus)
            for value, modulus in zip(
                unpacked["shear_overstress"],
                material.shear_branch_moduli,
                strict=True,
            )
            if modulus > 0.0
        )
        + sum(
            value**2 / (2.0 * modulus)
            for value, modulus in zip(
                unpacked["bulk_overstress"],
                material.bulk_branch_moduli,
                strict=True,
            )
            if modulus > 0.0
        )
    )
    assert previous_energy == pytest.approx(
        update.stored_energy_density + update.dissipated_energy_increment,
        rel=2.0e-13,
        abs=1.0e-13,
    )


def test_isotropic_tensor_maxwell_algorithmic_tangent_matches_fixed_state_difference():
    material = IsotropicGeneralizedMaxwell.from_prony(
        instantaneous_young_modulus=1500.0,
        instantaneous_poisson_ratio=0.32,
        shear_relaxation_ratios=[0.25, 0.15],
        bulk_relaxation_ratios=[0.05, 0.1],
        relaxation_times=[0.1, 8.0],
    )
    old_strain = np.asarray(
        [[0.01, 0.002, 0.0], [0.002, -0.003, 0.001], [0.0, 0.001, 0.004]]
    )
    state = material.initial_state(old_strain, condition="instantaneous")
    new_strain = old_strain + np.asarray(
        [[0.003, -0.001, 0.0005], [-0.001, 0.001, 0.0], [0.0005, 0.0, -0.002]]
    )
    update = material.update(state, new_strain, 0.7)
    direction = np.asarray(
        [[0.7, -0.2, 0.1], [-0.2, -0.4, 0.3], [0.1, 0.3, 0.2]]
    )
    step = 1.0e-7
    plus = material.update(state, new_strain + step * direction, 0.7).stress
    minus = material.update(state, new_strain - step * direction, 0.7).stress
    numerical = (plus - minus) / (2.0 * step)
    analytical = np.einsum("ijkl,kl->ij", update.consistent_tangent, direction)
    np.testing.assert_allclose(analytical, numerical, rtol=2.0e-8, atol=2.0e-7)


def _global_viscoelastic_relaxation_patch(
    *, shift=None, temperature=None, step_options=None
):
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(
        study=studies.viscoelastic_solid(dimension=3),
        mesh=domain,
        name="generalized_maxwell_relaxation",
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        IsotropicGeneralizedMaxwell.from_prony(
            instantaneous_young_modulus=1000.0,
            instantaneous_poisson_ratio=0.0,
            shear_relaxation_ratios=[0.4],
            bulk_relaxation_ratios=[0.4],
            relaxation_times=[1.0],
            shift=shift,
        )
    )
    model.constraint(
        constraints.fixed_component(
            displacement,
            0,
            on=mesh.face(domain, axis="x", value=0.0, name="left"),
        )
    )
    model.constraint(
        constraints.fixed_component(
            displacement,
            1,
            on=mesh.face(domain, axis="y", value=0.0, name="y_symmetry"),
        )
    )
    model.constraint(
        constraints.fixed_component(
            displacement,
            2,
            on=mesh.face(domain, axis="z", value=0.0, name="z_symmetry"),
        )
    )
    model.constraint(
        constraints.fixed_component(
            displacement,
            0,
            value=0.01,
            on=mesh.face(domain, axis="x", value=1.0, name="right"),
        )
    )
    options = {
        "steps": 2,
        "amplitude": amplitudes.tabular(
            [0.0, 1.0, 2.0], [0.0, 1.0, 1.0]
        ),
        "temperature": temperature,
        "solver_options": solvers.newton(
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-11,
            maximum_iterations=4,
        ),
        "progress": False,
    }
    options.update(step_options or {})
    step = model.step(
        target=displacement,
        material=material,
        duration=2.0,
        **options,
    )
    return step


def test_global_generalized_maxwell_relaxation_matches_exact_solution():
    step = _global_viscoelastic_relaxation_patch()
    step.solve(until=1.0)
    stress_after_ramp = results.average(
        step.state.stress.function[0, 0], measure=step.state.measure
    )
    simulation = step.solve_result()
    final_stress = results.average(
        step.state.stress.function[0, 0], measure=step.state.measure
    )
    branch = 0.4 * 1000.0
    equilibrium = 0.6 * 1000.0
    expected_ramp = 0.01 * (equilibrium + branch * (1.0 - np.exp(-1.0)))
    expected_final = 0.01 * (
        equilibrium + branch * (1.0 - np.exp(-1.0)) * np.exp(-1.0)
    )
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.global_viscoelastic_relaxation"
    )

    assert step.procedure.algorithm == "exact_generalized_maxwell_equilibrium"
    assert step.last_solve_info.completed_step
    assert stress_after_ramp == pytest.approx(expected_ramp, rel=2.0e-10)
    assert final_stress == pytest.approx(expected_final, rel=2.0e-10)
    assert all(
        golden.verify(
            {
                "mean_axial_stress_after_ramp": stress_after_ramp,
                "mean_axial_stress_after_hold": final_stress,
            }
        ).values()
    )
    assert final_stress < stress_after_ramp
    assert {"S", "E", "SENER", "VDENER", "MISES", "RF"} <= set(
        simulation.fields
    )
    assert simulation.histories["viscous_dissipation"].latest > 0.0
    assert abs(simulation.histories["constitutive_energy_residual"].latest) < 1.0e-12


def test_global_generalized_maxwell_adaptive_time_control_rejects_and_recovers():
    quadratic = amplitudes.Amplitude(
        name="quadratic_loading",
        value=lambda time: (time / 2.0) ** 2,
        serializable=False,
    )
    adaptive = _global_viscoelastic_relaxation_patch(
        step_options={
            "steps": None,
            "incrementation": steps.automatic(
                initial=1.0,
                minimum=1.0 / 256.0,
                maximum=1.0,
                max_increments=256,
                max_cutbacks=10,
                cutback_factor=0.5,
                growth_factor=1.5,
            ),
            "time_error_tolerance": 2.0e-3,
            "amplitude": quadratic,
        }
    )
    reference = _global_viscoelastic_relaxation_patch(
        step_options={
            "steps": 256,
            "amplitude": quadratic,
        }
    )

    adaptive.solve()
    reference.solve()
    simulation = adaptive.solve_result()

    assert adaptive.last_solve_info.completed_step
    assert len(adaptive.attempted_increments) > len(adaptive.accepted_increments)
    assert any(not item.converged for item in adaptive.attempted_increments)
    assert all(
        item.time_error_estimate <= adaptive.time_error_tolerance
        for item in adaptive.accepted_increments
    )
    assert adaptive.summary()["time_grid"]["kind"] == "automatic"
    assert adaptive.summary()["attempted_increments"] == len(
        adaptive.attempted_increments
    )
    assert {
        "equilibrium_iterations",
        "time_increment",
        "time_error_estimate",
    } <= set(simulation.histories)
    assert simulation.metadata["solve"]["attempted_increments"] == len(
        adaptive.attempted_increments
    )
    assert any(
        not item["converged"] for item in simulation.metadata["solve"]["attempts"]
    )
    np.testing.assert_allclose(
        adaptive.state.stress.values,
        reference.state.stress.values,
        rtol=5.0e-3,
        atol=1.0e-8,
    )


def test_global_generalized_maxwell_adaptive_restart_preserves_next_increment(
    tmp_path,
):
    control = steps.automatic(
        initial=0.25,
        minimum=0.01,
        maximum=0.25,
        max_increments=40,
        cutback_factor=0.5,
    )
    options = {
        "steps": None,
        "incrementation": control,
        "time_error_tolerance": 1.0e-3,
    }
    reference = _global_viscoelastic_relaxation_patch(step_options=options)
    reference.solve()
    partial = _global_viscoelastic_relaxation_patch(step_options=options)
    partial.solve(until=1.0)
    checkpoint = partial.save_checkpoint(tmp_path / "adaptive_viscoelastic.npz")
    restarted = _global_viscoelastic_relaxation_patch(step_options=options)
    restarted.load_checkpoint(checkpoint)

    assert restarted.next_increment_size == pytest.approx(
        partial.next_increment_size
    )
    restarted.solve()
    np.testing.assert_allclose(
        restarted.state.state.committed_state_vectors(),
        reference.state.state.committed_state_vectors(),
    )
    assert [item.end_time for item in restarted.accepted_increments] == pytest.approx(
        [item.end_time for item in reference.accepted_increments]
    )


def test_global_generalized_maxwell_restart_matches_uninterrupted_path(tmp_path):
    reference = _global_viscoelastic_relaxation_patch()
    reference.solve()

    partial = _global_viscoelastic_relaxation_patch()
    partial.solve(until=1.0)
    checkpoint = partial.save_checkpoint(tmp_path / "viscoelastic_restart.npz")
    assert checkpoint.with_suffix(checkpoint.suffix + ".checkpoint.json").is_file()
    restarted = _global_viscoelastic_relaxation_patch()
    restarted.load_checkpoint(checkpoint)
    restarted.solve()

    assert restarted.last_solve_info.completed_step
    np.testing.assert_allclose(restarted.solution.x.array, reference.solution.x.array)
    np.testing.assert_allclose(
        restarted.state.state.committed_state_vectors(),
        reference.state.state.committed_state_vectors(),
    )
    np.testing.assert_allclose(restarted.state.stress.values, reference.state.stress.values)
    np.testing.assert_allclose(
        [item.constitutive_energy_residual for item in restarted.energy_history],
        [item.constitutive_energy_residual for item in reference.energy_history],
    )


def test_global_generalized_maxwell_schedules_and_retains_checkpoints(tmp_path):
    directory = tmp_path / "scheduled"
    policy = checkpointing.every(
        2,
        directory=directory,
        final=True,
        keep_last=1,
    )
    step = _global_viscoelastic_relaxation_patch(
        step_options={"steps": 4, "checkpoint": policy}
    )

    result = step.solve_result()

    assert len(result.checkpoints) == 1
    record = next(iter(result.checkpoints.values()))
    assert record.coordinate_value == pytest.approx(2.0)
    assert record.metadata["role"] == "scheduled_checkpoint"
    assert record.path.is_file()
    assert record.path.with_suffix(record.path.suffix + ".checkpoint.json").is_file()
    assert len(tuple(directory.glob("*.npz"))) == 1
    assert len(tuple(directory.glob("*.checkpoint.json"))) == 1
    assert result.metadata["step"]["checkpoint_policy"]["keep_last"] == 1
    assert result.metadata["step"]["checkpoint_policy"][
        "effective_portable"
    ] is False


def test_global_generalized_maxwell_checkpoint_failure_is_atomic(tmp_path):
    policy = checkpointing.every(1, directory=tmp_path / "scheduled")
    step = _global_viscoelastic_relaxation_patch(
        step_options={"checkpoint": policy}
    )
    initial_solution = step.solution.x.array.copy()
    initial_state = step.state.snapshot()

    def fail_checkpoint(*_args, **_kwargs):
        raise OSError("injected checkpoint failure")

    step.save_checkpoint = fail_checkpoint
    with pytest.raises(RuntimeError, match="checkpoint failed collectively"):
        step.solve(until=1.0)

    np.testing.assert_array_equal(step.solution.x.array, initial_solution)
    for name, values in initial_state["state"].items():
        np.testing.assert_array_equal(step.state.snapshot()["state"][name], values)
    assert step.accepted_time == 0.0
    assert step.accepted_increments == []
    assert step.attempted_increments == []
    assert step.energy_history == []
    assert step.checkpoints == []


def test_global_generalized_maxwell_portable_restart_matches_uninterrupted_path(
    tmp_path,
):
    control = steps.automatic(
        initial=0.25,
        minimum=0.01,
        maximum=0.25,
        max_increments=40,
        cutback_factor=0.5,
    )
    options = {
        "steps": None,
        "incrementation": control,
        "time_error_tolerance": 1.0e-3,
    }
    reference = _global_viscoelastic_relaxation_patch(step_options=options)
    reference.solve()
    partial = _global_viscoelastic_relaxation_patch(step_options=options)
    partial.solve(until=1.0)
    checkpoint = partial.save_checkpoint(
        tmp_path / "portable_viscoelastic", portable=True
    )
    restarted = _global_viscoelastic_relaxation_patch(step_options=options)
    restarted.load_checkpoint(checkpoint)

    assert restarted.next_increment_size == pytest.approx(partial.next_increment_size)
    np.testing.assert_allclose(restarted.state.stress.values, partial.state.stress.values)
    np.testing.assert_allclose(
        restarted.state.stored_energy.values,
        partial.state.stored_energy.values,
    )
    restarted.solve()
    np.testing.assert_allclose(restarted.solution.x.array, reference.solution.x.array)
    np.testing.assert_allclose(
        restarted.state.state.committed_state_vectors(),
        reference.state.state.committed_state_vectors(),
    )


def test_global_generalized_maxwell_portable_restore_is_atomic(tmp_path):
    source = _global_viscoelastic_relaxation_patch()
    source.solve(until=1.0)
    manifest = source.save_checkpoint(tmp_path / "portable_corrupt", portable=True)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    quadrature = manifest.parent / payload["quadrature_state"]["path"]
    with quadrature.open("ab") as stream:
        stream.write(b"corrupt")

    target = _global_viscoelastic_relaxation_patch()
    target.solution.x.array[:] = 0.123
    target.solution.x.scatter_forward()
    displacement_before = target.solution.x.array.copy()
    state_before = target.state.snapshot()
    with pytest.raises(ValueError, match="size|checksum"):
        target.load_checkpoint(manifest)

    np.testing.assert_array_equal(target.solution.x.array, displacement_before)
    for name, values in state_before["state"].items():
        np.testing.assert_array_equal(target.state.snapshot()["state"][name], values)
    assert target.accepted_time == 0.0


def test_global_generalized_maxwell_consumes_temperature_shift_and_guards_restart(
    tmp_path,
):
    shift = WLFShift(reference_temperature=293.15, c1=8.0, c2=80.0)
    reference = _global_viscoelastic_relaxation_patch(
        shift=shift,
        temperature=293.15,
    )
    hot = _global_viscoelastic_relaxation_patch(
        shift=shift,
        temperature=313.15,
    )
    reference_result = reference.solve_result()
    hot_result = hot.solve_result()
    reference_stress = results.average(
        reference.state.stress.function[0, 0], measure=reference.state.measure
    )
    hot_stress = results.average(
        hot.state.stress.function[0, 0], measure=hot.state.measure
    )

    assert hot_stress < reference_stress
    assert hot_result.quantity("minimum_viscoelastic_temperature") == pytest.approx(
        313.15
    )
    assert hot_result.quantity("maximum_viscoelastic_temperature") == pytest.approx(
        313.15
    )
    checkpoint = reference.save_checkpoint(tmp_path / "reference_temperature.npz")
    incompatible = _global_viscoelastic_relaxation_patch(
        shift=shift,
        temperature=303.15,
    )
    with pytest.raises(ValueError, match="temperature"):
        incompatible.load_checkpoint(checkpoint)
    assert reference_result.metadata["step"]["temperature"]["unit"] == "K"


def test_global_generalized_maxwell_consumes_regional_materials():
    domain = dolfinx_mesh.create_box(
        MPI.COMM_SELF,
        [np.zeros(3), np.ones(3)],
        [1, 2, 1],
        cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    model = models.create(
        study=studies.viscoelastic_solid(dimension=3),
        mesh=domain,
        name="regional_viscoelastic_patch",
    )
    displacement = model.field(fields.displacement(domain))
    regions = mesh.partition_cells(
        domain,
        lower=lambda x: x[1] <= 0.5,
        upper=lambda x: x[1] > 0.5,
    )
    for young, region in ((1000.0, regions.lower), (2000.0, regions.upper)):
        model.material(
            IsotropicGeneralizedMaxwell.from_prony(
                instantaneous_young_modulus=young,
                instantaneous_poisson_ratio=0.0,
                shear_relaxation_ratios=[0.4],
                bulk_relaxation_ratios=[0.4],
                relaxation_times=[1.0],
            ),
            region=region,
        )
    model.fix(displacement, on=mesh.face(domain, axis="x", value=0.0), component=0)
    model.fix(displacement, on=mesh.face(domain, axis="y", value=0.0), component=1)
    model.fix(displacement, on=mesh.face(domain, axis="z", value=0.0), component=2)
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=1.0),
        component=0,
        value=0.01,
    )

    step = model.step(target=displacement, duration=1.0, steps=1, progress=False)
    step.solve()
    axial_stress = step.state.stress.values[:, 0, 0]

    assert isinstance(step.material, QuadratureMaterialMap)
    assert len(step.material.materials) == 2
    assert np.max(axial_stress) == pytest.approx(2.0 * np.min(axial_stress))


def _abaqus_viscoelastic_rod_step():
    """Three-dimensional form of the public Abaqus viscoelastic-rod benchmark."""

    k1 = 6.89
    k2 = 62.01
    bulk = 689.0
    instantaneous_extensional = k1 + k2
    equilibrium_shear = 3.0 * bulk * k1 / (9.0 * bulk - k1)
    instantaneous_shear = (
        3.0
        * bulk
        * instantaneous_extensional
        / (9.0 * bulk - instantaneous_extensional)
    )
    shear_relaxation_time = (
        (9.0 * bulk - instantaneous_extensional) / (9.0 * bulk - k1)
    )
    material = IsotropicGeneralizedMaxwell(
        equilibrium_bulk_modulus=bulk,
        equilibrium_shear_modulus=equilibrium_shear,
        shear_branch_moduli=(instantaneous_shear - equilibrium_shear,),
        bulk_branch_moduli=(0.0,),
        relaxation_times=(shear_relaxation_time,),
        name="abaqus_viscoelastic_rod",
    )
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (254.0, 1.0, 1.0),
        (2, 1, 1),
        comm=MPI.COMM_SELF,
        cell_type="hexahedron",
    )
    model = models.create(
        study=studies.viscoelastic_solid(dimension=3),
        mesh=domain,
        name="abaqus_viscoelastic_rod",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(material)
    model.fix(displacement, on=mesh.face(domain, axis="x", value=0.0), component=0)
    model.fix(displacement, on=mesh.face(domain, axis="y", value=0.0), component=1)
    model.fix(displacement, on=mesh.face(domain, axis="z", value=0.0), component=2)
    model.traction(
        (0.689, 0.0, 0.0),
        on=mesh.face(domain, axis="x", value=254.0),
    )
    time_points = np.concatenate(
        ([0.0, 0.001], np.geomspace(0.005, 50.0, 120))
    )
    return model.step(
        target=displacement,
        material=material,
        duration=50.0,
        time_points=time_points,
        amplitude=amplitudes.tabular(
            [0.0, 0.001, 50.0],
            [0.0, 1.0, 1.0],
            name="sudden_constant_traction",
        ),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-11,
            maximum_iterations=6,
        ),
        progress=False,
    )


def test_global_generalized_maxwell_matches_public_abaqus_viscoelastic_rod():
    step = _abaqus_viscoelastic_rod_step()
    step.solve(until=0.001)
    initial_axial_strain = results.average(
        step.state.accepted_strain.function[0, 0], measure=step.state.measure
    )
    step.solve()
    final_axial_strain = results.average(
        step.state.accepted_strain.function[0, 0], measure=step.state.measure
    )
    final_lateral_strain = results.average(
        step.state.accepted_strain.function[1, 1], measure=step.state.measure
    )
    exact_final = 0.1 * (1.0 - 0.9 * np.exp(-5.0))
    final_poisson = -final_lateral_strain / final_axial_strain
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.abaqus_viscoelastic_rod"
    )

    assert step.time_increment is None
    assert step.summary()["time_grid"]["kind"] == "nonuniform"
    assert final_axial_strain == pytest.approx(exact_final, rel=2.0e-3)
    assert all(
        golden.verify(
            {
                "axial_strain_at_0.001_seconds": initial_axial_strain,
                "axial_strain_at_50_seconds": final_axial_strain,
                "effective_poisson_ratio_at_50_seconds": final_poisson,
            }
        ).values()
    )


def test_global_generalized_maxwell_rejects_invalid_declared_time_grid():
    step = _global_viscoelastic_relaxation_patch()
    with pytest.raises(ValueError, match="time_points"):
        replace(step, time_points=(0.0, 1.0, 0.5))


def test_global_generalized_maxwell_rejects_ambiguous_time_controls():
    control = steps.automatic(initial=0.1, minimum=0.01, maximum=0.25)
    with pytest.raises(ValueError, match="incrementation or a prescribed"):
        _global_viscoelastic_relaxation_patch(
            step_options={"incrementation": control}
        )


def test_global_generalized_maxwell_requires_adaptive_control_for_time_error():
    with pytest.raises(ValueError, match="requires adaptive incrementation"):
        _global_viscoelastic_relaxation_patch(
            step_options={"time_error_tolerance": 1.0e-3}
        )
