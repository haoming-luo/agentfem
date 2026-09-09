from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import fields, mesh, models, operators, procedures, results, studies
from agentfem.constitutive import isotropic_elastic


def _rayleigh_bar(*, cells: int = 12):
    length = 2.0
    traction = 3.0
    density = 1.0
    young = 1000.0
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (length, 1.0, 1.0),
        (cells, 1, 1),
        comm=MPI.COMM_SELF,
        cell_type="hexahedron",
    )
    model = models.create(
        study=studies.harmonic_solid(dimension=3),
        mesh=domain,
        name="rayleigh_harmonic_bar",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        isotropic_elastic(
            young=young,
            poisson=0.0,
            density=density,
        )
    )
    model.fix(displacement, on=mesh.face(domain, axis="x", value=0.0), component=0)
    model.fix(displacement, on=mesh.face(domain, axis="y", value=0.0), component=1)
    model.fix(displacement, on=mesh.face(domain, axis="z", value=0.0), component=2)
    loaded_end = mesh.face(domain, axis="x", value=length)
    model.traction((traction, 0.0, 0.0), on=loaded_end)
    stiffness = model.stiffness(displacement)
    mass = model.mass(displacement)
    damping = operators.rayleigh_damping(
        mass,
        stiffness,
        mass_coefficient=0.4,
        stiffness_coefficient=2.0e-3,
    )
    force = model.external_force(displacement)
    return model, displacement, loaded_end, stiffness, mass, damping, force


def _tip_phasor(step, loaded_end):
    return results.average(
        step.solution_real[0], measure=loaded_end.measure
    ) + 1j * results.average(
        step.solution_imaginary[0], measure=loaded_end.measure
    )


def _rayleigh_bar_exact(frequency: float) -> complex:
    omega = 2.0 * np.pi * frequency
    effective_young = 1000.0 * (1.0 + 1j * omega * 2.0e-3)
    wave_number = np.sqrt((omega**2 - 1j * omega * 0.4) / effective_young)
    return 3.0 * np.tan(wave_number * 2.0) / (
        effective_young * wave_number
    )


def test_direct_harmonic_operator_contract_is_explicit_and_inspectable():
    _model, _u, _end, stiffness, mass, damping, force = _rayleigh_bar(cells=2)
    system = operators.direct_harmonic_system(
        stiffness,
        force,
        M=mass,
        C=damping,
    )

    assert system.K is stiffness
    assert system.M is mass
    assert system.C is damping
    assert system.F is force
    assert system.validate().is_valid
    assert system.summary()["dissipation_channels"] == {
        "material_loss": False,
        "viscous_damping": True,
    }
    assert "omega C" in system.equation


def test_generic_direct_harmonic_rayleigh_bar_matches_complex_wave_solution():
    model, displacement, loaded_end, stiffness, mass, damping, force = _rayleigh_bar()
    omega = 8.0
    step = model.step(
        target=displacement,
        K=stiffness,
        M=mass,
        C=damping,
        F=force,
        angular_frequency=omega,
    )
    result = step.solve_result()
    tip = results.average(
        step.solution_real[0], measure=loaded_end.measure
    ) + 1j * results.average(step.solution_imaginary[0], measure=loaded_end.measure)

    young = 1000.0
    density = 1.0
    alpha = 0.4
    beta = 2.0e-3
    effective_young = young * (1.0 + 1j * omega * beta)
    wave_number = np.sqrt(
        density * (omega**2 - 1j * omega * alpha) / effective_young
    )
    expected = 3.0 * np.tan(wave_number * 2.0) / (
        effective_young * wave_number
    )

    assert tip == pytest.approx(expected, rel=4.0e-3)
    assert result.quantity("material_dissipated_energy_per_cycle") == pytest.approx(0.0)
    assert result.quantity("viscous_dissipated_energy_per_cycle") > 0.0
    assert result.quantity("relative_cycle_energy_balance_error") < 1.0e-9
    assert result.quantity("relative_residual_norm") < 1.0e-9
    assert result.metadata["step"]["system"]["damping"]["family"] == (
        "rayleigh_damping"
    )


def test_generic_harmonic_prepared_problem_reuses_allocation_across_frequency():
    model, displacement, _end, stiffness, mass, damping, force = _rayleigh_bar(
        cells=4
    )
    step = model.step(
        target=displacement,
        K=stiffness,
        M=mass,
        C=damping,
        F=force,
        frequency=0.5,
    )
    step.solve()
    first = step.complex_dofs.copy()
    step.set_frequency(frequency=1.25)
    step.solve()

    assert not np.allclose(first, step.complex_dofs)
    execution = step.summary()["backend_execution"]
    assert execution["problem_allocation_count"] == 1
    assert execution["matrix_allocation_count"] == 1
    assert execution["solve_count"] == 2
    assert execution["ksp_object_reused"] is True
    assert execution["factorization_reuse_claimed"] is False


def _rayleigh_sweep(*, execution_order="forward"):
    model, displacement, loaded_end, stiffness, mass, damping, force = (
        _rayleigh_bar()
    )
    response = results.harmonic_response(
        "tip_x",
        lambda point: _tip_phasor(point, loaded_end),
        unit="m",
        description="Mean loaded-end axial displacement phasor.",
    )
    step = model.step(
        target=displacement,
        K=stiffness,
        M=mass,
        C=damping,
        F=force,
        frequencies=(1.25, 0.5, 0.875),
        responses=(response,),
        execution_order=execution_order,
    )
    return model, step


def test_direct_harmonic_sweep_is_canonical_bounded_and_matches_reference():
    model, step = _rayleigh_sweep()

    assert model.steps == [step]
    assert step.procedure.algorithm == "real_block_complex_harmonic_sweep"
    result = step.solve_result()

    axis = np.asarray(result.histories["tip_x_AMPLITUDE"].abscissa)
    actual = np.asarray(result.histories["tip_x_REAL"].values) + 1j * np.asarray(
        result.histories["tip_x_IMAG"].values
    )
    expected = np.asarray([_rayleigh_bar_exact(value) for value in axis])
    assert axis == pytest.approx((0.5, 0.875, 1.25))
    assert actual == pytest.approx(expected, rel=4.0e-3)
    assert np.max(
        result.histories["relative_cycle_energy_balance_error"].values
    ) < 1.0e-9
    assert np.max(result.histories["relative_residual_norm"].values) < 1.0e-9
    assert np.all(result.histories["linear_converged"].values == 1.0)
    assert result.quantity("peak_tip_x_frequency") in axis
    assert "maximum_displacement_vector_amplitude" in result.histories
    execution = step.point_step.summary()["backend_execution"]
    assert execution["problem_allocation_count"] == 1
    assert execution["matrix_allocation_count"] == 1
    assert execution["solve_count"] == 3
    assert len(step.records) == 3


def test_direct_harmonic_sweep_forward_reverse_and_chunk_resume_are_equivalent():
    _forward_model, forward = _rayleigh_sweep(execution_order="forward")
    forward_result = forward.solve_result()
    _reverse_model, reverse = _rayleigh_sweep(execution_order="reverse")
    reverse.solve(max_points=1)
    assert not reverse.completed
    with pytest.raises(RuntimeError, match="partial harmonic sweep"):
        reverse.canonical_records()
    with pytest.raises(TypeError, match="integer"):
        reverse.solve(max_points=1.5)
    reverse_result = reverse.solve_result()

    for name in (
        "tip_x_REAL",
        "tip_x_IMAG",
        "tip_x_AMPLITUDE",
        "relative_residual_norm",
        "relative_cycle_energy_balance_error",
    ):
        assert reverse_result.histories[name].values == pytest.approx(
            forward_result.histories[name].values,
            rel=1.0e-11,
            abs=1.0e-13,
        )


def test_direct_harmonic_sweep_rejects_axis_procedure_and_output_mismatches(tmp_path):
    model, displacement, _end, stiffness, mass, damping, force = _rayleigh_bar()
    with pytest.raises(ValueError, match="procedure disagree"):
        model.step(
            target=displacement,
            K=stiffness,
            M=mass,
            C=damping,
            F=force,
            frequencies=(0.5, 1.0),
            procedure=procedures.direct_harmonic(),
        )
    assert model.steps == []

    _model, step = _rayleigh_sweep()
    step.execution_context = step.execution_context.__class__(
        model=step.execution_context.model,
        target=step.execution_context.target,
        material=step.execution_context.material,
        policy=step.execution_context.policy.__class__(output=tmp_path / "fields.xdmf"),
    )
    with pytest.raises(NotImplementedError, match="snapshot frequencies"):
        step.solve_result()


@pytest.mark.parametrize(
    "frequencies",
    [(), (1.0, 1.0), (np.nan,), (-1.0,)],
)
def test_direct_harmonic_sweep_rejects_invalid_frequency_axes(frequencies):
    model, displacement, _end, stiffness, mass, damping, force = _rayleigh_bar(
        cells=2
    )
    with pytest.raises((TypeError, ValueError)):
        model.step(
            target=displacement,
            K=stiffness,
            M=mass,
            C=damping,
            F=force,
            frequencies=frequencies,
        )
