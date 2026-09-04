from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import dynamics, fields, mesh, models, studies
from agentfem.constitutive import elasticity
from agentfem.results import HistoryResult


def _left(x):
    return np.isclose(x[0], 0.0)


def test_fft_recovers_amplitude_frequency_and_phase_contract():
    time = np.arange(0.0, 2.0, 0.001)
    signal = 3.0 * np.sin(2.0 * np.pi * 12.0 * time + 0.2)
    result = dynamics.spectrum(time, signal, window="none")

    index = int(np.argmax(result.amplitude[1:]) + 1)
    assert result.frequency[index] == pytest.approx(12.0)
    assert result.amplitude[index] == pytest.approx(3.0, rel=1.0e-12)
    assert result.dominant_frequency == pytest.approx(12.0)
    assert result.to_result().histories["amplitude"].abscissa_unit == "Hz"


def test_signal_tools_accept_structured_histories_directly():
    time = np.arange(0.0, 2.0, 0.001)
    excitation = HistoryResult(
        "force",
        time,
        np.sin(2.0 * np.pi * 10.0 * time),
        unit="N",
    )
    response = HistoryResult(
        "displacement",
        time,
        2.0 * np.sin(2.0 * np.pi * 10.0 * time + 0.3),
        unit="m",
    )

    spectrum = dynamics.spectrum(excitation, window="none")
    frf = dynamics.frequency_response(excitation, response, window="none")

    assert spectrum.dominant_frequency == pytest.approx(10.0)
    selected = np.argmin(np.abs(frf.frequency - 10.0))
    assert abs(frf.response[selected]) == pytest.approx(2.0)


def test_frequency_response_marks_unexcited_bins_instead_of_dividing_by_zero():
    time = np.arange(0.0, 2.0, 0.001)
    excitation = np.sin(2.0 * np.pi * 10.0 * time)
    response = 2.0 * np.sin(2.0 * np.pi * 10.0 * time + 0.3)
    frf = dynamics.frequency_response(time, excitation, response, window="none")
    selected = np.argmin(np.abs(frf.frequency - 10.0))

    assert frf.valid[selected]
    assert abs(frf.response[selected]) == pytest.approx(2.0)
    assert np.angle(frf.response[selected]) == pytest.approx(0.3)
    assert np.count_nonzero(~frf.valid) > 0
    assert frf.to_result().metadata["frequency_response"]["valid_bins"] > 0


def test_free_decay_returns_standard_damping_quantities():
    samples = np.zeros(41)
    samples[[5, 15, 25, 35]] = np.exp(-0.2 * np.arange(4))
    estimate = dynamics.damping_from_free_decay(samples)

    assert estimate.logarithmic_decrement == pytest.approx(0.2)
    assert estimate.quality_factor == pytest.approx(1.0 / (2.0 * estimate.damping_ratio))


def test_dense_modal_reference_and_modal_superposition():
    stiffness = np.diag([4.0, 9.0])
    mass = np.eye(2)
    basis = dynamics.solve_dense_modes(stiffness, mass)
    response = dynamics.modal_frequency_response(
        basis,
        [0.1, 0.2],
        [1.0, 0.0],
        damping_ratio=0.02,
    )

    np.testing.assert_allclose(basis.angular_frequencies, [2.0, 3.0])
    assert response.shape == (2, 2)
    assert basis.to_result().quantity("frequencies").shape == (2,)


def test_modal_step_uses_public_model_language_and_removes_fixed_dofs():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (8, 2),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.modal_solid(dimension=2, assumption="plane_stress"),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain, degree=2))
    model.material(
        elasticity.isotropic_elastic(
            young=210.0e9,
            poisson=0.3,
            density=7800.0,
        )
    )
    model.clamp(
        displacement,
        on=mesh.boundary(domain, _left, name="left", tag=1),
    )

    capability = models.step_capability(
        model,
        target=displacement,
        options={"modes": 3},
    )
    result = model.step(target=displacement, modes=3).solve_result()

    assert capability["provider"]["name"] == "linear_structural_modes"
    assert result.quantity("frequencies").shape == (3,)
    assert np.all(np.diff(result.quantity("frequencies")) > 0.0)
    assert np.max(result.quantity("residual_norms")) < 1.0e-7
    assert tuple(result.fields) == ("Mode_1", "Mode_2", "Mode_3")
    euler_bernoulli = (
        1.875104068711961**2
        / (2.0 * np.pi)
        * np.sqrt(210.0e9 * 0.2**2 / (12.0 * 7800.0))
    )
    assert result.quantity("frequencies")[0] == pytest.approx(
        euler_bernoulli,
        rel=0.035,
    )
