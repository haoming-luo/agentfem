from __future__ import annotations

import h5py
import numpy as np
import pytest
from mpi4py import MPI
import ufl

from agentfem import (
    dynamics,
    fields,
    mesh,
    models,
    operators,
    problems,
    studies,
    verification,
)
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
    assert estimate.quality_factor == pytest.approx(
        1.0 / (2.0 * estimate.damping_ratio)
    )


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
    assert np.max(basis.residual_norms) < 1.0e-14
    assert basis.metadata["residual_definition"] == "relative_backward_error"
    for column, anchor in enumerate(basis.metadata["orientation_anchor_dofs"]):
        assert basis.modes[anchor, column] > 0.0


def test_dense_modal_reference_rejects_truncation_and_missing_positive_modes():
    stiffness = np.diag([0.0, 4.0])
    mass = np.eye(2)
    with pytest.raises(ValueError, match="positive integer"):
        dynamics.solve_dense_modes(stiffness, mass, modes=1.5)
    with pytest.raises(ValueError, match="found 1 positive modes"):
        dynamics.solve_dense_modes(stiffness, mass, modes=2)


def test_repeated_modes_compare_as_an_invariant_subspace():
    reference = dynamics.solve_dense_modes(
        np.diag([4.0, 4.0, 9.0]),
        np.eye(3),
    )
    angle = np.pi / 5.0
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    rotated_modes = reference.modes.copy()
    rotated_modes[:, :2] = rotated_modes[:, :2] @ rotation
    candidate = dynamics.ModalBasis(
        reference.eigenvalues,
        rotated_modes,
        reference.residual_norms,
    )

    comparison = reference.compare(candidate)

    assert comparison.accepted
    assert tuple(cluster.multiplicity for cluster in reference.clusters()) == (2, 1)
    repeated = comparison.clusters[0]
    assert repeated.modal_assurance is None
    assert repeated.projection_distance == pytest.approx(0.0, abs=1.0e-14)
    assert repeated.maximum_principal_angle_degrees == pytest.approx(
        0.0,
        abs=1.0e-6,
    )
    assert comparison.clusters[1].modal_assurance == pytest.approx(1.0)
    assert reference.metadata["repeated_modes_compare_as"] == "invariant_subspace"


def test_modal_comparison_detects_a_different_repeated_subspace():
    reference = dynamics.ModalBasis(
        [4.0, 4.0],
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
    )
    candidate = dynamics.ModalBasis(
        [4.0, 4.0],
        np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
    )

    comparison = reference.compare(candidate)

    assert not comparison.accepted
    assert comparison.clusters[0].projection_distance == pytest.approx(1.0)
    assert comparison.clusters[0].maximum_principal_angle_degrees == pytest.approx(
        90.0
    )


def test_dense_modal_metadata_marks_a_cluster_cut_by_mode_count():
    basis = dynamics.solve_dense_modes(
        np.diag([4.0, 4.0, 9.0]),
        np.eye(3),
        modes=1,
    )

    assert not basis.metadata["selected_clusters_complete"]
    assert basis.metadata["eigenvalue_clusters"][0]["multiplicity"] == 1


def test_modal_basis_rejects_nonfinite_modes_and_unsorted_eigenvalues():
    with pytest.raises(ValueError, match="finite"):
        dynamics.ModalBasis([1.0], [[np.nan]])
    with pytest.raises(ValueError, match="sorted"):
        dynamics.ModalBasis([2.0, 1.0], np.eye(2))


def test_dynamic_postprocessors_reject_unobservable_or_singular_requests():
    time = np.arange(0.0, 1.0, 0.01)
    with pytest.raises(ValueError, match="observable excitation"):
        dynamics.frequency_response(time, np.zeros_like(time), np.ones_like(time))
    basis = dynamics.solve_dense_modes(np.diag([4.0, 9.0]), np.eye(2))
    with pytest.raises(ValueError, match="singular"):
        dynamics.modal_frequency_response(
            basis,
            [basis.frequencies[0]],
            [1.0, 0.0],
            damping_ratio=0.0,
        )


def test_modal_step_uses_public_model_language_and_removes_fixed_dofs(tmp_path):
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
    step = model.step(target=displacement, modes=3)
    result = step.solve_result(
        output=tmp_path / "modes.xdmf",
        strict_output=True,
    )

    assert capability["provider"]["name"] == "linear_structural_modes"
    assert result.quantity("frequencies").shape == (3,)
    assert np.all(np.diff(result.quantity("frequencies")) > 0.0)
    assert np.max(result.quantity("residual_norms")) < 1.0e-7
    assert tuple(result.fields) == ("Mode_1", "Mode_2", "Mode_3")
    assert result.artifacts["fields_xdmf"].exists()
    assert result.metadata["field_output_fields"]["included"] == (
        "Mode_1",
        "Mode_2",
        "Mode_3",
    )
    assert result.metadata["field_output"]["warp_field"] == "Mode_1"
    assert result.metadata["field_output"]["warp_field_semantic"] == "Mode shape"
    assert result.metadata["field_output"]["field_aliases"] == {"Mode shape": "Mode_1"}
    with h5py.File(result.artifacts["fields_hdf5"], "r") as h5:
        assert h5.attrs["primary_field"] == "Mode_1"
        assert h5.attrs["primary_semantic_name"] == "Mode shape"
        assert "Mode_1" in h5["Frames/0000/Point"]
    solve = result.metadata["solve"]
    assert solve["mass_orthogonality_error"] < 1.0e-10
    assert solve["stiffness_diagonalization_error"] < 1.0e-10
    assert solve["orientation_convention"] == "largest_global_component_positive"
    assert solve["stiffness_symmetric"]
    assert solve["mass_symmetric"]
    assert solve["operator_symmetry_relative_tolerance"] > 0.0
    assert solve["stiffness_symmetry_absolute_tolerance"] > 0.0
    assert solve["mass_symmetry_absolute_tolerance"] > 0.0
    assert len(solve["orientation_anchor_dofs"]) == 3
    assert solve["selected_clusters_complete"]
    assert solve["repeated_modes_compare_as"] == "invariant_subspace"
    assert sum(
        cluster["multiplicity"] for cluster in solve["eigenvalue_clusters"]
    ) == 3
    for mode in result.fields.values():
        assert operators.quadratic_form(step.mass, mode.field) == pytest.approx(
            1.0,
            rel=1.0e-10,
            abs=1.0e-12,
        )
        assert mode.unit is None
        assert mode.processing["comparison_object"] in {
            "individual_mode",
            "invariant_subspace",
        }
        assert mode.processing["eigenvalue_cluster"] >= 1
    for left_index, left in enumerate(result.fields.values()):
        for right_index, right in enumerate(result.fields.values()):
            expected = 1.0 if left_index == right_index else 0.0
            assert operators.bilinear_form(
                step.mass,
                left.field,
                right.field,
            ) == pytest.approx(expected, rel=1.0e-10, abs=1.0e-12)
    for mode, anchor in zip(
        result.fields.values(),
        solve["orientation_anchor_dofs"],
    ):
        assert mode.field.x.array[int(anchor)] > 0.0
    euler_bernoulli = (
        1.875104068711961**2
        / (2.0 * np.pi)
        * np.sqrt(210.0e9 * 0.2**2 / (12.0 * 7800.0))
    )
    assert result.quantity("frequencies")[0] == pytest.approx(
        euler_bernoulli,
        rel=0.035,
    )


def test_slender_cantilever_frequency_has_analytical_and_mesh_convergence_evidence():
    frequencies = []
    for longitudinal_cells in (8, 16, 32):
        domain = mesh.rectangle(
            (0.0, 0.0),
            (1.0, 0.05),
            (longitudinal_cells, max(1, longitudinal_cells // 8)),
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
        result = model.step(target=displacement, modes=1).solve_result()
        frequencies.append(float(result.quantity("frequencies")[0]))

    euler_bernoulli = (
        1.875104068711961**2
        / (2.0 * np.pi)
        * np.sqrt(210.0e9 * 0.05**2 / (12.0 * 7800.0))
    )
    convergence = verification.ConvergenceStudy(
        name="slender_cantilever_modal_mesh_convergence",
        observable="first_natural_frequency",
        samples=tuple(
            verification.ConvergenceSample(
                1.0 / longitudinal_cells,
                frequency,
                label=f"q2_{longitudinal_cells}x{max(1, longitudinal_cells // 8)}",
            )
            for longitudinal_cells, frequency in zip((8, 16, 32), frequencies)
        ),
    )
    convergence_claim = convergence.verify(
        maximum_relative_change=7.0e-4,
        minimum_observed_order=1.5,
    )
    analytical_claim = verification.VerificationClaim.compare(
        name="slender_cantilever_euler_bernoulli",
        observable="first_natural_frequency",
        actual=frequencies[-1],
        expected=euler_bernoulli,
        reference="Euler--Bernoulli clamped-free bending frequency",
        relative_tolerance=2.0e-3,
        validity_domain="slender homogeneous plane-stress cantilever",
    )

    np.testing.assert_allclose(
        frequencies,
        [41.9907373893, 41.8934372526, 41.8649074705],
        rtol=1.0e-7,
    )
    assert convergence_claim.status == "passed"
    assert convergence.observed_order > 1.5
    assert analytical_claim.status == "passed"


def test_modal_target_frequency_selects_nearest_mode_not_lowest_mode():
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

    low_modes = model.step(target=displacement, modes=5).solve_result()
    reference = low_modes.quantity("frequencies")[2]
    targeted = model.step(
        target=displacement,
        modes=1,
        target_frequency=1.01 * reference,
    ).solve_result()

    assert targeted.quantity("frequencies")[0] == pytest.approx(reference, rel=1.0e-8)


def test_modal_step_rejects_an_unsymmetric_custom_operator():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.modal_solid(dimension=2, assumption="plane_stress"),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain, degree=1))
    model.clamp(
        displacement,
        on=mesh.boundary(domain, _left, name="left", tag=1),
    )
    trial = displacement.trial
    test = displacement.test
    unsymmetric = operators.OperatorForm(
        name="K_unsymmetric",
        kind="test_unsymmetric_stiffness",
        role="matrix",
        family="test",
        expression=(
            trial[0] * test[0]
            + trial[1] * test[1]
            + trial[0] * test[1]
        )
        * ufl.dx,
    )
    mass = operators.mass_operator(displacement, density=1.0)

    step = model.step(
        target=displacement,
        modes=1,
        K=unsymmetric,
        M=mass,
    )
    with pytest.raises(ValueError, match="stiffness operator must be symmetric"):
        step.solve()


@pytest.mark.parametrize(
    ("keyword", "value"),
    (("modes", 1.5), ("modes", True), ("maximum_iterations", 3.5)),
)
def test_modal_step_rejects_values_that_would_be_silently_truncated(
    keyword,
    value,
):
    options = {
        "target": object(),
        "mass": object(),
        "stiffness": object(),
        "modes": 1,
        keyword: value,
    }
    with pytest.raises(ValueError, match="positive integer"):
        problems.modal_analysis(**options)
