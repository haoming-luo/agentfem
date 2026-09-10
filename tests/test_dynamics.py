from __future__ import annotations

import h5py
from dolfinx import fem
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from agentfem import (
    constraints,
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


def _scaled_modal_step(*, stiffness_scale: float = 1.0):
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
    live_scale = fem.Constant(domain, PETSc.ScalarType(stiffness_scale))
    stiffness = operators.scale(model.stiffness(displacement), live_scale)
    return model.step(target=displacement, modes=1, K=stiffness), live_scale


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
    manifest = result.scientific_input_manifest()
    assert manifest["complete"] is True
    identity = result.scientific_inputs["modal_executable_identity"]
    assert identity["complete"] is True
    assert identity["record"]["mesh"]["global_cells"] == 16
    assert identity["record"]["homogeneous_dirichlet"]["global_scalar_dofs"] > 0
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


def test_modal_problem_facade_delegates_to_the_mechanics_owner():
    step = problems.modal_analysis(
        target=object(),
        mass=object(),
        stiffness=object(),
        modes=1,
    )

    assert type(step) is problems.ModalAnalysisStep
    assert type(step).__module__ == "agentfem.mechanics.modal"


def test_modal_constraint_failure_names_the_actual_procedure():
    step = problems.modal_analysis(
        target=object(),
        mass=object(),
        stiffness=object(),
        modes=1,
        constraints=(object(),),
    )

    with pytest.raises(
        TypeError,
        match="modal analysis received.*not a strong Dirichlet constraint",
    ):
        step.solve()


def test_modal_step_rejects_nonzero_strong_support_before_eigensolve():
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
    model.material(
        elasticity.isotropic_elastic(
            young=210.0e9,
            poisson=0.3,
            density=7800.0,
        )
    )
    model.clamp(
        displacement,
        on=mesh.boundary(domain, _left, name="moving_support", tag=1),
        value=(0.1, 0.2),
    )

    step = model.step(target=displacement, modes=1)
    with pytest.raises(NotImplementedError, match="AFM-MODAL-BC-003"):
        step.solve()
    assert step.last_solve_info is None


@pytest.mark.parametrize(
    ("asset", "code"),
    (
        (
            constraints.TimeDependentDirichlet(
                constant=object(),
                bc=object(),
                amplitude=object(),
            ),
            "AFM-MODAL-BC-001",
        ),
        (
            constraints.RemoteDisplacementConstraint(
                bc=object(),
                value=object(),
                reference_values=np.zeros(2),
                reference_point=object(),
                translation=(0.0, 0.0),
                rotation=0.0,
            ),
            "AFM-MODAL-BC-002",
        ),
    ),
)
def test_modal_step_rejects_history_and_remote_support_semantics(asset, code):
    step = problems.modal_analysis(
        target=object(),
        mass=object(),
        stiffness=object(),
        modes=1,
        constraints=(asset,),
    )

    with pytest.raises(NotImplementedError, match=code):
        step.solve()


def test_modal_rejects_an_affine_reduction_instead_of_dropping_its_mpc():
    reduction = constraints.DistributedAffineReduction(
        mpc=object(),
        bcs=(),
        original_space=object(),
        full_size=2,
        reduced_size=1,
        slave_count=1,
        control_dof_count=0,
    )
    step = problems.modal_analysis(
        target=object(),
        mass=object(),
        stiffness=object(),
        modes=1,
        constraints=(reduction,),
    )

    with pytest.raises(TypeError, match="not a strong Dirichlet constraint"):
        step.solve()


@pytest.mark.parametrize("residual", (float("inf"), 1.0e-3))
def test_modal_solve_info_requires_finite_accepted_eigenpair_residuals(residual):
    info = dynamics.ModalSolveInfo(
        converged_eigenpairs=1,
        requested_modes=1,
        accepted_modes=1,
        constrained_dofs=1,
        free_dofs=2,
        residual_norms=(residual,),
        eigensolver="synthetic",
        mass_orthogonality_error=0.0,
        stiffness_diagonalization_error=0.0,
        residual_tolerance=1.0e-7,
        stiffness_symmetric=True,
        mass_symmetric=True,
    )

    assert info.converged is False
    assert info.as_dict()["residual_tolerance"] == pytest.approx(1.0e-7)


def test_modal_result_fingerprint_tracks_live_executable_coefficients():
    first_step, _first_scale = _scaled_modal_step(stiffness_scale=1.0)
    second_step, _second_scale = _scaled_modal_step(stiffness_scale=1.1)

    first = first_step.solve_result()
    second = second_step.solve_result()

    assert (
        first.scientific_input_manifest()["fingerprint"]
        != second.scientific_input_manifest()["fingerprint"]
    )
    first_identity = first.scientific_inputs["modal_executable_identity"]
    second_identity = second.scientific_inputs["modal_executable_identity"]
    assert first_identity["fingerprint"] != second_identity["fingerprint"]
    assert (
        first_identity["record"]["operators"]["stiffness"]["constants"]
        != second_identity["record"]["operators"]["stiffness"]["constants"]
    )


def test_failed_modal_resolve_retains_the_last_accepted_result_atomically():
    step, _scale = _scaled_modal_step()
    step.solve()
    accepted_values = step.eigenvalues.copy()
    accepted_modes = tuple(mode.x.array.copy() for mode in step.mode_shapes)
    accepted_info = step.last_solve_info
    accepted_identity = step._executed_identity
    accepted_request = step._executed_request_manifest

    step.modes = 10_000
    with pytest.raises(ValueError, match="free dofs.*requests"):
        step.solve()

    np.testing.assert_array_equal(step.eigenvalues, accepted_values)
    for mode, values in zip(step.mode_shapes, accepted_modes, strict=True):
        np.testing.assert_array_equal(mode.x.array, values)
    assert step.last_solve_info is accepted_info
    assert step._executed_identity == accepted_identity
    assert step._executed_request_manifest == accepted_request
    with pytest.raises(RuntimeError, match="mode count.*changed after solve"):
        step.scientific_inputs()


def test_modal_result_publication_fails_closed_without_executable_identity(
    monkeypatch,
):
    from agentfem.operators import identity as operator_identity
    from agentfem.results import _modal as modal_results

    step, _scale = _scaled_modal_step()
    step.solve()
    monkeypatch.setattr(
        operator_identity,
        "modal_executable_identity",
        lambda **_kwargs: {
            "complete": False,
            "missing": ({"path": "modal_system.stiffness"},),
            "record": {},
            "fingerprint": "unavailable",
        },
    )

    with pytest.raises(ValueError, match="cannot establish.*executable identity"):
        modal_results.from_modal_step(step)


def test_three_dimensional_modal_cantilever_matches_beam_limit():
    length = 1.0
    width = 0.1
    height = 0.05
    young = 210.0e9
    density = 7800.0
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (length, width, height),
        (8, 1, 1),
        comm=MPI.COMM_SELF,
        cell_type="hexahedron",
    )
    model = models.create(
        study=studies.modal_solid(dimension=3),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain, degree=2))
    model.material(
        elasticity.isotropic_elastic(
            young=young,
            poisson=0.3,
            density=density,
        )
    )
    model.clamp(
        displacement,
        on=mesh.boundary(domain, _left, name="fixed_end", tag=1),
    )

    result = model.step(target=displacement, modes=2).solve_result()
    frequencies = np.asarray(result.quantity("frequencies"))
    beta_1 = 1.875104068711961
    beam_limit = (
        beta_1**2
        / (2.0 * np.pi)
        * np.sqrt(young * height**2 / (12.0 * density * length**4))
    )

    assert frequencies[0] == pytest.approx(42.31395028, rel=1.0e-7)
    assert frequencies[0] == pytest.approx(beam_limit, rel=1.5e-2)
    assert frequencies[1] > frequencies[0]
    assert result.metadata["solve"]["converged"] is True
    assert max(result.quantity("residual_norms")) < 1.0e-7
    executable = result.scientific_inputs["modal_executable_identity"]
    assert executable["record"]["mesh"]["geometry_dimension"] == 3
