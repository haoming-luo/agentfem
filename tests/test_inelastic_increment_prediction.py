# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Accepted material evidence caps future proposals, not acceptance criteria."""

from dataclasses import dataclass

import numpy as np
import pytest

from agentfem import steps
from agentfem.step_providers import StepOptionContract


def _control(**options):
    return steps.automatic(maximum_inelastic_increment=.01,
                           inelastic_prediction_safety=.8, **options)


@pytest.mark.parametrize("value", [True, np.bool_(True), "0.8", np.nan, np.inf, 0., 1., -1.])
def test_prediction_safety_is_an_explicit_finite_fraction(value):
    with pytest.raises(ValueError, match="inelastic_prediction_safety"):
        steps.automatic(maximum_inelastic_increment=.01, inelastic_prediction_safety=value)


def test_prediction_requires_a_declared_acceptance_limit():
    with pytest.raises(ValueError, match="requires maximum_inelastic_increment"):
        steps.automatic(inelastic_prediction_safety=.8)


@pytest.mark.parametrize("measure", [None, True, "0.001", -1., np.nan, np.inf, .01001])
def test_prediction_cannot_accept_missing_invalid_or_rejected_evidence(measure):
    with pytest.raises(ValueError, match="inelastic_increment"):
        _control().after_convergence(.1, 3, inelastic_increment=measure)


@pytest.mark.parametrize("iterations", [2, 6, 15])
def test_prediction_only_caps_the_newton_proposal(iterations):
    ordinary = steps.automatic(maximum_inelastic_increment=.01)
    control = _control()
    proposal = ordinary.after_convergence(.1, iterations)
    for measure in (0., 1e-320, .001, .005, .01):
        actual = control.after_convergence(.1, iterations, inelastic_increment=measure)
        expected = proposal if measure < 1e-300 else max(control.minimum, min(proposal, .1 * .8 * .01 / measure))
        assert actual == pytest.approx(expected)
        assert actual <= proposal


def test_prediction_minimum_and_failure_cutback_have_distinct_meanings():
    control = _control(initial=.1, minimum=.1, maximum=.25)
    assert control.after_convergence(.1, 3, inelastic_increment=.01) == .1
    # A later rejected attempt still requests a cutback below the minimum;
    # the Procedure must report exhaustion, not force acceptance at minimum.
    assert control.after_failure(.1) < control.minimum


def test_prediction_identity_is_opt_in_and_support_is_explicit():
    control = _control()
    assert "inelastic_prediction" not in steps.automatic().summary()
    assert control.summary()["inelastic_prediction"] == {
        "algorithm": "accepted_increment_proportional_cap_v1", "safety": .8,
    }
    with pytest.raises(NotImplementedError, match="AFM-INCREMENT-FEEDBACK"):
        steps.normalize(control)
    assert steps.normalize(control, inelastic_feedback=True) is control
    unsupported = StepOptionContract(accepted=("incrementation",))
    assert unsupported.issues({"incrementation": control})[0]["code"] == "AFM-INCREMENT-FEEDBACK"
    supported = StepOptionContract(accepted=("incrementation",), inelastic_increment_feedback=True)
    assert not supported.issues({"incrementation": control})
    assert supported.summary()["inelastic_increment_feedback"] is True


def test_existing_custom_two_argument_policy_is_still_used():
    @dataclass(frozen=True)
    class Custom(steps.AutomaticIncrementation):
        def after_convergence(self, size, iterations):
            return .123

    assert steps._after_convergence(Custom(), .1, 3, inelastic_increment=.1) == .123


def test_affine_prediction_requires_an_explicit_measure_before_setup():
    from agentfem import problems
    with pytest.raises(NotImplementedError, match="AFM-INCREMENT-FEEDBACK"):
        problems.affine_nonlinear(None, None, jacobian=None, constraint=None,
                                 incrementation=_control())


def test_affine_prediction_rejects_unvalidated_mpi_before_assembly():
    from types import SimpleNamespace
    from agentfem import problems
    solution = SimpleNamespace(function_space=SimpleNamespace(mesh=SimpleNamespace(comm=SimpleNamespace(size=2))))
    with pytest.raises(NotImplementedError, match="AFM-INCREMENT-FEEDBACK"):
        problems.affine_nonlinear(None, solution, jacobian=None, constraint=None,
                                 incrementation=_control(), acceptance_check=lambda: {},
                                 inelastic_increment_measure="increment")


def test_public_prediction_capability_agrees_with_execution():
    from test_finite_strain_j2_standard import _standard_patch
    from agentfem.step_providers import step_capability, step_providers
    model, target, material, _ = _standard_patch(incrementation=steps.fixed(1))
    selected = step_capability(model, target=target, options={"material": material, "incrementation": _control()})
    assert selected["supported"], selected
    supported = {provider.name for provider in step_providers()
                 if provider.option_contract is not None and provider.option_contract.inelastic_increment_feedback}
    assert supported == {"finite_strain_j2_strong_static", "finite_strain_j2_affine_static"}


def test_single_regional_assignment_does_not_advertise_prediction_support():
    from dolfinx import mesh as dm
    from mpi4py import MPI
    from agentfem import constitutive, fields, mesh, models, studies
    from agentfem.step_providers import step_capability
    domain = dm.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain)
    target = model.field(fields.displacement(domain))
    region = mesh.partition_cells(domain, all_cells=lambda x: np.ones(x.shape[1], dtype=bool)).all_cells
    model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000., poisson=.3, yield_stress=2., hardening_modulus=100.), region=region)
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.), name="left")
    model.clamp(target, on=left)
    assert step_capability(model, target=target, options={"incrementation": steps.fixed(1)})["supported"]
    capability = step_capability(model, target=target, options={"incrementation": _control()})
    assert not capability["supported"], capability
    with pytest.raises((ValueError, NotImplementedError), match="AFM-INCREMENT-FEEDBACK"):
        model.step(target=target, incrementation=_control(), progress=False)


def test_distributed_prediction_is_rejected_before_material_allocation():
    from dolfinx import mesh as dm
    from mpi4py import MPI
    from agentfem import constitutive, fields, mechanics, mesh, models, studies
    from agentfem.step_providers import step_capability
    if MPI.COMM_WORLD.size == 1:
        pytest.skip("Run with agentfem mpi-run -n 2 to check distributed admission.")
    domain = dm.create_unit_cube(MPI.COMM_WORLD, 2, 1, 1)
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain)
    target = model.field(fields.displacement(domain))
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000., poisson=.3, yield_stress=2., hardening_modulus=100.))
    model.clamp(target, on=mesh.boundary(domain, lambda x: np.isclose(x[0], 0.), name="left"))
    assert not step_capability(model, target=target, options={"incrementation": _control()})["supported"]
    with pytest.raises((ValueError, NotImplementedError), match="AFM-INCREMENT-FEEDBACK"):
        model.step(target=target, incrementation=_control(), progress=False)
    with pytest.raises(NotImplementedError, match="AFM-INCREMENT-FEEDBACK"):
        mechanics.finite_strain_j2_standard_problem(displacement=target, material=material,
                                                    incrementation=_control())


def test_prediction_does_not_accept_an_oversized_increment_at_minimum():
    from test_finite_strain_j2_standard import _standard_patch
    control = steps.automatic(initial=.25, minimum=.25, maximum=.25,
                              maximum_inelastic_increment=1e-10, inelastic_prediction_safety=.8)
    step = _standard_patch(incrementation=control)[-1]
    initial = step.response.state.committed_state_vectors().copy()
    with pytest.raises(RuntimeError, match="minimum|cutback"):
        step.solve()
    assert step.accepted_load_factor == 0.
    assert not step.accepted_increments
    assert not step.energy_history
    np.testing.assert_array_equal(step.response.state.committed_state_vectors(), initial)


@pytest.mark.parametrize("measure", [None, -.001, np.nan])
def test_invalid_affine_measure_cannot_publish_a_converged_attempt(monkeypatch, measure):
    from test_affine_j2_automatic_restart import _automatic_periodic_j2_step
    step = _automatic_periodic_j2_step(incrementation=_control())
    step.solve(until=.5)
    displacement = step.solution.x.array.copy()
    state = step.response.state.committed_state_vectors().copy()
    accepted = list(step.accepted_increments)
    next_size = step.next_increment_size
    duals = step.constraint_dual_history.snapshot_runtime_state()
    original = step.acceptance_check
    def malformed():
        result = original()
        assert result["accepted"]
        if measure is None:
            result.pop("maximum_plastic_increment")
        else:
            result["maximum_plastic_increment"] = measure
        return result
    monkeypatch.setattr(step, "acceptance_check", malformed)
    with pytest.raises(ValueError, match="inelastic_increment"):
        step.solve()
    np.testing.assert_allclose(step.solution.x.array, displacement, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(step.response.state.committed_state_vectors(), state, rtol=2e-8, atol=2e-10)
    assert step.accepted_load_factor == .5
    assert step.next_increment_size == next_size
    assert step.accepted_increments == accepted
    assert step.constraint_dual_history.snapshot_runtime_state() == duals


def test_prediction_checkpoint_binds_the_measure_name(tmp_path):
    from test_affine_j2_automatic_restart import _automatic_periodic_j2_step
    partial = _automatic_periodic_j2_step(incrementation=_control())
    partial.solve(until=.5)
    checkpoint = partial.save_checkpoint(tmp_path / "named_measure")
    changed = _automatic_periodic_j2_step(incrementation=_control())
    changed.inelastic_increment_measure = "different_scientific_measure"
    untouched = changed.solution.x.array.copy()
    with pytest.raises(ValueError, match="identity"):
        changed.load_checkpoint(checkpoint)
    assert changed.accepted_load_factor == 0.
    np.testing.assert_array_equal(changed.solution.x.array, untouched)


@pytest.mark.parametrize("kind", ["standard", "affine", "augmented_moving_cyclic"])
@pytest.mark.parametrize("portable", [False, True])
def test_public_prediction_restart_and_proposal_failure_are_atomic(tmp_path, monkeypatch, kind, portable):
    from test_affine_j2_automatic_restart import _automatic_periodic_j2_step
    from test_finite_strain_j2_standard import _standard_patch

    def make(safety=.8):
        control = steps.automatic(initial=.25, minimum=1e-4, maximum=.5,
                                  max_increments=100, max_cutbacks=12, cutback_factor=.5,
                                  maximum_inelastic_increment=.0045,
                                  inelastic_prediction_safety=safety)
        if kind == "augmented_moving_cyclic":
            from agentfem import amplitudes
            from test_finite_j2_normal_augmentation import _problem
            return _problem(increment_control=control, tool_translation=(-.002, 0., 0.),
                            load_amplitude=amplitudes.tabular((0., .25, .75, 1.), (0., 1., .2, 1.)))[1]
        return (_automatic_periodic_j2_step(incrementation=control) if kind == "affine"
                else _standard_patch(incrementation=control)[-1])

    def arrays(step):
        return (step.solution.x.array.copy(), step.response.state.committed_state_vectors().copy(),
                step.response.first_piola_stress.values.copy(),
                step.response.strain_energy_density.values.copy())

    def same(actual, expected):
        for value, reference in zip(actual, expected, strict=True):
            assert np.isfinite(value).all() and np.isfinite(reference).all()
            np.testing.assert_allclose(value, reference, rtol=2e-8, atol=2e-8)

    reference = make()
    assert reference.checkpoint_capabilities().rank_count_portability == "unsupported"
    reference.solve(until=.5)
    boundary = arrays(reference)
    if portable and kind == "augmented_moving_cyclic":
        # Prediction does not widen the existing contact restart boundary.
        with pytest.raises(NotImplementedError, match="Portable restart"):
            reference.save_checkpoint(tmp_path / kind, portable=True)
        same(arrays(reference), boundary)
        assert not (tmp_path / kind).exists()
        return
    checkpoint = reference.save_checkpoint(tmp_path / kind, portable=portable)
    next_size = reference.next_increment_size
    reference.solve()
    restarted = make()
    restarted.load_checkpoint(checkpoint)
    same(arrays(restarted), boundary)
    assert restarted.next_increment_size == next_size
    assert restarted.accepted_load_factor == .5
    previous_count = len(restarted.accepted_increments)
    previous_energy = list(getattr(restarted, "energy_history", ()))
    if kind == "augmented_moving_cyclic":
        previous_contact = restarted.contact_provider.snapshot_accepted_state()
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise ValueError("injected accepted prediction failure")
        patch.setattr(steps.AutomaticIncrementation, "after_convergence", fail)
        with pytest.raises((ValueError, RuntimeError), match="injected accepted prediction failure"):
            restarted.solve()
    same(arrays(restarted), boundary)
    assert restarted.accepted_load_factor == .5
    assert restarted.next_increment_size == next_size
    assert len(restarted.accepted_increments) == previous_count
    assert list(getattr(restarted, "energy_history", ())) == previous_energy
    if kind == "augmented_moving_cyclic":
        assert restarted.contact_provider.snapshot_accepted_state() == previous_contact
    restarted.solve()
    same(arrays(restarted), arrays(reference))
    np.testing.assert_allclose(restarted.response.tangent.values, reference.response.tangent.values,
                               rtol=2e-8, atol=2e-8)
    assert restarted.next_increment_size == pytest.approx(reference.next_increment_size)
    if kind != "affine":
        from dataclasses import asdict
        def same_record(actual, expected):
            if isinstance(actual, dict):
                assert actual.keys() == expected.keys()
                for key in actual:
                    same_record(actual[key], expected[key])
            elif isinstance(actual, (tuple, list)):
                for a, e in zip(actual, expected, strict=True):
                    same_record(a, e)
            elif isinstance(actual, (int, float)):
                assert actual == pytest.approx(expected, rel=2e-8, abs=2e-10)
            else:
                assert actual == expected
        for actual, expected in zip(restarted.energy_history, reference.energy_history, strict=True):
            same_record(asdict(actual), asdict(expected))
    for name in ("accepted_increments", "attempted_increments"):
        actual = [(x.load_factor, x.iterations, x.converged) for x in getattr(restarted, name)]
        expected = [(x.load_factor, x.iterations, x.converged) for x in getattr(reference, name)]
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)
    for item in restarted.accepted_increments:
        measure = (item.checks["maximum_plastic_increment"] if kind == "affine"
                   else item.maximum_plastic_increment)
        assert measure <= .0045
    if kind == "augmented_moving_cyclic":
        assert all(any(abs(item.load_factor - knot) < 1e-12 for item in restarted.accepted_increments)
                   for knot in (.25, .75))
        assert restarted.energy_history_complete
        assert abs(restarted.energy_history[-1].tool_work) > 0.
        assert restarted.energy_history[-1].pair_tool_work
    wrong_policy = make(None)
    untouched = arrays(wrong_policy)
    with pytest.raises((ValueError, RuntimeError), match="identity"):
        wrong_policy.load_checkpoint(checkpoint)
    same(arrays(wrong_policy), untouched)
    assert wrong_policy.accepted_load_factor == 0.
