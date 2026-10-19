"""The call-local reuse must not change the discrete law or old-state identity."""

from dataclasses import replace

import numpy as np
import pytest

from agentfem import constitutive
from agentfem.constitutive.material_array_batch import MaterialPointArrayBatchInput


def _request(material, gradients, states):
    return MaterialPointArrayBatchInput(
        deformation_gradient_old=np.tile(np.eye(3), (len(gradients), 1, 1)),
        deformation_gradient_new=gradients,
        state_old=states,
        state_schema=material.state_schema,
        time=0.0, time_increment=0.1,
    )


@pytest.mark.parametrize("hardening", [0.0, 1000.0])
def test_trial_spectrum_matches_recomputed_and_numerical_derivative(hardening):
    material = constitutive.finite_strain_j2_logarithmic(
        young=210_000, poisson=0.3, yield_stress=250,
        hardening_modulus=hardening,
    )
    # Repeated / nearly repeated eigenvalues, shear, elastic and plastic return.
    gradients = np.array([
        np.eye(3), np.diag([1.0001, 1., 1.]), np.diag([1.08, 1., 1.]),
        [[1.04, .03, .01], [0., .99, .02], [0., 0., .98]],
        np.diag([1.02, 1.02 + 1e-12, 1.02]),
    ])
    states = np.tile(material.state_schema.initial_state(), (len(gradients), 1))
    history = material._integrate_batch(gradients, states)["state"]
    for old in (states, history):
        before = old.copy()
        baseline = material._integrate_batch(gradients, old)
        reused = material._analytic_algorithmic_tangent_batch(
            gradients, old, baseline=baseline,
        )
        recomputed = material._analytic_algorithmic_tangent_batch(
            gradients, old, baseline={"first_piola_stress": baseline["first_piola_stress"]},
        )
        np.testing.assert_array_equal(reused, recomputed)
        np.testing.assert_array_equal(old, before)
    # Use a fresh nonproportional trial strictly away from a yield switch.
    next_gradients = gradients.copy()
    next_gradients[:, 0, 1] += .025
    request = _request(material, next_gradients, history)
    analytic = material.update_array_batch(request)
    numerical = replace(material, tangent_evaluation="central_difference").update_array_batch(request)
    np.testing.assert_allclose(analytic.consistent_tangent, numerical.consistent_tangent,
                               rtol=4e-6, atol=4e-4)
    np.testing.assert_array_equal(analytic.cauchy_stress, numerical.cauchy_stress)
    np.testing.assert_array_equal(analytic.state_new, numerical.state_new)


def test_array_update_uses_one_svd_per_call_and_never_reuses_another_trial(monkeypatch):
    material = constitutive.finite_strain_j2_logarithmic(
        young=210_000, poisson=.3, yield_stress=250, hardening_modulus=1000,
    )
    gradients = np.tile(np.diag([1.03, 1., 1.]), (8, 1, 1))
    states = np.tile(material.state_schema.initial_state(), (8, 1))
    request = _request(material, gradients, states)
    svd = np.linalg.svd
    calls = []

    def counted(value, *args, **kwargs):
        calls.append(value.copy())
        return svd(value, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", counted)
    first = material.update_array_batch(request)
    changed = gradients.copy()
    changed[:, 0, 1] = .04
    second = material.update_array_batch(_request(material, changed, first.state_new))
    retried = material.update_array_batch(request)
    assert len(calls) == 3
    assert all(value.shape == (8, 3, 3) for value in calls)
    assert not np.array_equal(first.cauchy_stress, second.cauchy_stress)
    np.testing.assert_array_equal(first.consistent_tangent, retried.consistent_tangent)
    np.testing.assert_array_equal(first.state_new, retried.state_new)
    np.testing.assert_array_equal(request.state_old, states)
