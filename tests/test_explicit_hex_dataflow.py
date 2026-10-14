"""Prepared response reuse must retain fields, detached gathers and rollback."""

import numpy as np
import pytest
from dolfinx import fem
from agentfem.constitutive._explicit_material_driver import ExplicitQuadratureResponse
from agentfem.constitutive.quadrature import QuadratureField
from agentfem.elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual
from test_explicit_material_quadrature import fixture


def test_prepared_piola_is_assembled_without_field_readback(monkeypatch):
    material, state, options = fixture()
    response = ExplicitQuadratureResponse.create(state.domain, material.state_schema)
    u = fem.Function(fem.functionspace(state.domain, ("Lagrange", 1, (3,))))
    residual = FiniteUniformHexResidual(
        u, response, density=1, hourglass_modulus=1, hourglass_scale=0.1
    )
    u.interpolate(lambda x: 0.01 * x)
    getter = QuadratureField.owned_values.fget

    def checked(field):
        if field is response.first_piola_stress:
            raise AssertionError(
                "Prepared stress must not be copied back from its field"
            )
        return getter(field)

    monkeypatch.setattr(QuadratureField, "owned_values", property(checked))
    options.pop("deformation_gradient_new")
    with residual.explicit_trial_evaluation(material, **options) as (vector, trial):
        expected = np.linalg.det(trial.deformation_gradient)[:, None, None] * (
            trial.material_response.cauchy_stress
            @ np.linalg.inv(trial.deformation_gradient).swapaxes(1, 2)
        )
        np.testing.assert_allclose(response.first_piola_stress.values, expected)
        assert np.linalg.norm(vector.array) > 0
    vector.destroy()
    response.commit()
    np.testing.assert_allclose(response.first_piola_stress.values, expected)


def test_gather_is_detached_and_uses_cell_order():
    owner = object.__new__(FiniteUniformHexResidual)
    owner.cell_nodes = np.array([[3, 2, 1, 0, 7, 6, 5, 4]], dtype=np.int32)
    values = np.arange(24, dtype=float).reshape(8, 3)
    actual = owner._gather_cell_vectors(values)
    expected = values[owner.cell_nodes].copy()
    values[:] = 0
    np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(actual, values)


def test_gather_failure_preserves_trial_fields(monkeypatch):
    material, state, options = fixture()
    response = ExplicitQuadratureResponse.create(state.domain, material.state_schema)
    u = fem.Function(fem.functionspace(state.domain, ("Lagrange", 1, (3,))))
    residual = FiniteUniformHexResidual(
        u, response, density=1, hourglass_modulus=1, hourglass_scale=0.1
    )
    saved = {
        key: field.x.array.copy() for key, field in response.trial_fields().items()
    }

    def fail(_):
        response.cauchy_stress.function.x.array[:] = 5
        raise RuntimeError("injected gather failure")

    monkeypatch.setattr(residual, "_gather_cell_vectors", fail)
    options.pop("deformation_gradient_new")
    with pytest.raises(RuntimeError, match="gather failure"):
        with residual.explicit_trial_evaluation(material, **options):
            pytest.fail("Failed gather must not yield")
    for key, field in response.trial_fields().items():
        np.testing.assert_array_equal(field.x.array, saved[key])


def test_prepared_force_failure_does_not_publish_state(monkeypatch):
    material, state, options = fixture()
    response = ExplicitQuadratureResponse.create(state.domain, material.state_schema)
    u = fem.Function(fem.functionspace(state.domain, ("Lagrange", 1, (3,))))
    residual = FiniteUniformHexResidual(
        u, response, density=1, hourglass_modulus=1, hourglass_scale=0.1
    )
    saved = {
        key: field.x.array.copy() for key, field in response.trial_fields().items()
    }

    def fail(*args, **kwargs):
        for key, field in response.trial_fields().items():
            np.testing.assert_array_equal(field.x.array, saved[key])
        raise RuntimeError("detached force preparation failed")

    monkeypatch.setattr(residual.cells, "_response_from_checked_displacement", fail)
    options.pop("deformation_gradient_new")
    with pytest.raises(RuntimeError, match="detached force preparation"):
        with residual.explicit_trial_evaluation(material, **options):
            pytest.fail("Incomplete response must not be published")
    for key, field in response.trial_fields().items():
        np.testing.assert_array_equal(field.x.array, saved[key])
