# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Built-in arithmetic is independent of material state width and cell shape."""
import numpy as np
import pytest

from agentfem import constitutive, elements
from agentfem.constitutive.material_array_batch import MaterialPointArrayBatchInput
from agentfem.constitutive.user_material import MaterialStateSchema, MaterialStateVariable
from test_response_material_step import contact_step


@pytest.mark.parametrize('width', [1, 3, 29])
def test_builtin_body_matches_independent_work_conjugate_reference(width):
    pytest.importorskip('agentfem._hex_native')
    rng = np.random.default_rng(929)
    n = 73
    schema = MaterialStateSchema('generic', (MaterialStateVariable('history', shape=(width,)),))
    f = np.eye(3) + rng.normal(scale=.07, size=(n, 3, 3))
    request = MaterialPointArrayBatchInput(np.tile(np.eye(3), (n, 1, 1)), f,
        rng.normal(size=(n, width)), schema, .1, .001)
    stress = rng.normal(size=(n, 3, 3))
    stress = (stress + stress.swapaxes(1, 2)) / 2
    def update(r):
        return constitutive.ExplicitMaterialBatchOutput(stress, r.state_old + .01,
            schema, np.arange(n, dtype=float), requested_time_scale=np.full(n, .5))
    material = constitutive.explicit_user_material(state_schema=schema, update=update,
        identity={'law': 'independent-test', 'width': width}, density=1.)
    geometry = elements.ExplicitHexGeometry(rng.normal(size=(n, 8, 3)),
        rng.normal(size=(n, 8, 4)), rng.uniform(.1, 3, n), rng.uniform(0, 2, n))
    u = rng.normal(size=(n, 8, 3))
    p = np.linalg.det(f)[:, None, None] * (stress @ np.linalg.inv(f).swapaxes(1, 2))
    modes = geometry.hourglass_modes.swapaxes(1, 2) @ u
    force = geometry.volume[:, None, None] * (geometry.average_gradient @ p.swapaxes(1, 2))
    force += geometry.hourglass_coefficient[:, None, None] * (geometry.hourglass_modes @ modes)
    for backend in ('numpy', 'native'):
        actual = elements.explicit_hex_kernel(backend=backend).evaluate(material, request, geometry, u)
        np.testing.assert_allclose(actual.first_piola_stress, p, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(actual.internal_force, force, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(actual.hourglass_energy,
            .5 * geometry.hourglass_coefficient * np.sum(modes**2, axis=(1, 2)), rtol=2e-14)
        np.testing.assert_array_equal(actual.material_response.state_new, request.state_old + .01)
        np.testing.assert_array_equal(actual.material_response.requested_time_scale, .5)
        assert not actual.internal_force.flags.writeable
        assert not actual.material_response.state_new.flags.writeable


@pytest.mark.parametrize('backend', ['numpy', 'native'])
@pytest.mark.parametrize('storage', ['fields', 'owned_arrays'])
def test_builtin_step_and_checkpoint(tmp_path, backend, storage):
    if backend == 'native':
        pytest.importorskip('agentfem._hex_native')
    reference = contact_step(material_storage=storage)
    reference.run()
    first = contact_step(material_storage=storage,
        execution_kernel=elements.explicit_hex_kernel(backend=backend))
    first.run(until_step=2)
    checkpoint = first.save_checkpoint(tmp_path/'builtin')
    resumed = contact_step(material_storage=storage,
        execution_kernel=elements.explicit_hex_kernel(backend=backend))
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    np.testing.assert_allclose(resumed.state.u.value.x.array, reference.state.u.value.x.array,
                               atol=1e-16, rtol=1e-13)
    np.testing.assert_allclose(resumed.material_residual.accepted_gradient,
                               reference.material_residual.accepted_gradient, atol=1e-16, rtol=1e-13)


def test_native_body_rejects_bad_buffers_without_output():
    native = pytest.importorskip('agentfem._hex_native')
    args = [np.eye(3)[None], np.eye(3)[None], np.zeros((1,8,3)),
            np.zeros((1,8,3)), np.zeros((1,8,4)), np.ones(1), np.ones(1)]
    for index, bad in ((0, -np.eye(3)[None]), (1, np.full((1,3,3), np.nan)),
                       (5, np.zeros(1)), (6, -np.ones(1)), (3, np.zeros((1,7,3))),
                       (2, np.zeros((1,8,3), dtype=np.float32))):
        wrong = args.copy()
        wrong[index] = bad
        with pytest.raises(ValueError):
            native.body_response(*wrong)
    empty = [np.empty((0,)+a.shape[1:]) for a in args]
    assert native.body_response(*empty) == (b'', b'', b'')


def test_builtin_backend_selection():
    with pytest.raises(ValueError, match='backend'):
        elements.explicit_hex_kernel(backend='silent-fast-math')
    assert elements.explicit_hex_kernel(backend='numpy').summary()['state_owner'] == 'core'


def test_native_auxiliary_step_geometry_viscosity_and_rollback():
    pytest.importorskip('agentfem._body_pipeline')
    from dataclasses import replace
    from agentfem.elements.body_auxiliary import auxiliary_callbacks
    from agentfem.elements.explicit_execution import prepare_kernel_kinematics
    kinematics, viscosity = auxiliary_callbacks()
    kernel = replace(elements.explicit_hex_kernel(backend='native'),
                     prepare_kinematics=kinematics, evaluate_viscosity=viscosity)
    options = {'bulk_viscosity': elements.BulkViscosityPolicy(1.,10.,.5)}
    plain, fast = contact_step(**options), contact_step(execution_kernel=kernel, **options)
    plain.run(); fast.run()
    np.testing.assert_allclose(fast.state.u.value.x.array, plain.state.u.value.x.array, atol=1e-15)
    assert fast.material_residual.viscosity_work == pytest.approx(plain.material_residual.viscosity_work, rel=1e-12)
    geometry = fast.material_residual.internal._execution_geometry
    folded = np.zeros_like(geometry.coordinates)
    folded[:,:,0] = -2*geometry.coordinates[:,:,0]
    with pytest.raises(ValueError):
        prepare_kernel_kinematics(kernel, geometry, folded)
