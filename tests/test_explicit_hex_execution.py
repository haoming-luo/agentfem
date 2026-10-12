# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem import elements
from test_response_material_step import contact_step


def reference_kernel(calls, identity="reference-v1", fail=False):
    def evaluate(material, request, geometry, displacement):
        calls.append(request.time)
        assert not request.state_old.flags.writeable
        assert not displacement.flags.writeable
        assert not geometry.volume.flags.writeable
        if fail and request.time > 5e-5:
            raise RuntimeError("injected kernel failure")
        response = material.update_explicit_array_batch(request)
        f = request.deformation_gradient_new
        p = np.linalg.det(f)[:, None, None] * (response.cauchy_stress @ np.linalg.inv(f).swapaxes(1, 2))
        modes = geometry.hourglass_modes.swapaxes(1, 2) @ displacement
        force = geometry.volume[:, None, None] * (geometry.average_gradient @ p.swapaxes(1, 2))
        force += geometry.hourglass_coefficient[:, None, None] * (geometry.hourglass_modes @ modes)
        hg = .5 * geometry.hourglass_coefficient * np.sum(modes**2, axis=(1, 2))
        return elements.ExplicitHexBodyOutput(response, p, force, hg)
    return elements.ExplicitHexBodyKernel(evaluate, {"implementation": identity})


def test_instance_local_kernel_matches_reference_and_restart(tmp_path):
    plain = contact_step()
    plain.run()
    first_calls, second_calls = [], []
    first = contact_step(execution_kernel=reference_kernel(first_calls))
    second = contact_step(execution_kernel=reference_kernel(second_calls))
    first.run(until_step=2)
    assert first_calls and not second_calls
    checkpoint = first.save_checkpoint(tmp_path / "kernel")
    second.load_checkpoint(checkpoint)
    second.run()
    np.testing.assert_array_equal(second.state.u.value.x.array, plain.state.u.value.x.array)
    np.testing.assert_array_equal(second.material_residual.accepted_gradient, plain.material_residual.accepted_gradient)
    assert second_calls


def test_kernel_failure_does_not_commit():
    step = contact_step(execution_kernel=reference_kernel([], fail=True))
    step.run(until_step=1)
    before = step.residual.snapshot()
    u = step.state.u.value.x.array.copy()
    with pytest.raises(RuntimeError, match="injected kernel failure"):
        step.run()
    np.testing.assert_array_equal(step.state.u.value.x.array, u)
    assert step.residual.snapshot() == before


def test_kernel_identity_change_refuses_restart(tmp_path):
    step = contact_step(execution_kernel=reference_kernel([]))
    step.run(until_step=2)
    path = step.save_checkpoint(tmp_path / "identity")
    other = contact_step(execution_kernel=reference_kernel([], identity="different"))
    with pytest.raises(ValueError):
        other.load_checkpoint(path)


def test_kinematic_screen_preserves_reference_and_rejects_fold():
    from dataclasses import replace
    from agentfem.elements._hex_validity import admits_reference_perturbation
    from agentfem.elements.explicit_execution import prepare_kernel_kinematics
    def geometry_kernel(geometry, u):
        assert not u.flags.writeable
        f = np.eye(3) + u.swapaxes(1,2) @ geometry.average_gradient
        mask = admits_reference_perturbation(geometry.coordinates, u,
                                             geometry.reference_perturbation_radius)
        return f, mask
    kernel = replace(reference_kernel([]), prepare_kinematics=geometry_kernel)
    plain = contact_step()
    fast = contact_step(execution_kernel=kernel, material_storage='owned_arrays')
    plain.run()
    fast.run()
    np.testing.assert_array_equal(fast.state.u.value.x.array, plain.state.u.value.x.array)
    geometry = fast.material_residual.internal._execution_geometry
    folded = np.zeros_like(geometry.coordinates)
    folded[:,:,0] = -2*geometry.coordinates[:,:,0]
    with pytest.raises(ValueError):
        prepare_kernel_kinematics(kernel, geometry, folded)
    malformed = replace(kernel, prepare_kinematics=lambda geometry,u: (np.tile(np.eye(3), (len(u),1,1)),
                                                                     np.ones(len(u))))
    with pytest.raises(ValueError, match='boolean reference certificate'):
        prepare_kernel_kinematics(malformed, geometry, np.zeros_like(folded))


def test_bulk_execution_equivalence_and_invalid_output_rollback():
    from dataclasses import replace
    viscosity = elements.BulkViscosityPolicy(1., 10., .5)
    calls = []
    def bulk(geometry, gradient, velocity, coefficients, reference):
        calls.append(len(gradient))
        assert not velocity.flags.writeable
        assert not coefficients['density'].flags.writeable
        return reference()
    kernel = replace(reference_kernel([]), evaluate_viscosity=bulk)
    plain = contact_step(bulk_viscosity=viscosity)
    fast = contact_step(bulk_viscosity=viscosity, execution_kernel=kernel, material_storage='owned_arrays')
    plain.run(); fast.run()
    np.testing.assert_array_equal(fast.state.u.value.x.array, plain.state.u.value.x.array)
    assert fast.material_residual.viscosity_work == plain.material_residual.viscosity_work
    assert calls
    attempts = []
    def invalid(*args):
        attempts.append(1)
        result = args[-1]()
        if len(attempts) == 1:
            return result
        return replace(result, artificial_power=np.full_like(result.artificial_power, -1.))
    failing = contact_step(bulk_viscosity=viscosity, material_storage='owned_arrays',
        execution_kernel=replace(kernel, evaluate_viscosity=invalid))
    failing.run(until_step=1)
    before = failing.residual.snapshot()
    with pytest.raises(ValueError, match='nonnegative'):
        failing.run()
    assert failing.residual.snapshot() == before
