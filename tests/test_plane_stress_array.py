# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Batched transverse equilibrium against the independent scalar solver."""
from dataclasses import replace
import numpy as np
import pytest
from agentfem import constitutive as c
from agentfem.constitutive.material_array_batch import MaterialPointArrayBatchInput, MaterialPointArrayBatchOutput
from agentfem.constitutive.plane_stress import reduce_plane_stress, reduce_plane_stress_array

POLICY = c.PlaneStressReduction(symmetry="reflection_symmetric")


class ArrayNeoHookean:
    """Independent compressible hyperelastic oracle plus a history clock."""
    name = "neo_hookean_clock"
    tangent_convention = c.MaterialTangentConvention.first_piola_deformation_gradient()
    state_schema = c.MaterialStateSchema(name, (c.MaterialStateVariable("AGE", unit="s"),))

    def __init__(self, *, array=True):
        if not array:
            self.update_array_batch = None

    def summary(self):
        return {"kind": self.name, "mu": 50., "lame": 100., "version": 1}

    def _array(self, request):
        F = request.deformation_gradient_new
        inverse = np.linalg.inv(F).transpose(0, 2, 1)
        J = np.linalg.det(F)
        b = 100*np.log(J)-50
        P = 50*F + b[:, None, None]*inverse
        identity = np.eye(3)
        A = (50*np.einsum("ik,JL->iJkL", identity, identity)[None]
            +100*np.einsum("niJ,nkL->niJkL", inverse, inverse)
            -b[:, None, None, None, None]*np.einsum("niL,nkJ->niJkL", inverse, inverse))
        return MaterialPointArrayBatchOutput(P @ F.transpose(0, 2, 1)/J[:, None, None],
            A.reshape(-1, 9, 9), request.state_old+request.time_increment,
            self.tangent_convention, self.state_schema,
            strain_energy_density=25*(np.sum(F*F, axis=(1, 2))-3)-50*np.log(J)+50*np.log(J)**2)

    update_array_batch = _array

    def update(self, point):
        request = MaterialPointArrayBatchInput(point.deformation_gradient_old[None],
            point.deformation_gradient_new[None], point.state_old[None], self.state_schema,
            point.time, point.time_increment)
        output = self._array(request)
        return c.MaterialPointOutput(output.cauchy_stress[0], output.consistent_tangent[0],
            output.state_new[0], tangent_convention=self.tangent_convention, state_schema=self.state_schema,
            strain_energy_density=output.strain_energy_density[0])


def material_request(count=9):
    material = c.finite_strain_j2_logarithmic(young=190000., poisson=.3,
        yield_stress=280., hardening_modulus=1100.)
    F = np.tile(np.eye(3), (count, 1, 1))
    request = MaterialPointArrayBatchInput(F, F, np.tile(material.state_schema.initial_state(), (count, 1)),
        material.state_schema, time=.1, time_increment=.1)
    return material, request


def scalar(material, request, i):
    return c.MaterialPointInput(request.deformation_gradient_old[i], request.deformation_gradient_new[i],
        request.time, request.time_increment, request.properties, request.state_old[i], request.state_schema)


def test_loading_unloading_hold_preserves_history_and_full_tangent():
    material, request = material_request()
    for step, stretch in enumerate((1.02, 1.07, 1.07, 1.04, 1.01, 1.08)):
        F = request.deformation_gradient_new.copy()
        F[:, 0, 0] = np.linspace(1., stretch, request.point_count)
        F[:, 0, 1] = .015
        request = replace(request, deformation_gradient_new=F, time=(step+1)*.1)
        old_state = request.state_old.copy()
        output, reduced, calls = reduce_plane_stress_array(material, request, reduction=POLICY)
        assert calls > 0
        for i in range(request.point_count):
            expected = reduce_plane_stress(material, scalar(material, request, i), reduction=POLICY)
            np.testing.assert_allclose(reduced.deformation_gradient[i], expected.deformation_gradient, rtol=2e-10, atol=1e-12)
            np.testing.assert_allclose(reduced.in_plane_tangent[i], expected.in_plane_tangent, rtol=2e-8, atol=1e-5)
            np.testing.assert_allclose(output.state_new[i], expected.material_response.state_new, rtol=2e-9, atol=1e-12)
            np.testing.assert_allclose(output.consistent_tangent[i], expected.material_response.consistent_tangent, rtol=2e-8, atol=1e-5)
        np.testing.assert_array_equal(request.state_old, old_state)
        request = replace(request, deformation_gradient_old=reduced.deformation_gradient,
            deformation_gradient_new=reduced.deformation_gradient, state_old=output.state_new)


def test_condensed_array_tangent_by_fixed_state_finite_difference():
    material, request = material_request(3)
    F = request.deformation_gradient_new.copy()
    F[:, 0, 0] = [1.002, 1.03, 1.1]
    request = replace(request, deformation_gradient_new=F)
    _, baseline, _ = reduce_plane_stress_array(material, request, reduction=POLICY)
    components = np.array([0, 1, 3, 4])
    for j, component in enumerate(components):
        values = []
        for sign in (1., -1.):
            gradient = F.copy().reshape(3, 9)
            gradient[:, component] += sign * 1e-6
            output, reduced, _ = reduce_plane_stress_array(material,
                replace(request, deformation_gradient_new=gradient.reshape(3, 3, 3)), reduction=POLICY)
            actual = reduced.deformation_gradient
            P = np.linalg.det(actual)[:, None, None] * output.cauchy_stress @ np.linalg.inv(actual).transpose(0, 2, 1)
            values.append(P.reshape(3, 9)[:, components])
        np.testing.assert_allclose(baseline.in_plane_tangent[:, :, j], (values[0]-values[1])/2e-6, rtol=2e-5, atol=2e-3)


def test_invalid_provider_batch_is_not_silently_repaired():
    material, request = material_request(2)
    class Broken:
        name = material.name
        tangent_convention = material.tangent_convention
        state_schema = material.state_schema
        update = material.update
        def update_array_batch(self, request):
            return None
    with pytest.raises(TypeError, match="MaterialPointArrayBatchOutput"):
        reduce_plane_stress_array(Broken(), request, reduction=POLICY)


def test_standard_step_batch_parity_and_safe_scalar_fallback(monkeypatch):
    from test_prescribed_material_fields import build
    from agentfem import results
    from agentfem.constitutive import plane_stress as reducer
    scalar, edge = build(material=ArrayNeoHookean(array=False))
    fast, edge2 = build(material=ArrayNeoHookean())
    scalar.solve()
    fast.solve()
    for key in ("F", "S", "P", "AGE", "SENER", "MATERIAL_DPDF"):
        np.testing.assert_allclose(fast.accepted_fields[key], scalar.accepted_fields[key], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(results.reaction_resultant(fast, on=edge2), results.reaction_resultant(scalar, on=edge), atol=1e-10)
    assert np.min(fast.accepted_fields['MATERIAL_DPDF'][:, 8, 8]) > 0
    np.testing.assert_array_equal(fast.accepted_fields['DPDF'][:, 2, 2, 2, 2], 0.)
    def difficult(*args, **kwargs):
        raise reducer._PlaneStressBatchFallback("injected local numerical difficulty", 1)
    monkeypatch.setattr(reducer, "reduce_plane_stress_array", difficult)
    fallback, _ = build(material=ArrayNeoHookean())
    fallback.solve()
    np.testing.assert_allclose(fallback.accepted_fields["AGE"], scalar.accepted_fields["AGE"])
    np.testing.assert_allclose(fallback.accepted_fields["F"], scalar.accepted_fields["F"], atol=1e-10)


def test_plane_stress_chunks_publish_only_after_all_material_calls_succeed():
    from mpi4py import MPI
    from agentfem import mesh
    material = ArrayNeoHookean()
    domain = mesh.rectangle((0., 0.), (1., 1.), (4, 4), comm=MPI.COMM_SELF)
    response = c.MaterialQuadratureResponse.create(domain, material.state_schema, degree=2)
    count = response.state.reference_field.point_count
    old = np.tile(np.eye(3), (count, 1, 1))
    F = old.copy()
    F[:, 0, 0] = np.linspace(1., 1.1, count)
    result = response.update(material, deformation_gradient_old=old, deformation_gradient_new=F,
        time=.1, time_increment=.1, max_batch_points=7, plane_stress=POLICY)
    assert result.provider_batch_calls > 1 and result.scalar_fallback_points == 0
    assert not np.any(response.state.committed_state_vectors())
    response.commit()
    before = response.state.committed_state_vectors().copy()
    stress = response.cauchy_stress.values.copy()
    calls = 0
    def broken(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected provider failure")
        return material._array(request)
    material.update_array_batch = broken
    with pytest.raises(RuntimeError, match="injected provider failure"):
        response.update(material, deformation_gradient_old=result.plane_stress.deformation_gradient,
            deformation_gradient_new=F, time=.2, time_increment=.1,
            max_batch_points=7, plane_stress=POLICY)
    np.testing.assert_array_equal(response.state.committed_state_vectors(), before)
    np.testing.assert_array_equal(response.state.trial_state_vectors(), before)
    np.testing.assert_array_equal(response.cauchy_stress.values, stress)


@pytest.mark.parametrize('scale', [1e-200, 1e-12, 1., 1e12, 1e200])
def test_batch_reduction_preserves_stress_unit_scaling(scale):
    base = ArrayNeoHookean()
    class Scaled(ArrayNeoHookean):
        def update_array_batch(self, request):
            out = base._array(request)
            return replace(out, cauchy_stress=scale*out.cauchy_stress,
                consistent_tangent=scale*out.consistent_tangent,
                strain_energy_density=scale*out.strain_energy_density)
    F = np.tile(np.diag([1.12, .99, 1.]), (3, 1, 1))
    request = MaterialPointArrayBatchInput(np.tile(np.eye(3), (3, 1, 1)), F,
        np.zeros((3, 1)), base.state_schema, .1, .1)
    _, expected, _ = reduce_plane_stress_array(base, request, reduction=POLICY)
    _, actual, _ = reduce_plane_stress_array(Scaled(), request, reduction=POLICY)
    np.testing.assert_allclose(actual.deformation_gradient, expected.deformation_gradient, rtol=1e-12)
    np.testing.assert_allclose(actual.in_plane_tangent/scale, expected.in_plane_tangent, rtol=1e-12)


def test_batch_rejects_out_of_plane_shear_and_incompatible_transverse_response():
    material = ArrayNeoHookean()
    F = np.tile(np.diag([1.1, .99, 1.]), (2, 1, 1))
    request = MaterialPointArrayBatchInput(np.tile(np.eye(3), (2, 1, 1)), F,
        np.zeros((2, 1)), material.state_schema, .1, .1)
    bad = F.copy()
    bad[1, 0, 2] = .01
    with pytest.raises(ValueError, match='block-diagonal'):
        reduce_plane_stress_array(material, replace(request, deformation_gradient_new=bad), reduction=POLICY)
    class WrongSymmetry(ArrayNeoHookean):
        def update_array_batch(self, request):
            out = self._array(request)
            stress = out.cauchy_stress.copy()
            stress[:, 0, 2] = stress[:, 2, 0] = .01
            return replace(out, cauchy_stress=stress)
    with pytest.raises(ValueError, match='AFM-PLANE-STRESS-SYMMETRY'):
        reduce_plane_stress_array(WrongSymmetry(), request, reduction=POLICY)
