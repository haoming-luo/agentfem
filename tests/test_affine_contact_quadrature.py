# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Quadrature controls reach the same total-residual affine contact route."""
import numpy as np
import pytest

from mpi4py import MPI

from agentfem import boundary_models, constitutive, fields, mesh, models, steps, studies
from agentfem.step_providers import step_capability
from periodic_cube_fixture import periodic_unit_cube
from test_finite_strain_j2_periodic import _periodic_j2_contact_step


def make(degree=None, **kwargs):
    return _periodic_j2_contact_step(
        surface=boundary_models.rigid_plane(
            point=(1.01, 0., 0.), normal=(-1., 0., 0.)),
        contact_quadrature_degree=degree, **kwargs)


@pytest.mark.parametrize("degree", [5, 7, 9, 11])
def test_affine_quadrature_agrees_with_capability_and_integrates_plane(degree):
    _, _, reference = make()
    domain, constraint, candidate = make(degree)
    assert f"degree_{degree}" in candidate.contact_provider.adapter.quadrature_rule
    reference.solve()
    candidate.solve()
    assert candidate.contact_provider.evaluate().active_points > 0
    np.testing.assert_allclose(candidate.solution.x.array, reference.solution.x.array,
                               rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(candidate.response.state.committed_state_vectors(),
                               reference.response.state.committed_state_vectors(),
                               rtol=1e-7, atol=1e-9)
    full = candidate.assemble_full_residual()
    transformation = constraint.reduction(1.).matrix(domain.domain.comm)
    reduced = transformation.createVecRight()
    try:
        transformation.multTranspose(full, reduced)
        assert reduced.norm() < 1e-5
    finally:
        reduced.destroy()
        transformation.destroy()
        full.destroy()


def test_affine_quadrature_restart_preserves_rule_and_rejects_changed_rule(tmp_path):
    _, _, reference = make(5)
    reference.solve()
    _, _, partial = make(5)
    partial.solve(until=1/3)
    saved = partial.save_checkpoint(tmp_path / "degree5")
    _, _, resumed = make(5)
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(resumed.solution.x.array, reference.solution.x.array,
                               rtol=1e-10, atol=1e-12)
    _, _, changed = make(7)
    before = changed.solution.x.array.copy()
    with pytest.raises(ValueError):
        changed.load_checkpoint(saved)
    np.testing.assert_array_equal(changed.solution.x.array, before)
    assert changed.accepted_load_factor == 0.


def rejection_model(*, contact=True):
    fixture = periodic_unit_cube(MPI.COMM_SELF)
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True),
                          mesh=fixture.domain)
    u = model.field(fields.displacement(fixture.domain))
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000., poisson=.3, yield_stress=2., hardening_modulus=100.))
    constraint = model.constraint(fixture.constraint(u))
    if contact:
        right = mesh.boundary(fixture.domain, lambda x: np.isclose(x[0], 1.), name="right")
        model.add_boundary_model(boundary_models.rigid_contact_pair(
            right, boundary_models.rigid_body(boundary_models.rigid_plane(
                point=(1.01, 0., 0.), normal=(-1., 0., 0.))), penalty=1000.))
    return model, u, dict(material=material, constraints=constraint)


@pytest.mark.parametrize("degree", [3, True, 5.0])
def test_invalid_degree_rejected_by_capability_and_execution(degree):
    model, u, options = rejection_model()
    options["contact_quadrature_degree"] = degree
    assert not step_capability(model, target=u, options=options)["supported"]
    with pytest.raises((ValueError, NotImplementedError)):
        model.step(target=u, **options)


@pytest.mark.parametrize("scope", ["absent_contact", "nodal_augmentation", "active_partition"])
def test_quadrature_override_never_silently_discarded(scope):
    model, u, options = rejection_model(contact=scope != "absent_contact")
    options["contact_quadrature_degree"] = 5
    if scope == "nodal_augmentation":
        options["normal_augmentation"] = steps.normal_augmentation(
            penetration_tolerance=1e-7, complementarity_tolerance=1e-6,
            length_unit="m", pressure_unit="Pa")
    elif scope == "active_partition":
        options["contact_integration"] = "planar_active_partition"
    assert not step_capability(model, target=u, options=options)["supported"]
    with pytest.raises((ValueError, NotImplementedError)):
        model.step(target=u, **options)
