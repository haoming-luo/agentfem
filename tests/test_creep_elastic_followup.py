# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Series shear analogue of elastic follow-up, using ordinary regional creep."""

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import constitutive, fields, mesh, models, solvers, steps, studies, units


def _series_step(*, followup, exponent, increments):
    # Two unit-length slabs, equal area. Creep in the left slab, elastic
    # right slab: f = 1 + G1/G2. A zero creep coefficient is the exact
    # elastic limit of the same existing regional material contract.
    shear, initial = 100_000., 100.
    displacement = initial * followup / shear
    domain = mesh.cuboid((0., 0., 0.), (2., 1., 1.), (4, 1, 1),
                         cell_type="hexahedron", comm=MPI.COMM_SELF)
    model = models.create(study=studies.creep_solid(), mesh=domain, units=units.n_mm_mpa())
    u = model.field(fields.displacement(domain))
    regions = mesh.partition_cells(domain, creeping=lambda x: x[0] <= 1.,
                                   elastic=lambda x: x[0] > 1.)
    for region, modulus, coefficient in (
        (regions.creeping, shear, 1.e-4), (regions.elastic, shear / (followup - 1.), 0.),
    ):
        model.material(constitutive.isotropic_power_law(
            young=2. * modulus, poisson=0., density=1., coefficient=coefficient,
            reference_stress=100., stress_exponent=exponent), region=region)
    # Antiplane support: retain the x reactions needed on z faces, not
    # fictitious traction-free surfaces incompatible with pure xz shear.
    for end in (0., 1.):
        face = mesh.face(domain, axis="z", value=end, tag=1 + int(end))
        for component in (0, 1):
            model.fix(u, on=face, component=component, value=0.)
    model.fix(u, on=mesh.face(domain, axis="x", value=0., tag=3), component=2, value=0.)
    model.fix(u, on=mesh.face(domain, axis="x", value=2., tag=4), component=2, value=displacement)
    step = model.step(target=u, duration=2., incrementation=steps.fixed(increments),
                      solver_options=solvers.newton(relative_tolerance=1.e-10, absolute_tolerance=1.e-9),
                      progress=False)
    step.solve()
    stress = np.array([initial] + [frame.generalized_reaction / displacement for frame in step.energy_history])
    # Equal shear traction across the material interface; only one slab creeps.
    np.testing.assert_allclose(step.state.stress.values[:, 0, 2], stress[-1], rtol=1.e-9, atol=1.e-7)
    x = step.state.stress.owned_physical_points[..., 0].reshape(-1)
    ce = step.state.creep_strain.owned_values.reshape((-1, 3, 3))
    np.testing.assert_array_equal(ce[x > 1.], 0.)
    np.testing.assert_allclose(ce[x < 1., 0, 2],
        .5 * followup / shear * (initial - stress[-1]), rtol=1.e-8, atol=1.e-12)
    return stress


@pytest.mark.parametrize("exponent", [1., 3.])
@pytest.mark.parametrize("followup", [2., 5.])
def test_regional_elastic_followup_relaxation_matches_continuous_series_solution(exponent, followup, record_property):
    # Compatibility: (1/G1+1/G2)*tau_dot + gamma_creep_dot = 0.
    # Associative Mises flow: gamma_creep_dot=sqrt(3)*A*(sqrt(3)*tau/s_ref)^n.
    # Hence tau_dot=-k*tau^n with G_eff=G1/f. No FE update enters the oracle.
    initial, shear, coefficient, reference = 100., 100_000., 1.e-4, 100.
    rate = shear / followup * np.sqrt(3.) * coefficient * (np.sqrt(3.) / reference)**exponent
    def exact(time, factor=1.):
        if exponent == 1.:
            return initial * np.exp(-rate * factor * time)
        return (initial**(1. - exponent) + (exponent - 1.) * rate * factor * time)**(1. / (1. - exponent))
    errors = []
    for count in (16, 32, 64):
        stress = _series_step(followup=followup, exponent=exponent, increments=count)
        time = np.linspace(0., 2., count + 1)
        errors.append(float(np.max(np.abs(stress - exact(time)))))
        assert np.all(np.diff(stress) < 0.)
        assert stress[-1] > exact(2.)
    assert 1.7 < errors[0] / errors[1] < 2.2
    assert 1.7 < errors[1] / errors[2] < 2.2
    assert errors[-1] / initial < .01
    # Full elastic restraint is a different problem, not an interchangeable
    # prescribed-strain material test: follow-up slows this stress relaxation.
    assert exact(2.) > exact(2., factor=followup)
    record_property("maximum_stress_errors_MPa_16_32_64", errors)
    record_property("continuous_final_stress_MPa", float(exact(2.)))
