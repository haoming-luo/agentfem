# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent nonuniform fields for pure creep and simultaneous plasticity."""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from agentfem import amplitudes, constitutive, fields, mesh, models, solvers, steps, studies


def _antiplane_step(cells, increments, material, body_force, amplitude, *, temperature_profile=None):
    domain = mesh.cuboid((0., 0., 0.), (1., 1., 1.), (cells, 1, 1),
                          cell_type="hexahedron", comm=MPI.COMM_SELF)
    model = models.create(study=studies.creep_solid(), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(material)
    temperature = None
    if temperature_profile is not None:
        temperature = model.field(fields.temperature(domain))
        temperature.value.interpolate(temperature_profile)
        temperature.value.x.scatter_forward()
    # Constrain x/y on both z end faces; the analytical solution satisfies
    # these constraints. z traction there is zero. x ends have u_z=0.
    for end in (0., 1.):
        z_face = mesh.face(domain, axis="z", value=end, tag=1 + int(end))
        for component in (0, 1):
            model.fix(u, on=z_face, component=component, value=0.)
        model.fix(u, on=mesh.face(domain, axis="x", value=end, tag=3 + int(end)),
                  component=2, value=0.)
    model.body_force((0., 0., body_force), target=u)
    step = model.step(
        target=u, duration=1., amplitude=amplitude,
        incrementation=steps.fixed(increments), progress=False,
        temperature=temperature,
        solver_options=solvers.newton(relative_tolerance=1.e-10, absolute_tolerance=1.e-10),
    )
    step.solve()
    return u, step


def _relative_displacement_error(u, exact):
    measure = ufl.dx(metadata={"quadrature_degree": 12})
    delta = u.value - exact
    error = fem.assemble_scalar(fem.form(ufl.inner(delta, delta) * measure))
    norm = fem.assemble_scalar(fem.form(ufl.inner(exact, exact) * measure))
    return np.sqrt(error / norm)


def _shear_field(cells, increments):
    # Exact continuous solution: u=(0,0,s*x*(1-x)*t), lambda=3GA/s_ref,
    # sigma_xz=G*s*(1-2x)*h(t), h=(1-exp(-lambda*t))/lambda.
    # div(sigma)+b=0 requires b_z=2G*s*h(t). Other strains/stresses vanish.
    young, poisson, scale = 200_000., .3, .001
    shear = young / (2. * (1. + poisson))
    rate = 3. * shear * 1.e-4 / 100.
    time = np.linspace(0., 1., increments + 1)
    forcing = -np.expm1(-rate * time) / rate
    material = constitutive.CreepPlasticityMaterial(
        creep=constitutive.isotropic_power_law(
            young=young, poisson=poisson, density=1., coefficient=1.e-4,
            reference_stress=100., stress_exponent=1.),
        plastic=constitutive.J2LinearIsotropicHardening(
            young=young, poisson=poisson, yield_stress=1.e9, hardening_modulus=0.),
    )
    u, step = _antiplane_step(cells, increments, material, 2. * shear * scale,
                              amplitudes.tabular(time, forcing))
    x = ufl.SpatialCoordinate(u.value.function_space.mesh)
    exact = ufl.as_vector((0., 0., scale * x[0] * (1. - x[0])))
    assert step.state.plastic_fields["equivalent_plastic_strain"].global_max() == 0.
    assert step.state.equivalent_creep_strain.global_max() > 0.
    return _relative_displacement_error(u, exact)


def test_nonuniform_shear_field_spatial_refinement_against_continuous_solution(record_property):
    errors = [_shear_field(cells, 64) for cells in (2, 4, 8)]
    # Q1 approximation of a quadratic field: expect second-order L2 decay
    # before the finite time-step error floor. Temporal accuracy has separate
    # analytical tests; this does not establish active-plastic spatial accuracy.
    assert 3.5 < errors[0] / errors[1] < 4.5
    assert 3.5 < errors[1] / errors[2] < 4.5
    assert errors[-1] < .02
    record_property("relative_displacement_l2_2_4_8_cells", list(map(float, errors)))


def test_nonuniform_shear_field_spatial_limit_is_not_zero_error():
    assert _shear_field(2, 32) > .1


def _active_joint_field(cells, increments, *, return_creep_error=False):
    # Prescribe equilibrium stress tau_xz=S*t*(1-2x) through b_z=2*S*t.
    # For monotonically increasing |tau|: p=max(sqrt(3)*|tau|-Y,0)/H,
    # gamma_p=sqrt(3)*sign(tau)*p, gamma_c=sqrt(3)*sign(tau)*integral(A*(q/ref)^n dt).
    # Integrate gamma=du_z/dx independently, using reflection about x=1/2.
    young, poisson, stress, yield_stress, hardening = 200_000., .3, 60., 50., 10_000.
    coefficient, reference, exponent = 1.e-3, 100., 3.
    shear = young / (2. * (1. + poisson))
    material = constitutive.CreepPlasticityMaterial(
        creep=constitutive.isotropic_power_law(
            young=young, poisson=poisson, density=1., coefficient=coefficient,
            reference_stress=reference, stress_exponent=exponent),
        plastic=constitutive.J2LinearIsotropicHardening(
            young=young, poisson=poisson, yield_stress=yield_stress, hardening_modulus=hardening),
    )
    u, step = _antiplane_step(cells, increments, material, 2. * stress,
                              amplitudes.tabular([0., 1.], [0., 1.]))
    x = ufl.SpatialCoordinate(u.value.function_space.mesh)[0]
    left = ufl.min_value(x, 1. - x)
    plastic_front = .5 * (1. - yield_stress / (np.sqrt(3.) * stress))
    extent = ufl.min_value(left, plastic_front)
    elastic = stress / shear * (left - left ** 2)
    plastic = (3. * stress * (extent - extent ** 2) - np.sqrt(3.) * yield_stress * extent) / hardening
    creep = (np.sqrt(3.) * coefficient * (np.sqrt(3.) * stress / reference) ** exponent
             * (1. - (1. - 2. * left) ** (exponent + 1.)) / (2. * (exponent + 1.) ** 2))
    exact = ufl.as_vector((0., 0., elastic + plastic + creep))
    plastic_state = step.state.plastic_fields["equivalent_plastic_strain"].values
    assert np.max(plastic_state) > .001
    assert np.min(plastic_state) == pytest.approx(0., abs=1.e-12)
    assert step.state.equivalent_creep_strain.global_max() > 1.e-4
    # In the one-dimensional Q1 reduction the element stress is its exact
    # midpoint value. This independent equilibrium reference also checks the
    # joint state, not merely a visually close displacement field.
    peak_q = np.sqrt(3.) * stress * (1. - 1. / cells)
    assert np.max(plastic_state) == pytest.approx((peak_q - yield_stress) / hardening, rel=1.e-7)
    exact_peak_creep = coefficient * (peak_q / reference) ** exponent / (exponent + 1.)
    relative_creep_error = step.state.equivalent_creep_strain.global_max() / exact_peak_creep - 1.
    # Sum of the right-endpoint cubes has a known closed form. This separately
    # exposes the temporal error instead of attributing all error to the mesh.
    assert relative_creep_error == pytest.approx(2. / increments + 1. / increments ** 2, rel=2.e-6)
    if return_creep_error:
        return relative_creep_error
    return _relative_displacement_error(u, exact)


def test_active_joint_nonuniform_field_refines_to_independent_integrated_solution(record_property):
    errors = [_active_joint_field(cells, 128) for cells in (4, 8, 16)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[2] > 6.
    assert errors[-1] < .02
    record_property("active_joint_relative_l2_4_8_16_cells", list(map(float, errors)))


def test_active_joint_creep_state_time_refines_at_fixed_mesh(record_property):
    errors = [_active_joint_field(8, count, return_creep_error=True) for count in (32, 64)]
    assert 1.9 < errors[0] / errors[1] < 2.1
    record_property("active_joint_peak_creep_relative_error_32_64_steps", list(map(float, errors)))


def _thermal_joint_field(cells, increments):
    # Manufactured prescribed T(x) makes the Arrhenius factor exactly
    # 1+a*(1-2x)^2 in the continuum. No production update builds the reference.
    young, poisson, stress, yield_stress, hardening = 200_000., .3, 60., 50., 10_000.
    coefficient, reference, exponent, contrast = 1.e-3, 100., 3., 2.
    activation, gas_constant, reference_temperature = 120_000., 8.31446261815324, 800.
    elastic = constitutive.thermoelastic(
        young=young, poisson=poisson, density=1., thermal_expansion=0.,
        conductivity=1., specific_heat=1., reference_temperature=reference_temperature)
    material = constitutive.CreepPlasticityMaterial(
        creep=constitutive.isotropic_arrhenius_power_law(
            elastic=elastic, coefficient=coefficient, reference_stress=reference,
            stress_exponent=exponent, activation_energy=activation,
            reference_temperature=reference_temperature),
        plastic=constitutive.J2LinearIsotropicHardening(
            young=young, poisson=poisson, yield_stress=yield_stress, hardening_modulus=hardening))
    def temperature(coordinates):
        factor = 1. + contrast * (1. - 2. * coordinates[0])**2
        return 1. / (1. / reference_temperature - gas_constant / activation * np.log(factor))
    u, step = _antiplane_step(cells, increments, material, 2. * stress,
        amplitudes.tabular([0., 1.], [0., 1.]), temperature_profile=temperature)
    x = ufl.SpatialCoordinate(u.value.function_space.mesh)[0]
    left = ufl.min_value(x, 1. - x)
    y = 1. - 2. * left
    front = .5 * (1. - yield_stress / (np.sqrt(3.) * stress))
    extent = ufl.min_value(left, front)
    elastic_strain_integral = stress / (young / (2. * (1. + poisson))) * (left - left**2)
    plastic_integral = (3. * stress * (extent - extent**2)
                        - np.sqrt(3.) * yield_stress * extent) / hardening
    creep_integral = (np.sqrt(3.) * coefficient * (np.sqrt(3.) * stress / reference)**exponent
        / (2. * (exponent + 1.))
        * ((1. - y**(exponent + 1.)) / (exponent + 1.)
           + contrast * (1. - y**(exponent + 3.)) / (exponent + 3.)))
    exact = ufl.as_vector((0., 0., elastic_strain_integral + plastic_integral + creep_integral))
    assert step.state.plastic_fields["equivalent_plastic_strain"].global_max() > .001
    assert step.state.equivalent_creep_strain.global_max() > 1.e-4
    minimum = step.accepted_increments[-1].minimum_temperature
    maximum = step.accepted_increments[-1].maximum_temperature
    assert 800. <= minimum < maximum < 860.
    return _relative_displacement_error(u, exact)


def test_nonuniform_temperature_joint_field_refines_to_independent_solution(record_property):
    errors = [_thermal_joint_field(cells, 128) for cells in (4, 8, 16)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[2] > 6.
    assert errors[-1] < .02
    record_property("thermal_joint_relative_l2_4_8_16_cells", list(map(float, errors)))
