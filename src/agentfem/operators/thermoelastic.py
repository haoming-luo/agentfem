# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit reversible thermal feedback for three-dimensional small strain."""

from math import isfinite
from dataclasses import dataclass

import ufl
from dolfinx import fem

from agentfem import fields
from .core import OperatorForm


@dataclass(frozen=True)
class _ThermoelasticBlocks:
    """Physical block forms, with no solver, geometry factory or oracle."""

    mechanical_matrix: object
    mechanical_rhs: object
    thermal_matrix: object
    thermal_rhs: object
    beta: float
    capacity: float


def _thermoelastic_blocks(
    displacement,
    temperature_departure,
    *,
    old_displacement,
    old_temperature_departure,
    material,
    dt,
    heat_load=None,
    mechanical_load=None,
):
    """Lower constant homogeneous 3D small-strain thermoelastic blocks.

    Temperature unknown and history are departures from material T0, not
    absolute temperatures. Loads are already integrated linear forms. This
    private seam deliberately does not select boundaries, mesh or time path.
    """
    from .elasticity import elastic_stiffness, thermal_expansion_vector
    from agentfem.materials.properties import (
        ThermoElasticIsotropicProperties,
        constant_volumetric_heat_capacity,
    )

    if not isinstance(material, ThermoElasticIsotropicProperties):
        raise ValueError(
            "AFM-THERMO-005: constant isotropic thermoelastic material required."
        )
    u, theta = displacement, temperature_departure
    domain = u.space.mesh
    if (
        theta.space.mesh is not domain
        or old_displacement.function_space != u.space
        or old_temperature_departure.function_space != theta.space
    ):
        raise ValueError(
            "AFM-THERMO-003: current/history fields require matching spaces on one mesh."
        )
    beta = float((3 * material.lambda_ + 2 * material.mu) * material.thermal_expansion)
    capacity = constant_volumetric_heat_capacity(material)
    if not isfinite(capacity) or capacity <= 0:
        raise ValueError(
            "AFM-THERMO-005: positive constant-strain heat capacity required."
        )
    dx = ufl.Measure("dx", domain=domain)
    feedback = thermoelastic_heat_source(
        theta.test,
        u.value - old_displacement,
        coupling_coefficient=beta,
        reference_temperature=material.reference_temperature,
        dt=dt,
        measure=dx,
    )
    a_u = elastic_stiffness(u, material).expression
    f_u = thermal_expansion_vector(
        u, material.reference_temperature + theta.value, material
    ).expression
    a_t = (
        capacity / dt * theta.trial * theta.test
        + material.conductivity * ufl.inner(ufl.grad(theta.trial), ufl.grad(theta.test))
    ) * dx
    f_t = (
        capacity / dt * old_temperature_departure * theta.test * dx
        + feedback.expression
    )
    for load, expected in ((heat_load, theta.test), (mechanical_load, u.test)):
        if load is not None:
            form = getattr(load, "expression", load)
            if not isinstance(form, ufl.Form) or form.arguments() != (expected,):
                raise ValueError(
                    "AFM-THERMO-006: loads must be linear forms on their target test space."
                )
            if any(item != domain.ufl_domain() for item in form.ufl_domains()):
                raise ValueError("AFM-THERMO-003: load uses a different mesh.")
    if heat_load is not None:
        f_t += getattr(heat_load, "expression", heat_load)
    if mechanical_load is not None:
        f_u += getattr(mechanical_load, "expression", mechanical_load)
    return _ThermoelasticBlocks(a_u, f_u, a_t, f_t, beta, capacity)


def thermoelastic_heat_source(
    test,
    displacement_increment,
    *,
    coupling_coefficient,
    reference_temperature,
    dt,
    measure=ufl.dx,
):
    """Return RHS ``-beta*T0*div(delta_u)/dt`` (not plastic dissipation).

    ``beta=3*K*alpha`` for isotropic 3D elasticity. T0 is an absolute
    temperature and capacity must be the constant-strain heat capacity.
    This primitive does not select a coupled solution procedure or convert
    units. Constant real coefficients and a shared 3D mesh are required.
    """
    beta, t0, step = map(float, (coupling_coefficient, reference_temperature, dt))
    if not all(isfinite(v) for v in (beta, t0, step)) or t0 <= 0 or step <= 0:
        raise ValueError(
            "AFM-THERMO-001: finite beta, positive absolute T0 and dt required."
        )
    test = getattr(test, "test", test)
    increment = fields.unwrap(displacement_increment)
    domain = ufl.domain.extract_unique_domain(test)
    if (
        domain.topological_dimension != 3
        or domain.geometric_dimension != 3
        or increment.ufl_shape != (3,)
        or test.ufl_shape != ()
    ):
        raise ValueError(
            "AFM-THERMO-002: this feedback requires a 3D vector and scalar test."
        )
    if ufl.domain.extract_unique_domain(increment) != domain:
        raise ValueError("AFM-THERMO-003: feedback fields must share one mesh.")
    if measure.ufl_domain() is not None and measure.ufl_domain() != domain:
        raise ValueError("AFM-THERMO-003: integration measure uses a different mesh.")
    if measure.integral_type() != "cell":
        raise ValueError(
            "AFM-THERMO-004: reversible heat production requires a cell measure."
        )
    # A Constant retains form arity even when the coupling is exactly zero.
    coefficient = fem.Constant(domain, -beta * t0 / step)
    return OperatorForm(
        name="Q_thermoelastic",
        kind="thermoelastic_heat_source",
        role="vector",
        family="thermal_feedback",
        expression=coefficient * ufl.div(increment) * test * measure,
        metadata={
            "kinematics": "small_strain_3d",
            "beta": beta,
            "absolute_reference_temperature": t0,
            "dt": step,
            "sign": "negative_for_positive_beta_and_dilatation",
            "energy_semantics": "reversible_thermoelastic_exchange",
            "capacity_convention": "constant_strain",
            "not_included": ["plastic_dissipation", "frictional_heat"],
        },
    )
