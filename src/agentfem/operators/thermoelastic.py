# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit reversible thermal feedback for three-dimensional small strain."""

from math import isfinite

import ufl
from dolfinx import fem

from agentfem import fields
from .core import OperatorForm


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
