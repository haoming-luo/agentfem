"""Elasticity operators in engineering FEM notation."""

from __future__ import annotations

import ufl

from agentfem import forms
from agentfem.constitutive import elasticity
from agentfem.operators.core import OperatorForm


def stiffness_operator(displacement, test_function=None, properties=None, *, measure=ufl.dx) -> OperatorForm:
    """Create an elastic stiffness/internal virtual-work operator ``K``."""

    trial, test, props = _elastic_args(displacement, test_function, properties)
    return OperatorForm(
        name="K",
        kind="elastic_stiffness_operator",
        expression=forms.stiffness_form(
            elasticity.stress(trial, props),
            elasticity.strain(test),
            measure=measure,
        ),
    )


def internal_force_vector(displacement, test_function=None, properties=None, *, measure=ufl.dx) -> OperatorForm:
    """Create an elastic internal-force vector contribution."""

    trial, test, props = _elastic_args(displacement, test_function, properties)
    return OperatorForm(
        name="F_int",
        kind="elastic_internal_force_vector",
        expression=forms.stiffness_form(
            elasticity.stress(trial, props),
            elasticity.strain(test),
            measure=measure,
        ),
    )


def _elastic_args(displacement, test_function=None, properties=None):
    if hasattr(displacement, "trial") and hasattr(displacement, "test"):
        if properties is None:
            properties = test_function
        if properties is None:
            raise ValueError("properties are required for an elastic operator.")
        return displacement.trial, displacement.test, properties
    if test_function is None or properties is None:
        raise ValueError(
            "stiffness_operator requires either (UnknownField, properties) "
            "or (trial_function, test_function, properties)."
        )
    return displacement, test_function, properties
