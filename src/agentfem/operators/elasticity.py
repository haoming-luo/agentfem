"""Elasticity operators in engineering FEM notation."""

from __future__ import annotations

import ufl

from agentfem import _axisymmetric
from agentfem.constitutive import elasticity
from agentfem.operators.core import OperatorForm


def stiffness_operator(displacement, test_function=None, properties=None, *, study=None, temperature=None, measure=ufl.dx) -> OperatorForm:
    """Create an elastic stiffness/internal virtual-work operator ``K``."""

    if study is not None and hasattr(study, "require"):
        study.require(physics="solid_mechanics")
    trial, test, props = _elastic_args(displacement, test_function, properties)
    weight = _axisymmetric.integration_weight(trial, study)
    return OperatorForm(
        name="K",
        kind="elastic_stiffness_operator",
        role="matrix",
        family="elasticity",
        expression=ufl.inner(
            elasticity.stress(trial, props, study=study, temperature=temperature),
            elasticity.strain(test, study=study),
        )
        * weight
        * measure,
    )


def elastic_stiffness(displacement, properties, *, study=None, temperature=None, measure=ufl.dx) -> OperatorForm:
    """Create an elastic stiffness operator ``K`` from a displacement unknown."""

    return stiffness_operator(displacement, properties, study=study, temperature=temperature, measure=measure)


def internal_force_vector(displacement, test_function=None, properties=None, *, study=None, measure=ufl.dx) -> OperatorForm:
    """Create an elastic internal-force vector contribution."""

    if study is not None and hasattr(study, "require"):
        study.require(physics="solid_mechanics")
    trial, test, props = _elastic_args(displacement, test_function, properties)
    weight = _axisymmetric.integration_weight(trial, study)
    return OperatorForm(
        name="F_int",
        kind="elastic_internal_force_vector",
        role="vector",
        family="elasticity",
        expression=ufl.inner(
            elasticity.stress(trial, props, study=study),
            elasticity.strain(test, study=study),
        )
        * weight
        * measure,
    )


def thermal_expansion_vector(
    target,
    temperature,
    properties,
    *,
    study=None,
    measure=ufl.dx,
    name: str = "F_thermal",
) -> OperatorForm:
    """Equivalent nodal load produced by isotropic thermal expansion."""

    if hasattr(target, "test"):
        test = target.test
        dimension = len(target.value)
    else:
        test = target
        dimension = len(test)
    stress = elasticity.thermal_expansion_stress(
        temperature,
        properties,
        study=study,
        dimension=dimension,
    )
    weight = _axisymmetric.integration_weight(test, study)
    return OperatorForm(
        name=name,
        kind="thermal_expansion_vector",
        role="vector",
        family="thermoelasticity",
        expression=ufl.inner(stress, elasticity.strain(test, study=study))
        * weight
        * measure,
        metadata={
            "reference_temperature": properties.reference_temperature,
            "thermal_expansion": (
                properties.thermal_expansion.as_dict()
                if hasattr(properties.thermal_expansion, "as_dict")
                else properties.thermal_expansion
            ),
        },
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
