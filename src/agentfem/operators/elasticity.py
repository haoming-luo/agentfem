# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Elasticity operators in engineering FEM notation."""

from __future__ import annotations

import ufl

from agentfem import _axisymmetric
from agentfem.constitutive import elasticity
from agentfem.operators.core import OperatorForm


def stiffness_operator(
    displacement,
    test_function=None,
    properties=None,
    *,
    study=None,
    temperature=None,
    measure=ufl.dx,
) -> OperatorForm:
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


def elastic_stiffness(
    displacement, properties, *, study=None, temperature=None, measure=ufl.dx
) -> OperatorForm:
    """Create an elastic stiffness operator ``K`` from a displacement unknown."""

    return stiffness_operator(
        displacement, properties, study=study, temperature=temperature, measure=measure
    )


def internal_force_vector(
    displacement, test_function=None, properties=None, *, study=None, measure=ufl.dx
) -> OperatorForm:
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

    from agentfem import eigenstrains

    return eigenstrain_vector(
        target,
        eigenstrains.thermal(temperature),
        properties,
        study=study,
        measure=measure,
        name=name,
    )


def eigenstrain_vector(
    target,
    source,
    properties,
    *,
    study=None,
    measure=ufl.dx,
    name: str = "F_eigenstrain",
) -> OperatorForm:
    """Equivalent nodal load produced by one explicit eigenstrain source."""

    if hasattr(target, "test"):
        test = target.test
        dimension = len(target.value)
    else:
        test = target
        dimension = len(test)
    stress = source.equivalent_stress(
        properties,
        study=study,
        dimension=dimension,
    )
    weight = _axisymmetric.integration_weight(test, study)
    base_properties = getattr(properties, "material", properties)
    metadata = {
        "eigenstrain": source.summary(),
        "material": getattr(properties, "name", type(properties).__name__),
    }
    if hasattr(base_properties, "reference_temperature"):
        metadata["reference_temperature"] = base_properties.reference_temperature
    if hasattr(base_properties, "thermal_expansion"):
        coefficient = base_properties.thermal_expansion
        metadata["thermal_expansion"] = (
            coefficient.as_dict()
            if hasattr(coefficient, "as_dict")
            else coefficient
        )
    return OperatorForm(
        name=name,
        kind="eigenstrain_vector",
        role="vector",
        family="eigenstrain",
        expression=ufl.inner(stress, elasticity.strain(test, study=study))
        * weight
        * measure,
        metadata=metadata,
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
