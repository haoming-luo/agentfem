"""Operator-level helpers for common FEM matrices and vectors."""

from __future__ import annotations

from dataclasses import dataclass

import ufl

from agentfem import assembly
from agentfem import forms


@dataclass(frozen=True)
class OperatorForm:
    """Named UFL form before compilation or assembly."""

    name: str
    expression: object
    kind: str

    def compile(self):
        """Compile the UFL expression into a DOLFINx form."""

        return assembly.make_form(self.expression)

    def assemble_matrix(self, *, bcs=None):
        """Compile and assemble this operator as a matrix."""

        return assembly.assemble_matrix(self.compile(), bcs=bcs)

    def assemble_vector(self):
        """Compile and assemble this operator as a vector."""

        return assembly.assemble_vector(self.compile())

    def summary(self) -> dict[str, str]:
        """Return a compact operator description."""

        return {"name": self.name, "kind": self.kind}


def compile_form(operator: OperatorForm):
    """Compile an ``OperatorForm`` or raw UFL form."""

    if isinstance(operator, OperatorForm):
        return operator.compile()
    return assembly.make_form(operator)


def assemble_matrix(operator, *, bcs=None):
    """Assemble an operator-level matrix from an ``OperatorForm`` or UFL form."""

    if isinstance(operator, OperatorForm):
        return operator.assemble_matrix(bcs=bcs)
    return assembly.assemble_matrix(assembly.make_form(operator), bcs=bcs)


def assemble_vector(operator):
    """Assemble an operator-level vector from an ``OperatorForm`` or UFL form."""

    if isinstance(operator, OperatorForm):
        return operator.assemble_vector()
    return assembly.assemble_vector(assembly.make_form(operator))


def mass_operator(trial_function, test_function=None, density=1.0, *, measure=ufl.dx) -> OperatorForm:
    """Create a consistent mass operator ``M``."""

    if hasattr(trial_function, "trial") and hasattr(trial_function, "test") and test_function is not None:
        if density == 1.0:
            density = test_function
            test_function = None
    trial, test = _trial_test(trial_function, test_function)
    return OperatorForm(
        name="M",
        kind="mass_operator",
        expression=forms.mass_form(density, trial, test, measure=measure),
    )


def damping_operator(
    trial_function,
    test_function=None,
    coefficient=None,
    *,
    measure=ufl.dx,
) -> OperatorForm:
    """Create a viscous damping operator ``C``."""

    if hasattr(trial_function, "trial") and hasattr(trial_function, "test") and test_function is not None:
        if coefficient is None:
            coefficient = test_function
            test_function = None
    if coefficient is None:
        raise ValueError("damping_operator requires a damping coefficient.")
    trial, test = _trial_test(trial_function, test_function)
    return OperatorForm(
        name="C",
        kind="damping_operator",
        expression=forms.damping_form(coefficient, trial, test, measure=measure),
    )


def diffusion_operator(
    trial_function,
    test_function=None,
    conductivity=None,
    *,
    measure=ufl.dx,
) -> OperatorForm:
    """Create a scalar diffusion/conduction operator."""

    if hasattr(trial_function, "trial") and hasattr(trial_function, "test") and test_function is not None:
        if conductivity is None:
            conductivity = test_function
            test_function = None
    if conductivity is None:
        raise ValueError("diffusion_operator requires a conductivity.")
    trial, test = _trial_test(trial_function, test_function)
    return OperatorForm(
        name="K",
        kind="diffusion_operator",
        expression=forms.diffusion_form(
            conductivity,
            trial,
            test,
            measure=measure,
        ),
    )


def body_force_vector(force, test_function, *, measure=ufl.dx) -> OperatorForm:
    """Create a body-force/source vector ``F``."""

    test = _test(test_function)
    return OperatorForm(
        name="F_body",
        kind="body_force_vector",
        expression=forms.body_load_form(force, test, measure=measure),
    )


def boundary_load_vector(load=None, test_function=None, *, value=None, target=None, measure=None, location=None) -> OperatorForm:
    """Create a boundary load vector ``F_boundary``."""

    if value is not None:
        load = value
    if target is not None:
        test_function = target
    if measure is None and location is not None:
        measure = location.measure
    if load is None:
        raise ValueError("boundary_load_vector requires load or value.")
    if test_function is None:
        raise ValueError("boundary_load_vector requires test_function or target.")
    test = _test(test_function)
    if hasattr(load, "form"):
        return OperatorForm(
            name="F_boundary",
            kind="boundary_load_vector",
            expression=load.form(test),
        )
    if measure is None:
        raise ValueError("boundary_load_vector requires measure, location, or load with a measure.")
    return OperatorForm(
        name="F_boundary",
        kind="boundary_load_vector",
        expression=forms.boundary_load_form(load, test, measure),
    )


def boundary_force_vector(*, target, value=None, location=None, load=None) -> OperatorForm:
    """Create a boundary force vector from a load object or value/location pair."""

    if load is not None:
        return boundary_load_vector(load=load, target=target)
    return boundary_load_vector(value=value, target=target, location=location)


def force_vector(target, loads=None, *, load=None) -> OperatorForm:
    """Create a total force/source vector from one or more load objects."""

    selected = _normalize_loads(loads, load)
    test = _test(target)
    expression = None
    for item in selected:
        if not hasattr(item, "form"):
            raise ValueError("force_vector expects load objects with a form(test_function) method.")
        term = item.form(test)
        expression = term if expression is None else expression + term
    return OperatorForm(
        name="F",
        kind="force_vector",
        expression=expression,
    )


def lumped_mass(V, density=1.0, *, measure=ufl.dx):
    """Assemble a lumped mass vector for explicit dynamics."""

    return assembly.assemble_lumped_mass(_space(V), density=density, measure=measure)


def lumped_operator(V, coefficient=1.0, *, measure=ufl.dx):
    """Assemble a generic lumped diagonal operator."""

    return assembly.assemble_lumped_operator(_space(V), coefficient=coefficient, measure=measure)


def _trial_test(field_or_trial, test_function=None):
    if hasattr(field_or_trial, "trial") and hasattr(field_or_trial, "test"):
        return field_or_trial.trial, field_or_trial.test
    if test_function is None:
        raise ValueError("test_function is required unless an UnknownField is provided.")
    return field_or_trial, _test(test_function)


def _test(field_or_test):
    return field_or_test.test if hasattr(field_or_test, "test") else field_or_test


def _space(V):
    return V.space if hasattr(V, "space") else V


def _normalize_loads(loads, load):
    if load is not None:
        if loads is not None:
            raise ValueError("Pass either loads or load, not both.")
        loads = (load,)
    if loads is None:
        raise ValueError("force_vector requires loads or load.")
    if hasattr(loads, "loads"):
        loads = loads.loads
    elif not isinstance(loads, (tuple, list)):
        loads = (loads,)
    if len(loads) == 0:
        raise ValueError("force_vector requires at least one load.")
    return tuple(loads)
