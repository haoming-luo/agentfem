"""Operator-level helpers for common FEM matrices and vectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from petsc4py import PETSc

from agentfem import assembly
from agentfem import fields
from agentfem import forms
from agentfem.kernel import dofs


@dataclass(frozen=True)
class OperatorForm:
    """Named UFL form before compilation or assembly."""

    name: str
    expression: object
    kind: str
    role: str = "operator"
    family: str = "generic"
    parts: tuple[object, ...] = ()

    def __add__(self, other):
        """Combine compatible operator forms by adding their expressions."""

        return combine(self, other)

    def __mul__(self, factor):
        """Scale an operator form by a scalar coefficient."""

        return scale(self, factor)

    def __rmul__(self, factor):
        """Scale an operator form by a scalar coefficient."""

        return scale(self, factor)

    def __truediv__(self, factor):
        """Scale an operator form by the inverse of a scalar coefficient."""

        return scale(self, 1.0 / factor)

    def compile(self):
        """Compile the UFL expression into a DOLFINx form."""

        return assembly.make_form(self.expression)

    def assemble_matrix(self, *, bcs=None):
        """Compile and assemble this operator as a matrix."""

        return assembly.assemble_matrix(self.compile(), bcs=bcs)

    def assemble_vector(self):
        """Compile and assemble this operator as a vector."""

        return assembly.assemble_vector(self.compile())

    def summary(self) -> dict[str, object]:
        """Return a compact operator description."""

        return {
            "name": self.name,
            "kind": self.kind,
            "role": self.role,
            "family": self.family,
            "parts": self.parts,
        }

    def renamed(
        self,
        name: str,
        *,
        kind: str | None = None,
        role: str | None = None,
        family: str | None = None,
    ):
        """Return the same expression with updated operator metadata."""

        return OperatorForm(
            name=name,
            expression=self.expression,
            kind=kind or self.kind,
            role=role or self.role,
            family=family or self.family,
            parts=self.parts,
        )


def combine(*operators, name: str = "combined_operator", kind: str = "combined_operator") -> OperatorForm:
    """Combine operator forms or raw UFL expressions into one operator form."""

    if len(operators) == 0:
        raise ValueError("combine requires at least one operator.")
    expression = None
    role = None
    families = []
    parts = []
    for operator in operators:
        expr = _expression(operator)
        expression = expr if expression is None else expression + expr
        if isinstance(operator, OperatorForm):
            if role is None:
                role = operator.role
            elif operator.role != role:
                raise ValueError(
                    f"Cannot combine operator roles {role!r} and {operator.role!r}."
                )
            families.append(operator.family)
            parts.append(operator.summary())
    return OperatorForm(
        name=name,
        kind=kind,
        role=role or "operator",
        family="+".join(families) if families else "generic",
        expression=expression,
        parts=tuple(parts),
    )


def scale(operator, factor, *, name: str | None = None, kind: str | None = None) -> OperatorForm:
    """Scale an operator or vector form while preserving its engineering role."""

    if isinstance(operator, OperatorForm):
        return OperatorForm(
            name=name or operator.name,
            expression=factor * operator.expression,
            kind=kind or operator.kind,
            role=operator.role,
            family=operator.family,
            parts=operator.parts,
        )
    return OperatorForm(
        name=name or "scaled_operator",
        expression=factor * operator,
        kind=kind or "scaled_operator",
        role="operator",
        family="generic",
    )


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


def action(operator, field):
    """Return the algebraic action of a matrix-like operator on a field.

    For a PETSc matrix or matrix-valued ``OperatorForm``, the result is a PETSc
    vector in the dual space. For a lumped diagonal vector, the result is an
    AgentFEM field with pointwise scaled dof values.
    """

    matrix_or_diagonal = _matrix_or_diagonal(operator)
    field_function = fields.unwrap(field)
    if isinstance(matrix_or_diagonal, np.ndarray):
        result = fields.empty_like(field_function, name=f"{getattr(operator, 'name', 'A')}_action")
        values = dofs.owned_array(result)
        source = dofs.owned_array(field_function)
        owned = len(source)
        diagonal = matrix_or_diagonal[:owned]
        if len(diagonal) != owned:
            raise ValueError("Diagonal action requires a diagonal matching the local owned dofs.")
        values[:] = diagonal * source
        result.x.scatter_forward()
        return result
    result = matrix_or_diagonal.createVecLeft()
    matrix_or_diagonal.mult(field_function.x.petsc_vec, result)
    return result


def bilinear_form(operator, left, right) -> float:
    """Return the algebraic scalar ``left^T operator right``.

    This is the discrete operator product. It is intentionally separate from
    field algebra and weak-form integrals so code can distinguish ``u * v``,
    ``u^T v``, and ``u^T M v``.
    """

    matrix_or_diagonal = _matrix_or_diagonal(operator)
    if isinstance(matrix_or_diagonal, np.ndarray):
        return fields.weighted_dot(left, matrix_or_diagonal, right)

    left_function = fields.unwrap(left)
    right_function = fields.unwrap(right)
    result = matrix_or_diagonal.createVecLeft()
    matrix_or_diagonal.mult(right_function.x.petsc_vec, result)
    value = left_function.x.petsc_vec.dot(result)
    result.destroy()
    return float(np.real(value))


def quadratic_form(operator, field) -> float:
    """Return the algebraic scalar ``field^T operator field``."""

    return bilinear_form(operator, field, field)


def xtmy(left, operator, right) -> float:
    """Cast3M-style alias for ``left^T operator right``."""

    return bilinear_form(operator, left, right)


def xtmx(field, operator) -> float:
    """Cast3M-style alias for ``field^T operator field``."""

    return quadratic_form(operator, field)


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
        role="matrix",
        family="mass",
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
        role="matrix",
        family="damping",
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
        role="matrix",
        family="diffusion",
        expression=forms.diffusion_form(
            conductivity,
            trial,
            test,
            measure=measure,
        ),
    )


def heat_conduction_operator(temperature, conductivity, *, measure=ufl.dx) -> OperatorForm:
    """Create a heat-conduction operator ``K`` for ``-div(k grad(T))``."""

    return diffusion_operator(temperature, conductivity, measure=measure).renamed(
        "K",
        kind="heat_conduction_operator",
        family="heat_conduction",
    )


def conduction_operator(temperature, conductivity, *, measure=ufl.dx) -> OperatorForm:
    """Create a conduction/diffusion stiffness operator ``K``."""

    return heat_conduction_operator(temperature, conductivity, measure=measure)


def heat_capacity_operator(temperature, capacity, *, measure=ufl.dx) -> OperatorForm:
    """Create a heat-capacity operator ``C`` for transient heat problems."""

    operator = mass_operator(temperature, capacity, measure=measure)
    return OperatorForm(
        name="C",
        kind="heat_capacity_operator",
        role="matrix",
        family="heat_capacity",
        expression=operator.expression,
    )


def capacity_operator(temperature, capacity, *, measure=ufl.dx) -> OperatorForm:
    """Create a capacity/storage operator ``C``."""

    return heat_capacity_operator(temperature, capacity, measure=measure)


def mass_action_vector(field, target, coefficient=1.0, *, measure=ufl.dx) -> OperatorForm:
    """Create a vector from a mass-like operator acting on a known field."""

    known_field = field.value if hasattr(field, "value") else field
    test = _test(target)
    return OperatorForm(
        name="F_mass_action",
        kind="mass_action_vector",
        role="vector",
        family="mass_action",
        expression=forms.mass_form(coefficient, known_field, test, measure=measure),
    )


def heat_capacity_vector(previous_temperature, temperature, capacity, *, measure=ufl.dx) -> OperatorForm:
    """Create the known heat-capacity vector ``C * T_previous``."""

    operator = mass_action_vector(previous_temperature, temperature, capacity, measure=measure)
    return OperatorForm(
        name="Q_capacity_history",
        kind="heat_capacity_vector",
        role="vector",
        family="heat_capacity",
        expression=operator.expression,
    )


def body_force_vector(force, test_function, *, measure=ufl.dx) -> OperatorForm:
    """Create a body-force/source vector ``F``."""

    test = _test(test_function)
    return OperatorForm(
        name="F_body",
        kind="body_force_vector",
        role="vector",
        family="body_load",
        expression=forms.body_load_form(force, test, measure=measure),
    )


def source_vector(source, target, *, measure=ufl.dx) -> OperatorForm:
    """Create a scalar or vector source/load vector for a target unknown."""

    return body_force_vector(source, target, measure=measure)


def heat_source_vector(source, temperature, *, measure=ufl.dx) -> OperatorForm:
    """Create a heat-source vector ``Q`` for a temperature unknown."""

    operator = source_vector(source, temperature, measure=measure)
    return OperatorForm(
        name="Q",
        kind="heat_source_vector",
        role="vector",
        family="heat_source",
        expression=operator.expression,
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
            role="vector",
            family="boundary_load",
            expression=load.form(test),
        )
    if measure is None:
        raise ValueError("boundary_load_vector requires measure, location, or load with a measure.")
    return OperatorForm(
        name="F_boundary",
        kind="boundary_load_vector",
        role="vector",
        family="boundary_load",
        expression=forms.boundary_load_form(load, test, measure),
    )


def boundary_force_vector(*, target, value=None, location=None, load=None) -> OperatorForm:
    """Create a boundary force vector from a load object or value/location pair."""

    if load is not None:
        return boundary_load_vector(load=load, target=target)
    return boundary_load_vector(value=value, target=target, location=location)


def boundary_model_vector(boundary_model, velocity, test_function=None) -> OperatorForm:
    """Create a vector contribution from a weak boundary model."""

    test = _test(test_function)
    return OperatorForm(
        name=getattr(boundary_model, "name", "F_boundary_model"),
        kind="boundary_model_vector",
        role="vector",
        family="boundary_model",
        expression=boundary_model.form(velocity, test),
    )


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
        role="vector",
        family="load",
        expression=expression,
    )


def load_vector(target, loads=None, *, load=None) -> OperatorForm:
    """Create a total external-load vector ``F`` for a target unknown."""

    return force_vector(target=target, loads=loads, load=load)


def stiffness(field, properties=None, *, law=None, study=None, measure=ufl.dx) -> OperatorForm:
    """Create the primary stiffness-like operator ``K`` for an unknown field.

    This is the beginner-facing K entry point. When a constitutive ``law`` is
    supplied, the law owns the dispatch. Without a law, AgentFEM currently
    supports displacement fields by creating an elastic stiffness operator.
    """

    if law is not None:
        if hasattr(law, "stiffness_operator"):
            return law.stiffness_operator(field, properties, study=study, measure=measure)
        if callable(law):
            return law(field, properties, study=study, measure=measure)
        raise ValueError("law must be callable or provide stiffness_operator(...).")
    if getattr(field, "kind", None) == "displacement":
        from agentfem.operators.elasticity import elastic_stiffness

        if properties is None:
            raise ValueError("operators.stiffness(displacement, ...) requires material properties.")
        _require_study_physics(study, "solid_mechanics")
        return elastic_stiffness(field, properties, study=study, measure=measure)
    raise ValueError(
        "operators.stiffness currently dispatches displacement fields to elastic stiffness. "
        "Use operators.conduction_operator(...) for scalar diffusion/conduction, or pass law=..."
    )


def _require_study_physics(study, physics: str) -> None:
    if study is not None and hasattr(study, "require"):
        study.require(physics=physics)


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


def _matrix_or_diagonal(operator):
    if isinstance(operator, OperatorForm):
        if operator.role != "matrix":
            raise ValueError(
                f"Expected a matrix operator for algebraic products, got role={operator.role!r}."
            )
        return operator.assemble_matrix()
    if hasattr(operator, "mass") and isinstance(operator.mass, np.ndarray):
        return operator.mass
    if isinstance(operator, np.ndarray):
        return operator
    if isinstance(operator, PETSc.Mat):
        return operator
    raise TypeError(
        "Expected an OperatorForm, PETSc.Mat, lumped diagonal ndarray, "
        "or object with a mass ndarray."
    )


def _expression(operator):
    return operator.expression if isinstance(operator, OperatorForm) else operator


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
