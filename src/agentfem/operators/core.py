"""Operator-level helpers for common FEM matrices and vectors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import ufl
from petsc4py import PETSc

from agentfem import assembly
from agentfem import fields
from agentfem import forms
from agentfem.kernel import dofs


_OPERATOR_ROLES = {"matrix", "vector", "residual", "scalar", "operator"}
_OPERATOR_OPERATIONS = {"primitive", "sum", "scale", "linearize"}
_ROLE_ARITY = {"matrix": 2, "vector": 1, "residual": 1, "scalar": 0}


@dataclass(frozen=True)
class OperatorForm:
    """Named scientific operator with a current backend expression.

    ``expression`` is intentionally still a UFL object in the FEniCSx-first
    implementation.  The remaining fields preserve enough scientific identity
    to inspect composition and to grow a separate semantic operator
    specification without breaking today's executable path.
    """

    name: str
    expression: object
    kind: str
    role: str = "operator"
    family: str = "generic"
    parts: tuple[object, ...] = ()
    operation: str = "primitive"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for attribute in ("name", "kind", "family"):
            value = str(getattr(self, attribute)).strip()
            if not value:
                raise ValueError(f"OperatorForm.{attribute} must not be empty.")
            object.__setattr__(self, attribute, value)
        role = str(self.role).strip().lower()
        if role not in _OPERATOR_ROLES:
            raise ValueError(
                f"Unknown operator role {self.role!r}; "
                f"expected one of {sorted(_OPERATOR_ROLES)}."
            )
        operation = str(self.operation).strip().lower()
        if operation not in _OPERATOR_OPERATIONS:
            raise ValueError(
                f"Unknown operator operation {self.operation!r}; "
                f"expected one of {sorted(_OPERATOR_OPERATIONS)}."
            )
        if self.expression is None:
            raise ValueError("OperatorForm.expression must not be None.")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "parts", tuple(self.parts))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def __add__(self, other):
        """Combine compatible operator forms by adding their expressions."""

        return combine(self, other)

    def __mul__(self, factor):
        """Scale an operator form by a scalar coefficient."""

        return scale(self, factor)

    def __sub__(self, other):
        """Subtract compatible operator forms with visible composition."""

        return combine(self, scale(other, -1.0), name=f"{self.name}_minus")

    def __neg__(self):
        """Return the sign-reversed operator with provenance."""

        return scale(self, -1.0, name=f"minus_{self.name}")

    def __radd__(self, other):
        """Allow ``sum(operators)`` without treating zero as an operator."""

        if isinstance(other, (int, float)) and other == 0:
            return self
        return combine(other, self)

    def __rmul__(self, factor):
        """Scale an operator form by a scalar coefficient."""

        return scale(self, factor)

    def __truediv__(self, factor):
        """Scale an operator form by the inverse of a scalar coefficient."""

        return scale(self, 1.0 / factor)

    def compile(self, *, backend=None):
        """Compile through the selected backend adapter."""

        from agentfem.backends import get_backend

        selected = get_backend() if backend is None else backend
        return selected.compile_form(self.expression)

    def assemble_matrix(self, *, bcs=None, backend=None):
        """Compile and assemble this operator as a matrix."""

        from agentfem.backends import get_backend

        selected = get_backend() if backend is None else backend
        return selected.assemble_matrix(self.expression, bcs=bcs)

    def assemble_vector(self, *, backend=None):
        """Compile and assemble this operator as a vector."""

        from agentfem.backends import get_backend

        selected = get_backend() if backend is None else backend
        return selected.assemble_vector(self.expression)

    def summary(self) -> dict[str, object]:
        """Return a compact operator description."""

        return {
            "name": self.name,
            "kind": self.kind,
            "role": self.role,
            "family": self.family,
            "operation": self.operation,
            "form_arity": form_arity(self.expression),
            "parts": self.parts,
            "metadata": dict(self.metadata),
        }

    def to_ir(self) -> dict[str, object]:
        """Return the serializable scientific portion of this operator."""

        return self.summary()

    def validate(self):
        """Check semantic role against the current weak-form arity."""

        from agentfem.validation import ValidationReport, issue

        issues = []
        actual = form_arity(self.expression)
        expected = _ROLE_ARITY.get(self.role)
        if actual is not None and expected is not None and actual != expected:
            issues.append(
                issue(
                    "AFM-OP-001",
                    f"operator.{self.name}.expression",
                    f"role={self.role!r} expects form arity {expected}, got {actual}.",
                    hint="Use matrix for bilinear forms, vector/residual for linear forms, and scalar for functional forms.",
                    expected_arity=expected,
                    actual_arity=actual,
                )
            )
        if self.operation != "primitive" and not self.parts:
            issues.append(
                issue(
                    "AFM-OP-002",
                    f"operator.{self.name}.parts",
                    f"operation={self.operation!r} has no recorded operands.",
                    hint="Construct derived operators with combine(), scale(), or linearize().",
                )
            )
        return ValidationReport.from_issues(
            issues,
            scope=f"operator:{self.name}",
        )

    def check(self) -> None:
        """Raise an addressable validation error when the operator is invalid."""

        self.validate().raise_if_errors()

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
            operation=self.operation,
            metadata=dict(self.metadata),
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
            parts.append(
                {
                    "relation": "operand",
                    "operator": operator.summary(),
                }
            )
    return OperatorForm(
        name=name,
        kind=kind,
        role=role or "operator",
        family="+".join(families) if families else "generic",
        expression=expression,
        parts=tuple(parts),
        operation="sum",
        metadata={"operand_count": len(operators)},
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
            parts=(
                {
                    "relation": "scaled_operand",
                    "factor": _coefficient_description(factor),
                    "operator": operator.summary(),
                },
            ),
            operation="scale",
            metadata={"source_name": operator.name},
        )
    return OperatorForm(
        name=name or "scaled_operator",
        expression=factor * operator,
        kind=kind or "scaled_operator",
        role="operator",
        family="generic",
        operation="scale",
        metadata={"factor": _coefficient_description(factor)},
    )


def residual_operator(
    expression,
    *,
    name: str = "R",
    family: str = "nonlinear",
    metadata: dict[str, object] | None = None,
) -> OperatorForm:
    """Wrap a nonlinear weak residual ``R(u; v)`` as a public operator."""

    return OperatorForm(
        name=name,
        expression=expression,
        kind="residual_operator",
        role="residual",
        family=family,
        metadata=dict(metadata or {}),
    )


def linearize(
    residual,
    unknown,
    direction=None,
    *,
    name: str = "K_t",
) -> OperatorForm:
    """Differentiate a residual to obtain its consistent tangent operator.

    This is an explicit bridge to UFL automatic differentiation.  AgentFEM
    records the scientific relationship ``K_t = dR/du`` while UFL performs the
    backend symbolic differentiation.
    """

    source = residual if isinstance(residual, OperatorForm) else residual_operator(residual)
    if source.role != "residual":
        raise ValueError(
            f"linearize expects role='residual', got {source.role!r}."
        )
    value = unknown.value if hasattr(unknown, "value") else unknown
    if direction is None:
        if hasattr(unknown, "trial"):
            direction = unknown.trial
        elif hasattr(value, "function_space"):
            direction = ufl.TrialFunction(value.function_space)
        else:
            raise ValueError(
                "linearize requires direction unless unknown exposes trial or function_space."
            )
    tangent = ufl.derivative(source.expression, value, direction)
    return OperatorForm(
        name=name,
        expression=tangent,
        kind="consistent_tangent_operator",
        role="matrix",
        family=source.family,
        parts=(
            {
                "relation": "derivative_of",
                "operator": source.summary(),
                "with_respect_to": getattr(unknown, "name", type(value).__name__),
            },
        ),
        operation="linearize",
        metadata={"equation": "K_t = dR/du", "source_name": source.name},
    )


def form_arity(expression) -> int | None:
    """Return the number of UFL arguments, or ``None`` for opaque backends."""

    if isinstance(expression, OperatorForm):
        expression = expression.expression
    arguments = getattr(expression, "arguments", None)
    if not callable(arguments):
        return None
    try:
        return len(arguments())
    except (TypeError, AttributeError):
        return None


def compile_form(operator: OperatorForm, *, backend=None):
    """Compile an ``OperatorForm`` or raw UFL form."""

    if isinstance(operator, OperatorForm):
        return operator.compile(backend=backend)
    from agentfem.backends import get_backend

    selected = get_backend() if backend is None else backend
    return selected.compile_form(operator)


def assemble_matrix(operator, *, bcs=None, backend=None):
    """Assemble an operator-level matrix from an ``OperatorForm`` or UFL form."""

    if isinstance(operator, OperatorForm):
        return operator.assemble_matrix(bcs=bcs, backend=backend)
    from agentfem.backends import get_backend

    selected = get_backend() if backend is None else backend
    return selected.assemble_matrix(operator, bcs=bcs)


def assemble_vector(operator, *, backend=None):
    """Assemble an operator-level vector from an ``OperatorForm`` or UFL form."""

    if isinstance(operator, OperatorForm):
        return operator.assemble_vector(backend=backend)
    if hasattr(operator, "assemble_vector"):
        return operator.assemble_vector()
    from agentfem.backends import get_backend

    selected = get_backend() if backend is None else backend
    return selected.assemble_vector(operator)


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


def dual_product(vector_operator, field) -> float:
    """Return the global discrete pairing ``field^T vector_operator``."""

    vector = assemble_vector(vector_operator)
    function = fields.unwrap(field)
    try:
        return float(np.real(function.x.petsc_vec.dot(vector)))
    finally:
        vector.destroy()


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


def inertial_force_vector(
    acceleration,
    target,
    density=1.0,
    *,
    measure=ufl.dx,
) -> OperatorForm:
    """Create the inertial virtual-work vector ``F_inertia = M a``."""

    known = acceleration.value if hasattr(acceleration, "value") else acceleration
    test = _test(target)
    return OperatorForm(
        name="F_inertia",
        kind="inertial_force_vector",
        role="vector",
        family="inertia",
        expression=forms.inertial_form(density, known, test, measure=measure),
    )


def flux_vector(
    flux,
    target,
    *,
    measure=None,
    location=None,
) -> OperatorForm:
    """Create a prescribed scalar boundary-flux vector."""

    selected_measure = measure if measure is not None else getattr(location, "measure", None)
    if selected_measure is None:
        raise ValueError("flux_vector requires measure or location.")
    return OperatorForm(
        name="Q_flux",
        kind="boundary_flux_vector",
        role="vector",
        family="boundary_flux",
        expression=forms.scalar_flux_form(flux, _test(target), selected_measure),
    )


def robin_operator(
    target,
    coefficient,
    *,
    measure=None,
    location=None,
) -> OperatorForm:
    """Create the boundary matrix ``K_R = integral(h trial test)``."""

    selected_measure = measure if measure is not None else getattr(location, "measure", None)
    if selected_measure is None:
        raise ValueError("robin_operator requires measure or location.")
    trial, test = _trial_test(target)
    return OperatorForm(
        name="K_robin",
        kind="robin_boundary_operator",
        role="matrix",
        family="robin_boundary",
        expression=forms.robin_form(
            coefficient,
            trial,
            test,
            selected_measure,
        ),
    )


def robin_source_vector(
    target,
    coefficient,
    reference_value,
    *,
    measure=None,
    location=None,
) -> OperatorForm:
    """Create the Robin environment vector ``F_R = integral(h x_ref test)``."""

    selected_measure = measure if measure is not None else getattr(location, "measure", None)
    if selected_measure is None:
        raise ValueError("robin_source_vector requires measure or location.")
    return OperatorForm(
        name="F_robin",
        kind="robin_boundary_source_vector",
        role="vector",
        family="robin_boundary",
        expression=forms.scalar_flux_form(
            coefficient * reference_value,
            _test(target),
            selected_measure,
        ),
    )


def rayleigh_damping(
    mass,
    stiffness,
    *,
    mass_coefficient=0.0,
    stiffness_coefficient=0.0,
) -> OperatorForm:
    """Create proportional damping ``C = alpha M + beta K``."""

    for name, value in (
        ("mass_coefficient", mass_coefficient),
        ("stiffness_coefficient", stiffness_coefficient),
    ):
        if isinstance(value, (int, float)) and value < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    if (
        isinstance(mass_coefficient, (int, float))
        and isinstance(stiffness_coefficient, (int, float))
        and mass_coefficient == 0.0
        and stiffness_coefficient == 0.0
    ):
        raise ValueError("Rayleigh damping requires a non-zero coefficient.")
    combined = combine(
        scale(mass, mass_coefficient, name="alpha_M"),
        scale(stiffness, stiffness_coefficient, name="beta_K"),
        name="C",
        kind="rayleigh_damping_operator",
    )
    return combined.renamed(
        "C",
        kind="rayleigh_damping_operator",
        role="matrix",
        family="rayleigh_damping",
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


def force_vector(target, loads=None, *, load=None, study=None) -> OperatorForm:
    """Create a total force/source vector from one or more load objects."""

    selected = _normalize_loads(loads, load)
    test = _test(target)
    if study is not None:
        from agentfem import _axisymmetric

        test = _axisymmetric.weighted_test(test, study)
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


def load_vector(target, loads=None, *, load=None, study=None) -> OperatorForm:
    """Create a total external-load vector ``F`` for a target unknown."""

    return force_vector(target=target, loads=loads, load=load, study=study)


def stiffness(field, properties=None, *, law=None, study=None, temperature=None, measure=ufl.dx) -> OperatorForm:
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
        return elastic_stiffness(
            field,
            properties,
            study=study,
            temperature=temperature,
            measure=measure,
        )
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


def _coefficient_description(value):
    """Describe a scale coefficient without serializing backend internals."""

    if isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "value"):
        coefficient = getattr(value, "value")
        if hasattr(coefficient, "tolist"):
            coefficient = coefficient.tolist()
        if isinstance(coefficient, (str, bool, int, float, list, tuple)):
            return coefficient
    value_type = type(value)
    return {
        "kind": "backend_coefficient",
        "python_type": f"{value_type.__module__}.{value_type.__qualname__}",
    }


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
