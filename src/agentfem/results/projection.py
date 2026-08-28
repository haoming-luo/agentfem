"""Reusable projection and standard small-strain result fields."""

from __future__ import annotations

import ufl
from dolfinx import fem

from .. import _axisymmetric
from .. import fields as field_api
from ..constitutive import elasticity
from .field_catalog import resolve_field_variables


def project(
    expression,
    *,
    domain=None,
    family: str = "DG",
    degree: int = 0,
    name: str = "ProjectedField",
    weight=1.0,
):
    """Return the global L2 projection of a UFL expression.

    ``DG0`` is the conventional default for element stress and strain output:
    it produces a cell-average field rather than a value at an arbitrary
    visualization point. Higher degrees remain available for users who need a
    richer post-processing space.
    """

    selected_degree = int(degree)
    if selected_degree < 0:
        raise ValueError("Projection degree must be non-negative.")
    selected_domain = domain or _expression_domain(expression)
    if selected_domain is None:
        raise ValueError(
            "Could not infer a mesh from the expression; pass domain=... explicitly."
        )
    return _project_terms(
        ((expression, ufl.dx(domain=selected_domain), weight),),
        domain=selected_domain,
        family=family,
        degree=selected_degree,
        name=name,
        weight=weight,
    )


def project_piecewise(
    terms,
    *,
    domain=None,
    family: str = "DG",
    degree: int = 0,
    name: str = "ProjectedField",
    weight=1.0,
):
    """Project region-dependent expressions into one finite-element field.

    Each term is ``(expression, measure)``.  This is the result-side analogue
    of assembling a multi-material stiffness from regional contributions: one
    mass problem is assembled over the same material partition, avoiding
    singular per-region projection spaces or case-specific field stitching.
    """

    selected_terms = tuple(terms)
    if not selected_terms:
        raise ValueError("project_piecewise requires at least one regional term.")
    selected_degree = int(degree)
    if selected_degree < 0:
        raise ValueError("Projection degree must be non-negative.")
    selected_domain = domain or _expression_domain(selected_terms[0][0])
    if selected_domain is None:
        raise ValueError(
            "Could not infer a mesh from the expressions; pass domain=... explicitly."
        )
    return _project_terms(
        selected_terms,
        domain=selected_domain,
        family=family,
        degree=selected_degree,
        name=name,
        weight=weight,
    )


def _project_terms(terms, *, domain, family, degree, name, weight=1.0):
    shape = tuple(getattr(terms[0][0], "ufl_shape", ()))
    element = (
        (str(family), degree)
        if not shape
        else (str(family), degree, shape)
    )
    space = fem.functionspace(domain, element)
    trial = ufl.TrialFunction(space)
    test = ufl.TestFunction(space)
    output = fem.Function(space, name=str(name))
    lhs = None
    rhs = None
    for term in terms:
        if len(term) == 2:
            expression, measure = term
            term_weight = weight
        elif len(term) == 3:
            expression, measure, term_weight = term
        else:
            raise ValueError("Projection terms must be (expression, measure[, weight]).")
        if tuple(getattr(expression, "ufl_shape", ())) != shape:
            raise ValueError("All piecewise projection expressions need one value shape.")
        lhs_term = term_weight * ufl.inner(trial, test) * measure
        rhs_term = term_weight * ufl.inner(expression, test) * measure
        lhs = lhs_term if lhs is None else lhs + lhs_term
        rhs = rhs_term if rhs is None else rhs + rhs_term
    from ..solvers import LinearSolverOptions, solve_linear_problem

    projected = solve_linear_problem(
        fem.form(lhs),
        fem.form(rhs),
        output,
        options=LinearSolverOptions(
            ksp_type="cg",
            pc_type="jacobi",
            rtol=1.0e-12,
            atol=1.0e-14,
            error_if_not_converged=True,
        ),
    )
    projected.x.scatter_forward()
    return projected


def small_strain_cell_fields(
    displacement,
    properties,
    *,
    study=None,
    variables=("S", "E", "MISES", "SENER"),
    degree: int = 0,
) -> tuple[object, ...]:
    """Create standard projected fields for linear small-strain elasticity.

    The returned fields use the public result-variable names ``S``, ``E``,
    ``MISES``, and ``SENER``. In two-dimensional plane strain, the von Mises
    calculation includes the constitutively implied out-of-plane stress.
    """

    function = field_api.unwrap(displacement)
    domain = function.function_space.mesh
    requested = resolve_field_variables(variables, finite_strain=False)
    expressions = _small_strain_expressions(
        function,
        properties,
        study=study,
        variables=tuple(item.key for item in requested),
    )
    weight = _axisymmetric.integration_weight(function, study)
    fields = []
    for variable in requested:
        if variable.key == "U":
            continue
        if variable.key not in expressions:
            raise NotImplementedError(
                f"Small-strain output does not provide {variable.key!r}."
            )
        fields.append(
            project(
                expressions[variable.key],
                domain=domain,
                family="DG",
                degree=degree,
                name=variable.key,
                weight=weight,
            )
        )
    return tuple(fields)


def small_strain_partition_fields(
    displacement,
    assignments,
    *,
    study=None,
    variables=("S", "E", "MISES", "SENER"),
    degree: int = 0,
) -> tuple[object, ...]:
    """Create standard fields for a complete regional material partition.

    ``assignments`` accepts AgentFEM material records or ``(material, region)``
    pairs.  Every assignment contributes its constitutive expression on its
    own cell measure, producing one coherent output field over the mesh.
    """

    function = field_api.unwrap(displacement)
    domain = function.function_space.mesh
    normalized = tuple(_material_assignment(item) for item in assignments)
    if not normalized:
        raise ValueError("small_strain_partition_fields requires material assignments.")
    if len(normalized) > 1 and any(measure is None for _, measure in normalized):
        raise ValueError(
            "Every material in a multi-material result needs a cell region."
        )
    requested = resolve_field_variables(variables, finite_strain=False)
    weight = _axisymmetric.integration_weight(function, study)
    regional = []
    for properties, measure in normalized:
        regional.append(
            (
                _small_strain_expressions(
                    function,
                    properties,
                    study=study,
                    variables=tuple(item.key for item in requested),
                ),
                ufl.dx(domain=domain) if measure is None else measure,
            )
        )
    fields = []
    for variable in requested:
        if variable.key == "U":
            continue
        terms = []
        for expressions, measure in regional:
            if variable.key not in expressions:
                raise NotImplementedError(
                    f"Small-strain output does not provide {variable.key!r}."
                )
            terms.append((expressions[variable.key], measure, weight))
        fields.append(
            project_piecewise(
                terms,
                domain=domain,
                family="DG",
                degree=degree,
                name=variable.key,
            )
        )
    return tuple(fields)


def _small_strain_expressions(
    function,
    properties,
    *,
    study=None,
    variables=("S", "E", "MISES", "SENER"),
):
    strain = elasticity.strain(function, study=study)
    stress = elasticity.stress(function, properties, study=study)
    expressions = {
        "S": stress,
        "E": strain,
        "SENER": 0.5 * ufl.inner(stress, strain),
    }
    if "MISES" in variables:
        expressions["MISES"] = _von_mises(
            stress,
            strain,
            properties,
            study=study,
        )
    return expressions


def _material_assignment(assignment):
    if hasattr(assignment, "item") and hasattr(assignment, "region"):
        region = assignment.region
        return assignment.item, None if region is None else region.measure
    try:
        properties, location = assignment
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "A material assignment must be an AgentFEM material record or "
            "a (material, region) pair."
        ) from exc
    return properties, getattr(location, "measure", location)


def _von_mises(stress, strain, properties, *, study=None):
    shape = tuple(stress.ufl_shape)
    if shape == (3, 3):
        deviator = stress - ufl.tr(stress) / 3.0 * ufl.Identity(3)
        return ufl.sqrt(1.5 * ufl.inner(deviator, deviator))
    if shape != (2, 2):
        raise ValueError("von Mises output requires a 2D or 3D stress tensor.")
    assumption = getattr(study, "assumption", None)
    if assumption == "plane_strain":
        if not hasattr(properties, "lambda_"):
            raise NotImplementedError(
                "Plane-strain von Mises output requires an isotropic material "
                "with an out-of-plane stress relation."
            )
        stress_zz = properties.lambda_ * ufl.tr(strain)
    else:
        stress_zz = 0.0
    xx = stress[0, 0]
    yy = stress[1, 1]
    xy = stress[0, 1]
    return ufl.sqrt(
        0.5
        * ((xx - yy) ** 2 + (yy - stress_zz) ** 2 + (stress_zz - xx) ** 2)
        + 3.0 * xy**2
    )


def _expression_domain(expression):
    domains = tuple(expression.ufl_domains()) if hasattr(expression, "ufl_domains") else ()
    if len(domains) != 1:
        return None
    return domains[0].ufl_cargo()


__all__ = [
    "project",
    "project_piecewise",
    "small_strain_cell_fields",
    "small_strain_partition_fields",
]
