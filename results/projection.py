"""Reusable projection and standard small-strain result fields."""

from __future__ import annotations

import ufl
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc

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
    shape = tuple(getattr(expression, "ufl_shape", ()))
    element = (
        (str(family), selected_degree)
        if not shape
        else (str(family), selected_degree, shape)
    )
    space = fem.functionspace(selected_domain, element)
    trial = ufl.TrialFunction(space)
    test = ufl.TestFunction(space)
    output = fem.Function(space, name=str(name))
    problem = fem_petsc.LinearProblem(
        ufl.inner(trial, test) * ufl.dx(domain=selected_domain),
        ufl.inner(expression, test) * ufl.dx(domain=selected_domain),
        u=output,
        petsc_options_prefix="agentfem_result_projection_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1.0e-12,
            "ksp_atol": 1.0e-14,
            "ksp_error_if_not_converged": True,
        },
    )
    projected = problem.solve()
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
    strain = elasticity.strain(function)
    stress = elasticity.stress(function, properties, study=study)
    energy = 0.5 * ufl.inner(stress, strain)
    equivalent = _von_mises(stress, strain, properties, study=study)
    expressions = {
        "S": stress,
        "E": strain,
        "MISES": equivalent,
        "SENER": energy,
    }
    fields = []
    for variable in resolve_field_variables(variables, finite_strain=False):
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
            )
        )
    return tuple(fields)


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


__all__ = ["project", "small_strain_cell_fields"]
