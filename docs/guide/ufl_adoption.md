# Reuse working FEniCSx formulations in AgentFEM

A working Python/UFL solver is a reusable numerical asset. Adopt it incrementally:
keep the same mesh, elements, boundary conditions, time scheme and solver first.
Do not replace a successful numerical method merely to use a named operator.

## Adopt a weak form

```python
from agentfem import operators

# a_ufl and L_ufl are the existing integrated UFL forms.
A = operators.from_ufl(a_ufl, name="A", family="my_physics")
F = operators.from_ufl(L_ufl, name="F", family="my_physics")
# A.expression is the original a_ufl object, not an approximation.
```

An integrated form with two arguments is a matrix; with one argument it is a
vector by default. For nonlinear equations explicitly declare a residual:

```python
R = operators.from_ufl(residual_ufl, name="R", role="residual")
Kt = operators.linearize(R, solution_function)
```

The original DOLFINx solve can consume `R.expression` and `Kt.expression`.
This preserves the ordinary Python workflow and supports custom physics before
it has a named AgentFEM factory. It does not automatically translate arbitrary
Python programs, select a physical model or infer missing boundary conditions.

## Replace reusable terms

For a backward-Euler scalar diffusion step:

```python
M = operators.mass_operator(u, v, measure=dx)
K = operators.diffusion_operator(u, v, conductivity=kappa, measure=dx)
H = operators.from_ufl(previous * v * dx, name="history")
Q = operators.source_vector(source, v, measure=dx)
a = (M / dt + K).expression
L = (H / dt + Q).expression
```

Compose the named operators before lowering to UFL. A UFL integrand supports
ordinary division, but an integrated UFL Form does not support `form / dt`;
use `(1 / dt) * form` for raw forms. Do not globally monkey-patch UFL to change
this contract. Do not pass `current - previous` to `ufl.action` as if that sum
were a finite-element coefficient; build the residual from its integrand.

For a lagged Burgers term, preserve the original lagging/time choice and use
`operators.burgers_convection_operator(previous, previous, v, measure=dx)`.
For a nonlinear Burgers residual use the current unknown in the original
formulation. These are distinct numerical schemes, not interchangeable aliases.

## Mesh input

```python
from agentfem import mesh

domain = mesh.from_arrays(
    cells=connectivity,
    coordinates=coordinates,
    coordinate_element=coordinate_element,
    comm=comm,
)
```

Explicit topology/geometry keywords avoid the positional order trap across
DOLFINx versions. Preserve the original cell ordering, coordinate element and
MPI ownership conventions. This helper does not manufacture geometry or
convert arbitrary external cell ordering.

## Numerical migration checks

1. Compare assembled residuals/matrices for the same mesh and fields.
2. Compare solution fields, finite-value masks and required observables.
3. For nonlinear/time-dependent cases, compare the consistent tangent and the
   state/time update; do not change tolerances or the time discretization.
4. Test a changed dimension, parameter or boundary condition before promoting
   a recurring formulation into a generic operator.

The regression tests in `tests/test_ufl_adoption.py` cover 2D/3D Burgers weak
forms, transient operator composition, residual/tangent preservation and mesh
geometry. Repository tests use the declared numerical environment; benchmarks
must also test the exact runtime version they freeze.

References: [DOLFINx 0.10 mesh API](https://docs.fenicsproject.org/dolfinx/v0.10.0/python/generated/dolfinx.mesh.html)
and [UFL form language](https://docs.fenicsproject.org/ufl/main/manual/form_language.html).

## Time-dependent residuals across abstraction levels

For a transient nonlinear problem, construct the time derivative before
integration whenever possible:

```python
residual_ufl = ((current - previous) / dt) * v * dx + spatial_residual_ufl
R = operators.from_ufl(residual_ufl, name="transient_residual", role="residual")
J = operators.linearize(R, current)
```

Here `current` and `previous` are Functions in the same space, `v` is the test
function, `dt` is the positive time increment, and `spatial_residual_ufl` is
an integrated spatial residual with the same test argument. Retain the original
initialization, boundary conditions and accepted-step state transfer.

When an existing solver already holds integrated forms, preserve them:

```python
residual_ufl = (1.0 / dt) * (current_mass_form - previous_mass_form) + spatial_residual_ufl
```

Do not divide an integrated form by `dt`; do not alter a correct time scheme
to work around this Python interface difference. For a matrix `mass_form`,
`ufl.action(mass_form, current) - ufl.action(mass_form, previous)` is valid
when both operands are Functions. Passing `current - previous` as the
coefficient to `ufl.action` is a different operation and is not interchangeable.

Compatibility fixes and formulation migrations should carry a runnable
original case, a minimal source diff, a comparison of fields and observables,
and a changed-input regression. Repaired cases remain development examples;
they do not retroactively change previously frozen benchmark results.

For a complete public thermal workflow, including initial fields, boundary
conditions, time advancement, sampling and output, see
[transient heat workflow](transient_heat_workflow.md). Use raw-form adoption
when the physical or numerical method requires it, rather than reproducing
standard time loops unnecessarily.
