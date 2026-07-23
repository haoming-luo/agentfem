# Application Tutorial Design

AgentFEM tutorials should match the user's finite-element maturity level.

## Level 1: Application And Operator Tutorials

Use this level for users who understand engineering FEM concepts such as
`K x = F`, `M a + C v + K u = F`, boundary conditions, and material properties,
but may not be comfortable deriving weak forms.

Recommended vocabulary:

- Mesh
- Mesh regions, such as fixed and loaded boundaries
- Unknown field, such as displacement or temperature
- Material properties
- Constraints
- Loads
- `K`, `M`, `C`, `F`
- `LinearSystem` or `SecondOrderSystem`
- Solve or time integration
- Output

Example style:

```python
displacement = fields.displacement(domain, degree=1)
left_boundary = mesh.boundary(domain, left_marker, name="left")
right_boundary = mesh.boundary(domain, right_marker, name="right")
fixed_left = constraints.fixed(
    displacement,
    location=left_boundary,
    value=0.0,
)
right_traction = loads.traction(
    value=(0.0, -1.0e6),
    location=right_boundary,
)
K = operators.stiffness_operator(displacement, properties)
F = operators.force_vector(
    target=displacement,
    loads=[right_traction],
)
system = operators.LinearSystem(stiffness=K, force=F)
problem = problems.LinearSystemProblem(
    system=system,
    unknown=displacement,
    bcs=fixed_left.bcs,
)
problem.solve()
```

Do not start these tutorials with `V`, `du`, `v`, or `sigma : epsilon(v)` unless
the tutorial is explicitly teaching weak forms.

## Level 2: Weak-Form Tutorials

Use this level for researchers who want to inspect, modify, or publish the
finite-element formulation.

Recommended vocabulary:

- Strain
- Stress
- Flux
- Test/trial functions
- Weak forms
- Assembly

Example style:

```python
sigma = elasticity.stress(du, properties)
a = forms.stiffness_form(sigma, elasticity.strain(v))
```

Weak-form tutorials may use `problems.LinearVariationalProblem`. Application
tutorials should prefer `problems.LinearSystemProblem`.

## Level 3: Solver And Algorithm Tutorials

Use this level for users modifying time integrators, nonlinear loops,
preconditioners, coupling algorithms, or parallel behavior.

Recommended vocabulary:

- Residual
- Tangent
- Predictor/corrector
- Lumped operator
- KSP/PC
- Diagnostics
- MPI checks

## Rule

Beginner application tutorials should be written in operator-level language.
Weak-form expressions remain available as the transparent lower layer.
