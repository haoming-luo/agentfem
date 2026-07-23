# Application Tutorial Design

AgentFEM tutorials should match the user's finite-element maturity level.

## Level 1: Application And Operator Tutorials

Use this level for users who understand engineering FEM concepts such as
`K x = F`, `M a + C v + K u = F`, boundary conditions, and material properties,
but may not be comfortable deriving weak forms.

Recommended vocabulary:

- Mesh
- Study context
- Model registry and checks
- Mesh regions, such as fixed and loaded boundaries
- Unknown field, such as displacement or temperature
- Material properties
- Constraints
- Loads
- `K`, `M`, `C`, `F`
- Analysis step, such as `K x = F` or `(C / dt + K) x_next = C x_old / dt + F`
- `SecondOrderSystem` when the tutorial is about dynamics
- Solve or time integration
- Output

Example style:

```python
study = studies.linear_static(
    physics="solid_mechanics",
    dimension=2,
    assumption="plane_strain",
)
model = models.create(study=study, mesh=domain)
displacement = model.field(fields.displacement(domain, degree=1))
left_boundary = mesh.boundary(domain, left_marker, name="left")
right_boundary = mesh.boundary(domain, right_marker, name="right")
model.fix(displacement, on=left_boundary, value=0.0)
model.traction(value=(0.0, -1.0e6), on=right_boundary)
model.material(properties)
step = model.linear_static_step(target=displacement)
step.solve()
print(model.tree())
```

Do not start these tutorials with `V`, `du`, `v`, or `sigma : epsilon(v)` unless
the tutorial is explicitly teaching weak forms.
For 2D solid mechanics, always show whether the study is `plane_strain` or
`plane_stress`, because that changes the elastic constitutive relation.

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
sigma = elasticity.stress(du, properties, study=study)
a = forms.stiffness_form(sigma, elasticity.strain(v))
```

Weak-form tutorials may use `problems.LinearVariationalProblem`. Application
tutorials should prefer model-owned step constructors such as
`model.linear_static_step(...)`. Tutorials that explicitly teach operator
notation may use `problems.linear_static(...)` and
`problems.first_order_transient(...)`.

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
