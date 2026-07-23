# Operators

Engineering-level namespace for reusable FEM operators such as mass, stiffness,
damping, force vectors, block systems, projection operators, and
preconditioner-ready blocks.

Use this layer when the user thinks in `K x = F` or `M a + C v + K u = F`
instead of weak-form expressions.

Examples:

```python
displacement = fields.displacement(domain, degree=1)
right_traction = loads.traction(
    value=(0.0, -1.0e6),
    location=right_boundary,
)
K = operators.stiffness_operator(displacement, properties, measure=dx)
F = operators.force_vector(
    target=displacement,
    loads=[right_traction],
)
system = operators.LinearSystem(stiffness=K, force=F)
```

## Algebra Layers

AgentFEM separates immediate field algebra from operator algebra.

Field algebra is dof-wise and returns a new field:

```python
u_pred = u + dt * v + 0.5 * dt**2 * a
speed_norm = fields.norm(v)
```

Operator algebra is finite-element matrix/vector algebra and returns vectors
or scalars:

```python
M = operators.mass_operator(displacement, density)
F_mass = operators.action(M, u)
q = operators.quadratic_form(M, u)  # u^T M u
q = operators.xtmx(u, M)            # Cast3M-style alias
q = operators.xtmy(u, M, v)         # u^T M v
```

Use explicit functions for these products instead of overloading `u @ v`; two
fields can represent pointwise products, algebraic dot products, weak-form
integrals, or mass-weighted products, and those meanings should stay visible.

Low-level weak forms remain available in `agentfem.forms`, and assembly remains
available in `agentfem.assembly`.
