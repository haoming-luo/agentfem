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

Low-level weak forms remain available in `agentfem.forms`, and assembly remains
available in `agentfem.assembly`.
