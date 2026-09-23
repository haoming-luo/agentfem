# High-order mixed-field interpolation

Use `spaces.independent_subspace(W, component)` for standalone boundary fields
and output fields on a mixed space. It preserves the component finite element
but creates independent degree-of-freedom numbering. Transfer values by
interpolation, never by copying coefficient arrays between these spaces.

```python
from dolfinx import fem
from agentfem import spaces

V = spaces.independent_subspace(W, 0)
g = fem.Function(V)
g.interpolate(boundary_values)
boundary_dofs = fem.locate_dofs_topological((W.sub(0), V), facet_dim, facets)
bc = fem.dirichletbc(g, boundary_dofs, W.sub(0))

velocity = fem.Function(V)
velocity.interpolate(solution.sub(0))
velocity.x.scatter_forward()
```

`VelocityPressureUnknown.collapsed_velocity()` and `.collapsed_pressure()`
perform this physical interpolation automatically. The mixed solid unknown's
corresponding methods follow the same route. Their returned fields retain the
same element degree and physical value shape; coefficient ordering is not a
public interchange format.

In a polygon-hole Stokes development replay, creating boundary and output
fields through raw collapsed mixed spaces produced approximately 3% error
with a fourth-order velocity element. Independent spaces with explicit field
interpolation reduced this to 2.30e-7 relative error on the same mesh. The
failure also reproduced in a local DOLFINx 0.11 environment. This observation
does not establish that all collapse operations are incorrect, or that all
backends share the same defect. It motivates the explicit transfer route and
polynomial reproduction regressions in two and three dimensions.

The original benchmark output is preserved. This is a post-development
numerical repair, not a revised first-attempt model score.
