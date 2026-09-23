# Initial and prescribed fields through one expression interface

Use `expressions.interpolate(target, source, parameters=...)` for declared
scalar, vector or tensor values. It evaluates constants and coordinate/time
expressions directly at interpolation points, avoiding the need to construct
a domain-bearing `fem.Expression` for a plain number.

```python
from agentfem import expressions
import numpy as np

expressions.interpolate(temperature, 300.0)
expressions.interpolate(temperature, "300 + sin(pi*x)*exp(-t)", parameters={"t": 0.2})
expressions.interpolate(velocity, np.array(["x+t", "0"]), parameters={"t": 0.2})
expressions.interpolate(tensor, [["x", 0], [0, "y"]])
```

The vector example requires a two-component target and the tensor example a
2-by-2 target. Lists and NumPy arrays must match the target's value shape;
components are not silently broadcast or flattened across incompatible shapes.
For 3D vector fields provide three components. `pi` is a built-in constant.

Interpolation assigns the current values. Updating `parameters` later does
not update the field automatically; call interpolate again at the required
time. For time-dependent boundary conditions in a standard Study prefer the
model's existing amplitude/boundary facilities, which own their update cadence.

This interface initializes or prescribes known fields. For an unknown in a
nonlinear residual use its symbolic finite-element Function, preserving the
dependency needed for the tangent. Do not replace a symbolic unknown with a
sampled array during residual construction.
