# Large-rotation elastic cantilever

Run with `agentfem run`. A 100 × 1 mm plane-stress StVK solid, E=1e6 and nu=0,
is clamped at the left and loaded vertically at the right by a reference dead
traction. The default nondimensional load PL²/(EI)=2 creates a large rotation
while material strains remain small. Geometry is in mm; force is per unit
reference thickness. Outputs include the actual displacement field and a
nondimensional load–tip displacement CSV.

The independent reference in `validation/geometric_elasticity/run_elastica.py`
solves theta''+alpha*cos(theta)=0 with theta(0)=0 and theta'(1)=0. It is an
inextensible beam approximation, so mesh convergence need not erase the small
physical difference from an extensible 2D solid.
