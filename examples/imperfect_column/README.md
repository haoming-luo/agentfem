# Mode-imperfect column under displacement control

Run `agentfem run` in this project with the FEM runtime. Outputs include the final
XDMF field, a CSV load–displacement curve and the standard result manifest.

The left end is clamped. At the right end, axial displacement is uniform while
transverse displacement is free. This suppresses end rotation and is a guided-end
condition, **not** a free-rotation cantilever. Linear buckling uses the same
perturbation constraints. Length 20, depth 1, E=1000, nu=0, plane strain and unit
out-of-plane thickness are used in consistent units. The first buckling mode is
scaled to a length amplitude of 0.02; shortening is 0.05.

`mesh.apply_mode_imperfection` preserves the original coordinates and rejects
folded/poor-quality geometry. Boundary facet tags are captured before perturbing
coordinates and explicitly selected through `mesh.tagged_boundary_region`.
Coordinate predicates such as x==L must not be reused on a moved end face.

The perturbed geometry is stress-free; this is a geometric imperfection, not an
initial displacement loaded with elastic stress. The existing neo-Hookean
finite-deformation step supplies consistent tangents, automatic increment
cutbacks and accepted-state rollback. All accepted states are saved. The new
`mechanics.displacement_controlled_response` reconstructs boundary reactions and
averaged displacements from those states, without changing the final solution.

Teaching prompt: compare amplitudes 0.02 and 0.10, refine nx from 24 to 40 and
maximum increment from 0.10 to 0.05. Explain why a larger imperfection reduces the
reaction at the same end shortening. Distinguish nonlinear load-path response
from a linear bifurcation load, and displacement control from arc-length control.
This example does not establish snap-back or arbitrary limit-point traversal.
