# Solid shallow arch with spherical arc-length continuation

Run `agentfem run` in this folder, in serial. The public workflow is
`model.step(procedure=procedures.arc_length(), arc_options=..., increments=...)`.
The case supplies geometry, material and boundary loading only; no continuation
or nonlinear solver is implemented inside the example.

The plane-strain neo-Hookean arch has span 2, rise 0.2, thickness 0.04,
E=1000 and nu=0. Both ends are clamped. A fixed downward reference traction is
applied on the crown patch x in [-0.1875,0.1875]. The multiplier is solved as an
unknown, not prescribed as increasing time. Defaults use 32x2 quadrilaterals,
quadratic displacement, arc increment 0.04 and 80 accepted increments.

The response passes through an upper and a lower load limit point. Output is
`arch.xdmf`, `load_displacement.csv` and the result manifest. The descending
branch is a static equilibrium path, not a prediction of the dynamic jump.

Teaching check: refine nx=32 to 64 and arc increment 0.04 to 0.02 (160
increments), retaining the same physical load patch. Compare curves at common
crown displacements. Do not compare loads at identical step indices. There is no
claim of branch switching, contact, plasticity, distributed execution or universal
convergence. The optional SciPy sparse adapter is a bounded reference route.
