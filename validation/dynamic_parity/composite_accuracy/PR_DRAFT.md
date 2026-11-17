# PR: Add explicit tetrahedral workflows, convergence evidence and material updates

## Problem and resulting behavior

Explicit dynamics lacked a complete public mesh/Step/result workflow for the
independently derived composite tetrahedron. This branch adds nodal loading,
verified tetrahedral kernels, parallel assembly, consistent loads, restart and
scientific output using the actual piecewise interpolation.

The latest increment adds whole-mesh convergence tests, eliminates repeated
reference integration and inefficient tensor contractions, and connects the
existing small-strain material protocol to the explicit Step. The J2 adapter
reuses the established radial-return law; accepted stress, plastic state, stored
energy and dissipation survive rollback and same-partition restart.

## Review areas

- `elements/composite_tet.py`: interpolation, mass, stabilization and research-only independent interior variables.
- `_step_composite_tet.py`, `_linear_explicit_body.py`: public Step integration and transactions.
- `constitutive/small_strain_explicit.py`, `elements/_composite_material.py`: generic provider bridge and native J2 adapter.
- `results/_composite_tet.py`: exact display interpolation, native coefficients, mesh metadata sent once.
- `validation/dynamic_parity/composite_accuracy/README.md`: convergence, timing and scope.

## Validation

- Related serial regression: 320 passed; 14 MPI-specific skips.
- Final material gate: 10 passed.
- MPI 2: material comparison passed on both ranks.
- MPI 8 over six cells: elastic output and J2 comparison passed, including empty partitions.
- Full-repository Ruff and whitespace checks passed.
- Bending refinement includes regular, distorted and nearly incompressible cases, plus ordinary P1/P2 reference calculations.
- Longitudinal frequency error: 1.417% → 0.122% over tested refinements.
- Public 162-cell/100-step benchmark, three repeats: median 1.320→0.146 s without output; 1.553→0.184 s with XDMF. Complete final displacement maximum difference 1.67e-16. This is a small synthetic workload, not an industrial-scale speedup claim.

## Scope

No C3D10M equivalence is claimed: the released mesh path has a tied centre;
the independent-centre 33-DOF kernel is research-only. Its interior inertia
cannot be dropped through static condensation and relabelled as dynamic parity.
C3D10M imported formulas remain unsupported.

The material route supports homogeneous rate-independent small-strain providers,
ordinary projected Tet10 and a checked elastic tangent envelope. Nonlinear mean-
dilatation stabilization, finite strain, contact, curved geometry and cross-
partition restart are not enabled. Output still gathers field values on root.
Ordinary projection locks severely in the near-incompressible test; unfavorable
evidence is retained alongside improvements.

No blade/application simulations or customer INP/CSV were used. Earlier commits
on this same branch include nodal explicit dynamics and Tet15 work; raw validation
records account for much of the diff size.

## Submission status

Branch: `feat/explicit-dynamic-parity`. Implementation commit: `a1d25051`.
This is a local draft; remote push and GitHub PR creation await explicit user
authorization. Neither the main branch nor client installations were changed.
