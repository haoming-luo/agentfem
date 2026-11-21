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

- `elements/composite_tet.py`: interpolation, mass, stabilization and independent interior variables with retained whole-mesh inertia.
- `_step_composite_tet.py`, `_linear_explicit_body.py`: public Step integration and transactions.
- `constitutive/small_strain_explicit.py`, `elements/_composite_material.py`: generic provider bridge and native J2 adapter.
- `results/_composite_tet.py`: exact display interpolation, native coefficients, mesh metadata sent once.
- `validation/dynamic_parity/composite_accuracy/README.md`: convergence, timing and scope.

## Nonlinear independent-centre extension

The generic material bridge now integrates the combined projected/deviatoric
strain at 48 positive-weight points for stabilized branches. Elastic limits
recover the existing stiffness. Plastic yielding updates the centre response
through the material law instead of retaining an artificial elastic correction.
Point histories, weighted output, rollback, restart and explicit tangent-envelope
checks use the same workflow. Ordinary tied-centre integration remains four-point.

## Validation

- Nonlinear-centre regression: 266 passed.
- Eight-rank nonlinear comparison and single-rank late-failure recovery: five passed per rank.
- External nonlinear material energy derivative, anisotropic elastic limit, centre-only yielding and cyclic plastic history verified.

- Latest independent-centre/material acceptance: 243 passed.
- Independent-centre MPI 8 comparison passed, including empty partitions.
- Forty-step cyclic plastic loading compared against native material-point updates.

- Related serial regression: 320 passed; 14 MPI-specific skips.
- Final material gate: 10 passed.
- MPI 2: material comparison passed on both ranks.
- MPI 8 over six cells: elastic output and J2 comparison passed, including empty partitions.
- Full-repository Ruff and whitespace checks passed.
- Bending refinement includes regular, distorted and nearly incompressible cases, plus ordinary P1/P2 reference calculations.
- Longitudinal frequency error: 1.417% → 0.122% over tested refinements.
- Public 162-cell/100-step benchmark, three repeats: median 1.320→0.146 s without output; 1.553→0.184 s with XDMF. Complete final displacement maximum difference 1.67e-16. This is a small synthetic workload, not an industrial-scale speedup claim.

## Scope

No C3D10M equivalence is claimed. The elastic mesh path supports both a tied
centre and an independent centre with all 33 element DOFs retained. Its interior inertia
cannot be dropped through static condensation and relabelled as dynamic parity.
C3D10M imported formulas remain unsupported.

The material route supports homogeneous rate-independent small-strain providers,
a checked elastic tangent envelope, and tied/independent centre with optional
mean dilatation. Stabilized nonlinear updates use 48 work-conjugate material
points. Finite strain, contact, curved geometry and cross-partition restart
are not enabled. Output still gathers field values on root.
Ordinary projection locks severely in the near-incompressible test; unfavorable
evidence is retained alongside improvements.

No blade/application simulations or customer INP/CSV were used. Earlier commits
on this same branch include nodal explicit dynamics and Tet15 work; raw validation
records account for much of the diff size.

## Submission status

Branch: `feat/explicit-dynamic-parity`. Implementation commit: `a1d25051`.
Draft PR: https://github.com/haoming-luo/agentfem/pull/195.
The main branch and client installations are unchanged.
