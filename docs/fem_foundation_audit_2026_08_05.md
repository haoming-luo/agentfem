# FEM Foundation Audit: 5 August 2026

## CTO verdict

AgentFEM now has a credible **continuum-solid linear elasticity foundation**
for selected laboratory and research workflows. It also has a credible but
deliberately bounded **3D small-strain J2 plasticity procedure**. These are not
the same maturity claim:

- linear elasticity is a general model-first FEM path for 2D plane stress,
  2D plane strain, and 3D solids, including regional materials and MPI;
- plasticity currently means one specific globally integrated procedure:
  rate-independent 3D J2 plasticity with linear isotropic hardening, serial
  quadrature state, consistent Newton tangent, adaptive cutback, and restart.

The product should say exactly that. It should not market the existence of one
J2 route as broad support for arbitrary plasticity.

## Evidence-based maturity matrix

| Foundation | Current evidence | Decision | Important boundary |
| --- | --- | --- | --- |
| Small-strain kinematics | automated 2D and 3D constant-strain fields | ready for supported continuum solids | axisymmetric, beams, shells, and mixed incompressible formulations are absent |
| Linear constitutive response | isotropic plane stress/strain/3D; selected 2D anisotropy | usable | no temperature/field-dependent property tables or general 3D orthotropy workflow |
| Regional materials | regional K/M and a two-material series-bar field test | usable | result projection assumes a complete nonoverlapping material partition |
| Essential conditions | fixed, prescribed, component, clamp, symmetry, roller, and periodic routes | usable for stated geometry | arbitrary inclined symmetry, contact, and general constraint reactions remain |
| Natural loads | traction, pressure, body force, gravity, and amplitudes | usable | concentrated nodal force/moment needs a deliberate discrete-load contract |
| Linear solution | PETSc direct/iterative configuration and MPI | usable on current scales | scalable 3D elasticity still needs an automatic rigid-body near-nullspace preset |
| Static results | one-call U/S/E/MISES XDMF/HDF5, opt-in SENER, named-boundary RF, energy helpers | usable | integration-point export, reviewed smoothing/recovery, weak/MPC/contact reaction and prescribed-motion work need separate definitions |
| J2 material update | yield-surface, complete cyclic path, tangent derivative tests, traceable quadrature S/PE/PEEQ/MISES | verified material point | linear isotropic hardening only |
| Global J2 | one- and multi-element 3D patches, Abaqus state comparison, work/energy, cutback, restart | FEM-integrated foundation | serial, one material, small strain; no plane stress or finite-strain plasticity |
| External verification | Abaqus homogeneous J2 state and fixed-mesh elastic release regression | partial | external structural geometry and mesh-convergence contracts remain urgent |

This audit follows the verification hierarchy used by mature finite-element
codes: constant-strain patch tests, exact load/reaction checks, material-point
paths, multi-element consumption, and external reference problems are distinct
obligations. Passing one does not substitute for the others.

## Closed in this implementation pass

1. Linear static steps now accept pure displacement control without requiring
   users to invent a zero load.
2. `AnalysisStep.solve_result(output=...)` writes the final static result in
   the same one-call style as transient procedures.
3. Model-generated isotropic elastic steps automatically expose `U`, `S`,
   `E`, and `MISES`; `SENER` is available as an explicit diagnostic request.
4. Regional materials are projected through one piecewise global mass problem,
   rather than singular per-region projections or ad hoc array stitching.
5. `results.reaction_resultant(..., on=..., component=...)` extracts a named
   strong-boundary reaction in serial and MPI.
6. Automated contracts now cover a displacement-controlled 3D patch, a
   two-material series bar, distributed boundary reactions, and a
   multi-element global J2 path.

## Next execution priorities

### Release-critical depth

1. Add a published linear-elastic structural benchmark, preferably NAFEMS
   LE10, with a mesh-convergence sequence and the corrected target stress
   location—not only a fixed-mesh displacement Golden.
2. Add a 3D elasticity iterative-solver preset that constructs translational
   and rotational near-nullspace modes for PETSc GAMG/Hypre-class workflows.
3. Add explicit equilibrium histories by named support/load region and define
   nonzero prescribed-displacement work for ordinary linear static steps.
4. Exercise triangle, quadrilateral, tetrahedral, and hexahedral families in a
   compact element/degree verification matrix, including distorted patches.

### Plasticity promotion

1. Add tabulated isotropic hardening before adding more yield surfaces; it is
   the most common bridge from test data to engineering J2 analysis.
2. Add kinematic/combined hardening for genuine cyclic response, with a
   published hysteresis benchmark rather than only a class implementation.
3. Generalize global state ownership to multiple material regions without
   multiplying public Step types.
4. Establish stable global cell/quadrature identities, then enable MPI J2 and
   restart with a different process count.
5. Add a nonuniform structural benchmark—such as a notched specimen or plate
   with a hole—covering localization pattern, load-displacement response,
   reaction work, mesh sensitivity, cutback, and restart equivalence.

### Important, but not before those gates

- axisymmetric elasticity and plasticity;
- mixed displacement-pressure treatment near incompressibility;
- beam/shell libraries and rotations;
- contact and friction;
- finite-strain plasticity and general UMAT execution.

The product principle remains depth before vocabulary: each promoted model
must carry a readable top-level workflow, formula, field/state contract,
failure path, benchmark, output semantics, and at least one independent
consumer.

## Primary references used for this audit

- [Abaqus element verification and patch-test rationale](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEVERRefMap/simaver-c-basicelemover.htm)
- [Abaqus static stress analysis, loads, boundary conditions, and output](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEANLRefMap/simaanl-c-static.htm)
- [Abaqus plasticity-model foundations](https://docs.software.vt.edu/abaqusv2024/English/SIMACAETHERefMap/simathe-c-plastoverview.htm)
- [DOLFINx 3D elasticity and algebraic multigrid demo](https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_elasticity.html)
- [NAFEMS standard benchmark collection](https://www.nafems.org/publications/resource_center/p18/)
