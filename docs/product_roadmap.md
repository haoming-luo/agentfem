# AgentFEM Product Roadmap

## Product Position

AgentFEM should first become an unusually clear, dependable, and extensible
finite-element tool on FEniCSx for a bounded set of real engineering problems.
Human readability and agent operability are product qualities of the same
public Python language; they are not reasons to postpone solver depth.

The near-term target is not feature-count parity with Abaqus or ANSYS. It is to
be better within selected workflows:

- less ceremony from model definition to a trustworthy result;
- engineering vocabulary and visible `K/M/C/F` or residual structure;
- direct escape hatches to UFL, DOLFINx, and PETSc;
- batch simulation and learning-data export as ordinary operations;
- explicit capability limits instead of silent approximations;
- benchmarks attached to every serious material/solver claim.

AF-IR remains an experimental record and research track. It must earn a larger
role through executable consumers and real cases. The Python product, numerical
core, results, verification, and documentation have priority.

## Current Capability Boundary

| Capability | Maturity | What is usable now | Important limit |
| --- | --- | --- | --- |
| Scientific operator layer | FEM-integrated foundation | K/M/C/F, static/first-/second-order systems, R/K_t linearization, composition and UFL role/arity validation | no mixed/block domain-range typing or physical-unit algebra |
| Linear elasticity | FEM-integrated | 2D plane stress/strain and 3D isotropic solids; regional materials; displacement-only steps; one-call U/S/E/MISES output with opt-in SENER and explicit processing metadata; named-boundary reactions; automatic assembled external-force/strong-reaction equilibrium evidence; serial/MPI patch evidence | external structural convergence, integration-point export/recovery, axisymmetry, mixed incompressibility, beams/shells, and affine/weak reactions remain |
| Thermoelasticity | FEM-integrated | steady/implicit-transient heat transfer, regional multi-material conductivity and capacity, amplitude-driven sources/ambient conditions, and sequential isotropic thermal stress in 2D/3D | no property tables or monolithic two-way coupling |
| Structural dynamics | FEM-integrated foundation | central difference, Newmark, and generalized-alpha through `dynamic_solid`; model-owned constraints and amplitude loads enter both Standard and Explicit paths; shared field output, mechanical-energy histories, integrity-checked pause/checkpoint/restart, truthful continuation output, and opt-in nodal checkpoint portability across MPI rank counts | implicit route is linear; portable NPZ is root-gathered and stateful quadrature portability is separate |
| Neo-Hookean solids | FEM-integrated | compressible displacement route plus monolithic P2/DG0 constant-pressure mixed route; 3D, 2D plane strain, and locally condensed finite-strain plane stress; automatic/fixed increments, consistent Newton tangent, cutback/rollback, positive-J acceptance; topology-preserving C3D10-to-C3D10H derivation; serial mixed affine-periodic solve, large-mesh Golden contract, and homogeneous affine plane-stress/thin-3D FEM cross-check | one material; distributed mixed affine-periodic MPC, general explicit locking control, full thin-3D fracture, Cook convergence suite, and independent external-code comparison remain |
| Dynamic cohesive fracture | experimental V4 mechanism + V5 evidence foundation | finite-strain plane-stress Explicit, prestrain transfer, fixed-path Mode-I cohesive transaction, complete energy ledger, V0--V3 guardrails, crack/supershear/spall V4 mechanism ladder, physical-length refinement contract, portable interface trace, multi-observer crack fronts, live SENER/KED/J saved fields, pinned Science Dryad manifest, affine publication-coordinate registration, sealed evidence bundle, cell-partition interface recovery, sparse physical-key MPI trace/force assembly, and two-rank-to-one-rank cohesive Explicit restart | V4 front speed remains above the ten-percent spatial gate; direct imported/3D interfaces, extreme-scale MPI profiling, full thin-3D fracture/general incompressibility, and independent V5 research comparison remain |
| J2 plasticity | FEM-integrated foundation | 3D shared quadrature transaction, analytical tangent, natural/displacement loading, non-monotone tabular amplitude, global Newton, physical-increment cutback, cumulative serial restart, traceable quadrature S/PE/PEEQ/MISES plus weighted DG0 recovery and nodal RF results, prescribed-work/energy histories, analytical and Abaqus states, homogeneous multi-element evidence, and nonuniform bending regression | linear isotropic hardening, serial single-material small strain only; no plane stress, multi-region driver, MPI-portable quadrature restart, or nonuniform external structural benchmark |
| Creep and creep damage | FEM-integrated power-law foundation + local damage assessment | 3D power-law backward Euler with shared quadrature state, analytical tangent, automatic physical-time cutback, CE/CEEQ/S/MISES/RF plus weighted DG0 recovery, dissipation and serial restart; scalar/field Arrhenius temperature consumed at quadrature points; official Abaqus constant-stress external contract; local K-R, Sinh, modified-theta and hot-wall assessment | global route is serial/single-material; no automatic transient thermal-history transfer, external component validation, or damage regularization |
| Stress-life fatigue | postprocessor | Basquin/tabulated S-N, rainflow, Goodman, Miner assessment from named result histories | no multiaxial critical-plane method |
| External CAE mesh | integrated Abaqus path + conversion interface | generic meshio conversion; SHA-256 conversion identity/cache invalidation; Abaqus node labels; NSET node regions; exterior SURFACE facet reconstruction for C3D4/C3D10/C3D8 families; verified C3D10 import; C3D10H mixed-pressure provider; linear equation parsing; simplex quality preflight | assembly-instance label scope, free/internal surfaces, more element families, mixed-topology solve domains, Jacobian quality for tensor-product cells, and full solver-deck semantics remain |
| Abaqus periodic equations | serial + two-rank displacement; serial mixed | exact chained affine elimination, distributed displacement `dolfinx_mpc`, 3D Neo-Hookean load path, and mixed P2/DG0 serial reduction that leaves pressure dofs independent | distributed mixed-space MPC, AMG near-nullspace transfer, reactions, and scaling studies remain |
| Abaqus user-material bridge | interface contract | solver-neutral material-point input/output and migration specification | no compiled adapter or quadrature-state global driver |
| Result/data flow | integrated foundation | declarative field/history/diagnostic/presentation plans; shared accepted-increment history and probe requests across heat, Standard, and Explicit procedures; serial compact single-grid XDMF/HDF5 plus collective MPI single-dataset PVD/PVTU presentation carrying point and cell fields; engineering-default U/S/E/MISES; explicit weighted integration-point-to-DG0 recovery; one structured event trace; atomic checkpoint cadence/retention and opt-in cross-rank nodal restart; strong-BC resultants, nonzero prescribed-motion work and energy closure; verification reports and trust-gated learning bridge | compact MPI single-grid/VTKHDF, direct quadrature export, smooth material-domain nodal recovery, affine/weak reactions, stateful quadrature portability, and broader conservation balances remain |
| Scientific trust and provenance | integrated foundation | computed/converged/verified/validated vocabulary; exploratory/engineering/release policies; automatic runtime checks; explicit claims and applicability domains; coarse-to-fine convergence evidence; automatically sealed result manifests and artifact hashes; attested tagged distributions; learning-data quality gates; orientation metamorphic regression | optional signed result identities, representative-family evidence inheritance, hole-stress and T-stiffener cliff families, GCI, and external-deck reproductions remain |
| Campaign-to-learning flow | workflow integrated | deterministic cases, resumable evidence, failure-aware dataset gate, reproducible train/validation workflow, ridge/POD/PyTorch adapters, applicability guard and FEM fallback; MPI-safe structured observation grids with units, layout, and geometry masks | no graph/basis field encoder, scheduler executor, active-learning governance, or calibrated epistemic uncertainty |
| Platform/install boundary | release foundation | Linux CI, macOS developer verification, WSL2 recommended for Windows, exact interpreter/import/distribution identity, versioned project schema, source-aware upgrade reports, Gmsh/meshio optional adapters | native Windows remains experimental; semantic Python migrations require human or agent review; AgentFEM is not yet a conda-forge package |
| Open-core extension boundary | integrated foundation | lazy Python entry-point discovery, explicit activation, API compatibility, staged provider/backend/material registration, project requirements, CLI inventory, and execution provenance | no arbitrary hook bus; new registration kinds require a stable public consumer and conflict semantics |

The same table is queryable in code through
`constitutive.capabilities()` and `benchmarks.list_benchmarks()`.

## Release Gates

A public solver/material capability advances through these levels:

1. **formula implemented** — typed parameters and declared assumptions;
2. **material point verified** — analytical/invariant checks and load paths;
3. **finite-element integrated** — state storage, tangent, nonlinear/time step,
   convergence evidence, and field output;
4. **benchmark verified** — mesh/time convergence and an external reference;
5. **workflow ready** — readable example, failure cases, MPI/output behavior,
   and user documentation.

A name in `constitutive/` does not imply level 3. The maturity catalog prevents
an agent, user, or README from confusing these levels.

## Delivery Sequence

### P0: harden the usable core

- make the installed product shell (`doctor/init/check/run/inspect`) pass a
  wheel-only empty-directory workflow, with versioned project, execution, and
  result contracts shared by humans, GUIs, and agents;
- preserve old-project operability through an independent project schema,
  stable upgrade diagnostic codes, dry-run JSON plans, and automatic changes
  limited to deterministic metadata;
- one `SimulationResult` contract and one structured execution-event stream
  for linear, nonlinear, and transient steps;
- standard QoIs: integrals, averages, norms, extrema, reactions, energies, and
  histories;
- attach global assembled load, strong reaction, force-balance residual, and
  relative equilibrium error to ordinary linear-static solid results;
- compact unified XDMF/HDF5 visualization and output manifests;
- automatically bind every published result manifest to its registered
  artifacts with a local provenance seal and one machine-readable verification
  command; keep optional authorship signatures as a later compatible layer;
- JSON-configured and Python-configured campaigns producing the same dataset;
- serial, MPI, docs, package, and example release gates;
- keep private workflow/material products in independent distributions using
  the explicit extension contract rather than long-lived core branches;
- clear solver convergence/failure evidence;
- reject Study/material/procedure combinations during model validation when no
  registered executable provider can consume them;
- share reusable time amplitudes across loads, prescribed data, and thermal
  boundary models, with automatic updates inside transient procedures.
- define amplitude coordinates once across single-solve static, normalized
  nonlinear static, and physical-time transient procedures; named histories
  must resolve identically for loads and prescribed values;
- make project execution fail collectively when any MPI rank fails, preserving
  rank-addressable evidence rather than hanging at a completion barrier;
- keep execution status distinct from scientific trust; release and training
  data may require explicit verification claims rather than successful exit;
- expand the `CAE Reliability Cliff Suite` from the automated orientation
  case to a perforated-plate resolution sweep and a beam/shell/solid
  theory-applicability family.

First-release closure additionally requires truthful installation commands,
an inspectable runtime/platform report, optional-dependency license boundaries,
operator/system contract checks, and a failure-aware campaign-to-learning
gate. Native Windows is not promoted until a compatible solver route passes an
installed-wheel Windows CI matrix; WSL2 is the recommended Windows route for
the first release.

### P1: nonlinear solid mechanics

- harden the implemented `SolutionProcedure` separation across validation,
  provider dispatch, result summaries, and future nonlinear/transient methods;
- build on the shared heat/Standard/Explicit `solve_result(output=...)`
  lifecycle with energy histories and restartable procedure state; field
  artifacts and accepted time increments are already unified;
- extend the minimal verified Newmark starter to larger meshes, MPI, and a
  tested transient XDMF/HDF5 lifecycle; the current macOS product smoke remains
  deliberately small after exposing a native PETSc/MPI failure at a larger
  starter size;
- extend the implemented ordinary Neo-Hookean automatic/fixed load path,
  forced-cutback rollback, positive-J acceptance, strain-energy evidence, and
  accepted-increment history to multi-region ownership and external
  load-path benchmarks;
- harden the implemented stateless periodic-cell automatic incrementation with
  forced-cutback regression cases and homogenized tangent checks; the serial
  affine and distributed `dolfinx_mpc` paths already share one public Newton
  policy and output contract;
- extend the implemented quadrature-state transaction, 3D J2 analytical
  tangent, analytical uniaxial Golden path, physical-increment forced cutback,
  cyclic tabular amplitude, reaction/work/energy history, cumulative serial
  restart, stable state identity, and published Abaqus homogeneous uniaxial
  verification to multi-region ownership, projected visualization fields,
  cross-partition MPI restart, and full external-deck reproduction;
- retain the implemented nonuniform 3D bending regression for partial yielding, state
  recovery, prescribed work, and energy closure; use the official Abaqus
  notched-beam case as the leading external candidate, but claim equivalence
  only after a 3D-extruded monotonic/isotropic subset or verified plane-strain
  return map matches its geometry, mesh, loading, and reported response;
- add reaction, internal/external work, and energy-balance histories with
  verified strong, weak, and affine-MPC definitions; proportional nonzero
  strong-Dirichlet work is implemented, while weak and affine duals remain;
- extend the implemented C3D10H constant-pressure mixed/hybrid procedure and
  serial periodic-MPC workflow to distributed mixed spaces, then add Cook's
  membrane convergence, locking diagnostics, and an independent external-code
  element path; continue to keep formulation identity separate from `tetra10`;
- then add tabulated hardening, kinematic hardening, and finite-strain
  plasticity only when driven by real applications.

For Abaqus material migration, implement UHYPER energy adaptation before the
more general UMAT path. UMAT requires quadrature state, trial/commit/rollback,
tensor and rotation conventions, compiler/ABI handling, and a consistent
tangent. Advance compatibility one restricted subroutine class at a time,
gated by material-point, one-element, and load-path comparisons. The public
AgentFEM model language should consume a neutral material-point protocol rather
than depend on Abaqus interfaces directly.

### P2: time-dependent materials and life

- harden the implemented global power-law step with time-step
  convergence, multi-element paths, natural-load work/energy balance, and an
  external benchmark;
- transfer accepted transient temperature histories into creep increments
  without hiding interpolation or time alignment; keep Sinh and K-R as
  separate material consumers rather than one flag-heavy solver;
- treat sequential temperature-to-creep as the first useful power-component
  route; add monolithic coupling only for cases with material heat generation
  or meaningful mechanical feedback;
- creep/relaxation single-element verification followed by NAFEMS cases;
- named result histories feed an auditable fatigue assessment now; add
  automatic stress extraction at named regions/points and fatigue fields;
- multiaxial fatigue only after a chosen engineering criterion and reference
  dataset are explicit.

The first global creep promotion has the following gate status:

1. **implemented:** the public step consumes `QuadratureTransaction`; no second
   private state store or copy-only rollback is used;
2. **implemented except an explicit local error estimator:** backward Euler
   returns stress, state, convergence evidence, local iterations, and an
   analytical algorithmic consistent tangent;
3. **implemented for power-law flow:** global/local failure or excessive CEEQ
   increment causes atomic rollback and deterministic cutback; damage remains
   outside the global driver;
4. **implemented in serial:** restart retains physical time, next increment,
   displacement, CE/CEEQ, temperature identity, energy/dissipation, events,
   and schema;
5. **partially implemented:** constant-stress material checks, one-element
   relaxation, consistent tangent, forced cutback, Golden observables, restart
   equivalence, and the official Abaqus held-stress case pass; time-step
   convergence, multi-element nonuniform paths, and an external component case
   remain;
6. **method decision retained:** automate transient thermal-history transfer
   next; add K-R/Liu--Murakami damage only after near-failure control and
   mesh-dependence policy are explicit.

### Shared transient and MPI state identity

The transient checkpoint envelope and portable quadrature identity are common
infrastructure, not J2 or creep features. A portable state is keyed by source
mesh fingerprint, stable global cell identity, quadrature-rule identity,
point number, material-region identity, state-layout schema, and physical
step coordinate. Acceptance requires:

- restart with a different MPI partition/process count reproduces global
  fields and material histories within declared tolerances;
- owned and ghost quadrature points are neither duplicated nor lost;
- incompatible mesh, quadrature, material, amplitude, or schema fingerprints
  fail before state is applied;
- Standard dynamics, Explicit dynamics, and heat use one checkpoint manifest
  envelope while retaining procedure-specific integrator history;
- energy components remain typed by procedure instead of being collapsed into
  one ambiguous scalar.

This work is urgent after the first release but must not be advertised from a
rank-local array serialization prototype. The first nodal-state slice is now
implemented: an opt-in physical-node-keyed NPZ written with two MPI ranks is
continued on one rank and checked against an uninterrupted reference.
Coincident independent nodes use durable source-node identity, and
physical-facet-keyed cohesive history follows the same two-rank-to-one-rank
continuation test. This is a laboratory-scale bridge, not the final collective
HDF5 path. The cell, quadrature-point, material-region, and state-layout keys
needed by J2/creep remain the next identity gate.

### Experimental finite-strain dynamic fracture

The first target is deliberately fixed-path rather than a generic fracture
framework: prestrained compressible Neo-Hookean dynamics plus a zero-thickness
Mode-I cohesive interface.  The current foundation includes auditable mesh
splitting, automatic independent interface-DOF recovery, irreversible
bilinear cohesive state, Total-Lagrangian central difference, stable-step
screening, preload transfer, crack observations, restart state, a complete
strong-Dirichlet/natural-load energy ledger, and an analytical small-on-large
wave oracle.  All remain experimental.

The first four promotion gates now have executable experimental evidence:

1. V1 finite-element arrival times under several homogeneous prestrains,
   checked against the acoustic tensor and mesh refinement;
2. V2 no-fracture and one-interface energy convergence, including smooth
   prescribed separation and exact cohesive dissipation;
3. V3 a classical sub-Rayleigh cohesive crack guardrail before any supershear
   exploration, repeated across mesh, time-step, and declared damping changes.
4. V4 a near-incompressible plane-stress, preloaded weak-interface mechanism
   ladder separating crack-like propagation, a resolved `c_s < v < c_d`
   front, and distributed spall-like separation under smooth remote impact.
5. an opt-in V4 two-dimensional refinement contract. Supershear and energy
   closure persist across 30x10, 40x14, and 60x20 meshes and a halved time
   increment. With a fixed physical fitting length, successive spatial speed
   changes decrease from 13.02% to 10.73% but remain above the 10% gate.
   Mechanism preservation therefore passes while fitted-speed convergence
   remains explicitly false.

These are scoped named benchmarks, not universal validation. V1--V3 remain
the compressible plane-strain guardrails; V4 is an experimental 2D membrane
mechanism gate. The next promotion work is:

6. diagnose the remaining front-speed sensitivity using cohesive-zone
   resolution and an alternative crack-tip observable, then add an impact-history
   family, loaded-interface wave reference, and full thin-3D fracture
   counterpart. The principal traction-free prestrained surface-wave secular
   oracle and homogeneous affine plane-stress/thin-3D FEM cross-check are
   implemented and independently checked;
7. direct Abaqus/Gmsh internal-surface ingestion. Ordered physical interface
   identity, cell-partition edge recovery, balanced facet ownership, sparse
   owner-scheduled trace/force exchange, globally reduced energy,
   coincident-node portable fields, and two-rank-to-one-rank Explicit
   continuation are implemented. Source ELSET/SURFACE and physical-group
   adapters, 3D surface pairing, and profiled neighborhood collectives remain;
8. author-deck/parameter acquisition for curve-level JMPS 2025 reproduction,
   followed by separated calibration and retained Science 2023 Dryad
   prediction cases.

The governing decisions and evidence boundaries live in
`docs/dynamic_cohesive_fracture_architecture.md`.  Phase field, free crack
paths, branching, and general contact do not enter this sequence early.

### P3: mesh and model interoperability

- verify Abaqus `.inp`, Nastran bulk-data, Gmsh, Exodus, and MED meshes;
- map volume sets and boundary sets to named AgentFEM regions;
- preserve source identities, checksums, conversion choices, and warnings
  (implemented for the Abaqus/XDMF cache path);
- extend the implemented NSET node-region and exterior element-face adapters
  beyond C3D4/C3D10/C3D8 to assembly/instance label scopes, automatically
  generated free surfaces, verified internal interfaces, and more element
  families;
- represent multi-topology imports as explicit solver-domain bundles while
  DOLFINx mixed-topology support remains incomplete;
- treat ANSYS CDB and full solver decks as separate adapters, not generic mesh
  conversion.
- keep Gmsh a separately installed adapter for direct model/`.msh` workflows;
  do not make it a prerequisite for structured, XDMF, Abaqus, or NASTRAN paths.

### P4: AI-native operation

- maintain one public API and one validation path for humans and agents;
- make errors addressable and capabilities queryable;
- stabilize the local process boundary before adding an asynchronous job
  service, report bundle, REST interface, or MCP adapter;
- pair every public function family with compact reference examples;
- add tool/service endpoints around the same campaign/result contracts;
- evolve AF-IR only when a loader, validator, migration, or independent
  consumer requires a stable semantic record.

## Definition of “Better”

Within a supported problem class, AgentFEM is competitive when an experienced
engineer can:

1. read the model without reconstructing generated backend code;
2. modify a material, region, load, or step locally;
3. inspect the governing operator/residual and solver evidence;
4. run one case or thousands with the same case builder;
5. obtain visualization, scalar histories, and training data without a second
   extraction project;
6. reproduce the result from a benchmarked open workflow.

This is a narrower and more defensible route to excellence than imitating the
entire feature surface of mature commercial CAE suites.
