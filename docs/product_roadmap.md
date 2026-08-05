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
| Linear elasticity | FEM-integrated | 2D isotropic/selected anisotropic, static/dynamics building blocks, strong-BC reaction field, M/K energy diagnostic | broader 3D verification and affine/weak reactions remain |
| Thermoelasticity | FEM-integrated | steady/implicit-transient heat transfer, regional multi-material conductivity and capacity, amplitude-driven sources/ambient conditions, and sequential isotropic thermal stress in 2D/3D | no property tables or monolithic two-way coupling |
| Structural dynamics | FEM-integrated foundation | central difference, Newmark, and generalized-alpha through `dynamic_solid`; model-owned constraints and amplitude loads enter both Standard and Explicit paths; shared field output, mechanical-energy histories, integrity-checked pause/checkpoint/restart, and truthful continuation output | implicit route is linear; checkpoints require the same mesh partition and MPI size |
| Compressible Neo-Hookean | FEM-integrated | 3D and 2D plane-strain nonlinear statics; automatic/fixed increments, Newton cutback/rollback, positive-J acceptance, energy and accepted-increment histories for ordinary loading; affine periodic-cell path | one-material convenience step; no 2D plane-stress local solve or external load-path benchmark |
| J2 plasticity | FEM-integrated foundation | 3D shared quadrature transaction, analytical tangent, natural/displacement loading, non-monotone tabular amplitude, global Newton, physical-increment cutback, cumulative serial restart including adaptive state and scientific identity, S/PE/PEEQ/RF results, prescribed-work/energy histories, analytical Golden path, and an automated Abaqus published-data verification state | no plane stress, kinematic hardening, multi-region driver, MPI-portable restart, or mesh-convergence external deck reproduction |
| Creep and creep damage | material-point/assessment verified | power-law and Arrhenius paths; exact K-R damage coupling; Sinh flow; modified-theta fitting; hot-wall sequential assessment | no global adaptive quadrature creep step or damage regularization |
| Stress-life fatigue | postprocessor | Basquin/tabulated S-N, rainflow, Goodman, Miner assessment from named result histories | no multiaxial critical-plane method |
| External CAE mesh | integrated Abaqus path + conversion interface | generic meshio conversion; Abaqus node labels; verified C3D10 import; linear equation parsing | full solver-deck sections/material cards are not imported |
| Abaqus periodic equations | serial + two-rank FEM-integrated | exact chained affine elimination, distributed `dolfinx_mpc`, and 3D Neo-Hookean load path | AMG near-nullspace transfer, reactions, and scaling studies remain |
| Abaqus user-material bridge | interface contract | solver-neutral material-point input/output and migration specification | no compiled adapter or quadrature-state global driver |
| Result/data flow | integrated foundation | declarative field/history/diagnostic/presentation plans; compact deformed XDMF/HDF5; one structured Standard/Explicit/thermal/J2 event trace; atomic and integrity-checked heat/Standard/Explicit checkpoint envelope; continuation-aware output; mechanical-energy and thermal-content histories; typed checkpoint records; MPI-safe point/path probes, region integrals/averages, boundary resultants and field extrema; verification reports and trust-gated campaign/PyTorch bridge | stress/strain projection, automatic checkpoint cadence, cross-partition MPI restart, and broader conservation balances remain |
| Scientific trust | integrated foundation | computed/converged/verified/validated vocabulary; exploratory/engineering/release policies; automatic runtime checks; explicit claims and applicability domains; coarse-to-fine convergence evidence; result manifests and learning-data quality gates; orientation metamorphic regression | representative-family evidence inheritance, hole-stress and T-stiffener cliff families, GCI, and external-deck reproductions remain |
| Campaign-to-learning flow | workflow integrated | deterministic cases, resumable evidence, failure-aware dataset gate, reproducible train/validation workflow, ridge/POD/PyTorch adapters, applicability guard and FEM fallback | no scheduler executor, active-learning governance, or calibrated epistemic uncertainty |
| Platform/install boundary | release foundation | Linux CI, macOS developer verification, WSL2 recommended for Windows, runtime dependency report, Gmsh/meshio optional adapters | native Windows remains experimental; AgentFEM is not yet a conda-forge package |

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
- one `SimulationResult` contract and one structured execution-event stream
  for linear, nonlinear, and transient steps;
- standard QoIs: integrals, averages, norms, extrema, reactions, energies, and
  histories;
- compact unified XDMF/HDF5 visualization and output manifests;
- JSON-configured and Python-configured campaigns producing the same dataset;
- serial, MPI, docs, package, and example release gates;
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
- add reaction, internal/external work, and energy-balance histories with
  verified strong, weak, and affine-MPC definitions;
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

- make the implemented Arrhenius, K-R, and Sinh local models consume the J2
  quadrature transaction in a global implicit creep step with adaptive time
  increments and restartable state;
- treat sequential temperature-to-creep as the first useful power-component
  route; add monolithic coupling only for cases with material heat generation
  or meaningful mechanical feedback;
- creep/relaxation single-element verification followed by NAFEMS cases;
- named result histories feed an auditable fatigue assessment now; add
  automatic stress extraction at named regions/points and fatigue fields;
- multiaxial fatigue only after a chosen engineering criterion and reference
  dataset are explicit.

The first global creep promotion has non-negotiable acceptance gates:

1. the public step consumes `QuadratureTransaction`; no second private state
   store or copy-only rollback is accepted;
2. a backward-Euler local update returns stress, state, convergence evidence,
   an algorithmic consistent tangent, and a local error/step recommendation;
3. global Newton failure, local failure, excessive equivalent creep increment,
   or excessive damage increment causes atomic rollback and deterministic
   cutback;
4. checkpoint/restart retains physical time, next proposed increment,
   temperature, displacement, committed material state, energy/dissipation,
   event cursor, and schema version;
5. constant-stress creep, stress relaxation, one-element paths, time-step
   convergence, forced cutback, and restart equivalence pass before a
   multi-element hot-wall example is advertised;
6. start with isothermal Sinh/power-law flow, add Arrhenius temperature
   dependence second, and K-R/Liu--Murakami damage only after near-failure
   step control and mesh-dependence limitations are explicit.

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
rank-local array serialization prototype.

### P3: mesh and model interoperability

- verify Abaqus `.inp`, Nastran bulk-data, Gmsh, Exodus, and MED meshes;
- map volume sets and boundary sets to named AgentFEM regions;
- preserve source identities, checksums, conversion choices, and warnings;
- extend the now-preserved Abaqus node labels and linear equations to common
  node/element-set semantics where meshio loses information;
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
