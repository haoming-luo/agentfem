# CTO Product Audit: 5 August 2026

## Executive decision

AgentFEM does not need an architectural rewrite before the first public testing
release. Its public model language, provider boundary, FEniCSx execution,
structured results, verification policy, project shell, and campaign-to-data
path form one coherent product. The correct strategy is now depth before
breadth: close trust and usability gaps in the supported workflows, freeze
their contracts with executable evidence, and postpone features that would
create a second private architecture.

This audit separates three questions that must not be conflated:

1. **Can a supported case be expressed and solved?**
2. **Can its progress, state, result, and failure be trusted and resumed?**
3. **Can a human, agent, GUI, campaign, and learning workflow consume the same
   contract without reverse-engineering files or terminal prose?**

The first-release work is the intersection of all three.

## Closed in this audit

- Transient checkpoint schema v2 publishes generation-specific rank shards and
  atomically replaces the manifest only after every rank succeeds.
- Each shard carries byte-size and SHA-256 evidence; missing or corrupted state
  fails collectively before any field is mutated.
- Restart identity now includes local mesh topology, cell type, geometry, and
  function layout. The contract remains honestly partition-bound.
- Heat, Standard dynamics, and Explicit dynamics resume through the normal
  `solve_result(output=...)` path. The output is explicitly labeled as a
  continuation segment with its physical start time.
- A completed transient step still refuses to invent unrecorded historical
  frames.
- J2 checkpoint publication is atomic, while its serial and quadrature-state
  portability boundary remains unchanged.
- Energy and thermal-content histories now carry definitions rather than only
  short names.
- The release smoke installs the just-built wheel into an isolated temporary
  target, checks representative source fingerprints, and runs both demos and
  an empty-directory project from that target. A stale installation with the
  same version number can no longer pass as the release candidate.

## Release-critical execution order

### R1 — freeze five demonstration contracts

Keep the existing simple cantilever, heat-transfer, inclusion-wave, periodic
Neo-Hookean, and creep-assessment workflows. Do not create replacement demos.
For each promoted demo, freeze:

- one readable top-level case;
- one numerical Golden manifest with declared tolerances;
- one deliberate failure or applicability-boundary check;
- one result/output contract that opens without case-specific repair;
- one documented serial command and, where supported, one MPI command.

The wave and Neo-Hookean demos have the highest visual value and therefore the
highest risk of being accepted by appearance alone. Their Golden contracts must
cover physical quantities and time/load coordinates, not screenshots.

### R2 — finish the result surface users expect

Implement named point/path probes and standard stress/strain projection before
adding more result container types. Extend reaction, external work, heat-input,
and balance histories using the existing `SimulationResult` and output-plan
contracts. Add automatic checkpoint cadence as a policy on the current Step
lifecycle, not as another runner.

Acceptance: direct Python, `agentfem run`, MPI, a future GUI, and an agent all
observe the same names, units, artifacts, execution events, and trust state.

### R3 — deepen nonlinear solids without multiplying public APIs

Promote the current J2 route through multi-region ownership, projected fields,
forced cyclic/cutback cases, and an external deck reproduction. Keep
`model.step(...)` as the public entry point and dispatch by Study, material
capability, and SolutionProcedure. Do not create material-named Step methods.

Neo-Hookean work should prioritize multi-region material ownership, load/work
balance, and an external load-path benchmark before additional hyperelastic
law names.

### R4 — global creep is the next new solver, not the next new formula

The first global creep procedure must reuse the quadrature transaction,
consistent-tangent convention, adaptive increment controller, rollback,
checkpoint record, event stream, and result contract already proven by J2.
Begin with isothermal power/Sinh creep, then temperature-dependent Arrhenius,
then K-R or Liu--Murakami damage.

Promotion requires constant-stress creep, stress relaxation, one-element
paths, time-step convergence, forced cutback, restart equivalence, and a
multi-element thermal application. Until those pass, material-point creep
models remain useful assessment components rather than a global FEM claim.

### R5 — protect the simulation-to-learning advantage

Keep one case builder for a single solve and a parameter campaign. Add a
scheduler executor and mesh-independent field sampling only after failed,
resumed, and verified cases preserve identical provenance. User PyTorch models,
built-in baselines, POD models, and later neural operators should consume the
same `ScientificDataset`; AgentFEM should not become a competing deep-learning
framework.

## Deliberately after the first testing release

- cross-partition/process-count restart, until stable global mesh and
  quadrature identities are proved;
- native Windows, while WSL2 provides the supported Windows route;
- a long list of constitutive model names without global integration evidence;
- a GUI-specific model schema or private service API;
- executable PINN/neural-operator trainers without benchmarked scientific
  contracts;
- broad AF-IR expansion without a loader, validator, migration, or independent
  consumer.

## Product invariants

Every new public capability must preserve these rules:

1. one human-readable Python workflow for humans and agents;
2. Study describes physics, SolutionProcedure describes the numerical route;
3. advanced users can inspect or replace UFL/PETSc-level components;
4. successful execution, convergence, verification, and validation remain
   distinct states;
5. no state is mutated after a failed identity or integrity check;
6. case-specific convenience stays in examples until two independent workflows
   prove it reusable;
7. a feature is advertised at the maturity level demonstrated by its tests and
   benchmarks, not by the existence of a class name.

These invariants are the foundation for a future GUI, agent service, MCP tool,
or commercial front end: each should orchestrate AgentFEM's existing public
contracts rather than fork the scientific core.
