# AgentFEM 0.3.8

AgentFEM 0.3.8 is the architectural-foundation release before 0.4. It
strengthens the boundaries between Model, Operator, Procedure, State, Backend,
and Result/Verification instead of adding another disconnected solver family.

## Foundation gates

The release closes six declared foundation gates:

1. one mesh--element--function-space compatibility contract used by
   inspection, validation, lowering, result identity, and checkpoint identity;
2. mesh-quality evidence and early rejection of unsupported topology or
   formulation combinations;
3. typed time inputs that distinguish right-hand-side, operator, state, and
   output-only effects before a Procedure reuses or rebuilds an operator;
4. provider-owned reaction, accepted-path work, and energy evidence for the
   declared MPC, weak-constraint, and bounded contact routes;
5. a typed MPI result, state-identity, and checkpoint lifecycle with verified
   cross-rank restoration where portability is declared;
6. an independently installed extension that adds behavior through the public
   provider boundary without modifying AgentFEM core.

## Contact and nonlinear state

The first contact routes remain intentionally bounded. The nonlinear-static
route covers small-strain solid mechanics, one analytical rigid plane and
frictionless penalty contact, with potential, residual, tangent, nodal
reaction, resultant, penetration diagnostics, accepted-path work semantics,
and conservative energy evidence.

The bounded explicit route now accepts one proportional prescribed rigid
translation/rotation schedule. It evaluates the moving surface at the
predicted displacement, publishes MPI-global force and moment, integrates only
accepted force--translation/moment--rotation stations, rolls failed increments
back, and restores the accepted work path from transient checkpoints. Serial
restart and two-rank canonical-State tests protect that lifecycle. The bounded
DOLFINx route can now derive a conservative contact spectral contribution
from its trace, both penalty channels, Coulomb pressure-cap coupling, and the
actual lumped mass for fixed-normal or piecewise-planar projectors; the
Procedure adds it to the body/material spectral bound before selecting the
time increment. Bounded rigid pairs
compose through the residual chain: accepted time, rollback, per-pair
force/moment, summed work/potential, and nested restart propagate without a
forming-specific solver; a mixed analytical-plus-triangulated pair test uses
that route without a two-tool special case. A separate reviewed path couples the same Procedure
to serial and routed distributed triangle BVHs; stable slave-point identity is
preserved while the closest rigid facet changes during finite sliding, and
pure tangential tool translation produces no spurious normal contact work.
The recommended construction binds surface, schedule, reference point, and
rank-independent scientific identity in one immutable `RigidBody`; serial or
distributed search remains a separate backend object rather than becoming
part of the body's identity.
`RigidContactPair` binds the slave boundary, body, and scalar local law as one
inspectable Model asset while leaving search and time-step stability with the
Backend and Procedure. Pair and body identities are verified before checkpoint
restoration.
Accepted transient events expose compact contact activity, maximum penetration,
force norm, prescribed-motion work, and interval power through the existing
bounded/throttled
progress lifecycle; no per-point history is accumulated implicitly.

The fixed public Abaqus/Explicit finite-sliding protocol now has a complete
AgentFEM solid bridge rather than only component tests. It uses the published
material, 500-unit normal load, friction coefficient 0.3, and 0.1-unit slide;
the ordinary Procedure reaches frictionless preload equilibrium, transfers
the accepted state atomically, and then closes normal force, the Coulomb cap,
action--reaction, friction dissipation, energy, and serial/two-rank endpoint
checks. The bounded tessellated tool also records real stable-facet crossings
without changing slave-point identity. Because the source discretization is
B31 and the bridge is a
tetrahedral CG1 solid, the evidence is labelled a protocol bridge rather than
an elementwise external reproduction.

The bridge additionally separates time and space refinement. It first halves
both stage time increments without changing physical duration, then refines
the in-plane mesh while retaining those time increments and durations. The
certificate requires stable friction dissipation, persistent facet crossings,
and non-growing energy error; it deliberately does not infer an observed order
from a two-level study.

General multi-tool candidate routing, arbitrary surface topology, free
rigid-body dynamics, deformable-to-deformable contact, and forming are not
inferred from this evidence. They remain explicit future capabilities.

The 0.4 foundation work now also defines a solver-neutral accepted/trial
closest-point State. It keeps stable 64-bit contact-point and surface-facet
identity, permits facet changes during sliding, rejects contact-pair identity
changes, and checkpoints only accepted projection boundaries. Search remains
geometry infrastructure; no general contact residual, Newton update policy, or
cross-rank projection checkpoint is claimed by this state contract.

A separate projection lifecycle now owns every-evaluation calls to analytical
or reviewed BVH projectors, keeps search diagnostics out of State, and binds
projection commit/rollback to the accepted increment. Distributed projectors
reach collective post-search acceptance, preventing one rank from entering
assembly after another rank has rejected invalid projection evidence. This is
still a lifecycle foundation, not a general contact Operator.

The same foundation now includes a solver-neutral local frictionless penalty
law. It consumes any reviewed `SurfaceProjection`, returns penetration,
pressure, conservative potential density, structural residual traction, and
the conjugate rigid-surface traction, and rejects invalid search evidence by
default. The response is deliberately labelled pointwise and unintegrated:
trace interpolation, quadrature, global assembly, and geometry-consistent
linearization remain required before AgentFEM can claim a general implicit
contact Operator.

A backend-neutral `ContactTrace` now defines the next hand-off without
hard-coding one element family. It binds stable point IDs to node interpolation,
positive quadrature weights, and an explicit reference/current measure, then
integrates a matching projection record and point response into nodal residual,
penalty potential, contact resultant, and optional tool moment. It rejects
identity, coordinate, or response mismatches before assembly. This is the
backend-neutral reference trace contract; consistent implicit linearization
remains outstanding.

The first DOLFINx trace adapter now closes that hand-off for one bounded route:
tagged owned exterior triangles on first-order tetrahedral geometry and a
continuous blocked three-component CG1 displacement space. It uses a positive
three-point reference-area rule, derives stable point identity from the
partition-independent facet IDs, synchronizes displacement ghosts before
interpolation, supports empty local shards, and rejects unsupported spaces
collectively. General/high-order topology and a consistent implicit contact
linearization are still not claimed.

## Time and restart semantics

Time-dependent inputs now declare whether they affect only the right-hand side,
the operator, constitutive state, or output. Linear implicit dynamics and
implicit-Euler heat transfer use this information to select safe reuse;
nonlinear heat transfer continues to assemble per step. Transient checkpoint
schema v5 binds the complete input plan and rejects changed or unidentifiable
callbacks before mutating a restored field.

The public checkpoint capability contract separates payload scope, save
boundary, atomicity, scientific identity, and MPI rank-count portability.
Supported transient, nonlinear, inelastic, viscoelastic, harmonic, and cyclic
owners expose that contract through `model.step()` and `SimulationResult`.

## Mesh and extension evidence

Reviewed higher-order triangle, quadrilateral, tetrahedral, and hexahedral
routes now preserve coordinate-element identity through import, quality audit,
patch testing, output, and supported MPI paths. The six-node wedge route is
also promoted for its declared path. Conditional or unsupported topology is
reported before assembly rather than failing inside a low-level reader.

The reference external material is built and installed as a separate wheel.
The extension acceptance proves provider discovery, ordinary model execution,
structured results, verification, and unchanged core-package hashes.

Release CI also creates a fresh-agent trial bundle bound to the exact candidate
wheel, source commit, task, review contract, required command sequence, and
required outputs. This prevents an earlier successful trial from being reused
as evidence for a different candidate.

## Upgrade

Install or upgrade with conda-forge after its feedstock update is available:

```bash
mamba install -c conda-forge agentfem=0.3.8
```

The Python distribution can also be upgraded after PyPI publication:

```bash
python -m pip install --upgrade agentfem==0.3.8
```

Existing projects keep the public
`Study -> Model -> scientific assets -> model.step(...) -> SimulationResult`
workflow. Checkpoint identity is stricter by design; incompatible or stale
archives fail before partial restoration.

## Explicit boundaries

AgentFEM 0.3.8 does not claim universal finite-element coverage, native Windows
solver support, general contact, forming-capable shells, arbitrary-path
fracture, universal nonlinear-material validation, or monolithic multiphysics.
Use `agentfem capabilities` for the exact installed maturity and evidence.
