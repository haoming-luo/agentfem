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

The first contact route remains intentionally bounded: small-strain solid
mechanics, one fixed rigid plane, frictionless penalty contact, and nonlinear
static equilibrium. Within that scope it now owns its potential, residual,
tangent, nodal reaction, resultant, penetration diagnostics, accepted-path
work semantics, and conservative energy ledger. Ordinary nonlinear state,
increment history, dual evidence, energy history, and execution events restart
atomically across one and two MPI ranks.

Time-varying loading, nonzero prescribed motion, moving tools, search, finite
sliding, multiple contact pairs, friction, deformable-to-deformable contact,
and forming are not silently inferred from that evidence. They remain explicit
future capabilities.

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
