# Stable ownership and state contracts

## Decision

AgentFEM fixes seven responsibility boundaries for its FEniCSx-first middle
layer: Model, Constitutive, State, Operator, Procedure, Backend, and
Result/Verification.

The dependency-free `_architecture_contract.py` is the machine source of this
inventory. Selected impossible cross-layer dependencies are rejected by AST
tests. `Model` cannot construct a discrete `problem`; first- and second-order
transient states belong to `state.py`; lumped mass belongs to `operators`.
Legacy imports from `problems` remain compatibility aliases.

State standardization is capability based. `RestartableState` requires
`snapshot/restore`; `ReplaceableState` additionally requires
`commit/rollback`. Trial creation stays procedure specific because a material
update, Newton increment, transient step, and ordered fatigue cycle require
different scientific inputs.

## Reason

AgentFEM's value lies in organizing, inspecting, and verifying engineering
computation above DOLFINx/PETSc. Growing Model, AI, campaign, and result
features without fixed ownership would create a numerical god object and make
future extensions ambiguous. Reimplementing finite-element assembly would not
solve that problem.

## Consequences

- `agentfem capabilities --json` publishes the same ownership inventory used
  by contributor tests.
- File size alone does not trigger a split; duplicated scientific decisions or
  upward dependencies do.
- Constitutive laws define response and state schema, while State owns the
  accepted/trial lifetime.
- Procedure owns algorithm and evolution; Backend owns compilation, assembly,
  DOFs, and algebra; Result/Verification owns acceptance evidence.
- A second numerical backend is deferred until it can implement the stable
  boundary without special cases in the public model language.

## Evidence

- architecture import tests and compatibility-identity tests;
- transaction tests for first- and second-order transient state;
- complete serial regression, targeted two-rank transient/cohesive regression,
  strict documentation build, and installed-wheel import smoke;
- executable G1--G4 platform-promotion audit.
