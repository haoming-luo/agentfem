# AgentFEM 0.2.6 promotion candidate

AgentFEM 0.2.6 was the immutable platform-foundation candidate used to prove
the seven 0.3 promotion gates. It fixes the ownership
boundaries between the readable engineering model and the FEniCSx/PETSc
runtime, while closing several workflows that now depend on those boundaries.

## A stable middle layer

Seven responsibilities are now explicit and machine readable:

```text
Model -> Constitutive -> State -> Operator
      -> Procedure -> Backend -> Result/Verification
```

`Model` remains the engineering facade. State owns accepted/trial history and
restart boundaries; Operator owns mathematical contributions; Procedure owns
increments and algorithms; the backend owns compilation, assembly, DOFs and
algebra; Result/Verification owns evidence and acceptance. Architecture tests
prevent selected upward dependencies from silently returning.

## Inelastic RVE foundation

The experimental finite-strain logarithmic J2 route now enters the ordinary
`model.step(...)` lifecycle under both strong kinematics and reviewed
affine-periodic constraints. It includes committed/trial quadrature state,
consistent tangents, cutback, MPI-portable restart, regional materials,
non-proportional loading, deterministic void and multi-void realizations, and
Golden RVE evidence.

This is a usable research foundation, not a claim of universally validated
finite-strain plasticity. Independent external structural validation remains
an explicit promotion gate.

## Migration and extension boundaries

Abaqus project migration preserves the recursive source graph, Part/Instance
scope, element formulation identity, sets, sections, materials, Steps and
unsupported assets. Native lowering remains a separately reviewed,
fail-closed operation.

The core also publishes framework-neutral neural-field contracts. PyTorch and
XDEM implementations remain in the separately installed AgentFEM-Learning
companion, demonstrating that an advanced provider can extend AgentFEM without
modifying its core.

## Evidence before promotion

The 0.3 promotion audit now binds Linux/macOS installed-wheel acceptance,
companion-provider evidence and a genuinely fresh AI-agent trial to an exact
AgentFEM version, source commit and wheel digest. Deterministic CI cannot
impersonate independent behavioral evidence. WSL2 remains the recommended
Windows route and is tracked separately rather than blocking this release.

## Compatibility

The recommended language remains:

```text
Study -> Model -> scientific assets -> model.step(...) -> SimulationResult
```

Historical 0.2.x helpers remain compatibility delegates. This release changes
ownership and evidence boundaries without requiring users to rewrite valid
existing projects mechanically.
