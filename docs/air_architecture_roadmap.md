# AgentFEM AIR Architecture and Engineering Roadmap

> Product priority note: the executable FEniCSx product, nonlinear mechanics,
> results, interoperability, and verification now have the governing
> near-term roadmap in `docs/product_roadmap.md`. AF-IR remains experimental
> and should advance only when a loader, migration, validator, or independent
> consumer creates evidence for a stable representation.

## Purpose

This document turns the long-term AF-IR/AIR argument into an engineering
sequence. It distinguishes:

- what AgentFEM already executes,
- what this release has made explicit,
- what should be built next,
- what should remain a research question,
- and what must not be claimed before evidence exists.

AgentFEM is not required to finish the final architecture before it becomes
useful. The near-term goal is a dependable, high-level, open finite-element
environment on FEniCSx. The long-term goal is a scientific layer through which
humans and agents can construct, inspect, validate, execute, and accumulate
finite-element knowledge while numerical backends continue to evolve.

## Core Position

Deep FEniCSx integration is the present strength of AgentFEM.

FEniCSx supplies mature weak-form, finite-element, PETSc, MPI, and output
machinery. AgentFEM should use those capabilities fully. A backend-neutral
interface that exposes only the intersection of imagined future solvers would
be less useful, less testable, and less scientifically honest than a strong
FEniCSx implementation.

At the same time, backend runtime objects must not become the only surviving
record of scientific intent. The architecture should therefore evolve along
two coordinated tracks:

1. **FEniCSx product depth**
   - broader supported analyses,
   - stable public workflows,
   - verification benchmarks,
   - deterministic diagnostics,
   - serial and MPI testing,
   - reliable output and restart behavior.

2. **AIR/AF-IR durability**
   - versioned scientific records,
   - stable object identities and references,
   - structured validation and repair,
   - explicit compilation boundaries,
   - execution provenance,
   - migrations,
   - and eventually independently tested lowering targets.

Neither track substitutes for the other. A beautiful schema without a strong
solver path is not a finite-element platform. A strong solver wrapper without
durable scientific records cannot become the shared human-agent layer proposed
by AgentFEM.

## Current Architectural Reality

The current executable path is:

```text
Human / agent-authored Python
        |
        v
AgentFEM Study / Model / Region / Field / Material / Load / Step
        |
        v
AgentFEM operators containing UFL expressions
        |
        v
DOLFINx form compilation and assembly
        |
        v
PETSc / MPI solve or AgentFEM time integration
        |
        v
Diagnostics and XDMF output
```

This already has important AIR properties:

- the top-level vocabulary follows finite-element practice;
- constraints, loads, and weak boundary models are distinct;
- operators retain engineering names such as `K`, `M`, `C`, and `F`;
- steps expose analysis and time-integration intent;
- summaries and model trees are inspectable;
- expert users can descend to UFL, DOLFINx, and PETSc.

The main technical coupling is also clear:

- fields contain DOLFINx functions and spaces;
- regions contain DOLFINx mesh tags and UFL measures;
- constraints contain DOLFINx boundary-condition objects;
- loads and boundary models may contain UFL coefficients and forms;
- `OperatorForm.expression` is currently a UFL expression;
- solving and output are directly implemented with DOLFINx/PETSc APIs.

This means AgentFEM 0.1 is best described as a human- and agent-readable
scientific layer with an experimental AF-IR record, executed through a
FEniCSx-first runtime. It is not yet a complete backend-independent compiler.

## Module-by-Module Assessment

### `studies.py`

This is one of the cleanest semantic modules. `Study` is immutable, normalized,
and validated before execution. It is already close to a persistable AF-IR
node. Its next need is a capability table linking analysis/physics combinations
to implemented operators and steps; expanding the vocabulary without this
table would let users declare unsupported studies too easily.

### `models.py`

`Model` is the central scientific registry and the strongest part of the public
language. It also carries the greatest architectural pressure: registration,
convenience constructors, regional assembly, validation, step construction,
inspection, and export now meet in one large module.

Do not split it merely to reduce line count. First stabilize the public
responsibilities, then move internal services behind the same `Model` API:

- assignment registry,
- validator,
- operator builder,
- step factory,
- AF-IR exporter.

The important boundary is responsibility, not file size.

### `mesh/`

Named boundary/cell regions and selectors provide the right public concept.
The current region objects remain runtime DOLFINx assets, and raw generated
meshes do not retain enough construction data for reconstruction. The AF-IR
export therefore marks runtime mesh summaries as non-reconstructable.

The next durable object is a `MeshSpec`/`MeshSource` that can represent:

- structured constructor parameters,
- imported file identity and checksum,
- Gmsh physical groups,
- conversion warnings,
- and the relationship between source labels and AgentFEM regions.

### `fields.py`

`UnknownField` provides excellent top-level readability. Eager `Field`
arithmetic is useful for explicit state updates and has a clear PyTorch/Cast3M
feel. The main risk is conceptual ambiguity between:

- eager dof-vector algebra,
- symbolic UFL expressions,
- and discrete operator action.

The current documentation distinguishes these, and this distinction should
become a validation/type rule before mixed and tensor fields expand.

### `materials/` and `constitutive/`

Separating data records, property containers, and response laws is correct.
Material properties already serialize well. The missing durable relations are:

- units,
- source/provenance at the property instance,
- applicable study assumptions,
- law/version identity,
- and explicit material-to-region references.

These should be added before a large material catalog.

### `constraints/`, `loads.py`, and `boundary_models/`

The conceptual separation is strong and should remain non-negotiable.
Scientific summaries now preserve common prescribed/load values and absorbing
impedances. Remaining gaps include target-field identity, full region
references, local coordinate systems, amplitudes on general loads, and clear
capability declarations that distinguish serial nodal projection from the
distributed Abaqus-equation MPC backend.

The periodic projection implementation is correctly explicit about being a
serial, nodal-averaging method rather than a general MPC solution. Future MPC
support should be a new implementation of the same semantic constraint, not a
silent change in the existing method.

### `operators/`

The K/M/C/F language is the clearest bridge between engineering review and
backend computation. Operator composition now retains sum/scale history, but
primitive operators still need explicit dependencies on fields, materials,
regions, coefficients, and measures. `OperatorSpec` should grow from these
common families while `OperatorForm.expression` remains the efficient UFL
runtime path.

### `problems.py`, `time/`, and `solvers.py`

The separation among systems, steps, states, and time integrators is sound.
Solver options are now inspectable and validated. The next missing interface is
an execution plan that records when compilation, constraint application,
state updates, diagnostics, and output occur. Numerical execution still calls
DOLFINx/PETSc directly in several places, which is appropriate until the plan
interface has real use.

### `diagnostics.py`, `results/`, and `io.py`

These modules now form the beginning of an execution evidence system.
`SolveEvent` is shared by terminal progress, status files, complete result
traces, and accepted-increment histories across Standard, Explicit, heat, and
J2 paths. Versioned Golden quantities add numerical acceptance criteria for
selected release workflows. Remaining work includes field/source checksums,
named regional reactions, external-work balances, transient checkpoints, and
MPI-portable state identity.

This clarifies an important architectural point: the proposed AIR layer is not
one JSON document. In the working product it is the coordinated public
language, validation contracts, structured execution evidence, scientific
results, benchmark cards, knowledge assets, and adapters. AF-IR is one durable
representation within that layer and should grow only with real consumers.

### Documentation skill and examples

The skill already gives agents a progressive reading order and important
concept boundaries. The next gain will come from pairing those instructions
with structured validation codes and small repair examples. Examples should
remain readable teaching artifacts; separate benchmarks must carry numerical
expected values and tolerances.

### Packaging and CI

The conda-forge FEniCSx environment is the correct reproducibility anchor for
the current stack. Unit/interface tests, serial solve, two-rank MPI solve,
package build, and documentation build are the minimum release gates. PyPI
metadata alone cannot install the full MPI/PETSc/DOLFINx stack reliably, so the
documentation must continue to make the conda path primary.

## Architecture Introduced in This Iteration

### Versioned AF-IR document envelope

`agentfem.ir` now defines:

- `AFIR_SCHEMA`,
- `AFIR_SCHEMA_VERSION`,
- `IRDocument`,
- deterministic JSON serialization,
- explicit opaque markers for runtime objects that do not yet have stable
  scientific serialization,
- `model.to_ir()` and `model.write_ir(...)`.

AF-IR 0.1 is marked `experimental`. It records supported public semantics and
does not imply that arbitrary UFL expressions can be reconstructed or lowered
to another backend.

### Structured validation and repair addresses

`model.validate()` now returns a `ValidationReport` containing
`ValidationIssue` objects. Every issue has:

- a stable code,
- a scientific-object path,
- a severity,
- an explanation,
- an optional repair hint,
- and optional structured context.

For example:

```text
[ERROR AFM-MATERIAL-002] model.materials[1].region:
Every material in a multi-material model needs a region.
Hint: Pass region=... when registering each material.
```

`model.check()` remains the fail-fast public guard, but it now raises a
`ModelValidationError` carrying the complete structured report.

### Honest backend seam

`agentfem.backends` now provides:

- `BackendDescriptor`,
- `BackendAdapter`,
- lazy backend registration,
- an explicit default backend,
- a `FEniCSxBackend` implementation.

`OperatorForm.compile`, `assemble_matrix`, and `assemble_vector` now cross this
seam. The seam is deliberately narrow. It does not claim that fields, meshes,
constraints, steps, or all operators are backend neutral.

### Preserved operator construction history

Operator scaling and addition now retain:

- operation kind (`primitive`, `scale`, or `sum`),
- operand summaries,
- scale factors when representable,
- source names,
- operand count.

This repairs a major scientific-record problem: `C / dt` and `K1 + K2` should
not become anonymous UFL expressions after composition.

### First test and continuous-integration foundation

The project now contains unit/interface tests for:

- validation reports,
- AF-IR serialization,
- backend registration,
- operator composition metadata,
- solver-policy validation.

The static FEniCSx example is also a serial and two-rank MPI smoke test and
writes an AF-IR record from rank zero. CI configuration uses a conda-forge
FEniCSx environment because MPI, PETSc, and DOLFINx should be tested as one
compatible stack.

## Target Architecture

The desired long-term flow is:

```text
Natural language / expert / domain model / general agent / imported case
                              |
                              v
                 AgentFEM public Python language
                              |
                              v
                Canonical semantic AF-IR graph
                 /            |             \
                v             v              v
        validation       transformation    scientific memory
        and repair       and composition   and provenance
                \             |              /
                 \            v             /
                  ---- execution plan -----
                              |
                  backend capability match
                              |
            +-----------------+------------------+
            |                 |                  |
            v                 v                  v
      FEniCSx lowering   future lowering   export/service target
            |                 |                  |
            v                 v                  v
      PETSc/MPI result   verified result    external execution
            \                 |                  /
             +-------- execution evidence ------+
                              |
                              v
                  human/agent review and reuse
```

The canonical graph is not required to contain every backend detail.
Backend-specific extensions are acceptable when they are:

- explicitly namespaced,
- capability checked,
- preserved during round trips,
- visible to reviewers,
- and rejected rather than silently approximated on unsupported targets.

## Design Laws

### Scientific meaning before serialization convenience

A JSON field should exist because it represents a meaningful finite-element
decision, not merely because a Python attribute was easy to dump.

### One strong backend before superficial portability

The FEniCSx path remains the reference implementation. A second backend should
be added to test the semantic boundary, not to create a marketing checklist.

### Semantic objects and compiled objects have different lifetimes

A `MaterialAssignment` or `OperatorSpec` should be persistable across sessions.
A DOLFINx `Form`, PETSc matrix, communicator, and dof map are runtime objects.
They may be cached or referenced in an execution record, but they should not
define the canonical scientific model.

### Unsupported behavior is data

An unsupported formulation must produce an addressable capability or validation
failure. It must not be silently dropped, translated approximately, or hidden
inside generated backend code.

### Progressive disclosure applies to architecture

Beginner code should remain short. Researchers should see operators and
solver policies. Backend specialists should reach UFL, PETSc, MPI, quadrature,
and dof-level behavior.

### Human and agent interfaces share one state

The agent API must not operate an invisible model different from the model
shown to the engineer. Python objects, AF-IR records, validation reports, and
execution evidence should describe the same scientific state.

## Missing Interfaces and Proposed Designs

### Stable object identity and references

Current names are useful but not always unique. Future AF-IR nodes need:

```python
ObjectId(namespace="model", kind="region", name="left", revision=1)
```

Requirements:

- deterministic within one model document;
- human-readable where possible;
- no dependence on Python memory addresses;
- references survive serialization and migration;
- duplicate display names produce a validation issue;
- renaming has explicit migration semantics.

Do not introduce global UUIDs everywhere in the public Python API. Begin with
document-local deterministic identifiers and retain friendly names.

### Semantic specification versus compiled artifact

The operator path should eventually distinguish:

```python
OperatorSpec(
    family="elastic_stiffness",
    role="matrix",
    target=Ref("field:displacement"),
    material=Ref("material:steel"),
    region=Ref("region:matrix"),
)

CompiledOperator(
    spec=...,
    backend="fenicsx",
    expression=ufl_form,
    cache_key=...,
)
```

This should be introduced family by family, beginning with operators that have
clear portable semantics:

- scalar mass/capacity,
- diffusion/conduction,
- isotropic elastic stiffness,
- body source,
- traction/flux,
- simple Dirichlet constraints.

Do not attempt to parse arbitrary UFL back into a universal semantic tree.
Custom UFL should remain an explicit backend extension with declared
capabilities and reduced validation coverage.

### Execution plan

A `Step` currently combines description and execution. A future
`ExecutionPlan` should resolve:

- selected backend,
- capability requirements,
- ordered compilation tasks,
- assembly tasks,
- constraint transformations,
- solver/integrator policy,
- diagnostic schedule,
- output schedule,
- restart and checkpoint policy.

The plan should be inspectable before execution and should have a stable hash
over scientific inputs plus declared numerical policy.

### Execution record and scientific evidence

Each run should be able to produce:

```text
case.agentfem.py
case.afir.json
case.plan.json
case.validation.json
case.run.json
results/
```

`case.run.json` should include:

- AgentFEM and backend versions;
- MPI size and scalar type;
- mesh identity and partition information;
- solver options and convergence reason;
- time-step and stability estimates;
- warnings and capability fallbacks;
- outputs with checksums;
- diagnostics and benchmark comparisons;
- start/end status;
- human approvals or edits when supplied.

This is the foundation for scientific memory. It is not a claim that every
successful run is scientifically correct.

### Schema migration

Before AF-IR is called stable, implement:

- semantic-version compatibility rules;
- `load_document(...)`;
- schema validation;
- migration functions such as `0.1 -> 0.2`;
- round-trip tests;
- golden example documents;
- preservation of unknown namespaced extensions;
- clear failure for unsupported future major versions.

The Python API and AF-IR schema may evolve at different rates. They need an
explicit compatibility table.

### Capability negotiation

Backend descriptors should eventually answer questions such as:

- supported mesh cell families;
- supported field value shapes;
- element/interpolation policies;
- supported constitutive/operator families;
- strong and weak constraint types;
- serial/MPI support;
- scalar/complex support;
- nonlinear/autodiff support;
- output formats;
- restart support.

Capability matching should return structured issues at paths such as:

```text
model.steps[0].operators.K
model.constraints[2]
model.outputs[0]
```

A backend must never advertise a broad capability such as `solid_mechanics`
when only one narrow formulation has been verified.

### Agent repair protocol

Documentation and a skill help an agent choose APIs, but reliable repair also
needs a machine-operable protocol:

```json
{
  "code": "AFM-MATERIAL-002",
  "path": "model.materials[1].region",
  "severity": "error",
  "message": "Every material in a multi-material model needs a region.",
  "allowed_repairs": [
    "assign_existing_region",
    "define_region",
    "remove_material_assignment"
  ]
}
```

Future validators may add `allowed_repairs`, but they should not automatically
apply scientific changes unless policy and user authority allow it.

### Extension protocol

Every custom extension should declare:

- semantic name and version;
- required backend capabilities;
- serializable parameters;
- validation hook;
- lowering hook;
- summary and documentation;
- benchmark or example evidence;
- whether it is deterministic;
- whether it supports MPI;
- fallback behavior, normally explicit rejection.

This is a stronger extension boundary than accepting arbitrary callables with
no record of their meaning.

### Batch, surrogate, and tool-service interfaces

Agent-native simulation will often operate collections rather than one case.
The long-term layer should support:

- parameter definitions with units and bounds;
- immutable case variants;
- deterministic case IDs;
- sweep and design-of-experiment plans;
- resumable execution;
- extracted quantities of interest;
- field sampling contracts;
- dataset manifests;
- links between samples and the exact AF-IR/run record;
- asynchronous local, cluster, or service execution.

The first local implementation now provides typed real/integer/choice
parameters, deterministic random/Latin-hypercube/factorial plans, stable case
IDs, resumable case records, deterministic plan shards, within-case MPI
coordination, scientific dataset manifests, NumPy ridge and POD baselines, an
optional PyTorch MLP adapter, independent validation reports, and a guarded
high-fidelity fallback. This advances part of the former Phase C/E route
without claiming that AF-IR reconstruction or remote execution is complete.

The implemented separation is:

```text
case factory -> FEM execution -> case/run evidence -> scientific dataset
             -> learning adapter -> validation -> applicability guard
             -> prediction or explicit FEM fallback
```

Neural-operator and PINN objects currently define contract-only records for
field encoding, geometry/mesh policy, physical residuals, conditions, and
required checks. They are intentionally not executable trainers. Arbitrary UFL
is not assumed to lower to a differentiable strong residual.

An MCP or other tool interface should expose operations such as:

```text
validate_model
explain_model
compile_plan
run_case
get_status
read_diagnostics
compare_runs
export_artifact
```

The service interface should wrap the same AF-IR and validation semantics used
locally. It should not create a second hidden representation.

## FEniCSx-First Product Priorities

The following work has higher near-term value than adding a second backend:

1. Linear static elasticity
   - verified plane strain and plane stress;
   - 3D isotropic elasticity;
   - reaction forces;
   - solver convergence records;
   - mesh-convergence benchmarks.

2. Transient heat transfer
   - model-owned step API;
   - time-dependent sources and boundary conditions;
   - convection/Robin boundaries;
   - energy balance;
   - restart.

3. Explicit dynamics
   - stable time-step estimation;
   - energy and momentum diagnostics;
   - clear residual sign conventions;
   - robust constraint behavior;
   - serial and MPI benchmark coverage;
   - absorbing-boundary validation.

4. Mesh and region reliability
   - partition completeness and overlap checks;
   - imported tag provenance;
   - empty-region detection;
   - region visualization;
   - deterministic region identities.

5. Materials and constitutive behavior
   - units and provenance;
   - validated model/assumption compatibility;
   - thermal properties;
   - carefully staged anisotropy and path dependence.

6. Output and restart
   - output manifests;
   - checksums;
   - consistent field naming;
   - checkpoint/restart contracts;
   - parallel behavior.

7. Verification assets
   - benchmarks separate from tutorials;
   - expected values and tolerances;
   - mesh/time-step convergence;
   - serial/MPI parity where meaningful.

## Phased Delivery Plan

### Phase A: strengthen the executable core

Target: current 0.1.x line.

- keep FEniCSx as the sole production backend;
- add unit tests for semantic containers;
- run static example in serial and with two MPI ranks in CI;
- add small numerical benchmarks;
- expand structured model validation;
- make solver policies inspectable;
- keep AF-IR explicitly experimental.

Exit evidence:

- reproducible environment;
- unit suite;
- serial and two-rank MPI static numerical smoke test;
- at least three benchmark quantities with tolerances;
- no silent unsupported behavior in demonstrated workflows.

### Phase B: AF-IR 0.2 scientific records

Target: v0.2.

- stable document-local object IDs;
- explicit references between core model assets;
- JSON Schema or equivalent formal validation;
- load/write and migration API;
- golden AF-IR documents;
- execution-plan prototype;
- run record with environment and output evidence;
- validation codes documented as public contracts.

Exit evidence:

- round trip for supported examples;
- migration test from AF-IR 0.1;
- deterministic document hashes;
- agent repair evaluation on planted model defects.

### Phase C: semantic/runtime separation

Target: v0.3.

- `OperatorSpec` and `CompiledOperator` for selected common families;
- backend capability matching;
- namespaced FEniCSx extensions;
- compilation caches keyed by semantic and numerical policy;
- broader benchmark suite;
- harden the implemented batch/sweep interface and link every case to
  content-addressed AF-IR/run records.

Exit evidence:

- supported model can be reconstructed without retaining live UFL objects;
- compiled FEniCSx result agrees with reference implementation;
- custom UFL survives as an explicit extension rather than being erased.

### Phase D: test the boundary

Target: research milestone after v0.3, not a calendar promise.

- choose one deliberately narrow second target;
- lower a small semantic subset;
- compare results and unsupported-capability behavior;
- revise AF-IR where the second implementation reveals accidental FEniCSx
  assumptions.

Good candidates are not necessarily full solvers. A second target might be:

- an export deck for a restricted static problem;
- a matrix/reference backend for tiny verification cases;
- a service adapter;
- or another open finite-element library for a narrow operator family.

Exit evidence:

- independent lowering of the same AF-IR subset;
- explicit capability differences;
- benchmark agreement within declared tolerances;
- no weakening of the FEniCSx path.

### Phase E: scientific memory and agent operations

- indexed run/evidence records;
- retrieval by scientific entities and assumptions;
- comparison and regression tools;
- approved reusable patterns;
- extend the implemented dataset/surrogate foundation with merge,
  deduplication, uncertainty calibration, active-learning proposals, and
  reviewed external trainers;
- tool-service/MCP interface;
- access-control and audit policies for consequential execution.

Exit evidence:

- a reviewed case can be found, understood, rerun, compared, and adapted;
- agent actions remain attributable to AF-IR nodes and execution records;
- human review does not require reconstructing intent from generated code.

## Verification Matrix

Every public capability should eventually carry a matrix with:

| Layer | Required evidence |
| --- | --- |
| Semantic | constructor and validation tests |
| Serialization | JSON safety, round trip, migration, golden file |
| Lowering | form shape and backend capability test |
| Assembly | tiny matrix/vector reference |
| Solve | analytical/manufactured/benchmark quantity |
| Parallel | MPI smoke and declared limitations |
| Human use | readable example and defect-localization task |
| Agent use | generation/repair task with structured feedback |
| Provenance | complete run record and output identity |

Not every early feature needs all rows immediately, but its documentation must
state which rows are missing.

## Decisions to Avoid

- Do not replace readable Python with JSON as the primary human language.
- Do not advertise multi-backend support when only descriptors exist.
- Do not serialize `repr()` values containing memory addresses.
- Do not promise arbitrary UFL portability.
- Do not make a monolithic `solve_everything(...)` API the beginner path.
- Do not hide analysis assumptions inside backend defaults.
- Do not auto-repair scientific intent without an explicit policy.
- Do not treat examples as verification benchmarks.
- Do not expand feature coverage faster than validation evidence.
- Do not describe successful execution as proof of physical correctness.

## Near-Term Implementation Queue

The next concrete code changes should be:

1. Unify linear, nonlinear, and transient outputs through `SimulationResult`;
   finish reactions, energies, point/path probes, projection, and restart.
2. Harden Neo-Hookean nonlinear-static solving and build the quadrature-state
   protocol required for a real J2 finite-element step.
3. Add validation for region emptiness, partition overlap/completeness, target
   mesh consistency, and step/operator completeness.
4. Complete external mesh volume/boundary region import with real Abaqus,
   Nastran, Gmsh, Exodus, and MED golden files.
5. Expand the test-linked benchmark registry with expected quantities,
   tolerances, and mesh/time convergence.
6. Record solver convergence reason and iteration count in every result.
7. Add a model-owned transient heat step and checkpoint/restart.
8. Integrate J2 plasticity only after local state, consistent tangent,
   increment cutback, and single-element path tests are complete.
9. Keep the Abaqus-equation serial/two-rank parity gate and extend focused MPI
   validation to transient state, diagnostics, scaling, and deformed output.
10. Add a global adaptive creep step and reproduce closed-form/NAFEMS cases.
11. Add fatigue stress-history extraction and field-result provenance.
12. Complete standard QoIs for reactions, energies, curves, and
    mesh-independent field samples.
13. Add an execution-service protocol so local plan shards, Slurm jobs, and
    hosted runners produce the same case records.
14. Add Gaussian-process/ensemble uncertainty adapters and active-learning
    proposal records before automatic retraining.
15. Implement field projection and reviewed residual families before enabling
    neural-operator or PINN training adapters.
16. Revisit AF-IR loading/migration/object IDs only alongside an executable
    consumer and golden cases; do not let schema work displace product gates.

## Success Criterion

AgentFEM succeeds in the near term if it becomes a finite-element environment
that engineers can actually use and trust for its supported analyses.

It succeeds in the longer term if a scientific model can outlive:

- the prompt that produced it,
- the particular model that authored it,
- the Python process that executed it,
- and eventually the first backend that compiled it,

while remaining understandable to the people responsible for the result.
