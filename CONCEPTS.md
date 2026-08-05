# AgentFEM Concepts

This file defines the vocabulary that humans and AI agents should use when
working with AgentFEM.

## Study

The early analysis context: analysis type, physics, dimension, and modeling
assumptions. Examples include linear static solid mechanics in 2D plane strain,
transient heat transfer in 2D, or second-order dynamics in 3D. A study is a
modeling declaration and validation object; it does not assemble or solve.
Constitutive laws and operators may use it to select modeling assumptions such
as 2D plane strain or plane stress.

## Solution Procedure

A numerical route used to solve a declared study. The study answers what
physical equations and assumptions define the problem; the procedure answers
how the discrete equations advance. Examples are linear static,
implicit-Euler heat transfer, Standard/Newmark dynamics,
Standard/generalized-alpha dynamics, and Explicit/central difference.

`SolutionProcedure` is initially a compact description used by summaries,
validation, and step-provider dispatch. It should not grow into a second study
object or force every algorithm parameter into beginner-facing code.

## Model

A lightweight registry of mesh, regions, fields, amplitudes, materials,
constraints, loads, and boundary models under a study. A model supports checks
and summaries for humans and agents. It should not hide operator construction
or solving.

## Mesh

The computational domain and its topology/geometry. Mesh generation may be
application-specific, but standard mesh import, reading, tagging, and boundary
measures should use reusable helpers when possible.

A mesh summary records dimension, local/global entity counts, and available
cell/facet tags. A required-tag check is a modeling guard: it should fail early
when a material region or boundary label is missing.

External CAE mesh formats such as Abaqus input, NASTRAN bulk data, VTK, and
COMSOL-exported neutral formats should be converted into a DOLFINx-readable
format before analysis. AgentFEM treats this as mesh-format conversion, not as a
physics or solver concern.

## Mesh Region

A named geometric location on the mesh, such as a boundary, material region, or
point set. Loads and constraints should be described as acting on regions and
targeting unknown fields.

Boundary regions provide restricted `ds(tag)` measures. Cell regions provide
restricted `dx(tag)` measures for material-dependent domain integrals.

AgentFEM should treat regions as the user-facing modeling concept and
`MeshTags`/integer tags as implementation details. The long-term mesh/domain
language should follow:

```text
Domain -> Region -> Assignment -> Step
```

Concrete geometry predicates such as disks, boxes, planes, or user functions
are selectors: they are ways to create regions, not the core concept. A region
may also originate from imported CAE mesh semantics such as Abaqus `NSET`,
`ELSET`, and `SURFACE`, NASTRAN sets/properties, or Gmsh physical groups. These
sources should be mapped into the same AgentFEM region interface so materials,
loads, constraints, boundary models, and outputs can all target named regions.

Application examples should prefer named region collections such as
`regions.matrix` or `regions.left_boundary` over exposing `tag=1`, `tag=2`, or
raw `MeshTags` unless the example is explicitly teaching the low-level mesh
tagging layer.

## Selector

A reusable rule for selecting mesh entities by coordinates, imported tags, or
another backend-specific source. Selectors do not generate geometry or meshes;
they answer whether coordinates or entities belong to a region.

Selectors may be composed with boolean logic, for example complement, union,
and intersection. Convenience selectors such as disks, boxes, planes, and
layers are useful for common examples, but `where(lambda x: ...)` should be a
fallback for custom advanced selection rather than the beginner-facing path.

## Region Set

A named collection of related regions, often representing a cell partition,
boundary grouping, or imported set/surface table. Region sets should support
attribute and dictionary-style access, summaries for agents, and optional
visualization fields when backed by cell tags.

For cell material partitions, AgentFEM should check that selected regions do
not overlap and do not leave unassigned cells unless the API explicitly allows
partial partitions.

## Function Space

The finite-element approximation space for scalar, vector, or mixed unknowns.
Examples include displacement, temperature, pressure, and internal variables.

## Unknown Field

An application-level finite-element unknown that bundles the space, solution
field, trial function, and test function. Unknown fields let beginner workflows
say `displacement` or `temperature` instead of manually managing `V`, `u`,
`du`, and `v`.

## Field

A finite-element function living in a function space. Fields may represent an
unknown, a state variable, a coefficient, or an output quantity.

AgentFEM fields support eager same-space field algebra. For compatible fields
on the same function space, operations such as `u + dt * v` immediately compute
and return a new field with numerical dof values, similar to Cast3M field
operations or tensor-style array arithmetic. This is intentionally different
from symbolic weak-form expressions.

## Amplitude

A named time history or scale factor used to drive prescribed data. Amplitudes
may be constant, ramped, tabular, sinusoidal, or application-defined. They are
model assets, but they are not finite-element fields: they do not own a
function space or spatial degrees of freedom.

Loads, constraints, sources, and prescribed data may reference amplitudes.
Model-owned amplitudes can be registered once and referenced by name. Their
independent coordinate follows the owning Step: a one-solve static analysis
evaluates the step-end value at 1; nonlinear statics evaluate the normalized
load coordinate from 0 to 1; transient procedures evaluate physical time.
An amplitude-driven nonlinear natural load supplies the complete load scale
and is not multiplied by a second hidden ramp. A natural load without an
amplitude follows the Step's default proportional ramp.

## Constitutive Law

A local response relation that maps state to stress, flux, tangent, or another
response quantity. Elastic stress-strain equations belong here. Future
viscoelasticity, plasticity, thermal conduction, and coupled response laws
should also live here.

A constitutive name does not imply a complete FEM analysis. AgentFEM records
whether a capability is FEM-integrated, material-point verified, or a
postprocessor. The first global J2 route has committed/trial quadrature state,
an algorithmically consistent tangent, rollback/cutback, and restart
equivalence evidence, but is deliberately serial-only while its distributed
state contract is still being verified. Temperature-dependent Arrhenius
power-law creep remains a local constitutive capability, not a claimed global
creep analysis. Path-dependent material FEM integration additionally requires
quadrature state, a consistent tangent or documented alternative, increment
control, convergence evidence, and restart behavior.

## Material Record

A reusable set of material constants with units and provenance. Material
records are data assets. Constitutive laws are equations. Keep these concepts
separate so one material record can be used by different solvers or law
families when appropriate.

## Material Properties

A typed parameter object created directly by a user or loaded from a material
record. For example, isotropic elastic properties store `young`, `poisson`, and
`density`; the constitutive relation uses those properties to compute stress.

## Constraint

An essential or algebraic restriction on degrees of freedom. Dirichlet data,
periodicity, and MPC relations are constraints.

Neumann force, flux, and traction terms are not constraints.

## Load

A weak right-hand-side source term, such as body force, heat source, Neumann
flux, or traction.

Natural boundary data is expressed as a load because it enters the weak form,
not as a direct dof restriction.

## Boundary Model

A weak boundary physics model that is not simply an external load. Robin,
impedance, convection, and absorbing boundaries are boundary models.

## Form

A UFL expression representing a weak-form contribution before assembly.

## Operator

An assembled vector, matrix, diagonal/lumped operator, or linear algebra object
that acts on a field or residual.

At the application layer, operators should read like engineering finite-element
notation: `K u = F`, `M a + C v + K u = F`, or
`(C/dt + K) T = C T_old/dt + Q`. AgentFEM operator constructors may still
store UFL expressions before assembly, but their names should communicate the
matrix/vector role.

Operator contributions should remain addable. A multi-material stiffness may be
represented explicitly as `K = K1 + K2` or, preferably in audited scripts, as
`operators.combine(K1, K2, name="K")`.

Model-first helpers such as `model.stiffness(u)` may generate these
contribution sums from registered materials and regions, but the generated
operator should still be inspectable.

## Analysis Step

A solve stage under a study. A step records the analysis method, time increment
when relevant, visible operators, boundary conditions, and solver options.

Examples include a linear static step solving `K U = F` and an implicit Euler
heat-transfer step solving `(C / dt + K) T_next = C T_old / dt + Q`.

Transient steps share one result lifecycle. Calling
`step.solve_result(output="results.xdmf")` advances the procedure, writes its
default primary/state fields to one XDMF/HDF5 series, records every accepted
time increment, and attaches those artifacts to the returned
`SimulationResult`.

The step should not hide the finite-element meaning. It is the place where
visible operators become a solveable algebraic problem.

## Increment, Attempt, and Iteration

An Increment advances load or time within one analysis Step. An Attempt is one
try at that Increment; a failed nonlinear attempt may be rolled back and
retried after a cutback. An Iteration is one Newton correction inside an
Attempt.

Automatic incrementation chooses accepted Increment sizes from convergence
behavior. Its `max_increments` value is a termination limit, not a requested
count. A fixed subdivision is a separate, explicit policy. Newton `max_it`
limits Iterations in one Attempt and therefore has different semantics.
For stateful materials, a converged global Attempt may still be rejected when
a declared inelastic-state increment is excessive; that is a physical/numerical
acceptance control, not a Newton tolerance.

## Output Interval and Frame

An output interval or time point requests when fields should be persisted. A
Frame is the saved result state produced at one such point or accepted
Increment. Frames do not drive the nonlinear algorithm. Exact output marks may
require the solver to land at those values, while cutbacks may introduce
additional internal Increments.

## Element Policy

A reusable interpolation, quadrature, or integration choice. DOLFINx should
remain the backend, but AgentFEM may eventually name common element policies for
consistent human and agent workflows.

## State

A collection of fields used by a solver or time integrator. Examples include
first-order transient states and second-order displacement/velocity/acceleration
states.

For explicit second-order dynamics, the state may also store a mid-step
velocity such as `v_mid`, because central-difference boundary damping and
absorbing boundaries often use half-step velocity data.

## Quadrature Transaction

The atomic committed/trial state mechanism used by integration-point material
models. A transaction begins from committed state, allows repeated trial
updates during global iterations, commits all state variables only after an
accepted increment, and rolls every trial variable back after rejection.

The transaction does not define stress or a material law. J2, creep, damage,
and user-material adapters supply distinct updates and consistent tangents
while sharing state-transition and restart semantics.

## Time Integrator

A time integrator advances a transient state according to a named numerical
method. AgentFEM provides implicit Newmark and generalized-alpha descriptions
for linear structural dynamics and
`time.explicit.central_difference(...)` for explicit central difference, i.e.
the Newmark family with `beta=0` and `gamma=1/2`.

Integrator names should describe the analysis route first, such as explicit
dynamics, and expose method parameters through `summary()` rather than forcing
beginner workflows to start from algorithm parameters.

## Problem

A discrete algebraic or transient system, such as `K x = F`,
`C xdot + K x = F`, or `M a + C v + K u = F`. Problems solve systems; steps
describe the analysis stage, studies declare context, and models register
assets.

## Diagnostic

A scalar or field quantity used to inspect correctness, convergence, stability,
or physical behavior.

## Simulation Result

The scientific view of one completed analysis: named quantities of interest,
fields, histories, solver/model metadata, and artifact links. XDMF, CSV, and
NumPy files are result artifacts rather than the result abstraction itself.
A simulation result can supply declared campaign outputs without serializing
live finite-element fields into a tabular dataset. Execution status and
scientific trust are separate attributes.

## Verification Claim and Trust Level

A verification claim binds an observable to a reference, criterion,
applicability domain, evidence record, and passed/failed/inconclusive decision.
The ordered result trust levels are `not_computed`, `computed`, `converged`,
`verified`, and `validated`. A result advances only when the required evidence
passes; an inapplicable theory remains inconclusive.

Routine users apply `exploratory`, `engineering`, or `release` with
`SimulationResult.verify(...)`. Automatic runtime checks remain distinct from
scientific claims, so convenience cannot turn finite output into validation.

`validated` is reserved for explicitly labeled physical or experimental
validation claims. A fixed-mesh Golden or cross-code comparison is verification
evidence, not experimental validation.

## Benchmark

A standard verification problem with expected quantities and tolerances.
Benchmarks validate platform capabilities; examples teach workflows.
The benchmark registry links a stable identifier, criterion, reference,
automated test, and capability maturity.

## AF-IR Document

A versioned, JSON-safe scientific record of the supported AgentFEM model
structure. AF-IR 0.1 is experimental: it records public semantics and marks
unresolved backend runtime objects as opaque. It is not yet a complete
backend-neutral executable serialization.

The readable Python program remains the primary authoring language. AF-IR is
the persistent exchange, validation, provenance, and future lowering artifact.

## Validation Issue

An addressable finding with a stable code, scientific-object path, severity,
message, repair hint, and optional context. Validation reports let people and
agents repair a model at the level of `model.materials[1].region` or
`model.steps[0]`, rather than inferring scientific intent from a Python
traceback.

## Backend Adapter

The explicit boundary through which a supported scientific operator is
compiled and assembled by a numerical backend. FEniCSx is the only production
backend in the current release. The adapter is a seam for progressive
architecture, not evidence of multi-backend coverage.

## Parameter Space

An ordered, typed schema for related simulation cases. Each parameter carries
its admissible values and may carry units, description, nominal value, and
linear or logarithmic scale. A parameter space is scientific input metadata,
not merely a matrix column list.

## Campaign

A reproducible collection of immutable-by-construction case variants. A
campaign binds a parameter space and sampling plan to deterministic case IDs,
fresh case construction, declared output quantities, execution evidence,
failure records, and resumable artifacts.

Within-case MPI and across-case distribution are distinct. All ranks may
cooperate on one FEniCSx solve; deterministic campaign-plan shards may be sent
to separate jobs. Python threads are not assumed to be a safe FEM executor.

## Scientific Dataset

Successful campaign samples together with their parameter schema, output
names, units, shapes, field encodings, case identities, provenance, and
artifact links. A scientific dataset must retain enough information to explain
what a learned mapping means and which simulations supplied its evidence.

## Surrogate Model

A learned or reduced-order approximation to a declared mapping. In AgentFEM,
the model is incomplete as a scientific asset until it has independent
validation evidence, a stated applicability domain, and defined behavior for
out-of-domain inputs.

Surrogates may substitute for repeated solves, accelerate components inside
FEM, or participate in hybrid screening/active-learning loops. These roles
require different evidence.

## Neural Operator

A learned function-to-function map. Its contract must specify field units and
components, mesh/grid/graph/sensor representation, geometry and boundary
encoding, projection to and from FEM spaces, and field/physics validation. An
architecture name such as FNO or DeepONet does not supply this scientific
contract by itself.

## Physics-Informed Model

A learning workflow with explicit strong, weak, or discrete physics residuals
and boundary, initial, interface, or observation conditions. Arbitrary UFL
forms are not presumed to convert automatically into valid PINN residuals.
