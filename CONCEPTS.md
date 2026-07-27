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

## Constitutive Law

A local response relation that maps state to stress, flux, tangent, or another
response quantity. Elastic stress-strain equations belong here. Future
viscoelasticity, plasticity, thermal conduction, and coupled response laws
should also live here.

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

The step should not hide the finite-element meaning. It is the place where
visible operators become a solveable algebraic problem.

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

## Time Integrator

A time integrator advances a transient state according to a named numerical
method. In AgentFEM, `time.explicit.central_difference(...)` represents the
explicit central-difference method, i.e. the Newmark family with `beta=0` and
`gamma=1/2`.

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

## Benchmark

A standard verification problem with expected quantities and tolerances.
Benchmarks validate platform capabilities; examples teach workflows.
