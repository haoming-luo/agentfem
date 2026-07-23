# AgentFEM Concepts

This file defines the vocabulary that humans and AI agents should use when
working with AgentFEM.

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

## Element Policy

A reusable interpolation, quadrature, or integration choice. DOLFINx should
remain the backend, but AgentFEM may eventually name common element policies for
consistent human and agent workflows.

## State

A collection of fields used by a solver or time integrator. Examples include
first-order transient states and second-order displacement/velocity/acceleration
states.

## Problem

A lightweight description of the finite-element workflow: mesh, spaces, fields,
materials, constraints, loads, boundary models, and forms. A problem description
is for auditability and orchestration; it should not hide the weak form.

## Diagnostic

A scalar or field quantity used to inspect correctness, convergence, stability,
or physical behavior.

## Benchmark

A standard verification problem with expected quantities and tolerances.
Benchmarks validate platform capabilities; examples teach workflows.
