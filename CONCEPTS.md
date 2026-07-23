# AgentFEM Concepts

This file defines the vocabulary that humans and AI agents should use when
working with AgentFEM.

## Mesh

The computational domain and its topology/geometry. Mesh generation may be
application-specific, but standard mesh import, reading, tagging, and boundary
measures should use reusable helpers when possible.

External CAE mesh formats such as Abaqus input, NASTRAN bulk data, VTK, and
COMSOL-exported neutral formats should be converted into a DOLFINx-readable
format before analysis. AgentFEM treats this as mesh-format conversion, not as a
physics or solver concern.

## Function Space

The finite-element approximation space for scalar, vector, or mixed unknowns.
Examples include displacement, temperature, pressure, and internal variables.

## Field

A finite-element function living in a function space. Fields may represent an
unknown, a state variable, a coefficient, or an output quantity.

## Constitutive Law

A material or field law that maps local state to stress, flux, or another
response quantity. Elasticity belongs here. Future viscoelasticity, plasticity,
thermal conduction, and coupled material laws should also live here.

## Constraint

An essential or algebraic restriction on degrees of freedom. Dirichlet data,
periodicity, and MPC relations are constraints.

Neumann force, flux, and traction terms are not constraints.

## Load

A weak right-hand-side source term, such as body force, heat source, Neumann
flux, or traction.

## Boundary Model

A weak boundary physics model that is not simply an external load. Robin,
impedance, convection, and absorbing boundaries are boundary models.

## Form

A UFL expression representing a weak-form contribution before assembly.

## Operator

An assembled vector, matrix, diagonal/lumped operator, or linear algebra object
that acts on a field or residual.

## State

A collection of fields used by a solver or time integrator. Examples include
first-order transient states and second-order displacement/velocity/acceleration
states.

## Diagnostic

A scalar or field quantity used to inspect correctness, convergence, stability,
or physical behavior.
