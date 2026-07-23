# AgentFEM Architecture Review

This note records the current architecture direction from the perspective of
traditional finite-element workflows and agent-oriented use.

## Current Strengths

- The first-level modules mostly match standard FEM steps: mesh import/read,
  spaces, dofs, constraints, loads, forms, assembly, time integration, solvers,
  diagnostics, and output.
- Constitutive laws are below `constitutive/`, which matches finite-element
  language better than placing material laws directly at the first level.
- Weak boundary physics is below `boundary_models/`, which separates Robin,
  impedance, absorbing, and convection-like terms from essential constraints
  and Neumann loads.
- Application-specific geometry and source definitions are outside AgentFEM,
  keeping the core package reusable.
- External CAE mesh conversion is separated into `mesh_formats.py`, keeping
  solver workflow code independent from Abaqus, NASTRAN, COMSOL, or VTK details.
- The package has both human-facing workflow docs and skill-ready progressive
  references for agents.

## Main Refinements Needed

1. Keep mesh import generic:
   `mesh.py` should own reusable Gmsh and XDMF import/read/write operations,
   while application packages own geometry construction and meshing parameters.
   External file conversion should stay in `mesh_formats.py`.

2. Separate boundary concepts more clearly:
   `boundary.py` currently means Dirichlet helper operations, while
   `boundary_models/` means weak boundary physics. This is acceptable, but the
   naming should stay explicit in docs and APIs.

3. Promote problem definition:
   `problems.py` should gradually become the standard place for reusable
   problem/state containers, not a bag of case-specific records.

4. Make form construction more discoverable:
   `forms.py` should expose small weak-form blocks with clear names, such as
   mass, stiffness, damping, body load, traction, and flux contributions.

5. Keep time integration generic:
   `time.py` should contain method-level kernels, while application solvers
   decide which fields, loads, and boundary models enter each step.

6. Add examples only after APIs stabilize:
   Examples should demonstrate the workflow order without becoming hidden
   framework logic.

## Agent-Oriented Refinements

- Every public helper should say what FEM concept it belongs to.
- Public function names should be explicit enough that an agent can choose them
  without reading implementation details first.
- Docs should prefer routing tables and decision rules over long prose.
- Error messages should explain modeling mistakes, such as mixing Dirichlet
  constraints with Neumann loads.
- Validation docs should name the smallest useful check for each layer:
  import check, form-shape check, assembly check, serial run, MPI run, and
  ParaView output check.

## Suggested Priority

1. Tighten docs links and module routing.
2. Add small examples for each standard workflow step.
3. Improve `forms.py` into a clearer library of reusable weak-form blocks.
4. Add typed containers for standard analyses: static, first-order transient,
   and second-order dynamics.
5. Add constitutive submodules only when at least one application needs them:
   anisotropic elasticity, viscoelasticity, thermal conduction, and coupled
   models.
