# AgentFEM Architecture Review

This note records the current architecture direction from the perspective of
traditional finite-element workflows and agent-oriented use.

For the full FEniCSx-first engineering plan, AF-IR schema boundary, backend
strategy, agent repair protocol, execution-evidence design, and phased roadmap,
see `docs/air_architecture_roadmap.md`.

## Current Strengths

- The first-level modules mostly match standard FEM steps: studies, mesh
  import/read, models, spaces, fields, constraints, loads, forms, assembly,
  operators, problems, time integration, solvers, diagnostics, and output.
- Constitutive response relations are below `constitutive/`, while material
  records and property containers are below `materials/`.
- Study assumptions now influence constitutive behavior where implemented,
  including 2D isotropic plane strain and plane stress elasticity.
- Weak boundary physics is below `boundary_models/`, which separates Robin,
  impedance, absorbing, and convection-like terms from essential constraints
  and Neumann loads.
- Application-specific geometry and source definitions are outside AgentFEM,
  keeping the core package reusable.
- External CAE mesh conversion is separated into `mesh/formats.py`, keeping
  solver workflow code independent from Abaqus, NASTRAN, COMSOL, or VTK details.
- The package has both human-facing workflow docs and skill-ready progressive
  references for agents.
- Study summaries, model summaries, mesh summaries, tag checks,
  material-property summaries, load summaries, constraint summaries, and
  boundary-model summaries now provide a first layer of agent-readable
  inspection.
- AF-IR 0.1 provides an explicitly experimental, versioned JSON-safe record of
  supported public model semantics.
- Structured validation issues now carry stable codes, object paths, severity,
  and repair hints.
- Operator compilation crosses a narrow backend adapter boundary while
  FEniCSx remains the only production backend.

## Main Refinements Needed

1. Keep mesh import and mesh regions generic:
   `mesh/` should own reusable Gmsh and XDMF import/read/write operations,
   named mesh regions, tag checks, summaries, and simple structured mesh
   constructors. Application packages still own problem-specific geometry
   construction and meshing parameters. External file conversion belongs in
   `mesh/formats.py`.

2. Keep boundary concepts separate:
   Public strong boundary conditions should enter through `constraints/`.
   `constraints/boundary.py` is a low-level implementation helper for
   Dirichlet constants and dof application. Weak boundary physics belongs in
   `boundary_models/`.

3. Keep study, model, and problem responsibilities separate:
   `studies.py` declares context, `models.py` registers assets and checks the
   model, and `problems.py` represents discrete systems to solve.

4. Make form construction more discoverable:
   `forms.py` should expose small weak-form blocks with clear names, such as
   mass, stiffness, damping, body load, traction, and flux contributions.

5. Keep time integration generic:
   `time/` should contain method-level kernels and step-cadence helpers, while
   application solvers decide which fields, loads, and boundary models enter
   each step.

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

1. Build verification benchmarks and continuous tests around the current
   FEniCSx execution path.
2. Expand addressable validation for regions, assignments, operators, steps,
   and solver policies.
3. Stabilize AF-IR object identity, references, loading, and migrations.
4. Introduce semantic/runtime separation for a small set of proven operator
   families.
5. Add execution records and provenance before pursuing broad multi-backend
   claims.
