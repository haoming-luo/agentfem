# AgentFEM Architecture Review

This note records the current architecture direction from the perspective of
traditional finite-element workflows and agent-oriented use.

For the durable FEniCSx-first architecture, AF-IR boundary, backend strategy,
agent repair protocol, and execution-evidence design, see
`docs/air_architecture_roadmap.md`. Public roadmaps communicate direction;
release gates, sequencing, risks, and competitive execution notes are kept in
the private engineering record.

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
   model, `_step_builders.py` constructs built-in scientific procedures,
   `step_providers.py` selects and lowers supported analysis/material
   protocols, and `problems.py` represents discrete systems to solve.
   `Model.step` remains the stable public entry point; adding a material family
   does not justify adding a case-specific method to every model. Historical
   builder methods remain thin 0.2.x compatibility delegates rather than
   parallel implementations.

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

## Current execution-contract decision

`model.step(...)` keeps separate, readable keywords for solver, output,
history, progress, and checkpoint choices. Internally they are normalized into
one immutable `StepExecutionPolicy` and retained by the Step execution
context. This is an inspection and provenance boundary, not a second user
configuration language.

The policy has three consequences:

1. providers receive one consistent cross-cutting contract while retaining
   their own scientific `StepOptionContract`;
2. `solve_result()` can consume construction-time output and transient-history
   requests without case-specific plumbing;
3. result metadata exposes the declared controls to humans, agents, GUIs, and
   provenance tools without serializing live PETSc or DOLFINx objects.

An omitted policy value means the selected provider owns the default. Result
metadata labels these values as `declared_policies`; the executable
`metadata["step"]` record remains authoritative for resolved numerical values.

## 0.3 boundary audit

The 0.3 audit treats file size as a maintenance signal, not as proof of a bad
abstraction. A split is required when two modules own the same scientific
decision or when a lower layer imports an orchestration layer.

- `Model` owns registration, inspection, and compatibility delegates. It does
  not own built-in solution construction. Transient heat lowering now follows
  the same `StepRequest -> provider -> _step_builders -> problem` route as the
  structural, nonlinear-material, and dynamics providers.
- A release regression test requires every historical `*_step()` method to
  remain a thin delegate and prevents providers from calling those
  compatibility methods.
- Constant volumetric heat capacity is resolved once in `materials`, rather
  than being reimplemented by the model facade and the thermal provider.
- Core discovery now contains the ordinary engineering language. Procedure
  and solver controls are advanced; raw `io`, discrete `problems`, and time
  kernels are expert. This changes presentation, not runtime availability.
- `results` owns scientific result semantics and the common completion path.
  Low-level `io.XDMFTimeSeries` remains an expert DOLFINx-compatible writer;
  ordinary workflows use the single-grid result writers and `SimulationResult`.
- Constitutive modules own material-point laws, while `mechanics` owns global
  finite-element integration. Generic checkpointing owns field/time state;
  cohesive checkpointing additionally owns physical interface identity. These
  are deliberate layer pairs, not duplicate implementations.
- `_architecture_contract.py` now defines the seven stable ownership
  boundaries consumed by machine capability discovery. AST regression tests
  reject selected upward dependencies rather than relying on this prose.
- Generic first- and second-order transient states moved from `problems.py` to
  `state.py`; the diagonal mass object moved to `operators`. Compatibility
  aliases preserve existing imports while new code follows the correct owner.
- State has one minimal structural contract for restart and atomic replacement.
  Procedure-specific trial creation remains explicit because a Newton
  increment and a fatigue cycle do not begin with the same scientific inputs.

The next structural split should be evidence-driven: `_step_builders.py` may
become a private builder package when independent provider families need
separate ownership. Splitting it before then would move code without changing
the architecture. WSL2 remains a supported Windows route and an independently
tracked acceptance item; lack of local WSL2 evidence does not block 0.3 after
Linux and macOS installed-wheel acceptance passes.

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

1. Harden the current FEniCSx execution, result, visualization, campaign, and
   dataset path around selected real engineering analyses.
2. Advance nonlinear materials through explicit maturity gates: material
   point, FEM integration, benchmark, then workflow.
3. Verify external mesh volume/boundary set preservation with real format
   fixtures.
4. Expand addressable validation for regions, assignments, operators, steps,
   solver policies, and result contracts.
5. Evolve AF-IR identity/loading/migration only when an executable consumer or
   golden case requires it; do not let schema breadth outrun product evidence.
