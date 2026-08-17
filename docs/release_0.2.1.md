# AgentFEM 0.2.1

AgentFEM 0.2.1 is a workflow-convergence release. It reduces the public path
for supported analyses to one readable sequence:

```python
step = model.step(target=u, output="results.xdmf")
result = step.solve_result()
```

The same completion verb now covers linear statics, nonlinear finite-strain
statics, transient heat transfer, Standard and Explicit structural dynamics,
J2 plasticity, and implicit creep. Each result retains the owning model,
target, material, execution evidence, and output contract. Existing
`solve_result(output=...)` calls remain valid.

Every built-in Step provider declares its accepted and required options.
Invalid or misspelled keywords are rejected before finite-element assembly,
and the same contract is available to people, IDEs, GUIs, and agents through
capability JSON. Bundled cases now use physical Study factories and the common
`model.step(...).solve_result()` lifecycle while the 0.2.x compatibility
surface remains executable.

## Public surface

- `public_api("core")` identifies the daily engineering language.
- `public_api("advanced")` adds campaigns, fracture, mechanics, results
  pipelines, and learning bridges.
- `public_api("expert")` exposes backend, operator-development, validation,
  and extension seams.
- `agentfem capabilities --json` and `docs/agentfem.json` publish the same
  progressive discovery contract for agents, IDEs, and future GUIs.
- `models.model_api("core" | "advanced" | "compatibility")` separates the
  daily model language from expert and migration surfaces.
- `skills/agentfem/` is a standards-compatible, progressively disclosed coding
  skill that is versioned with the API and scientific validation rules.

Imported `FEMMesh` and Abaqus mesh facades may be passed directly to public
field constructors and models. Constraint component selectors accept either
integer indices or `"x"`, `"y"`, and `"z"` where meaningful.

The scientific manual now waits for MathJax startup during instant page
navigation and typesets only newly introduced formula nodes, so equations no
longer require a browser refresh to appear completely.

## Release evidence

The candidate is accepted only after:

- the complete serial test suite;
- two- and three-rank MPI tests;
- checkpoint restore across MPI partition counts;
- source and wheel identity checks;
- execution of every installed project template;
- release-facing static, thermal, wave, C3D10H, J2, creep, and
  simulation-to-learning workflows;
- strict documentation and scientific-knowledge checks.

Scientific maturity remains capability-specific. Workflow convergence does
not promote experimental fracture formulations or unvalidated engineering
applications.
