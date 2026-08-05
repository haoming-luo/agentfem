# AgentFEM Repository Instructions

AgentFEM is a FEniCSx-first finite-element platform. Keep one readable public
Python workflow for humans, terminals, GUI clients, and AI agents.

## Environment and checks

- Use the `fenicsx-env` conda environment for local numerical tests.
- Run targeted tests before the full suite.
- Run examples from the source parent or install the built wheel; do not rely
  on source-path insertion in new installed-use templates.
- Preserve user changes in `README.md`, `CITATION.cff`, and unrelated files.

## Architecture rules

- `Study` states the physical analysis; `SolutionProcedure` states the
  numerical route.
- Keep materials, constraints, loads, boundary models, steps, and outputs as
  distinct concepts.
- Add common finite-element functionality to its reusable module, not to one
  example.
- GUI and agent integrations call the public Python API or the structured CLI;
  they must not depend on private DOLFINx objects.
- `case.py` is the modeling source of truth. `agentfem.toml` contains only
  operational project metadata.
- Every machine-facing command needs stable JSON, addressable failure, and a
  non-zero exit code on failure.
- A completed solver call is not automatically a scientifically verified
  result. Preserve the `SimulationResult` trust distinction.

## Public workflow

Prefer `studies -> mesh -> models -> fields -> materials -> constraints/loads
-> step -> SimulationResult`. Use `model.step(target=...)`, named regions,
`project.current_run()`, and `RunContext.publish(result)` in new application
templates.

Read `WORKFLOW.md`, `CONCEPTS.md`, `AGENT_GUIDE.md`, and focused files under
`docs/` before extending a scientific concept.
