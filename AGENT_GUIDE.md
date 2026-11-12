# AgentFEM Agent Guide

Help the user define, run and understand a finite-element model through the
same public workflow they can read and edit. A prepared project runs without
an AI service; the agent assists with setup and modeling, not the calculation.

## Choose the shortest path

| User task | Start with | Finish with |
| --- | --- | --- |
| Install | The platform route in `agentfem.json` or [INSTALL.md](INSTALL.md) | Environment readiness; offer an optional first example |
| Run an existing project | Its `AGENTS.md`, `case.py` and `agentfem.toml` | `check → run → inspect` |
| View a result | `agentfem show latest --json` | Requested quantities or artifact paths; no new solve |
| Modify a model | The nearest example and installed provider options | Targeted check, run and comparison |
| Upgrade or diagnose | `doctor --json`; `upgrade --json` for migration findings | Readiness/failure explanation; offer a relevant regression |
| Extend the solver | `WORKFLOW.md`, `CONCEPTS.md` and the relevant module guide | Implementation and numerical evidence |

Use `--json` when consuming CLI output. Reuse the known environment and
capability information within a task; refresh it after an environment/version
change or a relevant failure. `agentfem check` already includes upgrade
findings, so a routine run does not need a separate upgrade audit.
Read only the references needed for the current physics or failure.

## Installation and environment changes

Follow [INSTALL.md](INSTALL.md); on Windows use
[WSL → Ubuntu → AgentFEM](docs/windows_agent_installation.md).
Do not require a GitHub clone. Retain the selected mirror and environment
path in the local handoff note. Standard installation includes SciPy,
Matplotlib and Gmsh; reuse a working environment instead of rebuilding it.
Use the user's chosen channel, or the documented mirror if the canonical
source is unreachable. Keep the compiled numerical stack on one channel.

Do permitted work directly. If approval, restart or a private password entry
is needed, present one specific next action and resume from the completed
stage. Preserve projects and credentials; never bypass an agent's security
policy. On WSL, prepare `agentfem workspace --protect` once and verify project
custody before replacing/removing a runtime. Detailed recovery and alternate
installation routes belong in the installation guide, not every response.

On Windows, show three milestones: `1/3 WSL`, `2/3 Ubuntu and account`,
`3/3 AgentFEM`. Say what is ready and whether any user action is needed.
For a new imported Ubuntu, prepare the ordinary working account directly
when authorised; no online registration or shared password is required.
If a download needs a manual handoff, provide the verified architecture-matched
link and one action (for example, download and double-click the Microsoft
kernel installer). Distinguish legacy kernel repair from modern WSL updates.
Keep long download/check/import recipes as agent references. A novice receives
only the current short command block in the correct shell, then a checked
success message before the next step. Preserve checksum checks behind this
simple presentation.

For separate agent tool calls, use `mamba run -n agentfem-env ...`; activation
in a previous shell may not persist. Use the installed manager's path if
needed. On installation completion, promptly report the installed version,
environment location and readiness. Then ask: “Would you like to try a small
example, or work on your own model?” Installation is complete without either.
Create or run a project only when requested. If the user already asked to install
and run an example, that request is sufficient; do not ask again.

Offer one relevant next step, not an automatic task queue. A suggestion does not
authorize a solve, benchmark, tuning sweep or extra integration. Reuse the task's
known context; do not repeat discovery or checks without a relevant change.
When a requested example finishes, report its result path and useful checked
quantity, then stop.

## Modeling and results

Start with a physical Study, named mesh regions, materials and loads, then
`model.step(...)`. Keep physics in `case.py` and operational settings in
`agentfem.toml`. Prefer the installed public API and its provider option
contracts; inspect `agentfem capabilities materials` or another relevant
topic before requesting the full JSON inventory. The installed version is
authoritative when the website describes newer development work.

Publish through the standard Result lifecycle. Use the manifest's recommended
visualization artifact and the existing writers, rather than reconstructing
connectivity or XDMF in the case. `agentfem verify` checks result provenance;
numerical acceptance also needs the relevant convergence, balance or reference
comparison. Match the depth of that check to the user's question.

For accepted response-material histories and restart, read
[process output](docs/accepted_process_output.md). Output frames and restart
checkpoints have separate schedules; report available frames rather than
interpolating unsaved damage states.

## When more is needed

- Parallel projects: `agentfem run --mpi N`; bounded MPI diagnostics:
  `agentfem mpi-run -n N --timeout 600 -- ...`.
- Repeated failures: `agentfem diagnose --json`, then the relevant guidance
  from `agentfem assist --force`. Keep support reports private unless the user
  authorizes sharing.
- Runtime replacement: check `agentfem workspace --json`; use the documented
  lifecycle tools for a Complete Runtime.
- Privacy and voluntary support: `agentfem telemetry status --json` and
  `agentfem support`. These are not prerequisites for using the solver.
  Account actions and public reports require the user's explicit consent.

## Read only the relevant reference

- Ordinary modeling: [workflow](WORKFLOW.md) and the nearest example.
- Uncertain parameter: `agentfem capabilities --api MODULE.CALLABLE --json`.
- Unsupported Step: inspect `error.details()` or `execution.json`; use
  `models.step_capability(model, target=..., options=...)` for that model.
- Numerical extensions or specialized physics: search the topic headings in
  [scientific agent reference](docs/agents/scientific_reference.md), then read
  only the matching section and its linked source guide.
- Optional MCP: [connection and tool scope](docs/agents/mcp.md). Terminal users
  do not need to install it.

Query installed features rather than assuming the website and installed
version match. On older versions without scoped/API queries, consult the
installed help or request the full inventory explicitly; do not auto-upgrade.
