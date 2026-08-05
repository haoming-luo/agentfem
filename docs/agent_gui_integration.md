# Agent and GUI Integration Contract

## One solver-facing contract

AgentFEM does not need a separate finite-element implementation for a GUI or
an AI assistant. All frontends should use the same public application path:

```text
human script / IDE agent / visual GUI / remote service
                    |
       project + capability + run contracts
                    |
     AgentFEM Study -> Model -> Step -> Result
                    |
             FEniCSx / PETSc / MPI
```

This boundary also supports a possible commercial GUI. The frontend may own
interaction design, scene rendering, progress presentation, authentication,
and reports; it should not reconstruct weak forms or inspect private DOLFINx
solver objects.

## A practical conversational MVP

A demonstration application can accept a request such as:

> Create a two-dimensional steel plate, clamp the left side, apply a downward
> load on the right, solve it, and explain the largest displacement.

The surrounding agent performs five visible stages:

1. **Problem interpretation** — extract geometry, assumptions, material,
   constraints, loads, outputs, and missing decisions.
2. **Case construction** — create or edit a normal AgentFEM `case.py` from a
   version-matched template.
3. **Preflight** — call `agentfem capabilities --json` and
   `agentfem check --json`; present unsupported assumptions instead of silently
   replacing them.
4. **Deterministic execution** — call `agentfem run --json`, optionally with
   MPI, and monitor the execution artifacts.
5. **Result explanation** — read `execution.json`, `result.json`, fields,
   histories, images, and verification evidence; distinguish computed,
   converged, verified, and validated claims.

The language model is an author and orchestrator. AgentFEM remains the
deterministic engineering tool and evidence producer. An API token belongs to
the surrounding agent service; AgentFEM neither requires nor stores an LLM
credential.

## Current machine interface

The first stable boundary is intentionally process based:

```bash
agentfem doctor --json
agentfem templates --json
agentfem capabilities --json
agentfem init --template static-solid ./case --json
agentfem check --project ./case --json
agentfem run --project ./case --json
agentfem inspect ./case/outputs/case/latest.json --json
```

This can be called from Python, JavaScript, a desktop application, a web
backend, Codex, or another coding agent. It also isolates a frontend process
from PETSc/MPI lifetime and native-library failures.

The contracts are versioned independently:

- `agentfem.project` — operational case identity;
- `agentfem.project-check` — preflight status and addressable errors;
- `agentfem.run` — one execution identity and artifact root;
- `agentfem.solve-events` — solver progress and increment evidence;
- `agentfem.simulation-result` — scientific outputs and trust evidence;
- `agentfem.capabilities` — queryable installed functionality.

GUI code should switch on `schema` and `schema_version`, not on prose printed by
an example.

## Process and service evolution

The local CLI is the MVP boundary. A future asynchronous service can wrap the
same contracts with:

- `submit_run` returning a run ID;
- `get_run_status` and structured event streaming;
- `cancel_run` and `resume_run`;
- artifact and report retrieval;
- campaign submission and dataset status;
- scheduler adapters for workstations, clusters, and cloud jobs.

An MCP adapter becomes useful after that job service exists, especially for
remote or protected compute. It should expose a small tool set such as
`list_capabilities`, `validate_case`, `submit_run`, `get_run_status`, and
`get_result_summary`. It should not expose one tool per material or arbitrary
shell execution.

## Security and control

- Treat generated Python as code: run it in a user-approved local environment,
  container, or restricted service account.
- Keep API tokens outside case files, manifests, logs, and datasets.
- Restrict writable artifacts to the active `RunContext` directory.
- Require explicit user decisions for destructive changes, expensive job
  submission, external communication, and unsupported physical substitutions.
- Keep numerical execution possible with no network and no AI provider.

The next service milestones are structured event streaming, cooperative
cancellation, background job status, versioned report bundles, and a portable
artifact server. They build on the current CLI/run/result contract without
changing the public finite-element language.
