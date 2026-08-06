# Start with AgentFEM

The fastest useful path is not to read the entire architecture. Install the
numerical environment, run one small model, inspect its result, and then move to
the guide for your physics.

## The first 15 minutes

1. [Install AgentFEM and create a project](../getting_started.md).
2. Run `agentfem doctor` to verify the numerical environment.
3. Run the generated `case.py` with `agentfem run`.
4. Inspect `result.json` and the field output rather than relying only on a
   successful process exit.
5. Open the [example gallery](../examples/index.md) and choose the nearest
   physical problem.

## Choose a starting point

<div class="grid cards" markdown>

-   **Finite-element user**

    Continue to the [engineering guides](../guide/index.md) and start from the
    physics you already know.

-   **Python or scientific-computing user**

    Read the [core workflow](../guide/index.md#the-public-workflow), then inspect
    the generated [Python API](../reference/api.md).

-   **AI agent or agent builder**

    Use the [agent entry](../agents/index.md), structured CLI, `llms.txt`, and
    machine-readable documentation manifest.

</div>

## A useful mental model

```text
case.py                         human-readable scientific source of truth
agentfem.toml                   operational project metadata
output/run-id/result.json       structured result and trust state
output/run-id/*.xdmf + *.h5     finite-element fields
output/run-id/execution.json    reproducible execution record
```

AgentFEM keeps these responsibilities separate so that terminals, IDEs, future
GUIs, and AI agents can operate the same project without inventing different
simulation formats.
