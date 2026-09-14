# Connect AgentFEM with MCP

AgentFEM MCP is the official, lightweight connection between AgentFEM and
Codex, Claude, or another Model Context Protocol host. It is not an AI model
and it does not replace the finite-element engine.

```text
AI agent → seven typed MCP tools → AgentFEM CLI
         → Study → Model → Step → SimulationResult
         → FEniCSx / PETSc / MPI
```

## Install for Codex

Install AgentFEM first and confirm `agentfem doctor`. Then choose the project
root the agent may access:

```bash
codex mcp add agentfem \
  --env AGENTFEM_MCP_ROOTS=/absolute/path/to/AgentFEMProjects \
  -- uvx --from agentfem-mcp agentfem-mcp
codex mcp list
```

Begin with a concrete request:

> Use AgentFEM to create, validate, run, verify, and briefly explain a 2D
> cantilever. Keep the project and result paths visible.

The adapter discovers the normal AgentFEM executable, including the documented
Complete Runtime and common conda environments. If a desktop host cannot see
it, set `AGENTFEM_COMMAND` to the absolute executable path.

## Why the surface is small

The adapter exposes seven lifecycle tools: describe the runtime, create a
project, validate it, inspect it, submit a run, read run status, and read a
result. The surrounding agent already has normal file-editing ability; MCP
does not duplicate materials, elements, or every solver option as hundreds of
fragile tools.

Numerical work runs in a separate process so PETSc and MPI do not enter the
agent host's lifetime. Paths must remain under approved roots, no shell command
is accepted, and the server cannot access the network. A successful process is
reported as `completed`; it becomes `verified` or `validated` only when the
AgentFEM result contains the corresponding evidence.

See the [public companion repository](https://github.com/haoming-luo/agentfem-mcp)
for Claude-compatible configuration, security boundaries, release evidence,
and the portable AgentFEM workflow Skill.
