# Contributing to AgentFEM

AgentFEM welcomes focused contributions that make finite-element workflows more
transparent, reusable, and agent-readable.

## License of Contributions

Unless explicitly agreed otherwise in writing, all contributions submitted to
AgentFEM are licensed under the Apache License, Version 2.0.

By opening a pull request or submitting a patch, you confirm that you have the
right to contribute the work and that it can be distributed under the project
license.

## Developer Certificate of Origin

AgentFEM uses a lightweight Developer Certificate of Origin style rule. Add a
sign-off line to commits when contributing:

```text
Signed-off-by: Your Name <your.email@example.com>
```

This states that you wrote the contribution or otherwise have the right to
submit it under the project license.

## Contribution Direction

Good contributions should keep the public workflow readable:

```text
Study -> Model -> Mesh/Regions -> Fields -> Materials -> Loads/Constraints
      -> Operators -> Step -> Solve -> Diagnostics/Output
```

Prefer small, well-scoped changes. Public APIs should use finite-element
language before backend-specific language. Lower-level DOLFINx/PETSc details
are welcome when they are isolated behind clear AgentFEM concepts.

## Commercial Extensions

The open-source AgentFEM core is Apache-2.0 licensed. Commercial services,
validated industrial workflows, proprietary plugins, and hosted products may be
developed separately from the core package.
