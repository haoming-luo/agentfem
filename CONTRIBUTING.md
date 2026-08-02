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

## Start With Evidence

AgentFEM is deliberately growing from validated use cases rather than from an
attempt to enumerate every possible finite-element feature. A proposed
scientific capability should normally identify:

- the engineering problem and governing equation it serves;
- at least one reproducible example or benchmark;
- the public object that consumes it (for example a material, operator, step,
  output request, or learning workflow);
- checks for invalid or ambiguous use, not only a successful code path;
- the maturity claim that the available evidence supports.

New public operators should state their role and form arity, compose through
the existing operator algebra where possible, and include both numerical or
symbolic tests and misuse tests. Backend helpers that have no stable scientific
meaning should remain internal.

## Reporting Bugs and Requesting Features

Use the GitHub issue forms. For a solver or scientific-validity bug, include a
minimal reproducer, the expected physical or numerical behavior, and the
runtime report produced by:

```python
import agentfem as af
print(af.platforms.runtime_report().format())
```

Do not attach confidential meshes, material data, credentials, or licenses.
If the issue requires proprietary input, first reduce it to a synthetic case.

## Pull-request Checklist

Before opening a pull request, run the smallest relevant tests and then the
full suite when the environment permits:

```bash
python -m pytest -q
python build_knowledge.py --check --check-imports
python build_docs.py
python release_gate.py
```

Changes to public scientific behavior should update the corresponding user
documentation and, when appropriate, a knowledge card and benchmark contract.
Optional integrations must remain behind lazy imports with actionable missing-
dependency messages. MPI-sensitive changes should be exercised with at least
two ranks.

## Commercial Extensions

The open-source AgentFEM core is Apache-2.0 licensed. Commercial services,
validated industrial workflows, proprietary plugins, and hosted products may be
developed separately from the core package.
