# Extension Rules Reference

Add a helper to AgentFEM only when it is a reusable FEM concept or workflow
operation.

Keep problem-specific geometry, source histories, benchmark constants, and paper
parameters in application packages or examples.

When a new concept is added, update:

- `CONCEPTS.md`
- `WORKFLOW.md` if the workflow changes
- `docs/module_map.md`
- The relevant skill reference
