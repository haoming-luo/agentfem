# Validation Reference

Minimum validation after code changes:

1. Compile/import touched modules.
2. Check mesh summaries and required tags when a workflow depends on labels.
3. Run `model.validate()` and report issue codes and paths when public modeling
   assets changed.
4. Check AF-IR JSON safety and determinism when serialization changed.
5. Run a small serial case when the change affects execution.
6. Run a small MPI case when the change affects parallel behavior.
7. Report untested assumptions and unsupported backend capabilities.

For modeling changes, also check units, boundary-condition type, and output
availability.
