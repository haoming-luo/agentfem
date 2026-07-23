# Validation Reference

Minimum validation after code changes:

1. Compile/import touched modules.
2. Run a small serial case when the change affects execution.
3. Run a small MPI case when the change affects parallel behavior.
4. Report untested assumptions.

For modeling changes, also check units, boundary-condition type, and output
availability.
