# Validation Reference

Minimum validation after code changes:

1. Compile/import touched modules.
2. Check mesh summaries and required tags when a workflow depends on labels.
3. Check problem/material/load/constraint summaries when public modeling assets
   changed.
4. Run a small serial case when the change affects execution.
5. Run a small MPI case when the change affects parallel behavior.
6. Report untested assumptions.

For modeling changes, also check units, boundary-condition type, and output
availability.
