# AI-native workflow: choose existing contracts over new abstractions

## Priorities and acceptance

1. Preserve structured repair diagnostics through execution. The existing
   StepOptionContract already owns option issues; execution must not force agents
   to recover those fields from prose. Keep TypeError compatibility and a failing
   exit status. Test actual CLI persistence, not only a helper.
2. Audit current Model.summary, manifest and AF-IR coverage with a few ordinary
   tasks. Add only demonstrated missing semantics; do not introduce a second
   editable model, new modeling language or automatic arbitrary case execution.
3. Use representative agent tasks to measure repair attempts, source lookups,
   tokens and human interventions. Numerical tests do not establish these UX
   improvements by themselves. No token/success-rate gain is claimed yet.

Defer autonomous optimization, wholesale architecture replacement and new
material families. Preserve the physical roadmap and current P92 work rather
than mixing scientific experiments into this interface change.

## Implemented first increment

StepOptionError carries a defensive copy of the enforcing contract's issues.
It remains a TypeError. CLI rank-error collection and the primary execution
record retain its details, provider and first issue code under stage
`step_options`. Raw option values and material objects are not in details;
existing tracebacks are unchanged and may contain source context.

No automatic repair, rerun, installation, network action or solver change is
introduced. Unsupported combinations still require their existing scientific
admission checks; this error type is not a universal capability certificate.

Targeted evidence: provider/common-workflow/inelastic-control tests initially
120 passed, one MPI-only test skipped; project CLI and new structured-error tests
37 passed. These runs overlap and are not to be added as a unique test count.
The CLI test confirms nonzero exit, persisted provider/issue/suggestion fields
and that code following the rejected request is not executed.

Items 2 and 3 remain next work, not completed work. No five-hour continuous
execution or remote CI claim is made by this checkpoint.
