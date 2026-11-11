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

## Semantic inspection and real solve acceptance

The AF-IR model export now retains the declared unit system and engineering
stages, including their inheritance and previous-stage relationship. These were
already in Model.summary but missing from the export. Missing declarations stay
missing: exporting does not invent SI units or convert values.

Serialization now orders sets deterministically and rejects distinct mapping
keys that would collapse into the same JSON key, rather than losing data.
AF-IR remains an inspection document, not a promise of lossless reconstruction
of arbitrary Python callbacks or an alternative editable model.

A real finite-element acceptance test inspects a cantilever, rejects a misspelled
Step option using the same contract as capability discovery, verifies that the
model is unchanged, applies an explicit correction, and solves. Acceptance checks
finite nonzero displacement and relative force imbalance below 1e-10. This is a
workflow and equilibrium regression, not a mesh-convergence study or an LLM
benchmark.

Next: evaluate representative agent tasks (item 3) before claiming fewer tokens,
fewer interventions, or superiority over another CAE product. No five-hour
continuous execution or remote CI claim is made by this checkpoint.

## Inspection reliability follow-up

Manifest ordering and the text tree now include units, amplitudes and engineering
stages. No second modeling state is introduced.

The distributed IR writer previously let rank zero raise an I/O error while
other ranks waited at a barrier. The write outcome is now broadcast; every rank
raises an addressable `AFM-IR-WRITE` error on publication failure. This addresses
the root-owned write phase, not arbitrary rank-local failures during collective
model validation.

Evidence: 70 targeted serial tests passed. A separate two-rank regression with
a 90-second launcher timeout used an invalid file destination and verified
identical errors on both ranks; each rank passed. No production simulation or
performance improvement is claimed from this reliability fix.
