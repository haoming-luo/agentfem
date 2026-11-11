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

## Continued audit: data integrity and provider selection

Reproduced and corrected a provider-selection mismatch: capability inspection
could stop at a high-priority provider missing required inputs while execution
selected a ready lower-priority provider. Inspection now prefers the same ready
provider, preserving partial-input diagnostics when none is ready.

Unsupported combinations retain structured candidate/rejection evidence in
`StepCapabilityError` and CLI execution records. This remains compatible with
`NotImplementedError`; it does not authorize changing the modeled physics.

NumPy scalar/array conversion previously caught recursive serialization errors,
hiding nonfinite numeric data as an opaque object. Conversion now separates
conversion failure from invalid converted data. Tests cover NaN, infinities,
valid arrays and self-returning conversion methods. IR publication reuses the
existing atomic writer; a failed replace preserves the prior document and
cleans its temporary file.

The shared atomic text implementation is now runtime-independent; checkpointing
keeps its public wrapper. A subprocess test proves standalone IR publication
does not import DOLFINx or initialize MPI. The corresponding coefficient-value
description path also rejects invalid NumPy values instead of hiding them.

CLI `--json` argument errors now have an addressable error envelope before
runtime startup. Normal commands reuse the parsed request instead of parsing
twice; human errors and case arguments retain their existing behavior. Local
diagnosis points Step failures to their actual contract evidence rather than
recommending syntax-only checks for a model-level problem.

Additional evidence (overlapping suites, not additive counts): 192 tests passed
with one MPI-only skip for material, coupled and finite-strain Step lifecycles;
181 passed with one MPI-only skip for IR, checkpoint/output and project execution;
93 CLI/feedback/documentation checks passed after argument handling changes.
The separate two-rank IR failure test passed after atomic publication changes.

## Independent integration branch

The AI workflow changes are isolated on `feat/ai-native-workflow-contracts`
from verified main `10ab18b0`, without the pending P92/life-observation commits.
The combined inspection, CLI, project, feedback and workflow acceptance on that
base passed 246 tests with one intentional MPI-only skip; generated documentation
was current. Four complete history-material CLI examples also passed on the
updated command entry point before isolation.

The CLI JSON encoder now refuses nonstandard NaN/Infinity tokens. Invalid
numeric output produces a failed CLI record rather than a successful malformed
JSON response or an invented null value. The focused follow-up passed 129 tests.
These suite counts overlap; they are not a cumulative total.

## Scoped discovery follow-up

Before the change, `capabilities materials --json` still returned the full
record (about 94.5 KB). Human topics existed but machine topics did not select
sections. Topic JSON now preserves the same selected evidence and declares
`scope.complete=false`; no-topic JSON retains its full historical shape.
Only requested sections are built from their existing owners. Workflow discovery
does not probe the numerical runtime or import the solver merely to list methods.

Measured UTF-8 output in this checkout: full 94,531 bytes, workflow 20,521,
materials 28,423, named `abaqus_user_material_bridge` 1,092. Runtime version
strings can change the full size. This is a payload-size observation, not a
claim about LLM tokens, accuracy or simulation speed. Exact evidence equality,
scope markers, unknown topics, no unrelated runtime probe and standalone
workflow imports are covered by tests; the targeted suite passed 100 tests.

An optional `capabilities --api MODULE.FUNCTION --json` query addresses another
observed gap: knowing module names does not reveal accepted keywords. It reads
the installed Python signature instead of maintaining a second parameter catalog.
Declared `models.Model` methods retain lifecycle evidence and omit the implicit
`self` argument. No callable, annotation expression, default-object repr, case or
installation is executed. Public functions may import their numerical module;
this is not a claim of dependency-free inspection of every scientific function.
Provider-specific keywords still require the existing Step option contract.
