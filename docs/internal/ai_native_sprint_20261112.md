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

Coverage audit: all 477 current core function call paths and declared Model
methods produced strict-JSON descriptions. This counts inspected call paths,
not 477 scientific validation cases. A regression now enumerates the current
contract rather than freezing that count. The real MPI publication-failure test
was moved into the existing `test_parallel_cli.py` CI group; all three tests
passed on both local ranks with a 90-second launcher timeout.

Inspection robustness follow-up: named assets no longer evaluate `repr()`
unnecessarily, and non-callable summary metadata is not treated as a method.
Scientific summary failures still propagate. Extended-precision NumPy scalars
can return fresh same-type objects from both `item()` and `tolist()`; those
conversions now terminate with the existing explicit unsupported-value marker,
without silently narrowing precision. NumPy documents this behavior at
https://numpy.org/doc/stable/reference/generated/numpy.ndarray.item.html.
The focused semantic/CLI/discovery regression passed 82 tests after this fix;
generated documentation and correctness lint also passed. CI remains separately
tracked on the pull requests; local counts do not imply CI completion.

Model tree and manifest now reuse AF-IR's workflow ordering, avoiding divergent
human/machine presentation catalogs. A second real workflow exercises missing
thermal material admission, preserves addressable validation evidence and the
unchanged model, then explicitly supplies the material/boundaries and checks
the solved temperature against linear steady conduction. The combined semantic,
validation and engineering-workflow regression passed 89 tests. CLI startup,
project, documentation and architecture regression separately passed 86 tests.

Boundary found during review: topic construction avoids unrelated numerical
imports, but the CLI emitter still consults the real MPI rank. It is not yet a
runtime-free command path. Do not bypass that ownership by guessing rank from
launcher environment variables; no change to MPI emission or telemetry policy
was included in this follow-up.

Public constructor discovery uses Python's installed callable signature protocol
(https://docs.python.org/3.11/library/inspect.html#inspect.signature), restricted
to AgentFEM classes explicitly exported by the selected public module. It does
not instantiate objects or compile material sources. The local coverage audit
described 210 core exported constructors; two exception types expose no Python
signature and remain explicit query failures instead of fabricated parameter
lists. Ordinary functions and declared Model methods retain the same query.

## Operational record integrity

`execution.json` and `latest.json` now reuse the shared atomic text publisher
with unique temporary files. Two concurrent writers previously used the same
`.tmp` pathname; a forced simultaneous-replace regression exercises that race.
The last successfully published latest pointer may refer to either concurrent
run, but it must contain one complete record. This is per-file publication,
not a claim of a multi-file transaction or power-loss durability.

Operational JSON refuses nonfinite values, matching Result manifests and CLI
JSON. Tests preserve the prior record on encoding/replace failure and clean
temporary files. Extended numeric checks also avoid narrowing `longdouble` to
Python float, so invalid extended scalars are rejected without falsely rejecting
the largest finite value. A platform-specific test initially assumed NumPy's
`finfo.max` itself had longdouble type; the test now explicitly constructs that
type without changing the software policy.

Artifact resolution also checks containment after resolving links: a linked run
root (including protected WSL storage) and links within that root remain valid,
but a nested link cannot redirect an artifact outside it. This is path validation,
not a sandbox against hostile concurrent filesystem changes. The targeted
project/path/CLI suite passed 52 tests. A separate 106-test run covered records,
project execution, structured Step failures and Result publication before the
additional path regression. Counts overlap and are not a cumulative total.

## External design cross-check

- [MCP tool results](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2025-06-18/server/tools.mdx)
  separate structured result data and tool execution failures. AgentFEM keeps
  those distinctions in its existing CLI/Result contracts; this sprint does not
  introduce or claim an MCP server.
- [JSON Schema annotations](https://json-schema.org/understanding-json-schema/reference/annotations)
  distinguish defaults from validation-driven value insertion. Our signature
  defaults describe installed callables, not permission to infer material values,
  change physics or auto-fill a case. Type annotations alone are not a full
  scientific admission schema.

The installed MPI supervisor now rejects nonpositive rank counts and nonfinite
or nonpositive timeouts during argument parsing, before probing a vendor or
starting a launcher. Its rank parser is shared with ordinary `run --mpi` rather
than duplicated. No new MPI JSON mode or environment-rank inference was added.
After the operational-record changes, eight complete workflow checks passed:
four material/kinematic example combinations, two pause/restart comparisons,
sparse field output, and sequential thermal-stage reuse. They are bounded
regressions, not a full composite simulation or new physical validation claim.

## Distributed command lifecycle review

The root-owned run phase had two further asymmetric failure paths: directory
selection could raise before the first broadcast, and final execution-record
write/read could raise before status broadcast. Both phases now broadcast their
outcome before peers continue. Root still owns I/O; records are not copied to
every rank just to display a summary. Finalization failures carry
`AFM-RUN-RECORD`. Eight tests passed on each of two ranks with a 90-second launch
limit, including injected selection/preparation/write errors and malformed
records. These tests cover known root-owned phases, not arbitrary failure within
a user's own collective solver. The latest MPI run had no cleanup warning.

Final-record admission additionally rejects a different run identity or a
missing/invalid status rather than treating parseable JSON as completion.
The expanded two-rank group passed 11 tests on each rank. The final combined
local selection passed 312 tests in 49.77 seconds, including the eight complete
workflows; this is one selection, not a sum of overlapping earlier runs.
Generated documentation remained current. PR #185 completed all its CI jobs,
including FEniCSx and macOS installed-wheel acceptance. PR #186 still requires
CI on its final head; neither PR has been merged or released by this sprint.

## Publication ownership in ordinary cases

`RunContext.publish(..., comm=...)` now reuses `SimulationResult.write_manifest`
collective agreement and the existing `provenance.collective_call` for local
preparation and root-owned completion. It does not introduce another parallel
solver or infer a communicator. Serial/root-owned callers retain the default.
Summary text also uses atomic publication. Static-solid, steady-heat and
structural-dynamics templates no longer hand-write a root publication/barrier
pair. The campaign template guards its existing root-only dataset export with
the established collective-call helper instead.

All four templates solved and published verified manifests with two ranks.
Fault injection covers rank-local preparation, root summary/execution writes
and disagreement in result values. A valid scientific manifest remains available
if a later summary/operational write fails; no completed execution record is
invented. The complete parallel CLI group passed 19 tests on each rank.

Serial project/discovery/packaging/documentation compatibility passed 167 tests.
Result metadata now also rejects key collisions after JSON key conversion and
extended NumPy scalars without a lossless Python conversion. Previously the
latter recursed until failure; values are not silently narrowed or replaced.
Supported NumPy scalar/array metadata remains numeric. Invalid metadata leaves
an existing manifest untouched. The Result and metadata regression passed 71
tests; generated documentation and correctness lint remained clean.

Result manifests and checkpoint sidecars now use that same atomic text owner,
instead of shared `.tmp` names or direct overwrites. Existing replacement-failure
tests inject by destination, preserving their failure semantics with unique
temporary filenames. The provenance/history/Result group passed 127 tests with
one MPI-only skip; the actual two-rank CLI plus archive agreement group passed
20 tests on each rank. Checkpoint-sidecar failure retention is also covered.

## Isolated installed-use acceptance

A wheel was installed into a temporary venv reusing the existing numerical
dependencies, without replacing the user's installed AgentFEM. Imports were
confirmed under the venv's site-packages, outside the repository and without
PYTHONPATH or pytest's source-path configuration. This is installed-code
acceptance, not a clean-machine dependency installation or native-speed claim.

31 tests passed against that installed wheel: complete history-material CLI
examples, pause/restart versus continuous histories, rejected material updates,
atomic rejection of corrupt checkpoints, operational record retention and
metadata publication. Pytest emitted only a cache-location warning from the
temporary `/dev/null` configuration; future isolated runs should disable its
cache provider or choose a writable cache path.

The installed console command additionally created and solved a static-solid
project outside the repository. `agentfem verify` returned `verified: true` with
no issues for its result manifest. This verifies artifact integrity, not a new
independent scientific benchmark. The preceding remote head `27fa2721` also
completed all relevant CI, including Linux FEniCSx and macOS installed-wheel
acceptance; follow-up commits still require their own latest-head CI.
