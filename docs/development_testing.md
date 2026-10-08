# Development and verification strategy

AgentFEM uses layered verification. The goal is fast feedback during an edit
and broad evidence before shared code or a release changes—not blindly running
the most expensive command after every keystroke.

## Validation layers

| Moment | Required evidence | Typical command |
| --- | --- | --- |
| Inner development loop | Direct unit/interface tests for the changed owner | `python -m pytest -q tests/test_extensions.py` |
| Before committing | Related workflow tests, critical static analysis, misuse tests, and generated-asset checks | `ruff check . --no-cache`; `python build_knowledge.py --check --check-imports`; `python build_docs.py --check` |
| Before pushing a coherent code change | Related suites; complete serial for cross-cutting numerical changes | `python -m pytest -q` when the change can cross ownership boundaries |
| MPI-sensitive change | Relevant two-rank modules using the verified launcher and a bounded test budget | `agentfem mpi-run -n 2 --timeout 600 -- python -m pytest ...` |
| Test-only or orchestration pull request and `main` | Changed test modules, a built-wheel solve, and serial/two-rank smoke | GitHub Actions `targeted` tier |
| Core numerical pull request and `main` | Wheel installation, owner-selected serial/MPI suites, affected portable-restart drivers, smoke tests, and documentation | GitHub Actions `core` tier |
| Documentation-only pull request and `main` | Strict generated-document and site build; no FEniCSx or PyTorch environment rebuild | GitHub Actions `docs` tier |
| Release candidate/tag | All preceding checks plus distribution inspection and installed-wheel release smoke | `python release_gate.py --dist dist --smoke` |

An extension promotion additionally requires two independently built wheels.
`extension_gate.py` installs them into an isolated prefix, executes a declared
project outside the checkout, verifies the sealed result, and proves that the
installed core tree was unchanged. Simulated entry-point tests do not replace
this evidence.

The 0.4 foundation certificate is an aggregation step, not another expensive
test matrix. On the release tier, `foundation_gate.py` validates the installed
smoke and external-provider artifacts after the serial and representative MPI
stages have passed, then binds those stages to the clean source commit and
candidate wheel digest. Omitting any required stage, changing the wheel, or
mixing extension evidence from another commit fails closed. The resulting
record is consumed with `promotion_gate.py --target 0.4-foundation`.

## Transient lifecycle maintenance

The post-0.4 maintenance slice uses these focused checks in `fenicsx-env`:

```bash
python -m pytest -q tests/test_operator_lifecycle_ledger.py \
  tests/test_first_order_operator_lifecycle.py tests/test_implicit_dynamics_lifecycle.py \
  tests/test_transient_restart.py tests/test_sprint_transient_rollback.py tests/test_time_inputs.py
agentfem mpi-run -n 2 --timeout 600 -- python -m pytest -q \
  tests/test_operator_lifecycle_ledger.py tests/test_first_order_operator_lifecycle.py \
  tests/test_implicit_dynamics_lifecycle.py -k 'collective or live_transferred'
PYTHONPATH=src python tools/benchmark_operator_identity.py
```

The measurement compares the former Python-int tuple snapshot with an exact
array-byte snapshot, including equality comparison. It does not measure a
complete solve or promise an end-to-end speedup. On the development macOS ARM64
host (Python 3.11.15, NumPy 2.4.6), seven repeats of 100 checks at 100,000
boundary indices measured approximately 3.79 ms versus 0.012 ms per check;
single-snapshot peak Python allocation was 4.12 MB versus 0.40 MB. Reproduce
locally rather than enforcing wall-clock thresholds in CI. Both paths remain
linear in the number of boundary indices; this is not a constant-time cache.

The original 0.4.0 code was independently replayed for a four-increment heat
and dynamics run, releasing the prepared system after increment two. It
reported one matrix and two solves instead of the actual two matrices and
four solves. The shared ledger retains totals across prepared generations.
Checkpoint restoration starts a new execution-cost ledger, not fabricated
timings from the previous process. Integration equations and accepted-state
history remain owned by their existing Procedures.

Live-input tests additionally compare both RHS-only updates and operator
updates with the independent implicit-Euler recurrence for a spatially uniform
linear first-order problem. They check the computed field and assembly counts
in serial and MPI, not only the declared capability metadata. This protects a
coupling prerequisite; it is not a validation of a complete coupled solver.

## Source and installed-wheel evidence are separate

The repository uses the standard `src/agentfem/` package layout. Pytest is
configured with `pythonpath = ["src"]`, so `python -m pytest` exercises the
current checkout rather than an older `agentfem` already present in the
environment. The release gate follows the opposite rule: it installs
the candidate wheel into an isolated target, rejects a source-checkout import,
compares every packaged runtime file with the candidate source, and then runs
the flagship workflows and installed project templates. A release therefore
needs both source evidence and installed-artifact evidence.

Build a release candidate from a clean packaging workspace. Setuptools may
reuse files under an old local `build/` directory even after those files have
left the source tree. CI starts from a fresh checkout; a local maintainer should
remove or archive stale `build/` and `dist/` directories before `python -m
build`. The release gate's source-to-wheel digest comparison is the final guard:
an extra or stale runtime file is a release failure, even when the version
number is correct.

Targeted tests answer “did this edit break its owner?” Full tests answer “did
this apparently local edit violate another public contract?” Both are needed.

For an explicit source-tree check, keep the repository root as the working
directory and prepend its `src` directory:

```bash
PYTHONPATH="$(pwd)/src" python -c \
  'import agentfem; print(agentfem.__file__)'
PYTHONPATH="$(pwd)/src" python -m pytest -q
```

The printed path must point to the current checkout. Release CI instead builds
and force-installs the candidate wheel before testing, intentionally verifying
the artifact users receive.

Source evidence records two deliberately different hashes. The
`package_tree_sha256` binds every packaged byte and remains the exact audit
identity. The `scientific_runtime_sha256` excludes only
`knowledge/benchmarks/**`, whose records describe promotion state rather than
execute a numerical model. Promotion comparisons use the scientific runtime
identity while retaining every exact package identity. Consequently, changing
a solver, material, operator, knowledge card, schema, or any other packaged
resource invalidates the evidence; changing an archived benchmark declaration
from experimental to accepted does not create a circular demand to rerun the
calculation that justified that declaration.

`promotion_gate.py` is a source-checkout audit and therefore inserts the local
`src/` directory itself. It also rejects an AgentFEM import from outside that
checkout. This prevents an installed older wheel from being paired with the
current Git commit in a promotion report. Installed-wheel acceptance remains a
separate G5 record rather than being inferred by the source audit.

Direct MPI driver scripts do not pass through pytest's `pythonpath` setting.
When they are used against an uninstalled checkout, prefix both serial and MPI
commands with the checkout parent explicitly, for example:

```bash
PYTHONPATH="$(pwd)/src" agentfem mpi-run -n 2 --timeout 600 -- \
  python tests/portable_inelastic_step_driver.py write /tmp/agentfem-step
```

The timeout is a test-lifecycle guard, not a solver convergence parameter.  If
an MPI rank diverges before a collective while another rank waits inside it,
the launcher and every child rank are terminated as one process group and the
CLI reports `AFM-MPI-TIMEOUT`.  Do not remove the guard merely to make a stuck
test appear busy; first identify the mismatched rank path.  Long production
simulations remain unlimited unless the caller explicitly supplies a timeout.

Release CI deliberately omits this prefix after force-installing the candidate
wheel; those same drivers then provide installed-artifact evidence.

## Change-aware CI tiers

Cross-module coupling is high—changes to `Model`, providers, output, mesh
identity, or checkpointing can affect many workflows. Known core owners select
their serial suites, two-rank suites, and any executable cross-rank restart
drivers explicitly. An unmapped core owner escalates to the complete release
gate instead of inventing a partial test set. Test corrections and product
orchestration changes run the affected test files against a freshly built
wheel, followed by serial and two-rank solver smoke tests. Documentation
changes build the strict site without creating a FEniCSx environment. PyTorch
is installed only for learning changes or a release gate.

The always-running classifier reports one explicit tier:

| Tier | Automatic scope | Evidence |
| --- | --- | --- |
| `docs` | Documentation and generated site inputs only | Generated entrypoints and strict site build |
| `targeted` | Tests, examples, integrations, CI, CLI, campaign, dataset, or surrogate orchestration | Built wheel, affected tests, serial and two-rank smoke |
| `core` | Mapped FEM formulation, operator, constitutive, state, result, solver, or MPI owners | Owner-selected serial/MPI suites, portable drivers, and installed-wheel smoke |
| `release` | Release identity/runtime inputs or explicit dispatch | Core gate plus release smoke, promotion audit, and retained artifacts |

Unknown repository paths fail safe to `core`; unknown `src/agentfem` owners
escalate to `release`. A manual dispatch selects the requested minimum tier and
defaults to `release`. New commits cancel superseded development runs;
immutable tag and release evidence is never replaced this way.

Developers should still begin with the smallest relevant tests. Re-running the
entire environment and MPI matrix after every one-line edit wastes time and
delays diagnosis. Related edits should be collected into one reviewable commit
and pushed after local evidence is green; a remote quota or infrastructure
failure is recorded once rather than retried until green.

## CI resource policy

Validation depth follows the evidence claim, not the number of edited files:

1. use targeted local tests while developing;
2. push coherent changes rather than each intermediate edit;
3. choose the remote tier from scientific impact, with unknown changes failing
   safe to the core gate;
4. reserve hosted platform matrices, installers, and expensive scientific
   benchmarks for tags, explicit dispatch, or scheduled evidence;
5. let a newer development commit cancel an obsolete run;
6. reuse evidence only when it remains bound to the exact code and artifact
   digest that produced it.

[Standard hosted runners are free for public repositories and metered for
private repositories](https://docs.github.com/en/billing/concepts/product-billing/github-actions),
but queue time, compute, storage, and reviewer attention are still finite
engineering resources. Cost control must not weaken a release claim; equally,
repeated computation that proves no new claim is not verification.

## When the suite grows

Introduce registered pytest markers only when runtime measurements justify
them, for example `unit`, `fem`, `mpi`, `external`, and `release`. Markers must
describe evidence or runtime requirements, not vague importance. The fast gate
must never become a permanently weaker alternative to the complete gate.

A mature schedule is:

1. targeted tests on every edit;
2. fast deterministic gate on every commit;
3. full serial and affected MPI tests for core numerical pull requests;
4. complete platform/MPI/optional-dependency evidence for releases and
   scheduled audits;
5. external-code and large-mesh benchmarks on a scheduled or release gate.

If a check is required by branch protection, prefer a workflow that always
reports a conclusion. GitHub documents that an entire workflow skipped by path
filters can leave a required check pending. Job-level conditions or a small
always-running decision job are safer when selective CI is eventually needed.

## Failure policy

- A failed targeted test blocks the edit immediately.
- A failed full test is not dismissed because unrelated targeted tests pass.
- Flaky numerical tests should be diagnosed, not retried until green.
- Golden regression, external-code comparison, mesh/time convergence, and
  experimental validation remain different evidence classes.
- Optional integrations are tested in isolated jobs so a core developer does
  not need every dependency locally.

This strategy keeps the development loop efficient without weakening the
scientific claims attached to a release.

## Static-analysis adoption

The first Ruff gate deliberately checks correctness-sensitive rules: syntax,
undefined names, invalid control flow, loop-variable capture, and mutable
function-call defaults. It does not reformat the historical repository or
rewrite third-party reference scripts. Broader style rules may be adopted
module by module only when their review cost is justified.
