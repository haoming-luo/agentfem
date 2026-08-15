# Development and verification strategy

AgentFEM uses layered verification. The goal is fast feedback during an edit
and broad evidence before shared code or a release changes—not blindly running
the most expensive command after every keystroke.

## Validation layers

| Moment | Required evidence | Typical command |
| --- | --- | --- |
| Inner development loop | Direct unit/interface tests for the changed owner | `python -m pytest -q tests/test_extensions.py` |
| Before committing | Related workflow tests, critical static analysis, misuse tests, and generated-asset checks | `ruff check . --no-cache`; `python build_knowledge.py --check --check-imports`; `python build_docs.py --check` |
| Before pushing a coherent code change | Complete serial suite | `python -m pytest -q` |
| MPI-sensitive change | Relevant two-rank modules using the environment-matched launcher | `$CONDA_PREFIX/bin/mpiexec -n 2 python -m pytest ...` |
| Pull request and `main` | Wheel installation, full serial, MPI, checkpoint portability, examples, documentation, and optional PyTorch bridge | GitHub Actions `Test` workflow |
| Release candidate/tag | All preceding checks plus distribution inspection and installed-wheel release smoke | `python release_gate.py --dist dist --smoke` |

## Source and installed-wheel evidence are separate

The repository root is also the Python package directory. Pytest is configured
with `pythonpath = [".."]`, so `python -m pytest` and MPI pytest commands always
exercise the current checkout rather than an older `agentfem` already present
in the environment. The release gate follows the opposite rule: it installs
the candidate wheel into an isolated target, rejects a source-checkout import,
compares every packaged runtime file with the candidate source, and then runs
the flagship workflows and installed project templates. A release therefore
needs both source evidence and installed-artifact evidence.

Targeted tests answer “did this edit break its owner?” Full tests answer “did
this apparently local edit violate another public contract?” Both are needed.

This repository uses the project directory itself as the `agentfem` package.
When an older wheel is also installed, running Python from the repository root
can otherwise import that wheel while the tests still discover local files.
For an explicit source-tree check, keep the repository root as the working
directory and prepend its parent:

```bash
PYTHONPATH="$(pwd)/.." python -c \
  'import agentfem; print(agentfem.__file__)'
PYTHONPATH="$(pwd)/.." python -m pytest -q
```

The printed path must point to the current checkout. Release CI instead builds
and force-installs the candidate wheel before testing, intentionally verifying
the artifact users receive.

Direct MPI driver scripts do not pass through pytest's `pythonpath` setting.
When they are used against an uninstalled checkout, prefix both serial and MPI
commands with the checkout parent explicitly, for example:

```bash
PYTHONPATH="$(pwd)/.." "$CONDA_PREFIX/bin/mpirun" -n 2 \
  python tests/portable_inelastic_step_driver.py write /tmp/agentfem-step
```

Release CI deliberately omits this prefix after force-installing the candidate
wheel; those same drivers then provide installed-artifact evidence.

## Why AgentFEM still runs full CI frequently

AgentFEM currently has a compact suite: the local complete serial run is much
cheaper than a nonlinear simulation campaign. Cross-module coupling is also
high—changes to `Model`, providers, output, mesh identity, or checkpointing can
affect many workflows. Therefore every push and pull request currently earns a
full remote gate.

Developers should still begin with the smallest relevant tests. Re-running the
entire environment and MPI matrix after every one-line edit wastes time and
delays diagnosis.

## When the suite grows

Introduce registered pytest markers only when runtime measurements justify
them, for example `unit`, `fem`, `mpi`, `external`, and `release`. Markers must
describe evidence or runtime requirements, not vague importance. The fast gate
must never become a permanently weaker alternative to the complete gate.

A mature schedule is:

1. targeted tests on every edit;
2. fast deterministic gate on every commit;
3. full serial and affected MPI tests on every pull request;
4. complete platform/MPI/optional-dependency matrix on `main` and nightly;
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
