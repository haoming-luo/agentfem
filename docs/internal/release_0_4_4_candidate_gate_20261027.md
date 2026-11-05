# 0.4.4 candidate decision record — 2026-10-27

Status: **published** as immutable `v0.4.4` at
`7dbf936df18997bd8785a72f27ff230bcad68581`; no scientific promotion.
The initial plan and pre-publication checks below remain historical evidence.
Base: merged main `5746414856ff002b752b596b7e93504ad96c597d` (PR #124),
including Windows quickstart PR #125. The previous source gate passed Linux
native/FEniCSx, MPI, macOS and PowerShell checks. Final release evidence must
bind to this candidate, not reuse a previous wheel hash as if unchanged.

## Five-hour closeout

The original authorization covered 08:57:57–13:57:57 UTC on 2026-10-27,
with instructions to stop starting new work at the deadline and report actual outcomes.
No new physical formulation, large mesh sweep, full composite run or installer
bundle is part of this closeout. Preserve other worktrees and generated sites.

| Gate | Required result | Initial state |
| --- | --- | --- |
| Scope | Version, changelog, public matrix and packaged contract agree | Updating |
| Safety | Admission, increments, state/output and release-identity regressions pass | Pending candidate check |
| Installed use | Fresh native wheel; ordinary project and bounded contact/output/restart tests | Pending |
| Integration | Exact release PR head passes applicable remote CI | Pending |
| Publication | Tagged immutable artifacts pass publish gates; verify actual upload | Not started |

Scientific exclusions do not excuse a regression in an admitted route. A
numerical failure, state corruption, false acceptance or missing boundary
contribution blocks the candidate. A documented experimental accuracy study
does not block unrelated engineering improvements indefinitely.

## What is retained as experimental

The current contact matrix in `docs/guide/model_setup.md` controls admission.
Periodic J2/contact, selected faceted friction and plane augmentation remain
experimental. Distributed implicit contact, general sharp-feature friction
and unsupported material/boundary combinations are rejected.

The nonuniform partial-patch study has independent discrete elastic/MPC
agreement and bounded graded-cycle path sensitivity. It does not yet have
spatially independent plastic pressure or an independent same-input nonlinear
reference. The full scientific record and raw reports remain in
`docs/internal/contact_0_4_4_local_parity.md` and `evidence/contact/`.
Do not tune acceptance limits after seeing the result or claim peak-pressure
convergence from total-force balance. A future promotion must match material
law, loading path, penalty measure, periodic equations and physical outputs.

## Representative executable evidence

- `tests/test_finite_strain_j2_periodic.py`,
  `tests/test_periodic_contact_virtual_work_oracle.py` and
  `tests/test_contact_plastic_coaxial_reference.py`: total reduction,
  conjugacy, constitutive reference and lifecycle.
- `tests/test_accepted_energy_components.py` and
  `tests/test_affine_residual_only.py`: energy channel ownership and
  equivalent residual-only observations, including legacy-provider fallback.
- `tests/test_finite_j2_multi_contact.py`,
  `tests/test_dolfinx_implicit_friction_contact.py` and
  `tests/test_finite_strain_j2_contact_mpi_boundary.py`: bounded composition
  and unsupported distributed rejection.
- `tests/test_live_xdmf.py`, `tests/test_xdmf_layout.py` and
  `tests/test_quadrature_physical_recovery.py`: accepted frame publication,
  layout and physical recovered fields.
- `tests/test_increment_control_validation.py`, `tests/test_small_strain_material_step.py`
  and `tests/test_packaging.py`: early rejection and package identity.
- Installed project smoke plus the repository publication workflow: ordinary
  entry points, platform/runtime identity and exact-artifact acceptance.

Named tests are an evidence map, not a claim that this candidate has run them.
Record commands, counts, artifact hashes and failures below as obtained.

## Source checks before freezing the candidate

The candidate introduces no new numerical formulation. Its source changes are
release identity validation/reporting, version metadata and scope documentation.
On the local `fenicsx-env` (single thread per numerical library):

- Packaging, increment controls, lazy CLI startup and Windows quickstart:
  **126 passed**. This includes wrong-version contract content with/without a
  tag, malformed contract structure and truthful source/distribution/smoke
  check summaries.
- Accepted energy, residual-only observation, live XDMF/layout, physical cell
  recovery, small-strain admission, multiple tools, periodic virtual work and
  the confined plastic reference: **123 passed, 3 skipped**. The skips require
  two MPI ranks and are not counted as distributed coverage.
- Generated documentation is current; strict MkDocs and repository Ruff pass.
  The documentation build uses a separate temporary site, preserving the
  user's existing generated `site/` modifications. Local REUSE executable was
  absent; source-attribution acceptance remains required in remote CI.

Output scope is explicit: a restart writes a new result segment and preserves
the old files. It does not automatically append/merge an earlier XDMF index.
Live frame-file publication and continuous physical time do not imply otherwise.

The first frozen source `5dbc6cc7` also passes native installed-use acceptance:
four ordinary templates, bounded contact examples and two-rank elasticity;
339 installed regressions; and a separate two-rank recovery/rejection group
with 26 tests passing on each rank. The local wheel SHA-256 is
`0a2b7084b0f90597cb0951d42fecc7783af93d6f6f5babb69c0e7275ccec3551`.
Its sdist SHA-256 is
`c9bd1971a532a8fb34a92bb8b2976fde7fc6d52ad73068aecbffe80284d4f413`.
These are pre-publication artifacts, not the final tagged Linux distribution.
Additional installed execution-policy checks pass 64 tests, with the
distributed-only policy-rejection test reserved for the two-rank gate.

The closeout audit found that the existing distributed contact/policy rejection
tests were not all selected by the complete CI/publication workflows. Both now
select those gates explicitly; publication also selects physical quadrature
recovery, including an empty partition and collective invalid-measure rejection.
This closes an acceptance-coverage gap without changing runtime algorithms.
The exact three-file selection passes **27 tests per rank** locally (two
ranks, 120-second launcher limit, 1.02 seconds of pytest execution). Packaging
and CI-scope checks pass 76 tests after the workflow update.

## Follow-up, not release scope expansion

Obtain a same-input independent nonlinear reference and targeted contact-edge
spatial evidence before promoting the nonuniform periodic composition.
Collect a clean non-root Windows/WorkBuddy installation report. Resume
explicit composite acceleration only with its own fixed-workload evidence;
implicit fixture improvements are not a substitute.

## Publication receipt

PR #131 merged at the tagged commit above. The
[official publication workflow](https://github.com/haoming-luo/agentfem/actions/runs/40486982823)
completed successfully, including installed-use, distributed and foundation
gates. Its serial suite reports **5155 passed, 427 conditional skips**;
separate MPI/native gates, not the skipped tests, supply that coverage.
The [GitHub Release](https://github.com/haoming-luo/agentfem/releases/tag/v0.4.4)
was published at 2026-10-27 11:08:09 UTC; the version is also available on
[PyPI](https://pypi.org/project/agentfem/0.4.4/).
Publication establishes the bounded release contract, not the open spatial
or independent nonlinear-reference gates.

## Bounded post-release review

The next small maintenance change preserves the single native contact tangent
insertion while rejecting non-real inputs and nonfinite duplicate sums before
any bulk-matrix mutation. Regression coverage includes unchanged valid duplicate
summation, real PETSc matrix preservation and a failed augmented-contact
increment followed by successful continuation. The accepted displacement,
material state, multiplier and work ledger must survive the rejection.

Review also rechecks retained material response, attempt-owned linear resources,
managed preconditioning, live frame publication and binary-history readers.
Source tests and their subprocesses must use the same source identity:
`PYTHONPATH=src` is necessary when the numerical environment also contains an
older installed AgentFEM. An initial run without that setting passed its
in-process tests but failed two subprocess checks by importing the old package;
the same selection passed with consistent source selection. No runtime code was
changed to disguise that environment mismatch.

The final source selection passes **315 tests, 21 skipped** on local
FEniCSx/PETSc with one numerical-library thread. Sixteen skips are unsupported
refined tetrahedral-contact combinations; five require multiple MPI ranks.
This review did not rerun MPI. Seven new cases cover the assembly rejection,
matrix preservation and joint recovery; the four new malformed/overflow-input
cases failed before the fix. Existing tests retain exact duplicate summation
and compare accepted fields, material state, work and restart.

Reproduce from the checkout with `PYTHONPATH=src`, `OPENBLAS_NUM_THREADS=1`
and `OMP_NUM_THREADS=1`, using the numerical environment's Python:

```sh
python -m pytest -q \
  tests/test_contact_sparse_insertion.py tests/test_affine_residual_only.py \
  tests/test_accepted_energy_components.py tests/test_adaptive_linear_correction.py \
  tests/test_retained_material_trial.py tests/test_affine_linear_workspace.py \
  tests/test_affine_adaptive_preconditioner.py tests/test_live_xdmf.py \
  tests/test_history_archive.py tests/test_dolfinx_implicit_facet_contact.py \
  tests/test_dolfinx_implicit_smooth_contact.py \
  tests/test_quadrature_physical_recovery.py tests/test_increment_control_validation.py
```

Generated-documentation consistency, targeted Ruff and `git diff --check`
also pass. These are targeted maintenance checks, not a second full-release
qualification or an audit of every module.

This follow-up neither rebuilds an installer nor changes a constitutive law,
acceptance tolerance or public default. It reports no new full-composite timing.
