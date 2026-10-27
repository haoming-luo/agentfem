# 0.4.4 candidate decision record — 2026-10-27

Status: release preparation, not a publication or a scientific promotion.
Base: merged main `5746414856ff002b752b596b7e93504ad96c597d` (PR #124),
including Windows quickstart PR #125. The previous source gate passed Linux
native/FEniCSx, MPI, macOS and PowerShell checks. Final release evidence must
bind to this candidate, not reuse a previous wheel hash as if unchanged.

## Five-hour closeout

The authorized window begins at 08:57:57 UTC and ends at 13:57:57 UTC on
2026-10-27. Stop starting new work at the deadline and report actual outcomes.
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

## Follow-up, not release scope expansion

Obtain a same-input independent nonlinear reference and targeted contact-edge
spatial evidence before promoting the nonuniform periodic composition.
Collect a clean non-root Windows/WorkBuddy installation report. Resume
explicit composite acceleration only with its own fixed-workload evidence;
implicit fixture improvements are not a substitute.
