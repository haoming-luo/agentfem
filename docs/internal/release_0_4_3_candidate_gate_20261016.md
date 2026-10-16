# 0.4.3 candidate decision record — 2026-10-16

This is a release gate, not an announcement that a version has been tagged or
that a general forming solver has been validated. The initial moving-facet
evidence below came from `af7b52a3`; final release evidence must instead bind
to the integrated candidate's exact source and installed wheel.

## Proposed bounded claim

One serial, three-dimensional finite-strain logarithmic-J2 displacement Step
can compose with one frictionless penalty rigid tool through the ordinary
`model.step(...)` workflow. The tool may be an analytical plane or a
triangulated surface whose *active* contact remains on a planar facet or a
coplanar interior seam. Its prescribed rigid motion uses the Step's normalized
load path. The same accepted increment owns the J2 material state, tool pose,
reaction, contact potential, force--moment work, result, and restart identity.

This does **not** admit curved or creased active facet transitions, friction,
multiple pairs, contact/strong-BC overlap, mixed displacement--pressure,
periodic MPC plus contact, or distributed implicit contact. Provider discovery
and Step lowering must reject unsupported combinations before residual
assembly; no boundary model may be silently dropped. The composed result's
`computed` trust status is intentional. A package version cannot promote it
to industrial forming verification.

## Evidence already obtained on the candidate branch

- Analytical plane and fixed/moving coplanar facets agree on the bounded J2
  fixtures for displacement, committed state, contact potential, generalized
  force and accepted tool work. The new moving-facet dual matches the
  fixed-state potential derivative at two z-axis rotation magnitudes. A
  separate three-axis proportional rotation check also matches the analytical
  plane and its force--work derivative; its regression is included with the
  pending release candidate. A separate no-traction run confirms that
  prescribed tool motion alone drives the body, rather than relying on an
  unrelated applied force. Strong/affine
  boundary rejection, failed-attempt rollback, serial restart, and the
  two-rank fail-closed path have targeted regressions. See the
  [fixed-facet](implicit_facet_contact_candidate_20261016.md) and
  [moving-facet](moving_facet_contact_candidate_20261016.md) records for exact
  scope and tolerances.
- A 0.2-rad fixture has 3/6/12-increment and 1--4-cell integral-output
  sensitivity records. Consecutive tool-work and penalty-potential differences
  shrink in those scans; spatial independence is not established, the energy
  residual is not monotone, and the penalty scan changes the physical result.
- A native macOS arm64 CPython 3.11 wheel and sdist were built from
  `af7b52a3`. The wheel SHA-256 is
  `7e03d5b5c419afde39f32ef3a418858496aa3a26c9665ca4c2d6cc60bbb44b9d`;
  the sdist SHA-256 is
  `717a0e4990f5b38b0317d0ae169335784023c7acb52303b942ba23c904a1dac2`.
  The isolated installed-wheel `release_gate.py --dist ... --smoke
  --require-platform macos` passed. From that installed wheel outside the
  checkout, `examples/finite_j2_faceted_contact.py --tool-travel=-0.02
  --tool-rotation=0.2` completed, wrote XDMF/HDF5 and a result manifest,
  and reported contact potential `1.04376` and tool work `2.00687` in its
  dimensionless fixture. These are smoke observables, not calibrated forming
  predictions.

## Still required before tagging

The previous v0.4.2 GitHub release did not reach PyPI: its Linux repaired
wheel contained 37 ordinary ZIP directory records, which the installed-wheel
identity gate incorrectly compared as runtime files. The 0.4.3 candidate now
filters archive directory records and retains the actual file-hash comparison;
a regression exercises both ZIP and source archives. The exact tagged Linux
wheel must still pass the publication workflow's installed-use gate before
PyPI upload. A GitHub release page alone is not publication evidence.

1. The analytical-plane, fixed-facet and moving-facet PRs have merged. Required
   remote checks on the exact 0.4.3 release PR head, including installed
   FEniCSx, must finish successfully before that PR merges.
2. Freeze the integrated source and run candidate-bound installed-wheel checks
   after the release metadata merge; do not reuse a branch-wheel hash as
   evidence for a changed source tree.
3. The changelog, version, citation and scoped release notes move together.
   Tagging triggers the publishing workflow; verify that exact tagged wheel,
   then confirm actual PyPI publication rather than assuming a GitHub tag did it.
4. Preserve the known limits in capability discovery and user documentation.
   Windows/WSL or a nonplanar die needs separate installation and scientific
   evidence; a Mac wheel smoke cannot establish either.

## Next numerical gate, not a 0.4.3 release shortcut

For a real curved triangulated tool, add smooth-surface or reviewed
edge/vertex branch derivatives, then compare with an independent structure
reference while refining tool facets, solid mesh, increments and penalty.
Abaqus/Standard's [analytical-rigid-surface guidance](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMODRefMap/simamod-c-rigidsurf.htm)
also flags discontinuous normals and tangents as a convergence difficulty;
this supports the present fail-closed branch policy, not a claim that our
piecewise-planar law is equivalent to its general-contact implementation.
For periodic MPC, assemble the total material-plus-contact residual and
tangent *before* applying `T^T R` and `T^T K T` on every Newton and line-search
trial, with joint state/work/restart evidence. Passing a contact parameter to
the current material-only affine route would be incorrect.
