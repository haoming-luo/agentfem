# 0.4.3 candidate decision record — 2026-10-16

This is a release gate, not an announcement that a version has been tagged or
that a general forming solver has been validated. The exact tested source is
`af7b52a3` on the stacked moving-facet branch; merge commits and a final
release-version change will require their own candidate check.

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

1. Required remote checks on the exact integrated PR heads, including the
   installed FEniCSx job, must finish successfully. Merge the fixed-facet PR
   before the stacked moving-facet PR, then recheck the latter against main.
2. Freeze the final source and run candidate-bound installed-wheel checks
   after the merge and release metadata update; do not reuse the hash of the
   earlier branch wheel as evidence for a changed source tree.
3. Update the changelog, version, citation and scoped release notes together.
   A tag triggers the publishing workflow and must not precede its gates.
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
