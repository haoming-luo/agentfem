# Next sprint: guided setup, then material evidence

## Recorded user feedback

A real Windows installation reportedly completed quickly with personal
guidance. Remaining friction: unnecessarily delegated account/setup commands,
unclear missing-kernel download handoff, and an initial long Ubuntu command
block that failed while a shorter guided sequence succeeded. The exact error,
host version and package identity were not supplied: no root cause is assigned.
Retain no user identity or password in this record.

## Order and acceptance

1. Simplify Windows guidance: three visible stages, agent-owned execution,
   one-action manual fallback, architecture-appropriate kernel link, short
   staged commands with integrity checks retained. Installation ends at
   readiness. Documentation checks are not a clean-host field trial.
2. Close engineering release scope: main a5164b83 passed CI; #138–#141 are
   merged. Update historical pending statuses before preparing 0.4.5. Do not
   require another full composite simulation or installer build for this task.
3. Freeze the P92 Norton negative-transfer baseline. Screen existing Sinh and
   joint creep/plasticity first. Introduce recovery/softening only after a
   mechanism discriminator, with state, tangent and work conventions specified.
   Compare independent local integration, three temporal refinements, full
   loading/hold/reversal and restart. SLR has already informed diagnosis and
   cannot become blind validation again; reserve a new complete condition.
4. Reuse co-located stress/strain/temperature histories for structure and life
   assessment. Separate numerical, material and life-validation criteria.
   Temperature-dependent user materials precede separately scoped regional,
   contact or MPI extensions. Do not reopen all combinations simultaneously.

Public foundations: BAM P92 data (doi:10.5281/zenodo.7198219), MOOSE power-law
creep, NEML Chaboche recovery, and ASTM E2714 test-family guidance. These are
references, not completed cross-solver comparisons or regulatory certification.

This record starts the work; it does not assert an eight-hour execution,
completed material screening, or a new release.

## Subsequent roadmap progress (2026-11-05)

- #142/#143/#146 are integrated; prescribed uniform thermal history #144 was
  subsequently merged after its full CI and installed-wheel checks passed.
- #147 adds selected material-point histories to the ordinary finite-strain
  user-material Step and preserves them through restart. It is a separate PR,
  not yet a release claim.
- #148 adds the pinned public P92 source-figure screen and strengthens the
  independent local integration/tangent reference. It does not install a P92
  engineering material card. See the [scientific record](p92_dwell_figure_screen_20261105.md).
- #149 fixes partial small-strain checkpoint restoration and rejects
  inconsistent accepted ledgers/boundary states. It remains a separate
  engineering review, with serial, failure-injection and cross-rank evidence.

Next keep integration and material admission distinct. Finish these reviews;
then specify the small-strain physical-time consumer before removing its
current rate-independent restriction. Its initialization currently calls the
provider with a unit interval and its restore reconstructs responses with an
epsilon interval: neither is a valid generic initialization/restart rule for
an evolving material. Accepted response fields should be restored, not advanced
again. Physical endpoint times, cutback subdivisions, temperature and provider
failure must belong to the same accepted-state transaction.

The first numerical gate should reuse an analytically checkable viscoelastic
law and a second history law through the same structure path; compare holds,
reversals, refinement and interrupted continuation. The P92 uniaxial reference
then supplies a local target, but a multi-axial extension still needs an
explicit model derivation and structural tests. Do not turn an axial tangent
into a 3D tensor by padding components. Local fixed-partition substepping may
reduce global cutbacks, provided the complete chain tangent is retained and
an end-to-end experiment demonstrates the benefit.
