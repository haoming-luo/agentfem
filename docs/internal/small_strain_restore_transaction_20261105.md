# Small-strain user-material restore transaction

## Why this is on the roadmap

Before admitting a rate-dependent user material to a structural Step, its
accepted-state lifecycle must be trustworthy. Auditing the existing
rate-independent small-strain Step exposed an independent restore defect:
nodal displacement was loaded before quadrature state, with no enclosing
rollback. An injected quadrature read failure left 20 of 36 displacement
entries changed in the serial reproduction (maximum difference 0.001).

## Bounded change

Checkpoint loading snapshots the live displacement, committed and trial state,
stress, tangent, optional energy fields, response diagnostics, load constants,
accepted strain, path ledgers and continuation metadata. Failure restores these
directly, without invoking the material provider again. Normal solve steps do
not acquire these additional copies.

Manifest/identity/payload-digest and path-metadata validation is coordinated
before distributed field restoration. Existing portable nodal and quadrature
loaders are reused; the checkpoint format is unchanged.

This does not admit rate-dependent materials. The current small-strain factory
still requires `rate_independent=True`. A physical-time consumer must separately
define initial response evaluation, accepted endpoint time, cutback subdivisions,
temperature, restart response reconstruction and local failure coordination;
adding a `duration` parameter alone would not complete that contract.

## Verification

The regression starts with a live accepted quarter-path and attempts to restore
a three-quarter-path checkpoint. It injects quadrature-load failure, a late
response failure after deliberately modifying stress/tangent/trial history,
rank-zero-only provider failure, and rank-zero-only manifest read failure.
It checks exact restoration and then retries the valid checkpoint and completes
the path against uninterrupted execution.

The four cases pass serially and with two ranks through the bounded
`agentfem mpi-run -n 2 --timeout 120` launcher, one numerical thread per process.
The final targeted user-material, Step, learned-material, snapshot-ownership
and CI-routing suite passed 147 tests. These tests cover
the existing small-strain route, not a new finite-strain or creep capability.

`tests/portable_small_strain_step_driver.py` additionally wrote a half-path
checkpoint with one rank and continued with two, then wrote with two and
continued with one. Both continuations matched uninterrupted displacement,
committed material history and Cauchy stress at `rtol=1e-11, atol=1e-13`.
