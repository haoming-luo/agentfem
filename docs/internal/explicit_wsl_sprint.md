# Explicit execution and WSL usability sprint

Baseline: main 5370a6e3 (PR109 merged). No physical controls change.

## Ordered delivery gates

1. One visualization entrypoint: time series includes the final station;
   optional raw endpoint archive must not write a second XDMF/HDF5 pair.
   Reuse checked cell transport across many fields and frames without stale
   mesh/key mappings or mutable published snapshots.
2. Honest execution timing: separate initial frames/restart, observer output,
   integration (including scheduled checkpoints), terminal checkpoint/export,
   and publication. No micro-timing represented as complete solver speed.
3. WSL advisories in ordinary check/run: detect actual mounted Windows storage,
   report CPU/memory/thread environment, no automatic relocation, resource
   changes or override of workspace custody protection.
4. Profile the current installed body/contact route before selecting the next
   arithmetic optimization. Prior private-path profiles are context, not
   evidence that today's bottleneck is unchanged. Preserve fallback, ambiguity,
   damage, work and transaction semantics; compare identical accepted states.
5. Installed candidate tests, independent volume-reader checks and private
   full-grid comparison. Full-run speed is only established by a full run.

## Reference rationale

- https://learn.microsoft.com/en-us/windows/wsl/filesystems: Linux-native
  storage for Linux I/O; preserve backup/custody warning for distro removal.
- https://docs.h5py.org/en/stable/high/dataset.html: chunk/filter cost matters;
  never introduce lossy compression to improve timings.
- https://docs.lammps.org/stable/Speed_measure.html: phase measurements and
  neighbor rebuild/skin tradeoff. Such a search cache is only admissible with
  a conservative motion bound and explicit invalidation, not a fixed skip.

## Boundaries

No Windows hardware is attached. Mocked WSL diagnostics plus Linux CI are not
Windows end-to-end performance acceptance. No private mesh/law is committed.
No automation or background promise substitutes for active development time.

## First candidate evidence (2026-10-15)

- Installed candidate 2: 147 targeted tests passed; four two-rank checks skipped
  in serial and then exercised in one bounded MPI run (21 passed per rank).
  Pytest's repository `pythonpath=src` override was explicitly disabled for the
  installed gates. The earlier source-path gate is not installed-package proof.
- Native worker validation rejects nonfinite stress/history/work, asymmetric
  stress, unwritten output and force overflow before publication. Local repair
  and immutable old-state checks remain covered. No physics controls changed.
- Batched P1/Q1 contact preparation retains owner-local node matching, the same
  integration rules and point identities. Warped/high-order/empty-rank tests
  pass; higher-order evaluation remains on the existing path.
- Private full-grid 600-step comparison: baseline 127.392 s; worker-validation
  and output candidate 124.925 s; additional batched preparation 119.220 s.
  These are one short run each, including setup/export, **not a full-run speedup**.
  Preparation was approximately 18.2, 18.4 and 13.4 s respectively; advancing
  600 increments remained approximately 104–105 s including initial fields.
  Do not attribute the startup saving to every increment or extrapolate it.
- All archived cell keys, material states, Cauchy stresses, gradients and times
  at increment 600 were bitwise identical across these three runs.
- One visualization series plus a raw endpoint archive is tested. Native
  worker changes have a new execution revision; no checkpoint identity bypass.

Next gates: independent ParaView check of the single-series project; frozen
full-run comparison and late-state investigation. A large solver throughput
improvement has **not** yet been demonstrated. WSL hardware timing is still
unavailable; the mount/resource diagnostics are covered by fixture tests only.
