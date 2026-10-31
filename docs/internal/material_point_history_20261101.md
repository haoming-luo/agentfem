# Co-located material histories: bounded implementation and evidence

## Ownership and admission

`results.material_point(name, cell=..., point=...)` is a declaration, not a
callback. The creep Step evaluates the candidate's raw stress, total strain,
trial internal variables and prescribed temperature at one reference quadrature
identity. Sampling validation happens before commit. Only an accepted frame
enters the recorder. Output/storage reuse ordinary tensor `HistoryResult`
objects and the existing NPZ result-manifest archive.

The first consumer admits serial Cartesian 3D homogeneous power-law creep,
homogeneous joint creep/Chaboche plasticity, and existing regional pure creep.
It rejects MPI recording and axisymmetric recording. It does not promote
regional joint plasticity or arbitrary finite-strain user materials.

Both creep checkpoint formats retain selection, full trajectory and integrity
digest. Selection and units also participate in outer scientific compatibility,
so an older reader cannot silently discard the new history contract. Restore
validates all frame times against the accepted increment ledger before publishing
the history; the existing whole-Step transaction restores old history along with
old displacement, material state and thermal time if anything fails.

## Defect found during acceptance

Legacy serial thermal checkpoints compared the active temperature sample/time
as identity, although a new receiver legitimately starts at another time.
The correction compares the full immutable temperature-path identity. Old v1
history descriptions are normalized to that identity; static temperature
fingerprints are unchanged. Current temperature is reconstructed at accepted
physical time. The portable-format identity already used the complete path.

## Executed evidence

Real `fenicsx-env`, one numerical process, thread limits one; process-local TCP
provider avoids the local VPN/OFI interface issue without changing machine
configuration.

- 17 new tests passed: point/tensor/temperature association; regional and joint
  compositions; both restart formats; corruption rollback; accepted-only
  recording; no fabricated past state; fixed-reference guard; legacy recovery.
- The affected creep, joint plasticity, thermal history, result archive,
  checkpoint and public structural benchmark suite passed **217 tests**, with
  **one MPI-only skip**, in **133.08 s**. This includes the existing NAFEMS
  axisymmetric creep cylinder sub-percent stress criterion; it is not a new
  Alloy 617 material fit.
- Documentation/architecture checks: **44 passed**. Strict docs build passed.
- After adding the explicit restored-endpoint/accepted-time consistency guard,
  the final point-history plus documentation/architecture rerun passed **61
  tests** in **29.60 s**. Generated docs and 58 scientific function cards passed
  their freshness checks; correctness lint passed.
- A subsequent continuation-identity audit refuses changed selections, time
  units and loading paths, as well as late activation of recording. The final
  focused suite passed **21 tests in 26.30 s**; the four additional tests check
  these boundaries without changing constitutive equations.
- Ordinary CLI example: **16 accepted increments / 16 attempts**, solver-reported
  **3.4 s**. Result trust stays `computed`; final temperature 900 K. This used
  the development checkout, not a new cross-platform installation claim.

The first broad regression exposed a child-process test loading an unrelated
old installed AgentFEM instead of the package under test. Its environment now
binds the parent package and asserts that identity. A second regression found
the new lazy exports missing from `__all__`; both issues were corrected before
the final passing run. Neither was hidden as an acceptable scientific failure.

## Open gates

No real-material fit/holdout prediction, lifetime qualification, generic
temperature-dependent UserMaterial admission, MPI history promotion or new
composite speed claim. The public Alloy 617 report was visually audited under
the PDF-read workflow; its model mismatch and staged admission plan are kept in
[the source audit](alloy617_validation_plan_20261101.md).
