# 0.4.2 candidate — execution and output consolidation

This is a preparation record, not a published release announcement. The moving
checkout is `0.4.2.dev0`, distinct from the published `0.4.1` package. Existing
citations continue to identify the actual published release until promotion.

## Included

- Packaged, material-independent Hex8 body execution and trusted compiled
  user-material dispatch, with separate explicit worker budgets.
- Reusable Q1 contact execution; exact fallback and ambiguity behavior remain
  distinct from application-specific geometry and material assumptions.
- Immutable accepted-state transport, local material repair and joined worker
  failure handling. No second state owner or hidden alternative time loop.
- One-grid result writing, explicit per-cell data mapping and independent
  volume/Warp acceptance. Cases do not own file-format connectivity.
- Accepted-history reporting and scheduled checkpoints through ordinary Steps.

## Distribution correction

The publication workflow previously built a native Linux wheel as though it
were platform independent and attempted to reuse it on macOS. The development
CI had already corrected the macOS side. Both now build a platform wheel from
the identical source archive through a shared tool. Linux publication repairs
the wheel with auditwheel **before** hashing and numerical acceptance; macOS
acceptance never attempts to load the Linux binary. The tested macOS artifact
is retained separately, not silently uploaded as another already-tested wheel.

Native body modules are mandatory for this platform build gate. Optional GMP
Q1 presence is reported, not inferred from build success. A source build in the
tested conda environment can include it; an absent optional module must not
be advertised as the complete fast contact path. Remote release-job execution
and native dependency portability remain acceptance obligations, not a claim
made by the new helper's unit tests.

## Current evidence and open gates

- Previous commit `1e42588e`: all applicable PR checks passed, including Linux
  numerical and macOS installed-wheel foundation acceptance.
- Installed revision-4 candidate: full original-model timing and final reader /
  state checks are in progress. Do not replace them with local pipeline timing.
- Private readable-case refactor: separate delivery directory; ordinary Model
  and Step calls, with runtime supervision separated. Its own construction and
  accepted-state checks remain required before handoff.
- Publication corrections, version metadata and candidate source need renewed
  CI. No release tag or public package has been emitted for this candidate.

Scope and acceptance are defined in
[the cahier des charges](release_ready_spec_20261013.md). No new physics or
experimental-capability promotion is required to close this release.
