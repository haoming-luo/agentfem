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

- Commit `146448bd`: all applicable PR checks passed, including portable Linux
  binary packaging, Linux numerical and macOS installed-wheel acceptance
  (workflow `38412091759`). A subsequent publication-test dependency error
  was isolated to an optional YAML import and removed without adding a runtime
  dependency. Remote status for subsequent changes is tracked on PR #102;
  merge/publication require its applicable checks to pass.
- Installed revision-4 candidate `b962a777` completed the fresh original-model
  run: 100,000 increments, 0.1 seconds, 6 mm prescribed tool travel, in
  15,465.054904 seconds (4 h 17 min 45 s), including preparation/export/shutdown.
  The prior migrated core took 19,119.111109 seconds (an observed 19.11% time
  reduction); the earlier private backend took 13,529.506969 seconds, so this
  candidate is still 14.31% slower than that private reference. These are local
  observations, not hardware-normalized vendor or isolated causal benchmarks.
- Final comparison passed: 3 nodal arrays, 57 auxiliary arrays, 20 numeric
  history channels, the damage subset and all 76 contact evidence entries.
  Compared numeric arrays are bitwise equal; all 10,000 journal records are
  exactly equal. Input/runtime manifest verification passed.
- Actual ParaView 6.1.1 reads the new file directly as one mesh with 296,074
  points and 255,041 three-dimensional Hex8 cells. Warp has zero coordinate
  error; no descriptor repair or interface-node welding is needed.
- Private readable-case refactor: ordinary Model/Step calls, with supervision
  separated; fresh 600-increment acceptance matches the frozen candidate.
  Final-checkpoint recovery and ordinary RunContext publication passed. All
  final arrays/evidence remain unchanged and the recovered visualization passes
  the same one-volume-grid check. Recovery advanced zero increments; its time
  is not another fresh full-run benchmark.
- A fresh `0.4.2.dev0` macOS wheel built from its source archive passed 165
  installed-package tests with checkout-path injection disabled. Installed
  source/native bytes match wheel SHA-256
  `d096179d2f0323f9c530c9228123ed6850efba727f3189254b0c15af08942e8e`.
  All three native modules are present. The unrelated public free-bar example
  completed 640 increments; relative linear-limit displacement error is
  0.000709678842 and relative energy-ledger error 0.00000193265549. It also
  passes actual ParaView volume/Warp acceptance. This is not large-strain or
  industrial-contact validation.
- No release tag or public package has been emitted for this candidate.

## Installed-project inspection boundary

Final ordinary-project inspection exposed an unrelated UX defect: it traversed
the frozen dependency installation and recommended modifying library-internal
compatibility calls as if they were case code. Inspection now prunes installed
wheel targets and Python environments before scanning. Ordinary user modules,
including a directory named `runtime`, remain inspectable; editable-source
egg-info does not hide code, and an explicit entrypoint is always inspected.
The 63 focused CLI/packaging/reader checks passed, and the actual delivery
project reports no migration findings with this change. It changes advisory
inspection only, not the frozen solver, scientific inputs or checkpoint identity.

The measured runtime is the frozen installed `b962a777` wheel, not a newly
published version. Through `ddc697d3`, subsequent changes under `src/agentfem`
only distinguish development-version metadata and the release contract; the
numerical source is unchanged. Lightweight source/doc/test work overlapped the
new run; the prior core run additionally overlapped builds/tests. No competing
simulation ran during the new timing. Offline comparison/reader review took
4.43 seconds and is excluded from solver time. Confidential fields, checkpoints,
source laws and detailed review records remain in the private project.

Scope and acceptance are defined in
[the cahier des charges](release_ready_spec_20261013.md). No new physics or
experimental-capability promotion is required to close this release.
