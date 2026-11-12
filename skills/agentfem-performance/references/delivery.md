<!-- SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Installed release and reproducible project delivery

Read when delivery, installed portability, checkpoint/export behavior or a
complete-run claim is actually in scope. Do not run this checklist as a
prerequisite to an exploratory optimization.

Keep case.py as modeling truth, private material implementation as a separate
provider, and operational settings separate from physics. Pin/hash the runtime,
source law and mesh. Verify the installed project without development-tree path
insertion, including execution from outside its directory. Scientific inputs
and executable identity must not include generated timings or audit reports.

Preserve accepted checkpoints and unambiguous output status on interruption.
Append-only reporting can reduce growing-history write cost; its journal is
not a checkpoint. Preserve the final digest tip externally if suffix loss must
be detectable. Do not assume I/O dominates without measuring its contribution.

Separate reproducibility of one frozen run from portability of the software.
Thread counts, timing thresholds, hardware names and case-specific final
counters are not scientific defaults. Audit operational fields in restart
identity; changes require versioned, tested compatibility rather than deleting
checks. Capability/workload-scoped autotuning may choose execution budgets with
a safe stale-cache fallback, not silently tune mass, material or accuracy.

Use the supported result writer and declared cell-order/field-layout contracts.
For visualization delivery or writer changes, inspect with an independent
reader: volume topology, complete displacement, field association and Warp
coordinates. Preserve unwarped source coordinates and scientific arrays.

Freeze the selected installed candidate for acceptance, not development forever.
Never replace the runtime loaded by a running reference. Independent lightweight
work may continue; record material resource interference. Compare relevant
scientific histories and lifecycle behavior, not only the final picture.

Report measured complete time only after completion, including startup, output
and shutdown. Until then label station-based estimates. State evidence scope:
numerical equivalence, independent physical validation and tested platform are
different claims. A Mac test does not establish Windows performance, but lack
of that hardware need not block an explicitly Mac-scoped delivery.

After the requested release gate passes, stop adding release work. Retain enough
to reproduce the baseline and interpret regressions, not every intermediate
build/archive indefinitely. Cleanup still needs exact targets and authorization.
