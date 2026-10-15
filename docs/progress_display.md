<!-- SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Terminal progress

Use the shared printer for human-facing English progress; keep structured
records in the existing result/history writers at full precision.

```python
from mpi4py import MPI
from agentfem.time import ProgressPrinter

printer = ProgressPrinter(total_steps=100_000, every=100, comm=MPI.COMM_WORLD)
printer.status("Model ready", quantities={"Cells": (255041, "")})
# Call only after acceptance. Values and units belong to the model/caller.
printer.update(400, time=0.0004, elapsed_seconds=92.056,
               quantities={"Max |u_i|": (3.683206e-6, "mm")})
printer.status("Partial run completed")
```

```text
Model ready | Cells: 255041
Step 400/100,000 (0.4%) | Time: 0.0004 s | Elapsed: 00:01:32 | Max |u_i|: 3.68321e-06 mm
Partial run completed
```

Physical quantities use six significant digits; elapsed wall time uses one
decimal place below a minute, then `HH:MM:SS` (hours may exceed 24).
`format_quantity(value, unit, precision=6)` is also
available for presentation. It does not convert units or round stored arrays.
Nonfinite values remain visible, not silently replaced by zero.

`update` prints on rank zero at the configured interval, first and final step.
Use `force=True` for a partial stop or restart station. `every=0` disables
progress, including forced updates; explicit `status` messages remain visible.
The caller supplies globally reduced quantities where required: printing must
not introduce collective operations that could deadlock rank-local callbacks.

The existing `print(step, message)` interface remains unchanged. Prefer
`update` in new integrations. Reporting does not claim that a completed run
is scientifically verified. Keep warnings visible and put large configuration
trees in the run report. This is a text presentation API, not a JSON formatter;
machine consumers must use structured records, not parse these lines.

## Output channels and layout

The new `update` and `status` methods default to **stderr**, keeping stdout
available for machine data. An explicit `stream=` can redirect presentation
to a notebook, test capture or application stream. Legacy `print` retains its
stdout behavior. Callers producing JSON must also avoid legacy stdout messages.

On a terminal, fields wrap as complete units at the detected width. Redirected
logs default to one line per station. `width=80` explicitly chooses a layout for
either destination. An individual long field is never truncated; it may exceed
the width. No colors, animation, carriage-return redraws or terminal clearing
are used. Control characters in labels/units are escaped so a provider cannot
overwrite progress or inject misleading extra lines.

Keep the hierarchy: a short preparation summary, accepted-step progress,
explicit warnings, then a completion/partial/failure message and result path.
Status is caller-declared: do not label a wall-limit stop as a complete solve.
Progress percentages refer to increment counts, not remaining wall time;
adaptive stepping and contact cost changes make automatic ETA unreliable.
No time-to-completion or scientific-validation claim is inferred.

Design reference: [Command Line Interface Guidelines](https://clig.dev/),
particularly human/machine output separation and non-interactive output.
