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
Step 400/100,000 (0.4%) | Time: 0.0004 s | Elapsed: 92.1 s | Max |u_i|: 3.68321e-06 mm
Partial run completed
```

Physical quantities use six significant digits; elapsed wall time uses one
decimal place in seconds. `format_quantity(value, unit, precision=6)` is also
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
