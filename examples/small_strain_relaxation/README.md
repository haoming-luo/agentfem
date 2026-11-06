# Local rate-dependent material, ordinary structural workflow

With the physical-time small-strain candidate installed:

```bash
agentfem check
agentfem run --name relaxation
```

`case.py` describes the mesh, constraints and two-second ramp/hold/unload.
`material.py` provides only a local standard-linear-solid stress, viscous
history and algorithmic tangent. AgentFEM owns assembly, Newton iteration,
accepted history, rollback, results and restart. Coefficients are synthetic,
in Pa and seconds; the Model explicitly declares SI (geometry in metres).
The declared initial state is stress-free with zero history.
The SI declaration is part of restart identity: checkpoints from an older copy
without declared Model units must not be reused with this updated case.

The run writes a midpoint checkpoint, a final checkpoint, final `results.xdmf`
and its H5 data, and the accepted load-amplitude/time history in `result.json`.
This small example writes the final field, not an animation. In ParaView use
`U` for Warp By Vector and `S_CELL` for cell-averaged stress coloring.

Add `-- --material-point` to record raw `sample.S`, `sample.E` and `sample.EV`
at every accepted increment. These compact tensor histories are independent of
field frames. Use the same option when restarting that run; history cannot be
backfilled from a final field. `E` is total strain, not a mechanical strain-life
input inferred from the material law.

To test continuation, pass the previous run's midpoint manifest:

```bash
agentfem run --name continued -- --restart outputs/001-relaxation/midpoint.checkpoint.json
```

Keep each manifest with its referenced NPZ payloads. Restart is currently
same-mesh-layout serial; this example does not enable MPI or temperature coupling.
The amplitude table uses normalized coordinates from 0 to 1, while the material
receives physical time in seconds. Changing duration changes a rate-dependent
problem, not merely its display units.
