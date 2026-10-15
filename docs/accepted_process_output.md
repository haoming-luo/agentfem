# Process fields and restart custody

Process visualization, scalar histories and restart checkpoints have different
jobs. Field output must not change the integration time grid or material state.

```python
frames = results.AcceptedResponseSeries(
    context.artifact("process.xdmf"),
    state_fields={"damage": ("damage", None)},
    velocity=True, acceleration=True, units={"time": "s"},
)
project.execute_step(step, context=context, frames=frames,
    policy=project.RunPolicy(field_intervals=40, completed_checkpoint_keep=1))
```

The bounded serial response-material route records the starting accepted state,
40 evenly distributed increment intervals and the endpoint (41 frames when the
step count is divisible by 40). Nondivisible stations round up to an existing
accepted increment; no interpolation, artificial extra increments or altered dt.
At most one frame per accepted macro increment is written. A partial run adds
its endpoint; a restart writes a new continuation segment on the original
schedule. Missing earlier frames are not reconstructed. Zero-state fields at
initialization remain the provider's responsibility, not the output adapter's.

Open `process.xdmf` and use ParaView's time controls. Apply Warp By Vector once.
Each frame is one volume grid with all selected point and cell fields. The
existing writer shares topology and reference coordinates, compresses field
arrays, and streams one station at a time rather than retaining all fields in
RAM. Output size is fixed mesh plus selected fields per frame, not a copy of
the whole checkpoint for every frame. On an ordinary solve/observer failure,
already written valid frames are closed as an incomplete segment. Storage
failures remain explicit; a partial/corrupt write is not declared valid.

`AcceptedResponseSeries` can also be written directly from restored accepted
states using `write(step)` and `close(complete=False)`. It never integrates or
loads a checkpoint itself. Restore with the original compatible model/runtime;
never remove identity checks to visualize a historical checkpoint.

## Checkpoint policy

Keep the existing `checkpointing.every(..., keep_last=2)` for interrupted runs.
The project wrapper reuses an integrity-checked scheduled endpoint instead of
writing another `accepted-final` copy. `report.json` and SimulationResult give
the authoritative checkpoint path; consumers must not assume its filename.

`RunPolicy(completed_checkpoint_keep=1)` explicitly removes older owned transient
checkpoints only after reaching the full horizon and successfully exporting
results. The final checkpoint's payload hashes are verified before pruning.
Only current-run recorded paths inside its output directory are eligible;
restart-source records, other directories and unrelated files are untouched.
Removal is permanent and reported in `checkpoint_retention`. Default `None`
retains the existing policy. Partial and failed runs never apply this completion
cleanup. This bounded cleanup supports current serial transient schemas only.
Checksum validation is storage integrity, not proof of scientific equivalence;
restart regression tests remain separate gates.

## Cross-software interpretation

Compare physical quantities, units, coordinate frames, locations and times,
not labels alone. A named material damage variable is not automatically Abaqus
SDEG, and internal work is not stored energy. Missing plastic strain or contact
traction fields must not be filled with zeros or invented aliases. Scalar tool
resultants do not equal a nodal contact traction distribution.

The Abaqus Explicit `*OUTPUT, FIELD` default is 20 intervals when no frequency
is supplied, not final-only output; see the
[official keyword reference](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKEYRefMap/simakey-r-output.htm).
Forty intervals is a project presentation choice, not a scientific core default.
