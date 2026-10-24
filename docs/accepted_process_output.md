# Process fields and restart custody

Process visualization, scalar histories and restart checkpoints have different
jobs. Field output must not change the integration time grid or material state.
Machine-readable XDMF/PVD time indices retain binary64 round-trip precision;
the compact numerical formatting of terminal progress does not apply to stored
times. Adjacent distinct accepted time values must not collapse in the index.

```python
frames = results.AcceptedResponseSeries(
    context.artifact("results.xdmf"), storage="frame_files",
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

The series owns its field-selection configuration, including nested component
selectors. Later edits to the caller's mapping do not change what the series
means. The first successfully written frame freezes an independent copy of
the material-state descriptor; an in-place provider schema change is rejected
before another frame is written. Rejected first-frame data does not establish
a successful schema. These copies contain small descriptors, not full state
or mesh arrays.
This response adapter requires one finite 3×3 Cauchy stress tensor per cell.
It does not reinterpret a different stress measure or silently export a scalar
under the tensor's name. The provider's source location is preserved in the
series metadata and cannot change between frames. These checks protect output
meaning; they do not independently validate the constitutive law.

Open `results.xdmf` and use ParaView's time controls. Apply Warp By Vector once.
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

If heavy-data close succeeds but the final JSON metadata publication fails,
the same series object may retry `close` with the original `complete` flag.
The retry publishes the already-frozen metadata only: it does not rerun the
solver, rewrite frames or change a partial analysis into a completed one.
Further writes remain closed. A heavy-data close failure is different: later
calls keep reporting `AFM-FRAMES-CLOSE-FAILED`, rather than silently returning
an empty successful-looking artifact set. This is an in-process publication
retry, not automatic recovery of damaged HDF5 or missing metadata after a crash.
The underlying serial `ResultWriter`/`UnifiedXDMFTimeSeries` objects also reject
writes and re-entry after close, whether the context completed or was aborted.
Repeated cleanup cannot promote an interrupted index to complete. Continue in
a new series rather than reusing a closed writer: the original heavy data must
not be reopened in truncating mode.

## Checkpoint policy

### Live viewing with immutable frames

The opt-in serial `storage="frame_files"` backend publishes one `results.xdmf`
entry point and a managed `results-data/` directory. It writes the shared mesh
once, closes each compressed frame file, then atomically updates the index.
Published heavy-data files are never modified. ParaView may read the index while
the solver continues; use **Reload Files** to discover newly published times.
Keep the index and its data directory together when moving the result.

An interruption during a later frame leaves earlier published frames readable;
unfinished temporary files are not included in the index. This is not an HDF5
SWMR writer, a power-loss guarantee, or a restart archive. The live index is
marked incomplete until clean close; the series JSON `complete` flag records
whether the analysis reached its horizon. A cleanly closed partial analysis is
still not a completed simulation. Existing output paths are rejected, not
overwritten. Restart currently writes a separate continuation segment; missing
earlier frames cannot be reconstructed from a final checkpoint.

For 100 frames including initial and final states, use
`RunPolicy(field_intervals=99)` with at least 99 increments. A partial run may
have an additional partial endpoint. Frame count does not change integration
steps. The legacy `storage="single_file"` remains the default for compatibility.

### Legacy single-file output

The serial unified HDF5 writer flushes storage after each successfully written
output frame, rather than holding object metadata until the end of a long run.
An abrupt exit **between completed writes** can therefore leave readable heavy
data. Each successful frame also atomically updates `fields.partial.xdmf`,
explicitly labelled incomplete. Read this index **only after the writer has
stopped**; concurrent live reading is not supported by this non-SWMR layout.
Clean close atomically publishes `fields.xdmf` and removes its own partial
index, so successful output still has one visualization entry point. An
existing partial index is preserved and requires a fresh output path.
A readable HDF5 file or partial index is not a completed result or a restart archive. This bounded
protection does not promise recovery from interruption during a write, disk
failure, or power loss, and cannot repair previously damaged files. Continue
to use verified checkpoints for resuming computation.

The implementation follows the documented [HDF5 buffer flush](https://docs.h5py.org/en/stable/high/file.html#h5py.File.flush)
boundary, not the stronger [SWMR](https://docs.h5py.org/en/stable/swmr.html)
protocol. SWMR would require a different predeclared/appendable dataset layout;
simply enabling its flag on the present per-frame group writer is not valid.

Keep the existing `checkpointing.every(..., keep_last=2)` for interrupted runs.
The project wrapper reuses an integrity-checked scheduled endpoint instead of
writing another `accepted-final` copy. `report.json` and SimulationResult give
the authoritative checkpoint path; consumers must not assume its filename.

`RunPolicy(completed_checkpoint_keep=1)` explicitly removes older owned transient
checkpoints only after reaching the full horizon, exporting fields and publishing
the structured result successfully. A result-construction or publication failure
preserves the existing checkpoints. The published result lists only checkpoints
selected for retention; physical removal happens afterwards. The final checkpoint's
payload hashes are verified before pruning.
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
