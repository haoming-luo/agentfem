# Co-located material histories

Local creep/fatigue assessment needs stress, strain and temperature at the
**same material point**. Separate maxima can come from different cells, and
cell-averaged visualization fields are not raw integration-point histories.

## Ordinary use

```python
step = model.step(
    target=u, duration=duration, temperature=temperature_path,
    material_points=(results.material_point("hot_spot", cell=12, point=0),),
)
result = step.solve_result()
stress = result.histories["hot_spot.S"]       # (accepted times, 3, 3)
strain = result.histories["hot_spot.E"]       # total infinitesimal tensor strain
temperature = result.histories["hot_spot.TEMP"]
```

The model still owns geometry, material and loading. The ordinary Procedure
owns evaluation and acceptance; the case does not implement a sampling loop.
The thermal cyclic-creep example demonstrates this option with a solved
temperature history and coupled creep/plastic state.

`cell` is the original input-cell ID, not the node ID or the current partition's
local cell index. `point` is the zero-based index in the actual reference
quadrature rule. Use the existing raw integration-point archive to inspect
`cell_keys` and quadrature coordinates; a physical coordinate probe would
instead need an explicit ambiguity/interpolation policy. Unknown IDs, duplicate
names and changed reference geometry/rules fail explicitly.

## Scope and scientific identity

The initial consumer is serial Cartesian 3D small-strain creep: homogeneous
power-law or joint creep/plastic material, and existing regional pure-creep
materials. Supplied scalar, FE or accepted-history temperature is sampled at
the same constitutive point. Joint regional plasticity, arbitrary user-material
Steps, axisymmetric point histories and MPI recording remain unadmitted.

Each accepted endpoint records `S`, `E`, `CE`, `CEEQ`, and supplied `TEMP`.
Joint plasticity also records `PE`, `PEEQ`, and `ALPHA` where present. Tensor
shear is not engineering shear. Stress is Cauchy; temperature is kelvin. Stress
and time units come from the model; absent unit declarations stay unknown.
`result.metadata["material_point_history"]` retains reference position,
original cell, quadrature index/rule, mesh identity, channel conventions and
the material/loading/thermal-path identity. No projected or smoothed values
enter these histories.

The first sample is the first accepted increment. The initial displacement
need not be an equilibrated preload, so no fictitious time-zero stress is
inserted. Restrict assessments to recorded time coverage; resolve holds and
reversals with adequate physical increments. The history itself does not
certify time discretization or supply an empirical rupture law.

## Pause, recovery and storage

`step.material_point_histories` returns ordinary `HistoryResult` copies even
while paused. A v1 serial or v2 portable-format Step checkpoint includes the
entire selected history, identity and integrity digest. In this consumer, v2
does **not** imply MPI admission. Changing or removing the selection on restart
is rejected; missing, corrupted or misaligned history rolls back the entire
restore, including displacement, constitutive state and thermal time.
After construction, changing the recording selection, units or scientific
loading/material/temperature identity is also refused before advance or save;
past values cannot be relabeled under a new contract.

Sampling is every accepted increment, independent of XDMF frames. Memory and
archive size scale with selected points and accepted increments, not with all
mesh points. Select a small scientifically motivated set; this is not a
full-field time archive. Existing result-manifest/NPZ history storage retains
the tensor arrays and their common time grid without a new result file format.

## Acceptance

`tests/test_material_point_histories.py` covers nonuniform physical-time
temperature, two distinct material locations, regional pure creep, homogeneous
joint plasticity, tensor association, pause/resume, old serial thermal-checkpoint
compatibility, corruption rollback, failed-trial exclusion and existing NPZ
result round-trip. These are integration checks, not real-material validation.
