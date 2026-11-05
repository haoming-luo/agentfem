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
Steps outside the bounded consumers below, axisymmetric point histories
and MPI recording remain unadmitted.

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

### Small-strain user materials

With a physical `duration`, this Step also accepts `temperature=300.0` (kelvin)
or a scalar `histories.temperature(...)` recorded against physical time. The
frozen input is passed to the local material as endpoint temperature and the
increment from the accepted start. Temperature knots are visited exactly.
The material owns thermal strain and must declare a stress-free initial state
at the initial temperature. This is prescribed thermal input, not a heat solve.

Selected material points then include `TEMP` in kelvin, co-located with `S`,
total `E` and named state. The final mesh output includes cell `TEMP` on the
same grid as displacement and stress. Restart requires the same frozen thermal
path; it does not invoke a fictitious material update to rebuild temperature.

The physical-time small-strain UserMaterial Step accepts the same
`material_points=` requests in its existing serial homogeneous 3D scope.
It records raw Cauchy `S`, total infinitesimal `E`, and named material state
at each accepted physical time. Tensor shear is retained; no thermal or
inelastic strain partition is invented. This option requires `duration` and
does not change the legacy rate-independent checkpoint route.

Selection, units, material and loading identity are frozen together. The v2
checkpoint includes the complete selected history and integrity digest;
missing history cannot be reconstructed from final fields. Failure during
frame preparation or state commit restores the previous accepted displacement,
constitutive response and history. `examples/small_strain_relaxation` offers
`--material-point` for ordinary run/restart use.

### Finite-strain user materials

The same `material_points=` option also works with the experimental ordinary
finite-strain UserMaterial Step, in serial 3D, plane strain and its declared
plane-stress reduction. It records `S` (Cauchy), `P` (first Piola), effective
three-dimensional `F`, `E_GL = (F.T @ F - I)/2`, named constitutive state and
supplied uniform `TEMP`. Plane-stress `F33` is the accepted thickness stretch.
All tensor channels are full 3×3 arrays; state shapes and units follow their
schema. Stress units follow the Model; undeclared units remain unknown.

`E_GL` is total Green–Lagrange strain, including any thermal or inelastic
deformation. It is not the infinitesimal `E` of the creep consumer and cannot
be substituted directly into a mechanical strain-life curve. No strain
partition or material-specific life model is inferred. State output aliases
must not collide with the recorded kinematic or thermal channels.

This consumer retains the Step's serial same-layout checkpoint format. A
checkpoint without the same selection cannot seed its missing history. Frames
are prepared before state commit and published only after acceptance; failed
commit, corrupt restore and rejected trials leave the accepted history intact.
Only selected points require derived strain arrays; tangent matrices and full
field histories are not stored. See the `--material-point` option in
[`examples/history_material`](https://github.com/haoming-luo/agentfem/tree/main/examples/history_material).

### Shared continuation behavior

`step.material_point_histories` returns ordinary `HistoryResult` copies even
while paused. A creep v1 serial or v2 portable-format Step checkpoint includes the
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
`tests/test_finite_material_point_histories.py` covers the finite-strain
consumer, stress/strain measures, both example laws, thermal input, plane
stress, original-cell identity, atomic continuation and sparse field output.
