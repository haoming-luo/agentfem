# P92 nonuniform structural screen

## Purpose and scope

Exercise the existing physical-time user-material consumer in bending and
stress redistribution, beyond homogeneous patches. The material is the
research-only isotropic tensor extension of the fixed 650 C parameters in
[Zhang et al., Metals 10 (2020) 307](https://doi.org/10.3390/met10030307).
The supplied Poisson ratio is 0.3. This is not the separate 2019 creep-fracture
law, a new calibration, or a component-life prediction.

The solver, accepted state and checkpoint belong to the ordinary `model.step`
workflow. The repository tool supplies geometry, loading and measurements;
it does not implement a second global solver.

## Test definition

A 4 x 1 x 1 mm solid is clamped on x=0. Its x=4 face has prescribed y
displacement, with the other components free. Displacement rises linearly
to 0.02 mm in 4 s and is held for 60 s. The temperature is fixed at 650 C.
This is a small-strain test, not a finite-rotation benchmark.

Hexahedral meshes have `(4r, r, r)` cells, for r=1,2,4,8. The base physical
time grid has eight ramp increments and twelve hold increments. Time
refinement doubles both counts without changing the waveform or duration.
Each solve is serial and limits numerical library threads to one.

Before interpreting the fine-grid result, set the numerical screening goals:
the final mesh doubling changes ramp-end and hold-end reactions by less than
5%; time-step halving at a fixed mesh changes them by less than 1%. These are
bounded numerical screening goals, not estimates of exact error or engineering
allowables. Report the actual differences even if a goal fails. Do not tune
material parameters to make them pass.

Measure reactions at the constrained support and loaded component separately.
Their vector sum tests force equilibrium. Summing the unconstrained internal
force over every node alone is insufficient: internal assembly may cancel
globally even when the nonlinear equations have not converged. Also retain
the accepted Newton residual. The clamp has boundary-condition transitions;
maximum sampled stress is diagnostic, not a converged hotspot quantity.

The selected material-point history uses cell 0 in each mesh; it is useful for
same-layout restart, not a common physical point for cross-mesh comparison.
Restore at the accepted end of the ramp into a fresh Step and require identical
selected histories and endpoint observables.

## Reproduction

With the repository research tools and AgentFEM available:

```bash
python -m tools.p92_nonuniform_structure --resolution 1 --checkpoint-dir bending-restart --output bending-r1.json
python -m tools.p92_nonuniform_structure --resolution 4 --refinement 2 --output bending-r4-t2.json
```

Use new paths: evidence is not overwritten. The 2048-cell mesh is an explicit
manual screen, not part of every CI test. CI uses the four-cell composition and
restart test. Run numerical screens sequentially and retain wall times without
calling them a general performance comparison.

## Reference practice

[Abaqus creep integration verification](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEBMKRefMap/simabmk-c-creep.htm)
separates constitutive integration checks against exact solutions from coupled
loading tests, and distinguishes stable integration from time accuracy. This
screen follows that separation: existing independent scalar/local checks are
retained; mesh refinement here does not replace experimental validation.

## Recorded numerical evidence

The six raw reports are retained in
`evidence/creep/2026-11-07-p92-nonuniform.json`, including tool identity,
constraint reactions, accepted residuals and wall times.

| Cells | Increments | Ramp-end reaction (N) | Hold-end reaction (N) |
| --- | --- | --- | --- |
| 4 | 20 | 15.837950 | 15.817883 |
| 32 | 20 | 11.787592 | 11.575751 |
| 256 | 20 | 10.596423 | 10.236817 |
| 2048 | 20 | 10.262701 | 9.854165 |
| 256 | 40 | 10.605809 | 10.234115 |
| 2048 | 40 | 10.273501 | 9.851034 |

Relative differences use the finer result as denominator. The last mesh
doubling changes total reactions by 3.25% and 3.88%; time refinement at
2048 cells changes them by 0.105% and 0.032%. These pass the stated bounded
total-reaction screens. The four-cell interrupted run reproduces selected
histories and endpoint observables exactly; other runs did not test restart.

The relaxation drop itself changes by about 12% between the last two meshes
at 20 increments (fine-drop denominator), so its spatial convergence remains
open. The maximum sampled clamp stress also continues to grow. Neither
quantity is certified by the total-reaction screen. Stop this mesh ladder
here rather than treating progressively larger solves as experimental
validation. Engineering multiaxial material validation remains a separate gate.
