# Figure-based mixed-energy follow-up

This investigation keeps the Figure 10(a) phase assignment, Q9/DPC1 space,
material constants and 40-increment loading path fixed. It does not change
the 3% external comparison or 0.5% mesh-energy convergence contracts.

## What accounts for the remaining mesh change?

The existing public mixed-energy diagnostic records the identity

`primal = condensed + pressure_orthogonality + pressure_constraint_defect`.

Between mesh targets 0.07 and 0.05, the signed changes are:

| Channel | Fine minus coarse |
| --- | ---: |
| Primal elastic energy density | -1.2947205037e-4 |
| Condensed elastic energy density | -1.8724816247e-7 |
| Pressure-constraint-defect energy density | -1.2928480221e-4 |
| Pressure orthogonality | -2.9398096908e-15 |

Thus 99.8554% of the primal change is algebraically attributable to the
pressure-constraint-defect channel. The condensed energy changes by 0.00795%.
This attribution is **not** a continuum error estimate, a claim that the
condensed channel equals the publication's energy, or permission to substitute
channels. The previous primal-energy convergence failure remains in force.

The interpretation is consistent with the independent pressure interpolation
in [Sussman and Bathe (1987)](https://doi.org/10.1016/0045-7949(87)90265-3):
discrete pressure is not generally identical pointwise to pressure recovered
from displacement. That reference does not settle the energy semantics of a
different paper's reported numbers. The quantitative attribution above comes
from AgentFEM's explicitly recorded algebraic channels, not from that reference.

`tests/zhang_2021_energy_sensitivity.py` consumes at least three candidate
assessment files, retains artifact hashes and all signed channels, rejects
mixed geometry/loading and malformed decompositions, and reports the original
mesh audit alongside the explanation. Its result never authorizes promotion.
For an unresolved/zero primal change it reports a null fraction rather than
inventing a percentage. Fractions are signed and may exceed one when channel
changes cancel.

## Clean-source repeat and quadrature control

Commit `597e8795dc0d5e1e8390798d94e56dff688f2f97` reproduced the 2859-cell,
40-increment figure-based solution with `tracked_dirty: false` and stable
start/end identities. Stress and primal energy are numerically identical to
the development candidate. External errors remain 0.099%, 0.014% and 0.507%
for stress, primal energy and tangent, respectively. This removes the dirty
checkout limitation for that candidate, not the mesh-convergence limitation.

On the identical 804-cell mesh, increasing quadrature degree from 6 to 8
changes stress by 0.01119%, primal energy by 0.08267% and tangent by 0.05629%.
The primal-energy error against the published number remains 9.919% at degree
8. Higher quadrature therefore does not explain away the spatial discrepancy
at this mesh size. The quadrature slice does not certify fine-mesh quadrature
or spatial convergence.

The completed 4/6/8-degree audit reports `content_bound: true` and
`quadrature_converged: true`, while overall promotion remains incomplete.
All three candidates retain identical executed discretization fingerprints
and clean, stable runtime/benchmark identities. The audit now also locks
macro gradient, cell repetitions, reference-cell area and prescribed path;
four regression cases demonstrate rejection of changes to these controls.

Local archive: `rve-clean-energy-20261007/clean-evidence.tar.gz`, SHA-256
`1d29978e3b09b0b0357ba24c1541caa7fd0ec49ee157020261b8fe6b08f9eac4`.
It contains the clean fine run, three quadrature candidates, quadrature audit
and energy-sensitivity report. The latter references the development mesh
sequence archived separately in `rve-geometry-20261007`. These are retained
local evidence, not a public Golden download.

## Next decision

Distinguish a clean reproduction of the paper's discrete target from spatial
convergence of the continuum approximation. Check quadrature on a fixed mesh,
then investigate the spatial distribution of the pressure defect before
choosing localized refinement or another explicitly declared interpolation.
Do not tune material data, replace the energy channel, or enrich pressure
without independently verifying stability and changing the formulation label.
