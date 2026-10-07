# Offline pressure-defect localization and interface refinement

The figure-based accepted raw integration-point archives contain coordinates,
physical quadrature weights, deformation gradient, Cauchy stress and condensed
elastic energy. No new nonlinear solve is needed for spatial diagnostics.

The bounded fixture diagnostic `tests/zhang_2021_pressure_localization.py`
recovers mean Kirchhoff pressure as `p = det(F) * trace(S) / 3`, not simply
`trace(S) / 3`. It reads the declared geometry/material constants from the
benchmark card, checks that analytic phase assignment is uniform in each cell,
and verifies all four reconstructed integrated energy channels against the
accepted candidate before reporting localization. Unknown geometry,
multi-cell replication, bad alignment, non-finite values, nonpositive Jacobians,
duplicate cell identities, negative weights or mismatched energies are rejected.
The report binds the input archive, candidate, card and diagnostic script hashes.

This analytic phase locator is only for this benchmark. It is not a general
replacement for material-region tags. If curved geometry makes a cell straddle
the analytic phase boundary, it refuses to assign a phase silently.

| Mesh | Cells | Worst 1% of cells | Worst 5% | Worst 10% | Inclusion share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform target 0.10 | 804 | 63.34% | 94.88% | 98.77% | 0.00110% |
| Uniform target 0.05 | 2859 | 73.24% | 97.19% | 99.38% | 0.00115% |

Percentages refer to the integrated **pressure-constraint-defect energy**, not
total strain energy or exact solution error. Counts use ceiling rounding.
Nearly all of this channel lies in the matrix. Cell IDs and quadrature-weighted
centroids are retained for inspection; they are not transferred by row number
to a remeshed problem.

## Controlled refinement experiment

The fixture adds an explicit optional `--interface-size`; default mesh behavior
is unchanged. It uses Gmsh's documented
[Distance/Threshold fields](https://gmsh.info/doc/texinfo/#t10) for all internal
curves, while preserving the same material geometry and periodic equations.
The candidate records the policy, transition distance and curve sampling.
Convergence, MPI and supercell controls reject mixed policies rather than
silently calling a different meshing strategy a uniform refinement level.

With background size 0.10 and interface size 0.04, the preflight mesh has
2448 Q9 cells, minimum scaled Jacobian 0.46037 and periodic pairing error
1.11e-16. These are mesh checks, not proof of improved solution accuracy.
The numerical comparison keeps 40 fixed increments and degree-4 quadrature.
No material, solver tolerance or external acceptance threshold is changed.

The experiment is not a general adaptive solver, an estimator-driven marking
loop or a proof of optimal refinement. Its purpose is to test whether the
spatial localization can reduce the remaining defect with fewer cells.

## Observed result

All 40 increments completed. External stress/energy/tangent errors are
0.053% / 0.470% / 0.942%, within the unchanged numerical comparison contracts.
This dirty-checkout experiment does not authorize full benchmark promotion.

| Quantity | Uniform 0.05 | Background 0.10 / interface 0.04 |
| --- | ---: | ---: |
| Q9 cells | 2859 | 2448 |
| Primal energy | 0.0024226703583 | 0.0024116134105 |
| Condensed energy | 0.0023560305044 | 0.0023565015575 |
| Pressure-defect energy | 0.0000666398539 | 0.0000551118530 |

The local policy uses 14.38% fewer cells and reduces the defect by 17.30%.
Primal energies differ by 0.4564% relative to the uniform candidate, but these
different policies are not a controlled refinement sequence. The solve took
421.9 seconds; earlier timings involved different concurrent workloads and
do not establish a measured speedup.

The refined energy is farther from the published number (0.470% versus
0.014%) despite a smaller pressure defect. A published discrete-mesh target
is not an exact continuum energy; do not select meshes to match it by chance.
Next: a same-policy refinement sequence, clean-source and lifecycle evidence.

Local archive `rve-localization-20261007/evidence.tar.gz` retains the experiment
and three localization reports. SHA-256:
`6089754565595860867edf8149bd5d14fb4e63541a90d68a56f6096bac02f920`.
It is local diagnostic evidence, not a promoted/public Golden archive.
