# Figure-based spatial follow-up — 2026-10-08

Status: **incomplete; no benchmark promotion**. Five fresh solves completed
40 accepted increments each, using Figure 10(a), Q2/DPC1, degree-4 quadrature,
unchanged material/solver settings, and clean source
`e1df87a2d15c7925c233fb11b6869fe83b5c6775` throughout execution.
Scientific runtime SHA-256:
`8e6c6e4695fb30fe18fdcda8ecac5e53518454d4d91671b7d2b1a0d226c7dd03`.

## Two controlled mesh families

All sizes below are mesh-policy inputs, not measured effective element sizes.
Distance sampling is 200 throughout. The shared coarse run is counted once.

| Family | Background / interface / transition | Cells | Table 5 stress / primal energy / tangent errors (%) |
| --- | --- | ---: | --- |
| Shared coarse | 0.10 / 0.04 / 0.10 | 2448 | 0.053 / 0.470 / 0.942 |
| Local middle | 0.10 / 0.03 / 0.10 | 3543 | 0.064 / 1.312 / 0.706 |
| Local fine | 0.10 / 0.0225 / 0.10 | 5538 | 0.085 / 1.770 / 0.590 |
| Global middle | 0.075 / 0.03 / 0.075 | 4215 | 0.075 / 1.658 / 0.783 |
| Global fine | 0.05625 / 0.0225 / 0.05625 | 7224 | 0.080 / 2.363 / 1.011 |

The global family scales all three lengths by 0.75; the local family changes
only interface size. The audit checks invariant policy, fixed size ratios and
strictly increasing actual cell counts. It does not equate local refinement
with global spatial convergence. This follows the controlled-spacing principle
in [NASA/NPARC spatial convergence guidance](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html),
without claiming a Richardson/GCI estimate or a continuum error bound.

| Final pair | Stress change (%) | Primal-energy change (%) | Tangent change (%) |
| --- | ---: | ---: | ---: |
| Local | 0.075890 | 0.465752 | 0.160360 |
| Global | 0.019064 | 0.722356 | 0.238598 |

Unchanged limits are 0.5%, 0.5%, 1%, **plus decreasing successive changes**.
Local energy and tangent pass, but local stress change increases from 0.068473%.
Global stress passes; global energy decreases from 1.208303% but remains over
0.5%, and global tangent increases from 0.171716%. Neither full axis passes.
All five external numerical comparisons pass their existing contracts; that
is not evidence of completed spatial convergence.

All five result manifests verify, and independent integration-point
reconstruction reproduces the energy channels. On the finest global mesh,
the worst 5% of cells contribute 94.8664% of the pressure defect. This remains
algebraic localization evidence, not a justification for replacing primal
energy with condensed energy.

## Reproducibility and publication safety

Original outputs, two audits and five reconstruction reports are retained in
the maintainer archive `outputs/rve-spatial-20261008/evidence.tar.gz`:

`e6078f68c40caabca0b9f30150b05f9a3a5ac88e6debe545466e23cd72bc94fd`

Raw outputs were not rewritten. Solve timings are not a performance comparison:
concurrency differed, and thread environment settings alone did not establish
single-thread execution.

A separate reproduced output bug silently returned a path after a serial
manifest write failure. Publication now raises consistently with or without an
MPI communicator. Injected replacement failures preserve the previous sealed
manifest. Targeted regression: 155 passed, one explicit unpromoted-benchmark
skip; two collective publication tests passed on each of two MPI ranks.

Remaining gates: full spatial convergence, fine-discretization quadrature and
loading-path checks, and figure-specific tangent/MPI/restart evidence. Historical
prose-geometry evidence must not be transferred automatically. No acceptance
tolerance or constitutive parameter was changed.
