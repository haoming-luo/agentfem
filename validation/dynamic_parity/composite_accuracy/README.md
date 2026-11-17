# Composite Tet10: convergence, execution cost and material updates

2026-11-17. Development on `feat/explicit-dynamic-parity`; synthetic automated
verification only. No blade geometry, external INP or customer CSV was used.

## Numerical evidence

`checks.py` assembles production element matrices in an independent static
verification harness. Its polynomial pure-bending solution has known stress and
energy. This is not a new static Step provider. `baseline.json` and
`optimized.json` retain every case, including unfavorable errors.

| Check | Coarse → fine result |
|---|---|
| Bending, ν=0.3, mean dilatation, uniform mesh | nodal error 5.15% → 1.73%; energy error 13.53% → 3.16% |
| Bending, ν=0.4999, mean dilatation, uniform mesh | nodal error 23.26% → 5.20%; energy error 39.70% → 6.76% |
| Bending, ν=0.4999, perturbed mesh | nodal error 22.59% → 5.19%; energy error 61.16% → 6.85% |
| First longitudinal frequency, 4 → 16 axial cells | error 1.417% → 0.122% |
| Central-difference temporal dispersion | halving dt reduces frequency error by approximately four |

The ordinary projected branch still has severe volumetric locking: at ν=0.4999
and four subdivisions its energy error is approximately 506%. Mean dilatation
is useful, not an unconditional accuracy guarantee. Independent ordinary P2
reproduces this quadratic manufactured field to roundoff; the composite element
is not claimed to be universally more accurate than P2. Ordinary P1 and P2
results are in `reference.json`. The nodal error is mass-weighted, not an exact
continuous L2 norm. Temporal dispersion is an eigenmode calculation, not a new
wave-pulse simulation.

## Implemented performance changes

1. Cache immutable reference integration; apply affine transformations per cell.
2. Choose efficient tensor contraction order in stiffness and stabilization.
3. Contract displacement with projected gradients before computing energy,
   rather than reconstructing per-cell B matrices each time.
4. Send mesh metadata only on the first XDMF frame; check geometry/ownership
   fingerprints on subsequent frames. Field values still gather on rank zero.

`performance.py` uses the public model/Step API: 162 cells, 100 explicit steps,
same material, mesh, supports and force, three runs per mode. Baseline is the
unmodified source from commit `afcc6ed4`, extracted into a separate temporary
directory. No baseline branch or user's files were changed. The first run in
each process includes cold initialization; all observations are retained.

| Complete workflow | Baseline median | Candidate median | Ratio |
|---|---:|---:|---:|
| No field output | 1.320 s | 0.146 s | 9.06× |
| XDMF every 10 increments | 1.553 s | 0.184 s | 8.44× |

Maximum difference of complete final nodal displacement arrays is 1.67e-16.
These are small synthetic single-process benchmarks on the recorded machine,
not predicted speedups for large industrial meshes. Root-output memory scaling
and Python constitutive-provider cost remain open performance limits.

Raw observations, platform and final fields are in `full_baseline.json` and
`full_optimized.json`. Local construction/force timing is retained separately;
do not multiply component speedups or sum overlapping tests.

## Constitutive and formulation delivery

- Generic rate-independent small-strain provider bridge, with density and a
  checked elastic stability envelope. It reuses `SmallStrainUserMaterial`.
- Native J2 adapter reuses the existing radial-return algorithm. Stress, plastic
  state, stored energy and dissipation enter the whole-mesh explicit Step.
- Trial/commit/rollback, same-partition restart, accepted stress output and raw
  `STATE_PE`/`STATE_PEEQ` result fields are covered by automated tests.
- Scope: homogeneous material, small strain, ordinary projected Tet10. Nonlinear
  mean-dilatation stabilization and finite strain are explicitly not enabled.
- Experimental 33-DOF local kernel releases the three centre variables.
  Stabilization restores six rigid modes; tying reproduces the existing kernel;
  static condensation satisfies internal equilibrium and energy minimization.
  Its positive internal inertia proves static condensation cannot simply replace
  dynamic elimination. The independent-centre elastic route now retains these variables in the
  whole-mesh Step, restart and output. It is not a C3D10M alias.

## Validation records

- `regression.txt`: 320 passed, 14 MPI-only tests skipped in serial.
- `mpi2_material.txt`: 1 parallel material comparison passed per rank.
- `mpi8.txt`: 2 tests passed per rank, including empty owned-cell partitions.
- `targeted_tests.txt`: focused development checkpoint; covered by later regression.

Theory, formulas, scope and source links:
[`composite_tetrahedron.md`](../../../docs/theory/composite_tetrahedron.md).

## Independent-centre and cyclic-material acceptance (2026-11-17)

- `acceptance_34.txt`: 243 related regression tests passed.
- `internal_mpi8.txt`: independent-centre serial/parallel displacement and energy
  agreement, including empty owned-cell partitions, passed on eight ranks.
- The independent centre retains all three inertial DOFs; a separately assembled
  global mass/stiffness reference verifies time integration and the stability bound.
- Exact native/display fields, probing and same-partition restart are checked.
- Forty-increment tension/unloading/reverse-loading checks compare accepted J2
  stresses and plastic state against the native material-point law.
- Independent-centre nonlinear materials remain explicitly unsupported; the
  supported plastic route uses tied-centre, ordinary projected interpolation.
