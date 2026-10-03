# Finite-strain J2 external beam gate

AgentFEM keeps the self-weight beam of Lewandowski et al. (2023) as an
external, structure-level promotion gate for the finite-strain J2 route.  The
problem is a three-dimensional beam with finite rotations, plasticity and
membrane stiffening; it is materially stronger evidence than another uniform
material-point or affine-cell test.

The machine-readable benchmark contract is
`agentfem.benchmark.finite_strain_j2_lewandowski_2023_beam`.

## Frozen problem

- geometry: `1.0 m x 0.04 m x 0.10 m`;
- elastic constants: `E = 210 GPa`, `nu = 0.3`;
- yield stress: `250 MPa`;
- linear isotropic hardening modulus: `1 MPa`;
- body-force path: `b(t) = -t 50 MN/m^3 e3`, `0 <= t <= 1`;
- left face: fully fixed;
- right face: `u_x = 0` symmetry condition;
- public-demo discretization: `30 x 5 x 8` box subdivisions, tetrahedra,
  quadratic displacement and 30 increments;
- public-demo observer: downward displacement at `(1, 0, 0)`.

The upstream driver and MFront behaviour are pinned to commit
`cb43561d5e36a9ef691ad2c308261448cef44e29`.  Their SHA-256 digests are stored
in `tests/lewandowski_2023_self_weight_beam_fixture.py`, so a mutable `master`
download cannot silently become the reference.

## What is and is not equivalent

The public MFront behaviour declares total Hencky strain, Hooke elasticity,
von Mises plasticity and linear isotropic hardening.  AgentFEM's current
experimental material uses a multiplicative decomposition, quadratic Hencky
elasticity, a Kirchhoff-stress J2 return and linear isotropic hardening.  The
paper reports essentially coincident responses for multiplicative and
logarithmic implementations, with a small post-yield hardening difference,
but the algorithms are not identical.  This is therefore a
**cross-formulation structural comparison**, not a bitwise reproduction.

The paper text describes point A as being at the top of the right edge, while
the executable public demo evaluates `(1, 0, 0)`, the middle of the right
extremity.  Promotion requires the observer choice to be reconciled and
recorded rather than silently mixing these two descriptions.

## Fail-closed promotion

The publication contains a plotted curve but no tabulated numerical oracle.
The pinned public driver and MFront behaviour have now been independently
reexecuted in a digest-pinned Linux/amd64 container, without installing the
legacy stack into AgentFEM's runtime. The resulting 31-point curve, resolved
package versions and image identity are bundled under
`agentfem.knowledge.external_data`. The reproducible environment recipe lives in
`tools/reference_environments/lewandowski_2023`. This closes the external-
oracle availability gap; it does **not** by itself promote the AgentFEM
candidate.

The clean, content-bound AgentFEM candidate uses the source-declared
`30 x 5 x 8` P2 tetrahedral mesh, 30 fixed increments and four MPICH ranks.
All 30 increments converged without cutback. Its final displacement was
`0.109793534 m` versus `0.109796364 m` for the independent reference;
normalized RMS and maximum curve errors were `7.71e-6` and `2.58e-5`.
Serial and four-rank curves agree to `1.28e-15` normalized RMS, and the
four-rank run was `2.99x` faster than serial. A scale-aware restart comparison
accepts the displacement, complete constitutive state, stresses, energies and
algorithmic tangent.

Three spatial levels pass the fixed mesh contract: the `30 x 5 x 8` to
`36 x 6 x 10` differences are `0.7054%` normalized RMS and `1.7861%` maximum.
A complete four-rank, central-difference 180-increment run now closes the
numerical refinement question. The `45 -> 90 -> 180` sequence is decreasing;
the 90-to-180 differences are `0.04460%` normalized RMS and `0.21498%`
maximum, both within the predeclared `0.2%` and `0.5%` contracts. All 180
increments were accepted without cutback, including a restart from the
portable 90-increment boundary. No tolerance was changed after seeing the
result. The compact formal archive and its machine-derived `promotion.json`
are under
`evidence/finite_strain_j2/lewandowski_2023_promotion_candidate`.

The observer discrepancy is now explicit rather than silently blurred: the
paper describes point A at the top of the right edge, whereas the pinned
public FEniCS executable samples the middle of the right extremity at
`(1, 0, 0)`. The bundled oracle and AgentFEM candidate both use the executable
observer, so this gate is strictly a comparison with the pinned executable
curve and does not claim to reproduce the plotted paper point A.

The numerical external, mesh, increment, serial/MPI and checkpoint/restart
gates now pass. Aggregate promotion remains **incomplete** only because the
new 180-increment artifact and the earlier compact evidence bind two different
executable package-tree identities. The fail-closed assessor will not merge
cross-revision evidence into one release claim. The next evidence maintenance
run must refresh the compact lower levels from one package identity; no new
algorithm or weaker tolerance is required. The promotion manifest binds the
independently generated CSV by
its SHA-256 digest; the candidate driver never fills source identity from its
own constants merely because a CSV was supplied. The fixture then applies fixed
AgentFEM comparison contracts of
3% normalized RMS error and 5% normalized maximum error.  These are project
promotion thresholds, not tolerances stated by the paper authors.

The AgentFEM candidate now uses the normal public workflow:
`model.material(...)`, `model.body_force(...)`, and the ordinary strong-
boundary `model.step(...)` finite-strain J2 provider. The external reference
and promotion requirements remain independent of that API migration. The
AgentFEM side can be executed without weakening this gate:

```bash
PYTHONPATH=src python tests/lewandowski_2023_self_weight_beam_driver.py \
  evidence/lewandowski-beam \
  --line-search basic \
  --reference-csv \
  src/agentfem/knowledge/external_data/lewandowski_2023_self_weight_beam.csv \
  --promotion-evidence-json /path/to/promotion-evidence.json
```

This writes `candidate_curve.csv` and an `assessment.json`. Use
`--reference-csv` with the bundled reference curve and provide a separate
promotion-evidence JSON; missing candidate convergence, MPI or restart gates
keep the status incomplete. Use `--line-search basic` for the public beam:
its valid full-Newton path is non-monotone near plastic onset, so a strictly
decreasing-residual backtracking policy can reject a convergent direction.
The evidence fixture, driver defaults, and promotion assessor share one frozen
solver contract: `basic` line search, 30 maximum Newton corrections,
`5e-6` absolute residual tolerance, and `1e-7` relative tolerance. A run that
changes these controls is retained as a diagnostic but cannot enter the
promotion set.
The candidate curve is replaced atomically after every accepted load point,
so a long interrupted run still leaves a readable accepted prefix; only a
completed run writes the final assessment.

Long refinement runs can additionally write portable accepted-state
checkpoints. The runner keeps the latest two scheduled states, and restart
uses the checkpoint coordinate as the authority: any newer CSV rows without a
matching restored constitutive state are discarded before the path continues.
The checkpoint policy is part of the Step identity and must remain unchanged:

```bash
PYTHONPATH=src mpiexec -n 4 python \
  tests/lewandowski_2023_self_weight_beam_driver.py \
  evidence/lewandowski-beam-i180 \
  --increments 180 \
  --checkpoint-directory evidence/lewandowski-beam-i180/checkpoints \
  --checkpoint-every 10 \
  --reference-csv \
  src/agentfem/knowledge/external_data/lewandowski_2023_self_weight_beam.csv

# After an interruption, select the newest complete manifest and retain the
# same output, increment count, mesh, solver controls, and checkpoint policy.
PYTHONPATH=src mpiexec -n 4 python \
  tests/lewandowski_2023_self_weight_beam_driver.py \
  evidence/lewandowski-beam-i180 \
  --increments 180 \
  --checkpoint-directory evidence/lewandowski-beam-i180/checkpoints \
  --checkpoint-every 10 \
  --resume-checkpoint /path/to/lewandowski-2023-beam-inc-XXXXXXXX.checkpoint.json \
  --reference-csv \
  src/agentfem/knowledge/external_data/lewandowski_2023_self_weight_beam.csv
```

Candidate output must never be recycled as its own reference. The assessment
records the candidate-curve digest, actual accepted load path, elapsed time,
Newton statistics, AgentFEM import path and runtime fingerprint so an installed
older version or changed output cannot silently stand in for the checkout under
test. It also records per-increment material-update, residual-assembly,
tangent-assembly, linear-solve and line-search timings together with PETSc
linear iteration counts and convergence reasons. The driver accepts
`--tangent-evaluation analytic_spectral` (the default) or
`--tangent-evaluation central_difference` for an explicit oracle comparison.

A local diagnostic on the same three-dimensional problem family, using a
`12 x 3 x 4` P2 tetrahedral mesh and 15 increments, produced indistinguishable
final displacement (`2.2e-14 m` absolute difference) while reducing the
maximum Newton count from 11 to 7 and wall time from about `43.0 s` to
`26.6 s` (`1.62x`). This workload-specific measurement explains the production
choice; it is not a portable speed guarantee and is not promotion evidence for
the external beam gate.

The same-rank restart driver compares the complete displacement curve, nodal
solution, accepted solution, every committed constitutive state, first-Piola
and Cauchy stress, deformation gradient, equivalent stress, stored energy and
algorithmic tangent:

```bash
PYTHONPATH=src python \
  tests/lewandowski_2023_self_weight_beam_restart_driver.py \
  evidence/lewandowski-beam-restart/checkpoint \
  --output evidence/lewandowski-beam-restart/report.json
```

After producing three mesh runs, three increment runs, one serial run, one MPI
run and the restart report from the same clean commit, the promotion assessor
derives every Boolean from those content-bound artifacts. It requires
decreasing three-level curve differences and fixed project tolerances; callers
cannot pass `mesh_converged=true` to bypass the calculation:

```bash
PYTHONPATH=src python \
  tests/lewandowski_2023_self_weight_beam_promotion.py \
  --mesh-run /path/coarse --mesh-run /path/medium --mesh-run /path/fine \
  --increment-run /path/i15 --increment-run /path/i45 \
  --increment-run /path/i90 \
  --rank-run /path/serial --rank-run /path/mpi \
  --restart-report /path/restart.json \
  --output /path/promotion.json
```

Primary sources:

- Lewandowski et al., *Multifield finite strain plasticity: Theory and
  numerics*, Computer Methods in Applied Mechanics and Engineering 414 (2023)
  116101, <https://doi.org/10.1016/j.cma.2023.116101>.
- Official MGIS/FEniCS finite-strain elastoplasticity demo,
  <https://thelfer.github.io/mgis/web/mgis_fenics_finite_strain_elastoplasticity.html>.
