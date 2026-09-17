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

The first full-size AgentFEM development candidate used the source-declared
`30 x 5 x 8` P2 tetrahedral mesh, 30 fixed increments and four MPICH ranks.
All 30 increments converged without cutback. Its final displacement was
`0.109793534 m` versus `0.109796364 m` for the independent reference;
normalized RMS and maximum curve errors were `7.71e-6` and `2.58e-5`.
This passes the curve-error contract by a wide margin, but the archived run is
deliberately labelled `incomplete`: its checkout was dirty and mesh,
increment, serial/MPI and restart equivalence gates remain open. The compact
development evidence is under
`evidence/finite_strain_j2/lewandowski_2023_mpi4_candidate`.

Promotion still requires observer reconciliation, candidate mesh and
increment convergence, serial/MPI equivalence, and checkpoint/restart
equivalence. The promotion manifest binds the independently generated CSV by
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
The candidate curve is replaced atomically after every accepted load point,
so a long interrupted run still leaves a readable accepted prefix; only a
completed run writes the final assessment.
Candidate output must never be recycled as its own reference. The assessment
records the actual AgentFEM import path and runtime fingerprint so an installed
older version cannot silently stand in for the checkout under test.

Primary sources:

- Lewandowski et al., *Multifield finite strain plasticity: Theory and
  numerics*, Computer Methods in Applied Mechanics and Engineering 414 (2023)
  116101, <https://doi.org/10.1016/j.cma.2023.116101>.
- Official MGIS/FEniCS finite-strain elastoplasticity demo,
  <https://thelfer.github.io/mgis/web/mgis_fenics_finite_strain_elastoplasticity.html>.
