# RVE homogenization and physical field statistics

This page defines the scientific meaning of AgentFEM's periodic-cell histories
and integration-point statistics.  The two capabilities share one rule:
reported reductions must follow the physical integration measure, not the
number or ordering of stored values.

## Physical weights for quadrature fields

For a quadrature value (a_q), AgentFEM constructs the owned physical weight

\[
w_q = w_q^{\mathrm{ref}}\left|\det J_q\right|m_q,
\]

where (w_q^{\mathrm{ref}}) is the reference quadrature weight,
(J_q) is the coordinate-map Jacobian and (m_q) is an optional declared
measure multiplier.  The default is (m_q=1).  Axisymmetric or otherwise
weighted measures must pass their multiplier explicitly; the software does not
infer it from a field name.

The weighted mean and variance are

\[
\bar a = \frac{\sum_q w_q a_q}{\sum_q w_q},
\qquad
s_a^2 = \frac{\sum_q w_q(a_q-\bar a)^2}{\sum_q w_q}.
\]

A weighted quantile is obtained from the cumulative physical measure after
sorting by (a_q).  This matters on distorted meshes, mixed cell sizes and
nonuniform quadrature: an unweighted percentile answers a question about stored
samples, not about material volume.

`QuadratureField.weighted_statistics()` and
`results.weighted_field_statistics()` return the measure, mean, standard
deviation, requested quantiles and threshold fractions.  MPI reductions use
owned cells only.  Exact weighted quantiles currently gather the compact scalar
value/weight arrays before broadcasting the result; they are intended for
scientific summaries, not for copying full tensor histories to every rank.

Tensor fields require an explicit component or invariant before reduction.
AgentFEM deliberately rejects an undeclared tensor-to-scalar conversion.

## Finite-strain periodic-cell averages

Let (Omega_0) be the complete reference cell and (V_0=|\Omega_0|).
For an affine-periodic finite-strain analysis, AgentFEM records

\[
\bar{\mathbf F}
=\frac{1}{V_0}\int_{\Omega_0}\mathbf F\,\mathrm dV,
\qquad
\bar{\mathbf P}
=\frac{1}{V_0}\int_{\Omega_0}\mathbf P\,\mathrm dV.
\]

When the computational mesh contains only the solid phase of a porous cell,
voids carry zero stress and the solid integral is still divided by the complete
cell volume.  The result is therefore an effective RVE stress, not a
matrix-phase average.

The macroscopic Cauchy stress is evaluated consistently in the current
configuration,

\[
\bar{\boldsymbol\sigma}
=\frac{1}{\bar J V_0}
  \int_{\Omega_0}J\boldsymbol\sigma\,\mathrm dV,
\qquad
\bar{\mathbf P}
=\bar J\,\bar{\boldsymbol\sigma}\,\bar{\mathbf F}^{-T}.
\]

The second equality is retained as a numerical consistency error rather than
assumed silently.

## Hill--Mandel evidence over accepted increments

For two consecutive accepted states (n) and (n+1), AgentFEM applies the same
trapezoidal stress rule at both scales:

\[
\Delta W_{\mathrm{micro}}
=\frac{1}{V_0}\int_{\Omega_0}
\frac{\mathbf P_n+\mathbf P_{n+1}}{2}
:\left(\mathbf F_{n+1}-\mathbf F_n\right)\,\mathrm dV,
\]

\[
\Delta W_{\mathrm{macro}}
=\frac{\bar{\mathbf P}_n+\bar{\mathbf P}_{n+1}}{2}
:\left(\bar{\mathbf F}_{n+1}-\bar{\mathbf F}_n\right).
\]

The stored residual is

\[
r_{\mathrm{HM}}
=\Delta W_{\mathrm{micro}}-\Delta W_{\mathrm{macro}},
\]

with a relative error normalized by the larger work magnitude.  This is an
accepted-increment audit, not an alternative equilibrium equation.

The current public request is intentionally fail-closed outside its verified
scope: quasistatic finite strain with affine-periodic kinematics and without
body-force, natural-load or inertia power.  The general Hill--Mandel relation
can include those terms, but they require an enlarged energy ledger and are not
silently omitted here.

## Stress-state convention

From the macroscopic Cauchy stress,

\[
\sigma_m=\frac{1}{3}\operatorname{tr}\boldsymbol\sigma,
\qquad
q=\sqrt{\frac{3}{2}\mathbf s:\mathbf s},
\qquad
J_3=\det\mathbf s,
\]

AgentFEM reports

\[
\eta=\frac{\sigma_m}{q},
\qquad
\bar\theta
=1-\frac{2}{\pi}
\arccos\left(\frac{27J_3}{2q^3}\right).
\]

This normalized Lode convention gives \(+1\) for axisymmetric tension, \(0\)
for pure shear and \(-1\) for axisymmetric compression. When \(q\) vanishes,
\(\eta\) and \(\bar\theta\) are undefined. NPZ output uses `NaN`; structured
result histories use a zero placeholder together with
`homogenized_stress_state_defined=0`.

## One scientific history, two output cadences

`results.periodic_cell_history(constraint)` attaches a lightweight recorder to
the affine nonlinear step.  It observes every accepted increment while the
spatial XDMF cadence may remain sparse.  It retains only one preceding
microscopic state plus compact macroscopic records, so its memory does not grow
with the number of spatial degrees of freedom.

Each macro frame aligns:

- deformation, strain, first-Piola and Cauchy tensors;
- energy density, phase fractions, triaxiality and Lode state;
- Hill--Mandel micro work, macro work and residual;
- accepted increment size, Newton iterations, final residual, periodic
  equation mismatch and accepted attempt number.

The NPZ artifact is the lossless numerical contract.  The CSV artifact is the
human-readable flattened view.  `result.json` records the history source,
scope, spatial-frame count and undefined-value convention.

## Minimal use

```python
output = results.output_plan(
    "outputs/cell",
    field=results.field_output("U", "S", "LE", every=5),
    requests=(results.periodic_cell_history(periodicity),),
)

step = model.step(
    target=displacement,
    material=material,
    constraints=periodicity,
    increments=20,
    output=output,
)

result = step.solve_result()
```

## Unload, reload and non-proportional macro paths

The step coordinate is an ordering coordinate, not a requirement that the
physical macroscopic deformation increase proportionally. Use one typed
piecewise-linear matrix path when an RVE must unload or change loading
direction:

```python
macro_path = constraints.deformation_gradient_path(
    coordinates=(0.0, 0.4, 0.7, 1.0),
    gradients=(F0, F_tension, F_unloaded, F_tension_shear),
    name="tension_unload_shear",
)
periodicity = constraints.abaqus_periodic_cell(
    displacement,
    nodes=nodes,
    equations=equations,
    anchor_node=anchor,
    reference_nodes=references,
    deformation_gradient_path=macro_path,
)
```

`F0` must be the identity. AgentFEM checks the determinant over each complete
linear segment, not only at its endpoints. Path knots cannot be skipped by the
global constitutive transaction, and their full matrix history is part of the
scientific and restart fingerprints. This separates a monotone execution
coordinate from a potentially non-monotone material history without hiding
the latter in a callback. These knots define the intended physical path, while
accepted subincrements between them control numerical integration accuracy;
constitutive path convergence must therefore be checked independently.

For an integration-point scalar:

```python
summary = quadrature_state.equivalent_plastic_strain.weighted_statistics(
    quantiles=(0.5, 0.95, 0.99),
    thresholds=(0.02,),
)
```

## True-void regression and refinement boundary

The finite-strain J2 true-void benchmark now has two deliberately separate
evidence layers. The ordinary automated layer is a **fixed-stack software
regression**. It freezes a geometric spherical cavity, an `h/L=0.25`
first-order tetrahedral mesh, a two-increment isochoric loading path, the
runtime stack, and a portable mesh identity. Only when those identities match
does the Golden compare the complete-cell-volume macroscopic first-Piola
stress, physical-weighted PEEQ mean and upper quantiles, and meshed solid
fraction. Maximum PEEQ remains a localization diagnostic rather than a Golden
quantity. This contract detects implementation or dependency drift; it is not
a cross-platform reference solution and does not establish mesh convergence.
The AgentFEM version stored in the card identifies the clean reference source;
it does not exempt a newer candidate release from comparison. On the declared
Darwin/arm64 reference stack, maintainers use a fail-closed mode:

```bash
AGENTFEM_REQUIRE_RVE_GOLDEN=1 python -m pytest -q \
  tests/test_periodic_void_fixture.py
```

If the declared numerical stack or optional Gmsh dependency is unavailable,
this command fails instead of reporting an ordinary skip. The portable mesh
identity must still match before the numerical quantities are compared. Linux
CI separately runs the two-rank driver with `--invariants-only`; that mode
enforces periodicity, admissibility, Hill--Mandel, energy-component and result
contracts without mislabeling a different platform as the Darwin/arm64 Golden.

The more expensive refinement layer is opt-in:

```bash
AGENTFEM_RUN_RVE_CONVERGENCE=1 python -m pytest -q \
  tests/test_periodic_void_fixture.py -k successive_refinement
```

It compares two against four increments on the fixed coarse mesh, then
compares successive `h/L=0.18` and `0.14` meshes using macroscopic stress,
physical-weighted PEEQ statistics, and improving geometric-volume error. A
passing result means only that these successive changes satisfy the declared
stability thresholds. The certificate does not identify an asymptotic regime,
compute an observed-order/GCI uncertainty estimate, or replace an independent
external benchmark. The Zhang--Feng--Khandelwal comparison below therefore
remains the fail-closed promotion gate.

## Deterministic multi-void regression

The multi-void contract extends the same public workflow to one versioned
four-sphere realization. The sampler, seed, clearance rules and complete void
list form a stable scientific identity; changing any of them creates a new
realization rather than silently reusing prior evidence. This is deliberately
a deterministic regression asset, not a claim that one cell statistically
represents a porous material.

The fixed `h/L=0.16` reference contains 2,368 first-order tetrahedra. Its
Golden quantities are the complete-cell-volume first-Piola tensor, the
physical-weighted PEEQ mean and 95th percentile, and the meshed solid fraction.
PEEQ P99 and maximum and minimum local \(J\) stay diagnostic because they are
more sensitive to local refinement. The independent `h/L=0.20`, `0.16` and
`0.12` certificate passes all invariant gates; from the medium to fine mesh,
the relative changes are approximately 0.195 percent for macroscopic stress,
0.044 percent for mean PEEQ and 0.572 percent for PEEQ P95. This is
successive-refinement stability, not formal asymptotic convergence or GCI.
The comparator removes only the realized mesh size and mesh-dependent equation
identity before hashing the case; it rejects a comparison if the material,
macroscopic path, increments, quadrature, solver, realization or geometry
changes between levels.

The same realization also passes a one-rank/two-rank comparison: the relative
first-Piola norm difference is about \(9.9\times10^{-14}\), all scalar
differences are below \(5.3\times10^{-16}\), and realization, scientific-input,
mesh and constraint identities agree. A midpoint checkpoint/restart performs
101 state and history comparisons with zero observed difference. The
environment-aware launcher avoids mixing OpenMPI and MPICH:

```bash
agentfem mpi-run -n 2 -- python tests/multi_void_rve_golden_driver.py \
  --mesh-size 0.16 --increments 2 \
  --output /tmp/agentfem-multi-void-mpi2.json

python tests/multi_void_rve_golden_driver.py --compare-ranks \
  /tmp/agentfem-multi-void.json \
  /tmp/agentfem-multi-void-mpi2.json \
  --output /tmp/agentfem-multi-void-rank-certificate.json

python tests/multi_void_rve_restart_driver.py \
  /tmp/agentfem-multi-void-restart --mesh-size 0.16 --increments 2 \
  --output /tmp/agentfem-multi-void-restart.json
```

These layers establish deterministic regression, spatial stability,
distributed equivalence and restart equivalence. They do not replace the
independent external promotion gate below or a multi-realization RVE-size and
statistical-convergence study.

## External finite-strain composite benchmark

The Zhang--Feng--Khandelwal (2021) nonlinear periodic-material benchmark is
the promotion target for the regional finite-strain J2 route. Its unit square
contains two stiff circular inclusions of diameter \(0.3\), centred at
\((-0.2,0.2)\) and \((-0.2,-0.2)\), and one circular void of the same diameter
centred at \((0.2,0)\). The matrix parameters are
\(\kappa=17.5\), \(\mu=8\), \(\sigma_y=0.45\), and \(H=0.1\); the inclusions are
100 times stiffer and remain elastic. Table 5 applies macroscopic simple shear
\(\bar F_{12}=0.1\) and reports, in column-major order
\((11,21,12,22)\),

\[
\bar{\mathbf P}=
(0.0128,\ 0.1893,\ 0.1953,\ 0.0598)^T,
\qquad
\bar\psi_e=2.423\times10^{-3}.
\]

In the same component order, the published effective tangent is

\[
\bar{\mathbb A}=
\begin{bmatrix}
26.1954 & -0.6689 & 0.3549 & 8.3450 \\
-0.6689 & 0.1601 & 0.0503 & -0.9698 \\
0.3549 & 0.0503 & 0.2038 & 0.9365 \\
8.3450 & -0.9698 & 0.9365 & 21.0161
\end{bmatrix}.
\]

The AgentFEM fixture retains these values as an external oracle with an
explicit component convention; none of them enters the solver logic.

The material equations have also been compared directly. Both routes use the
multiplicative split \(\mathbf F=\mathbf F_e\mathbf F_p\), a quadratic Hencky
elastic energy, a Kirchhoff-stress \(J_2\) surface and linear isotropic
hardening. The differing yield-function normalizations are algebraically
equivalent. The leading diagnosis is therefore the discretization:
the publication uses a two-dimensional mixed displacement--pressure high-order
element, while the current public AgentFEM route uses displacement-only P1
tetrahedra in a thin three-dimensional extrusion. Exact periodic pairing and a
small Hill--Mandel residual do not remove volumetric locking or geometric
approximation error. This diagnosis remains to be confirmed by a direct
locking-resistant plane-strain A/B comparison.

The current public finite-strain J2 provider is three-dimensional. The
published plane-strain cell is therefore lowered as a thin periodic extrusion
with \(F_{33}=1\), using the ordinary Gmsh import, physical material regions,
`model.step(...)`, accepted quadrature transaction, and periodic-cell history.
This is an **experimental benchmark fixture, not a passed benchmark**. The
paper used a mixed displacement--pressure high-order element, whereas the
current fixture uses the public low-order displacement route. Promotion
requires all of the following:

- mesh and numerical plane-strain-formulation convergence (for the current
  thin extrusion, this includes thickness convergence);
- periodic-cell-size invariance;
- agreement of first-Piola stress and published elastic energy;
- a homogenized algorithmic tangent compared in the published component order;
- serial/MPI and checkpoint/restart equivalence.

`tests/zhang_2021_periodic_composite_fixture.py` defines the geometry, material
translation, oracle and fail-closed assessment. A missing tangent or missing
convergence axis produces `incomplete`, even if one stress vector happens to
be close. The 3 percent oracle may be tightened but cannot be relaxed, and
acceptance requires explicit Boolean evidence for mesh, plane-strain
formulation, cell-size, serial/MPI, and restart equivalence.
`tests/test_zhang_2021_periodic_composite.py` verifies the fixture
semantics without claiming the external result has passed.

A separate [finite-strain J2 self-weight beam gate](finite_strain_j2_external_beam.md)
tests finite rotation and distributed body-force loading outside the periodic
RVE setting. It likewise remains fail-closed until an independently executed,
content-addressed reference curve and all declared convergence gates exist.

## Verification and present boundary

Current tests cover homogeneous finite-strain work equivalence, stress
invariant conventions, sparse spatial output with complete accepted history,
physical quadrature weights on distorted cells, validation failures and
two-rank MPI reductions. The experimental finite-strain J2 route now enters
this contract through public `model.step(...)` for single- or regional-material 3D
affine-periodic cell. Macro averages and Hill--Mandel work are integrated from
the provider-owned accepted quadrature `F`, `P`, `S`, `SENER`, `ELENER`, and
`HARDENER` fields rather than reconstructed from a history-free material
expression. Accepted-state
checkpoint/restart preserves that state across compatible MPI rank-count
changes.

These tests establish the software contract; an RVE used for a material claim
still requires its own mesh, loading-path, convergence, and reference-result
evidence. Multi-material finite-strain J2 dispatch is now part of the
experimental public affine route. The Zhang fixture makes the independent
external comparison executable, but it has not yet passed its convergence and
effective-tangent gates. Stress-state-controlled macro loading and a production
analytical tangent remain separate promotion gates.

## References

1. R. Hill, “Elastic properties of reinforced solids: Some theoretical
   principles,” *Journal of the Mechanics and Physics of Solids* 11 (1963),
   357--372. [doi:10.1016/0022-5096(63)90036-X](https://doi.org/10.1016/0022-5096(63)90036-X).
2. C. Liu and C. Reina, “Discrete averaging relations for micro to macro
   transition,” *Journal of Applied Mechanics* 83 (2016), 081006.
   [doi:10.1115/1.4033552](https://doi.org/10.1115/1.4033552),
   [open manuscript](https://arxiv.org/abs/1509.06621).
3. O. Hering, F. Kolpak and A. E. Tekkaya, “Flow curves up to high strains considering load
   reversal and damage,” *International Journal of Material Forming* 12
   (2019), 339--353.
   [doi:10.1007/s12289-018-01466-z](https://doi.org/10.1007/s12289-018-01466-z).
4. J. Zhang, X. Feng and K. Khandelwal, “A computational framework for
   homogenization and multiscale stability analyses of nonlinear periodic
   materials,” *International Journal for Numerical Methods in Engineering*
   122 (2021), 6527--6575.
   [doi:10.1002/nme.6802](https://doi.org/10.1002/nme.6802),
   [open manuscript](https://arxiv.org/abs/2010.02371).
