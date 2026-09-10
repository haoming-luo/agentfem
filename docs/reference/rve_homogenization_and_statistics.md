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

The fixed `h/L=0.16` mesh also has a separate 2/4/8-increment path
certificate. All three paths pass the invariant gates. From four to eight
increments, relative changes are approximately 0.00121 percent for the
macroscopic first-Piola tensor, 0.000993 percent for mean PEEQ and 0.0751
percent for PEEQ P95. The comparison removes only the increment count before
hashing the case and rejects any change in mesh, material, loading, solver,
quadrature, realization or constraint. It establishes final-state stability
for this monotonic path on this fixed mesh, not a general temporal error bound.

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

AGENTFEM_RUN_MULTI_VOID_RVE_LOAD_PATH=1 python -m pytest -q \
  tests/test_multi_void_rve_golden.py -k real_multi_void_load_path_certificate
```

These layers establish deterministic regression, spatial stability,
load-increment stability, distributed equivalence and restart equivalence.
They do not replace the
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
equivalent. Discretization must nevertheless be compared precisely. Zhang et
al. use a two-dimensional Q2 nine-node quadrilateral displacement field and a
three-mode discontinuous pressure space, which a direct AgentFEM route would
represent with DPC1 and which is commonly abbreviated 9/3. The pressure
interpolation spans three modes over a quadrilateral; it is not one constant
pressure value per tetrahedron.

The public mixed-field and finite-strain J2 lowering now support this
two-dimensional plane-strain quadrilateral Q2/DPC1 interpolation and embed the
material-point kinematics with \(F_{33}=1\). A homogeneous affine patch confirms
the three-degree-of-freedom pressure space and constant-pressure coupled solve;
it does not independently exercise all nonconstant DPC modes. Periodic-cell
history accepts the measured
2-by-2 macro gradient only on this provider-owned path, embeds it in the same
3-by-3 convention as the accepted quadrature tensors, and records
\(F_{33}=1\) for stress and Hill--Mandel evidence. The exact
two-inclusion/one-void geometry now has a direct Q2/DPC1 diagnostic driver. It
can report the Table 5 stress and an explicitly reconstructed primal Hencky
elastic-energy channel beside the condensed mixed channel. It also obtains the
current-state homogenized algorithmic tangent by condensing the converged full
Jacobian through the exact affine macro-gradient lift,

\[
 V\bar{\mathbb A}
 =B^T K B-B^T K T\left(T^TKT\right)^{-1}T^TKB.
\]

Here \(u=Tq+B\bar F\), and the provider must declare that no additional
explicit macro-gradient dependence is hidden in the residual. Each tangent
column records its linear-solver status and reduced-equilibrium sensitivity.
The recovery uses the final accepted increment only while a state-owned token
proves that the live quadrature tangent came from that increment; it otherwise
fails closed, including after a checkpoint reconstruction that did not persist
the macro tangent. It does not rerun or finite-difference the load path.
Homogeneous Q2/DPC1 and three-dimensional P2/DG0 Hencky-elastic patches verify
the component order and Schur condensation. This is still not a promoted Zhang
benchmark: Table 5 tangent agreement, load-path, mesh and formulation
convergence, replicated cells, restart/MPI equivalence, and content-bound
evidence remain open.

AgentFEM now has two deliberately distinct thin-3D diagnostic lowerings of the
published plane-strain cell with \(F_{33}=1\):

1. the older displacement-only P1 tetrahedral route, retained as a locking
   A/B diagnosis;
2. a real mixed route with P2 tetrahedral displacement and DG0 mean Kirchhoff
   stress, ordinary Gmsh physical regions, exact periodic kinematics,
   `model.step(...)`, accepted quadrature transactions, and periodic-cell
   histories.

The second route removes the false assumption that a high-order source mesh is
itself a hybrid formulation, and it provides an independent volumetric unknown
through a monolithic four-block Newton system. It is an experimental mixed
three-dimensional RVE formulation intended to mitigate volumetric locking; no
locking-convergence or inf-sup claim follows from the present tests. It is
**not** an exact
reproduction of the paper's 2D quadrilateral Q2/DPC1 9/3
element, so agreement from a coarse thin extrusion cannot by itself promote
the benchmark. Exact periodic pairing and a small Hill--Mandel residual likewise
do not establish spatial, path, or formulation convergence.

The fixture therefore remains an **experimental benchmark fixture, not a
passed benchmark**. Promotion requires all of the following:

- load-increment/path convergence with the same prescribed macroscopic
  history;
- mesh and numerical plane-strain-formulation convergence, including thickness
  convergence for a thin-3D route and converged execution of the complex
  fixture through the 2D Q2/DPC1 three-pressure-mode implementation;
- componentwise and vector-norm agreement of first-Piola stress, plus agreement
  of the published primal Hencky elastic energy;
- a homogenized current-state algorithmic tangent obtained from the linearized
  corrector with the pre-increment committed state fixed and the local return
  mapping consistently linearized, compared in the published component order;
- 1x1, 1x2, 2x1, and 2x2 periodic-cell replication invariance;
- serial/MPI and checkpoint/restart equivalence.

`tests/zhang_2021_periodic_composite_fixture.py` defines the geometry, material
translation, oracle and fail-closed comparison-completeness assessment. A
missing tangent or missing convergence axis produces `incomplete`, even if one
stress vector happens to be close. The AgentFEM-owned 3 percent relative and
componentwise absolute-plus-relative contracts may be tightened but cannot be
relaxed. At present, however, the assessor accepts caller-supplied Boolean
statements for load-increment/path, mesh, plane-strain formulation, cell-size,
serial/MPI, and restart equivalence. It is therefore a completeness schema, not
yet a content-bound scientific promotion gate. Promotion requires those flags
to be derived from identified evidence artifacts rather than asserted by a
caller. `tests/test_zhang_2021_periodic_composite.py` verifies the fixture
semantics without claiming the external result has passed.

One unarchived current-stack coarse diagnostic makes that boundary concrete.
On a 502-tetrahedron, thickness-0.10 P2/DG0 extrusion with 20 load increments,
the first-Piola vector has 1.4324 percent relative L2 error and passes that
global norm contract, but \(P_{11}\) and \(P_{22}\) fail the componentwise
absolute-plus-relative contract. Its condensed mixed `ELENER` is 0.002627074,
but Table 5 reports the primal Hencky elastic energy. Those different channels
must not be assigned a relative error; the thin-3D physical-energy comparison
therefore remains incomplete. The maximum Hill--Mandel relative residual is
\(1.054\times10^{-8}\) and the periodic mismatch is zero. These observations
have not yet been committed as a content-addressed result with runtime and input
identity. They are neither a Golden result nor a substitute for the required
path, mesh, formulation, tangent, replication, MPI, and restart evidence.

A separate [finite-strain J2 self-weight beam gate](finite_strain_j2_external_beam.md)
tests finite rotation and distributed body-force loading through the ordinary
strong-boundary provider outside the periodic RVE setting. It likewise remains
fail-closed until an independently executed,
content-addressed reference curve and all declared convergence gates exist.

## Verification and present boundary

Current tests cover homogeneous finite-strain work equivalence, stress
invariant conventions, sparse spatial output with complete accepted history,
physical quadrature weights on distorted cells, validation failures and
two-rank MPI reductions. The experimental displacement-only finite-strain J2
route enters this contract through public `model.step(...)` for single- or
regional-material 3D affine-periodic cells. Macro averages and Hill--Mandel
work are integrated from provider-owned accepted quadrature `F`, `P`, `S`,
`SENER`, `ELENER`, and `HARDENER` fields rather than reconstructed from a
history-free material expression. Its accepted-state checkpoint/restart
preserves that state across compatible MPI rank-count changes.

The mixed tetrahedral P2/DG0 route reuses the same constitutive state
transaction and adds `MEAN_KIRCHHOFF_STRESS`, an assembled discrete
pressure-block residual, and a separate `MIXED_POTENTIAL` quadrature diagnostic.
Its condensed mixed `ELENER` uses deviatoric Hencky storage plus
\(p^2/(2\kappa)\).
That term is pointwise equal to the primal volumetric storage only where
\(p=\kappa\ln J\) holds locally; integrating it does not make it the primal
observable. The mixed-energy diagnostic reconstructs the primal channel from
aligned accepted `F`, pressure, inverse-bulk-modulus and condensed `ELENER`
fields. With \(r_p=\ln J-p/\kappa\), it retains the signed
primal-minus-condensed gap, the signed pressure-orthogonality contribution
\(\overline{p r_p}\), the nonnegative constraint-defect contribution
\(\overline{\kappa r_p^2/2}\), and their decomposition residual. Table 5 can
consume only that explicit primal result, never condensed `ELENER`. The saddle
potential is not treated as pointwise stored energy. Portable checkpoints split
live and accepted mixed solutions into standalone displacement and
mean-Kirchhoff-stress fields before serialization, then let the state owner
reassemble them after identity validation. Fresh-Step checkpoint/continue equivalence is
verified for both serial mixed routes: 3D tetrahedral P2/DG0 and 2D plane-strain
quadrilateral Q2/DPC1. Independently, the underlying generic DPC cell-moment
identity has two-to-one and one-to-two MPI-rank acceptance coverage. This does
not promote the mixed equilibrium provider itself: distributed mixed MPC, an
MPI mixed-J2 solve, and cross-rank-count restart of a solved mixed J2 step remain
unverified.

These tests establish the software contract; an RVE used for a material claim
still requires its own mesh, loading-path, convergence, and reference-result
evidence. Multi-material finite-strain J2 dispatch is now part of the
experimental public affine routes. The Zhang fixture makes the independent
external comparison executable, but it has not yet passed its loading-path,
formulation, replication, effective-tangent, or distributed-execution gates.
Stress-state-controlled macro loading, full Zhang evidence through the direct
2D Q2/DPC1 route, and a production analytical deviatoric tangent remain
separate promotion gates.

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
4. G. Zhang, N. Feng and K. Khandelwal, “A computational framework for
   homogenization and multiscale stability analyses of nonlinear periodic
   materials,” *International Journal for Numerical Methods in Engineering*
   122 (2021), 6527--6575.
   [doi:10.1002/nme.6802](https://doi.org/10.1002/nme.6802),
   [open manuscript](https://arxiv.org/abs/2010.02371).
5. FEniCS Project, “Basix `create_element` and discontinuous DPC variant,”
   [official API reference](https://docs.fenicsproject.org/basix/main/python/_autosummary/basix.finite_element.html).
