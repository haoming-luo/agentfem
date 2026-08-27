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

This normalized Lode convention gives (+1) for axisymmetric tension, (0)
for pure shear and (-1) for axisymmetric compression.  When (q) vanishes,
(eta) and (ar\theta) are undefined.  NPZ output uses `NaN`; structured
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

For an integration-point scalar:

```python
summary = quadrature_state.equivalent_plastic_strain.weighted_statistics(
    quantiles=(0.5, 0.95, 0.99),
    thresholds=(0.02,),
)
```

## Verification and present boundary

Current tests cover homogeneous finite-strain work equivalence, stress
invariant conventions, sparse spatial output with complete accepted history,
physical quadrature weights on distorted cells, validation failures and
two-rank MPI reductions.  These tests establish the software contract; an RVE
used for a material claim still requires its own mesh, loading-path and
reference-result evidence.

Finite-strain inelastic state evolution and stress-state-controlled macro
loading are separate promotion gates.  The presence of this post-processing
contract must not be read as a claim that finite-strain J2 plasticity is already
validated.

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
