# Engineering life, uncertainty and reliability: reusable decision guide

Reviewed 2026-11-07. This is a research/design knowledge record, not a new
runtime capability, certified allowable, or implementation of a design code.
The recommendations below are our engineering synthesis; cited public scopes
are not substitutes for the licensed full standards or an application review.

## 1. Define the question before estimating a life

Life is conditional on material population, loading, environment and a failure
event. Distinguish crack initiation, specified crack size, stiffness/stress
drop, rupture, unacceptable deformation and loss of function. A specimen's
fracture count, a component's inspection interval and a design-code usage
limit are not interchangeable outputs. Specify cycles versus reversals and
calendar time versus accumulated dwell time.

Three complementary routes should remain visible:

- S–N / strain–life: an empirically defined fatigue outcome under its tested
  loading conditions; include mean stress/strain and environmental scope.
- Damage tolerance: evolve an existing or assumed defect to a defined limit;
  initial flaw distribution, crack-growth law, inspection probability of
  detection and residual stresses become important. E647 addresses crack
  growth measurements, not total life from an initially uncracked component.
- Creep / creep–fatigue: retain physical time, temperature, relaxation,
  hold direction, loading order and interaction assumptions. Initiation and
  propagation predictions may be combined only with compatible definitions,
  without counting the same damage twice.

The [ASTM E647 public scope](https://store.astm.org/standards/e647) identifies
loading ratio, temperature and environment as relevant to crack growth.
[ASTM E2714](https://store.astm.org/standards/e2714) concerns isothermal
specimen creep–fatigue deformation/crack formation, recommends complementary
fatigue and creep data, and excludes sequential block-cycle testing from its
stated scope. It is not component qualification.

## 2. Keep uncertainty sources separate

| Source | Examples | Appropriate treatment |
| --- | --- | --- |
| Population variability | heats, microstructure, surface condition, defects | conditional distributions and batch structure supported by observations |
| Future operation | starts/stops, temperature, load spectrum, dwell | measured stochastic processes or explicit operating scenarios |
| Measurement | force, temperature, strain, time and crack-size calibration | measurement uncertainty and correlations; not duplicate material scatter |
| Parameter knowledge | finite experimental sample, correlated fit parameters | likelihood/profile intervals, grouped bootstrap or Bayesian posterior |
| Model form / transfer | wrong damage mechanism, missing oxidation, specimen-to-component transfer | independent residual checks, competing models, explicit discrepancy or bounded scenarios |
| Numerical approximation | mesh/time error, solver tolerances, history sampling, surrogate error | convergence and numerical-error assessment, not unexplained random noise |

Aleatory/epistemic is a useful distinction, not an immutable label: the same
heat variation is population variability for a fleet but uncertainty about a
particular component before its heat is known. Unknown distributions should
remain bounded/scenario uncertainty until evidence supports probabilities.
Do not independently perturb correlated constitutive coefficients or add a
universal ±10% to everything. More simulations reduce sampling error, not
missing physics or experimental information. NASA separates implementation/
solution verification from empirical validation and use-specific acceptance:
[NASA-STD-7009B](https://standards.nasa.gov/standard/NASA/NASA-STD-7009).

For softening damage, mesh objectivity is a formulation issue as well as a
refinement issue. A local strain-to-failure parameter can produce
element-size-dependent dissipation; characteristic-length/fracture-energy
regularization is one established approach, not proof that all mesh dependence
vanishes. [Abaqus damage evolution documentation](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-damageevolductile.htm)
explains this distinction. A statistical scatter band must not conceal
unregularized localization or a singular peak stress used as a life input.

Budget numerical accuracy against the life quantity of interest. For the
illustrative relation N=C*sigma**(-m), d(log N)/d(log sigma)=-m. With m=10,
a 5% stress increase reduces predicted life by about 38.6%; this is algebra,
not measured P92 scatter. A small force or energy residual alone therefore
does not demonstrate a comparably small life error.

## 3. A defensible experimental data contract

Retain specimen and heat IDs, product/heat treatment/aging/weld region,
geometry and surface, environment, units, waveform, rate, R ratio, temperature,
failure definition, measurement uncertainty and original source/version.
For each observation retain its endpoint and status: failure, right-censored
runout/interruption, interval-censored detection, or a separately explained
invalid test. An interrupted test does not become a failure at its stop time.
Nor should a solver failure become experimental censoring or a physical failure.

For independent observations with non-informative censoring, the likelihood
uses density f(t) for exact failures, survival R(c) for right-censoring and
F(b)-F(a) for interval-censoring. A stop caused by visible damage or another
informative selection process needs explicit modeling; ordinary runout
likelihood assumptions may not hold. See [NIST censoring](https://www.itl.nist.gov/div898/handbook/apr/section1/apr131.htm)
and [censored likelihood](https://www.itl.nist.gov/div898/handbook/apr/section4/apr412.htm).

Repeated cycles of one specimen are correlated observations, not thousands
of independent life tests. Split and resample at the specimen/heat level
appropriate to the prediction target. Same-heat data are especially useful;
cross-heat data can still be used with explicit population/transfer modeling.
A hierarchical model is an option, not a cure for having too few heats.

## 4. Four outputs that must not be conflated

Let F_N(n | x) be the conditional life CDF and R_N=1-F_N.

| Quantity | Question answered |
| --- | --- |
| Median or fitted central curve | What is the central response under these conditions? |
| Confidence/credible interval for a parameter or quantile | How uncertain is our estimate? Bayesian credibility is conditional on priors and likelihood; frequentist confidence describes repeated-procedure coverage. |
| Predictive interval | What range should contain a future specimen's outcome, including relevant variability and estimation uncertainty? |
| One-sided tolerance bound | With stated confidence, what lower bound is exceeded by at least a stated fraction of the defined population? |

For B10, F_N(B10)=0.10: 90% survival at that life under the model. B10 is not
itself a 90%-confidence bound. A lower confidence bound for B10 is a separate
statistical calculation. A design allowable may use code adjustments and is
not automatically either of these. [NIST tolerance intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc263.htm)
distinguishes population coverage from parameter confidence.

Synthetic illustration, not P92 data: if log10(N) is normal with mean 4 and
known standard deviation 0.3, median life is 10,000 cycles but B10 is about
4,126 cycles. Estimating those parameters from a small sample adds uncertainty;
the latter number is not already a confidence-qualified design life.
For a lognormal distribution, mean and median differ; Weibull scale is its
63.2% failure quantile, not its median. Distribution choice must be checked,
not selected because a package defaults to it. See [NIST lognormal](https://itl.nist.gov/div898/handbook/apr/section1/apr164.htm).

## 5. Modeling and validation that match the decision

1. Start with log-life residual plots, conditional trends and censored-data
   exploration. Compare plausible lognormal/Weibull or other models rather
   than insisting on one universal distribution. Few failures give weak tail
   information even if the optimizer converges.
2. Treat fit-parameter uncertainty separately from observation variability.
   Bootstrap complete specimens/heats where appropriate. Bayesian models need
   physical priors, prior/posterior predictive checks and sensitivity to priors.
   Parameters and model discrepancy can compensate for each other: a narrow
   posterior alone does not establish the correct mechanism. This issue is
   discussed in [Bayesian calibration identifiability](https://doi.org/10.1137/19M1283707).
3. Hold out specimens, heats or operating conditions according to the intended
   transfer. Report log-life bias, factor-of-error distributions, unsafe
   overprediction, predictive coverage and interval width with sample counts.
   Do not call a same-source interpolation check independent-batch validation.
4. Predeclare application-specific acceptance. A factor-of-two band is a
   descriptive plot unless justified for the decision; passing it does not
   prove a prescribed reliability. Extremely wide intervals can cover every
   observation while being useless for engineering decisions.
5. Separate extrapolation (temperature, time, stress, mechanism) from
   interpolation. Long-term accelerated-test transfer requires mechanism and
   applicability evidence, not only a good Larson–Miller or Arrhenius fit.

Useful statistical planning reference: [ISO 12107:2012](https://www.iso.org/standard/50242.html),
confirmed current in 2026. [ASTM E739-23](https://store.astm.org/standards/e739)
was withdrawn in 2024 without replacement according to its official listing;
retain it only as historical guidance, not a current compliance claim.

## 6. Numerical damage is not failure probability

For an explicitly chosen assessment, D_f=sum(n_i/N_i) and
D_c=integral(dt/t_r(sigma(t),T(t))) are usage/damage measures. D=0.2 does not
mean a 20% probability of failure. A linear Miner sum omits sequence effects;
rainflow counts cycles but does not itself supply a damage law, multiaxial
criterion or dwell duration. Keep stress/strain/temperature at the same
material point and retain the accepted physical clock, not sparse images.
[ASTM E1049](https://store.astm.org/standards/e1049) covers counting methods.
The [COMSOL cumulative-damage theory](https://doc.comsol.com/6.3/doc/com.comsol.help.fatigue/fatigue_ug_sme.4.25.html)
also explicitly notes the Miner rule's lack of sequence effects. Reducing a
history to a cycle histogram can discard information needed by a
sequence-sensitive or creep–fatigue model; preserve the original history.

For multiaxial/nonproportional loading, choose and validate the stress/strain
reduction or critical-plane criterion. A uniaxial strain-life curve does not
automatically accept accumulated plastic strain, a positive tensor norm or
an unreviewed equivalent-stress history. Keep total, thermal and mechanical
strain distinct, and maintain paired stress/strain histories when applying a
mean-stress correction. An accurate statistical fit cannot repair a wrong
mechanical driving variable.

The [ASME Division 5 explanatory workshop](https://resources.asme.org/hubfs/CS-Codes-Standards/CS-Section%20III%20Div%205%20HTR%20V%20Workshop%20Files/HTR%20Presentations/HTR%20Day%202/6.%20Div%205-design-methods-metallic-Messner.pdf)
describes creep and fatigue fractions checked against an interaction envelope.
It does not establish a universal D_c+D_f=1 boundary or qualify P92 for a
particular code edition. Numerical damage variables in a constitutive model,
postprocessed usage fractions and statistical failure probability must remain
separate even if all are dimensionless.

Design-curve adjustment also differs from probabilistic tail estimation.
[NRC NUREG/CR-6909 Revision 1](https://www.nrc.gov/reading-rm/doc-collections/nuregs/contract/cr6909/r1/index)
reviews environmental effects and historical fatigue adjustments. Do not copy
a historic factor of 20 on life into every material, code edition or P92 study,
or apply both a lowered design curve and the same scatter reduction again.

## 7. Propagation, rare events and updating

Propagate justified joint inputs through the existing simulation and life
assessment; preserve correlations and separate conditional variability from
epistemic alternatives. [JCGM 101](https://www.bipm.org/documents/20126/2071204/JCGM_101_2008_E.pdf/325dcaad-c15a-407c-1105-8b7f322d651c)
provides a distribution-propagation basis for measurement models, not a
component certification rule. Sensitivity screening can identify whether more
experiments, load characterization or mesh refinement will be most valuable.

Monte Carlo is a reference, not always efficient for rare failure. Define a
physical limit state g(X), report P(g<=0) for an explicit mission, and assess
sampling error. FORM is a local approximation; multiple/non-smooth failure
regions require checking. Importance sampling needs likelihood weights and
failure-region coverage; surrogate classification errors near the boundary
must be checked against FEM. [OpenTURNS](https://openturns.github.io/openturns/latest/theory/reliability_sensitivity/importance_simulation.html)
offers existing algorithms rather than requiring a new AgentFEM UQ solver.

Synthetic iid Bernoulli example: zero failures in n independent trials gives
the exact one-sided 95% upper limit 1-0.05**(1/n). To place that limit at or
below 1e-4 requires at least 29,956 trials. Zero failures in 100 or 1,000
simulations is not evidence of zero risk. This argument presumes the sampled
model/input distribution is itself appropriate; it does not include model error.

Component reliability also requires a system event, not just local life maps.
Do not multiply the survival probabilities of every FE integration point as
if they were independent physical specimens: spatial correlation, common
loads/material parameters, interacting mechanisms and structural redundancy
matter. Refining a numerical mesh must not create more independent physical
failure sites. See [NASA system reliability discussion](https://ntrs.nasa.gov/api/citations/19990049421/downloads/19990049421.pdf).

For a fixed lifetime model, survival to age t0 changes remaining-life survival
to R(t0+u)/R(t0), where R(t0)>0. Real updated missions additionally require
accepted operating history, damage state and observations. A non-detection
depends on inspection probability of detection and is not proof of no crack.
Updating should retain the prior model/data version, avoid reusing evidence
twice, and distinguish Bayesian parameter updating from retraining a surrogate.
Remaining life is not generally unconditional mean life minus elapsed time.
NASA's [NDE methodology record](https://ntrs.nasa.gov/citations/20230008599)
illustrates the separate concepts of detection probability and confidence;
its fracture-critical inspection requirements are not a universal life target
for power-plant components. Inspection interval, probability of failure and
consequence-informed risk are related decisions, not identical quantities.

## 8. AgentFEM: minimal, reusable next implementation

Keep the deterministic solver and accepted-state lifecycle intact. Build
optional assessment/campaign consumers, not a second Model or global solver.
Reuse co-located histories, unit/source contracts and fresh per-case states.

Priority order:

1. Specimen-level life observations with censoring, failure definition, heat
   identity and units; retain raw replicates separately from interpolated curves.
2. Censored-data descriptive checks and explicitly scoped fits; report population,
   central/quantile estimates and interval type. Existing TabulatedRuptureCurve
   remains a deterministic reviewed curve, not a statistical fit to replicates.
3. Reuse campaigns for a small uncertainty study with fixed seeds, parameter
   correlations, sampling convergence and a defined holdout; no giant default
   Monte Carlo run. Only then consider hierarchy, rare events or active learning.

SciPy's [CensoredData](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.CensoredData.html)
works with continuous distributions' `.fit`; the documented
[fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html)
does not guarantee a global optimum. Its
[ecdf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ecdf.html)
supports right-censored Kaplan–Meier, not general interval-censored estimation.
These are reusable tools, not capabilities already exposed by AgentFEM.

Local smoke verification (SciPy in fenicsx-env): synthetic failures at 1,2,3
and right-censoring at 4,4 give exponential MLE scale 14/3; `.fit(floc=0)`
returned 4.666699 versus 4.666667 analytically, and Kaplan–Meier survival at
3 was 0.4. The B10 and zero-failure calculations above were also evaluated.
This checks statistical plumbing, not P92 applicability or a new core feature.

Reproduce the synthetic check without downloading data or running FEM:

```python
from math import ceil, log
from scipy.stats import CensoredData, ecdf, expon, norm

data = CensoredData(uncensored=[1., 2., 3.], right=[4., 4.])
_, scale = expon.fit(data, floc=0)
assert abs(scale - 14 / 3) < 1e-3
assert abs(float(ecdf(data).sf.evaluate(3.)) - 0.4) < 1e-12
assert abs(10000 * 10**(norm.ppf(.1) * .3) - 4126.0504164) < 1e-6
assert ceil(log(.05) / log(1 - 1e-4)) == 29956
```

Minimum acceptance for a future adapter: analytic/censored synthetic checks,
unit invariance, explicit unsupported censoring, repeatable sampling,
specimen-level split integrity, grouped resampling, no lost failed runs and
clear interval semantics. Scientific acceptance separately requires appropriate
data, meaningful coverage/precision and the selected engineering decision.

For the present P92 roadmap, stop asking only for one exact predicted life.
Choose either a declared research population with visible transfer uncertainty,
or an application-qualified design assessment. Missing hold stress histories
and unqualified rupture ranges cannot be repaired by adding probability bands.
The coherent DB cohort is useful, but sparse repeat summaries do not establish
its low-tail distribution. Numerical accuracy and experimental agreement need
separate budgets; scatter is not permission to conceal a deterministic bias.
