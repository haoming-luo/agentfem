# Creep, fatigue and interaction verification

## Evidence levels

Keep four decisions separate: (1) constitutive equations and their discrete
updates, (2) finite-element equilibrium and refinement, (3) material prediction
against independent tests, and (4) acceptance under an explicitly selected
design procedure. Passing the first two does not establish the last two.
The numerical fixtures below use synthetic coefficients, not a qualified
material database. No standard text, licensed design curve or private data is
embedded in AgentFEM.

## Reference map

These references were checked on 2026-10-29. ASTM entries identify test methods
and editions from the official public catalogue; their full normative text
has not been reviewed here. This is not a declaration of ASTM compliance.

| Question | Reference | Use and limit |
| --- | --- | --- |
| Creep and rupture measurements | [ASTM E139-24](https://store.astm.org/standards/e139/e139m.htm) | Organize constant-load creep/rupture observations; deformation and rupture are distinct outputs |
| Strain-controlled cyclic tests | [ASTM E606/E606M-21](https://store.astm.org/e0606_e0606m-21.html) | Select compatible fatigue data; retain amplitude/range, cycle/reversal, temperature and strain-rate conventions |
| Cycles with holds | [ASTM E2714-26](https://store.astm.org/standards/e2714) | Identify the creep-fatigue test family; does not supply a universal life law or component acceptance |
| Stress integration | [MOOSE power-law creep](https://mooseframework.inl.gov/source/materials/PowerLawCreepStressUpdate.html) | Independent algorithm reference; not an executed cross-solver comparison |
| Interaction assessment limitations | [JRC92093](https://publications.jrc.ec.europa.eu/repository/handle/JRC92093), DOI 10.1115/ICONE22-30640 | Study of 316/P91 assessment methods; relaxation and cyclic response matter, and design margins are not best-estimate life predictions |
| Hold-time and temperature sensitivity | [INL, Creep-fatigue behavior of Alloy 617 (2016)](https://inl.elsevierpure.com/en/publications/creep-fatigue-behavior-of-alloy-617/) | Candidate experimental comparison: reported hold-time effects differ between 850 and 950 degrees C; no Alloy 617 fit is claimed here |
| Loading-order sensitivity | [INL experimental verification of Division 5 rules](https://art.inl.gov/report/experimental-and-analytical-verification-of-asme-section-iiii-division-5-creep-fatigue-design-rules/) | Candidate variable-amplitude/sequence validation, beyond an order-independent damage sum |
| Traceable material measurements | [NIMS creep data sheets](https://cds.nims.go.jp/en/) | Candidate data source; check sheet revision, usage terms, material heat and condition before extracting or fitting data |
| Independent cycle-counting output | [MathWorks rainflow worked examples](https://www.mathworks.com/help/signal/ref/rainflow.html) | Compare numerical count/range/mean triples for nested reversals and residual half-cycles; not an executed MATLAB comparison or full ASTM certification |
| Nonuniform secondary creep | [NAFEMS R0027 Test 7 in the Abaqus benchmark manual](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEBMKRefMap/simabmk-c-creeptest7.htm) | Reuse the existing `creep_thick_cylinder_benchmark`; do not replace it with another project-level solver |
| P91 mechanism selection | [JRC94508](https://publications.jrc.ec.europa.eu/repository/handle/JRC94508), DOI 10.2790/49517 | Cyclic softening changes subsequent creep response; the public report summary also identifies limited long-hold/low-strain evidence. Do not present a hardening-only model or linear interaction as universal |
| Elastic follow-up and separated test families | [ANL-ART-168 (2019)](https://publications.anl.gov/anlpubs/2020/02/158589.pdf) | Section 2.1 provides a two-bar relaxation reference; Appendix A separates ordinary controlled strain from SMT effective strain and tensile/compressive holds. Do not mix those inputs as interchangeable fatigue samples |
| Traceable Grade 91 material scope | [NIMS Grade 91 sheet article](https://mdr.nims.go.jp/datasets/f9cdc009-32b5-4e6a-ba51-4b7338d799c5), DOI 10.1080/27660400.2025.2588872 | Public landing page identifies tube/plate/pipe grades, temperature/stress scope and CC BY 4.0. Candidate for matched-condition creep validation, not a fit already admitted to core |

Do not conflate the separately published
[INL/EXT-21-64314 Alloy 709 report](https://inldigitallibrary.inl.gov/sites/sti/sti/Sort_52942.pdf)
with Alloy 617 data. A similar test protocol is not the same material.

## Executable analytical references

The next experimental source gate is recorded in the
[Alloy 617 audit](internal/alloy617_validation_plan_20261101.md). Public
parameters are not interchangeable with another constitutive model. Local
assessment inputs can now use [co-located accepted material histories](material_point_histories.md);
this closes a data-lifecycle gap, not the experimental-validation gate.

The [executed P92 relaxation comparison](internal/p92_relaxation_validation_20261101.md)
uses public digital BAM records: a separate RF calibration and SLR conditional
holdout, source hashes, physical unit conversions and ordinary FEM time
refinement. It reports experimental errors without inventing an acceptance
tolerance or qualifying a full cyclic material/lifetime model.

Run the focused checks in the scientific environment:

```bash
python -m pytest -q tests/test_creep_fatigue_references.py tests/test_creep_damage_history.py tests/test_rainflow_references.py tests/test_assessments.py tests/test_strain_life.py
```

### Constant-strain Norton relaxation

A homogeneous 3D unit cube has a fixed axial strain and free lateral
contraction. The ordinary `model.step` solves a uniaxial stress state. For
`creep_rate = A (sigma / s_ref)^n`, set `k = E A / s_ref^n`. Independent
continuous solutions are

```text
n = 1: sigma(t) = sigma_0 exp(-k t)
n > 1: sigma(t) = [sigma_0^(1-n) + (n-1) k t]^(1/(1-n))
```

An instantaneous elastic preload defines sigma_0 at t=0; this fixture checks
the subsequent relaxation, not energy balance across that preload. E=200000
MPa, nu=0.3, A=1e-4 /s, s_ref=100 MPa, sigma_0=200 MPa and duration=2 s.
Reaction stress, lateral traction-free stress and accumulated creep strain
are checked. Maximum errors over the full sampled trajectory are:

| n | Exact final stress (MPa) | 16 increments | 32 increments | 64 increments |
| --- | ---: | ---: | ---: | ---: |
| 1 | 134.064009 | 0.660977 | 0.332805 | 0.166989 |
| 3 | 97.590007 | 3.360815 | 1.754748 | 0.897590 |

The approximately first-order decay checks backward-Euler accuracy against a
continuous solution, rather than comparing two copies of the same recurrence.
This homogeneous patch is not a nonuniform spatial-convergence claim.

### Elastic follow-up through ordinary regional equilibrium

`tests/test_creep_elastic_followup.py` is a shear analogue of the elastic/creep
series-bar mechanism discussed in ANL-ART-168 Section 2.1, not a reproduction
of its axial specimen or Alloy 617 tests. Two equal-area, unit-length slabs
carry equal shear traction. The left slab creeps; the right is elastic.
The ordinary regional power-law Step uses coefficient zero in the elastic
region, not a separate benchmark solver or an empirical follow-up correction.

For fixed total displacement, compatibility gives
`(1/G1+1/G2)*d(tau)/dt + d(gamma_creep)/dt = 0`. Define
`f=1+G1/G2` and use associative Mises shear flow to obtain

```text
d(tau)/dt = -k*tau^n
k = (G1/f)*sqrt(3)*A*(sqrt(3)/s_ref)^n
```

The exponential/power-law continuous solutions above apply to tau. G1=100000
MPa, tau_0=100 MPa, A=1e-4 /s, s_ref=100 MPa, duration=2 s. Four Q1 cells
resolve the piecewise-affine displacement field; antiplane support supplies
the required transverse reactions. Maximum stress-history errors are:

| f / n | Exact final tau (MPa) | 16 increments error (MPa) | 32 | 64 |
| --- | ---: | ---: | ---: | ---: |
| 2 / 1 | 74.08182 | 0.206073 | 0.103603 | 0.0519448 |
| 2 / 3 | 59.76143 | 0.981558 | 0.503437 | 0.255129 |
| 5 / 1 | 88.69204 | 0.0397219 | 0.0199082 | 0.0099660 |
| 5 / 3 | 76.24929 | 0.395441 | 0.200246 | 0.100769 |

Tests check interface traction balance, zero creep in the elastic region,
compatibility of the creeping strain and first-order time convergence. Elastic
follow-up slows relaxation compared with complete local strain restraint at
the same initial stress. This supports an existing regional **pure-creep**
route, not regional admission of the new joint creep/plastic composition.
Initial elastic preload work and experimental component qualification are
outside this benchmark.

### Time-fraction integration through a relaxing history

Use a deliberately synthetic rupture relation
`t_r(sigma) = 1000 (sigma_0/sigma)^n` seconds. Its independent continuous damage
integral is `Dc = (sigma_0-sigma_end)/(1000 k sigma_0^n)`.
The existing result consumer takes the maximum stress per declared dwell
block. Dividing the relaxation into accepted-time intervals gives:

| n | Exact Dc | Absolute error, 16 increments | 32 increments | 64 increments |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.001648400 | 2.42724e-5 | 1.21809e-5 | 6.10168e-6 |
| 3 | 0.000640062 | 9.43434e-5 | 4.73017e-5 | 2.36819e-5 |

This upper estimate converges for this monotone rupture relation and resolved
monotone stress history. A single maximum over a long hold is **not** the
time-resolved integral; a sample mean is not a time-weighted integral. Neither
is universally conservative for an arbitrary supplied rupture function or an
under-resolved history. This test verifies extraction and quadrature behavior,
not physical creep-fatigue interaction.

The time-resolved consumer `assessments.creep_damage_from_result` now retains
the actual sampled relaxation and integrates the reciprocal rupture time,
instead of assigning each hold a single maximum. Against the same continuous
reference, its absolute damage errors are:

| n | 16 increments | 32 increments | 64 increments |
| --- | ---: | ---: | ---: |
| 1 | 3.87400e-6 | 1.93036e-6 | 9.63480e-7 |
| 3 | 3.96548e-5 | 1.98208e-5 | 9.90696e-6 |

These errors remain first-order because the input FE stress is backward-Euler
accurate. Separate tests on exact sampled exponential stress and on a
synthetic temperature-dependent rate exhibit second-order trapezoidal
convergence. Do not infer universal conservatism or material accuracy from
either test. Explicit hour/second invariance, mismatched grids, domain failures,
nonoverlapping dwells, finite strict JSON and late-failure nonpublication are
also tested. Original sampling gaps remain in the assessment evidence.
The optional source-gap limit has a recorded 1e-12 relative roundoff allowance
so ordinary equal time increments are not rejected by subtraction noise. There
is no absolute or time-origin-scaled tolerance; large time offsets and very
small intervals cannot conceal a genuinely undersampled history.

### Independent rainflow counting

`tests/test_rainflow_references.py` checks the nine published count/range/mean
triples in the MathWorks algorithm example, including four full cycles and
five residual half-cycles. Duplicating dwell samples or inserting monotone
samples does not change the cycles. A scale-invariance regression fixes an
old numerical weakness: multiplying neighboring slopes could underflow and
erase a reversal. Direct direction comparisons preserve small-range cycles;
non-real histories and unrepresentable ranges are rejected. These checks
validate counting, not a fatigue-life calibration.

### Nonuniform spatial verification in a limiting case

`tests/test_creep_manufactured_field.py` checks the joint material with plastic
yield deliberately unreachable and linear creep active. On a unit cuboid,
choose `u_z=s*x*(1-x)*t`, `u_x=u_y=0`, `s=0.001`. With shear modulus G,
`lambda=3GA/s_ref` and `h(t)=(1-exp(-lambda*t))/lambda`, the continuous stress
is `sigma_xz=G*s*(1-2x)*h(t)`. The body force `b_z=2G*s*h(t)` balances its
divergence. Boundary constraints and tractions are those of this exact field.

At t=1 s, 64 equal physical increments and 2/4/8 Q1 hexahedra across x give
relative displacement L2 errors 0.248996, 0.0612490 and 0.0143323. This is
approximately second-order spatial decay before the temporal error floor,
not merely equality of homogeneous patch solutions. It verifies the
nonuniform creep-dominated limit. The existing axisymmetric NAFEMS
cylinder checks cover a different, pure-creep route and are not evidence of
axisymmetric admission for the joint material.

### Active simultaneous plasticity and creep in a nonuniform field

The same ordinary Step also solves an independently integrable force-driven
field: `tau_xz=S*t*(1-2x)`, `b_z=2S*t`, with S=60 MPa, E=200000 MPa,
nu=0.3, yield Y=50 MPa, isotropic hardening H=10000 MPa and
`creep_rate=A*(q/s_ref)^3`, A=0.001 /s, s_ref=100 MPa. Plane shear gives
`q=sqrt(3)*abs(tau)`. Under this monotonic loading, the exact plastic strain
is `p=max(q-Y,0)/H`, while equivalent creep is the time integral of that
known cubic stress law. The axial shear displacement follows by spatially
integrating `du_z/dx=tau/G+sqrt(3)*sign(tau)*(p+c)`, with zero displacement
at x=0 and x=1. This reference does not call an AgentFEM material update.

For t=1 s and 128 increments, 4/8/16 Q1 elements give relative displacement
L2 errors 0.128625, 0.0320049, 0.00781042. Plastic and creep variables are both
nonzero, and a central elastic region remains. Peak plastic strain is also
checked against the independent midpoint equilibrium stress. At fixed mesh,
32/64 increments give peak-creep relative errors 0.0634766 and 0.0314941,
matching the independently summed cubic endpoint rule `2/N+1/N^2` relative
to the continuous integral. Thus temporal bias remains visible rather than
being mislabeled as spatial error. This verifies bounded proportional,
isothermal small-strain behavior; it is not experimental validation, spatial
verification of nonproportional cyclic thermal loading, or a life prediction.

### Prescribed nonuniform temperature with active plasticity and creep

A second version retains the force-driven field above and introduces
Arrhenius dependence, with Q=120000 J/mol, R=8.31446261815324 J/(mol K),
T_ref=800 K and `a=2`. Prescribe

```text
T(x) = 1 / [1/T_ref - (R/Q) log(1+a*(1-2x)^2)]
```

so the continuous local rate coefficient is exactly
`A(x)=A_ref*(1+a*(1-2x)^2)`. The analytical stress/plastic field remains
as above. Let `l=min(x,1-x)`, `y=1-2l`; at t=1 the creep part of displacement is

```text
sqrt(3)*A_ref*(sqrt(3)*S/s_ref)^n / [2*(n+1)]
    * [(1-y^(n+1))/(n+1) + a*(1-y^(n+3))/(n+3)]
```

The Q1 temperature field is independently interpolated from this profile.
On 4/8/16 cells with 128 increments, relative displacement L2 errors are
0.137992, 0.0346244 and 0.00813975. Both plasticity and creep are active.
This checks spatial temperature sampling and its rate effect together with
equilibrium. Thermal expansion is deliberately zero to isolate that mechanism;
the profile is prescribed, not an independently verified heat-conduction
solution. Nonproportional thermal cycling and experimental material accuracy
remain separate gates.

### Rotating stress, distinct flow directions and thermal cycling

`tests/test_creep_plasticity_rotating_reference.py` constructs a continuous
material path independently of the discrete return map. This follows the
general verification principle of prescribed/manufactured solutions described
by [Salari and Knupp, SAND2000-1444](https://www.osti.gov/biblio/759450);
the particular material fixture here is not a published experimental test.

Two orthonormal deviatoric tensors define a direction
`e(t)=e1*cos(pi*t)+e2*sin(pi*t)` for 0 <= t <= 1 s. Prescribe accumulated
plastic strain `p=beta*t`, beta=0.002 /s, with plastic flow
`n_p=sqrt(3/2)*e(t)`. For linear isotropic hardening, stress is
`sqrt(2/3)*(Y+H*p)*e(t)`. For Chaboche, the relative stress has radius
`Y+Q*(1-exp(-b*p))`; add the exact continuous backstress convolution

```text
alpha_i(t) = sqrt(2/3)*C_i*beta *
    [real(z_i)*e1 + imag(z_i)*e2]
z_i = [exp(i*pi*t)-exp(-gamma_i*beta*t)] / (gamma_i*beta+i*pi)
```

The current stress and relative stress deliberately have different directions,
so creep and plastic flow are not coaxial in the Chaboche fixture. Integrate
the prescribed continuous creep/plastic rates to construct total strain,
then supply only that strain and temperature to the tested material update.
No AgentFEM return-map result enters the reference. Independent 64/128-point
Gauss integration agrees within 1e-12 relative and 1e-15 absolute tolerance.

Both isothermal and `T=800+50*sin(pi*t)^2` K heating/cooling paths are tested;
the latter includes alpha=1e-5 /K thermal expansion and Arrhenius creep with
Q_activation=120000 J/mol. E=200000 MPa, nu=0.3, Y=200 MPa,
A_ref=1e-5 /s, s_ref=100 MPa, n=3. Linear H=2000 MPa, or Chaboche
C=(12000,3000) MPa, gamma=(100,8), Q=60 MPa and b=10. Maximum stress
Frobenius errors over accepted time stations are:

| Plastic branch / temperature | 32 increments (MPa) | 64 | 128 |
| --- | ---: | ---: | ---: |
| Linear isotropic / constant | 7.01398 | 3.57182 | 1.80262 |
| Linear isotropic / heating-cooling | 7.07495 | 3.60098 | 1.81684 |
| Chaboche / constant | 6.69018 | 3.40388 | 1.71705 |
| Chaboche / heating-cooling | 6.76379 | 3.43956 | 1.73459 |

Creep/plastic tensors, their accumulated scalars, and all Chaboche backstress
branches also show first-order error reduction. This is a nonproportional
thermal material-point reference, not nonuniform structural verification or
an experimentally calibrated thermomechanical fatigue-life model. The initial
elastic preload to yield is prescribed; its work is outside this test.

### Perfect-plastic cyclic limit and strain-life conventions

The simultaneous material, with zero creep and perfect J2 plasticity, follows
a prescribed uniaxial strain cycle. For E=200000 MPa, sigma_y=200 MPa and
eps_a=0.003, the independent closed-loop work is
`4 sigma_y (eps_a - sigma_y/E) = 1.6 MPa` (energy per unit volume).
The assembled reaction loop gives 1.6 MPa; reversal stresses are +/-200 MPa.

A synthetic Coffin-Manson-Basquin curve with both exponents -0.5 has a closed
form life, not a numerical round-trip reference. This cycle gives
N=14168.055556 full cycles and Miner damage 7.05813156e-5. Residual half-cycle
counting, amplitude versus range and N versus 2N are tested separately.
This does not calibrate fatigue damage from plastic work.

## Fail-closed assessment

The complete damage point must lie inside the declared interaction-diagram
abscissa range and below its ordinate. In particular, Dc>1 and Df=0 must not
pass the reference Dc+Df=1 diagram. A partial diagram cannot certify values
outside its supplied range. Imported negative/nonfinite damage is rejected.
`margin` remains the vertical fatigue margin; consult `acceptable` and
`within_interaction_domain` together. `acceptable` only refers to that declared
diagram, not to a regulatory or component-life approval.

## Next material-validation gate

A first open-data audit inspected the experimental CSV from
[Rovinelli et al., Mendeley Data v1, DOI 10.17632/kcbcnywygr.1](https://data.mendeley.com/datasets/kcbcnywygr/1)
(CC BY 4.0, retrieved 2026-10-29). Its 77 records contain three principal
stresses in MPa and rupture time in hours: IN600 (41), SS316 (13), SS304 (23).
The raw file SHA-256 is
`f1c187779effc0b8230111f966f4ed885720495488d397c25060904d9f4add9a`, matching the
repository file metadata. Experimental and simulation tables were kept
distinct; only the experimental table was inspected. It contains no explicit
temperature, heat, condition, censoring flag or hold-cycle observations.
The landing-page description mentions IN100 while the table says IN600.
Consequently this is a **candidate rupture reference, not admitted calibration
or independent creep-fatigue validation data**. Resolve identities against
the original papers before fitting; do not infer temperatures or treat the
published correlation coefficients as an independent prediction.

The public [JRC P91 assessment recommendations](https://publications.jrc.ec.europa.eu/repository/handle/JRC94508)
make cyclic softening a mechanism-selection gate, not a cosmetic fit choice.
The current Chaboche asset admits nonnegative isotropic saturation and
plastic-flow-driven dynamic recovery. It does not evolve static backstress
recovery during a hold or represent cyclic softening. A future extension must
define physical-time state evolution, consistent tangent, temperature/domain
dependence, recoverable storage and dissipation, and joint rollback/restart.
Simply allowing a negative saturation parameter or changing a Norton
coefficient is not an accepted implementation. A fitted model must predict
held-out relaxation/cycling observations before a life assessment is promoted.

The OSTI record `974288` identifies a potentially useful experimental appendix,
but the attempted public PDF retrieval returned a rate-limit HTML page on
2026-10-29. Its tables have **not** been inspected or admitted. A search snippet
is not enough to establish strain conventions, material identities or reuse
rights, and no missing values were inferred.

A subsequent review obtained the CC BY 4.0 NIMS Grade 91 article linked
above and visually checked Tables 7–9. The reported rupture times are in
**hours**, temperatures in **degrees C**, stress in **MPa**, and minimum
creep rates in **percent/hour**, not strain/second. Tube, plate and pipe fits
and their stress regimes are separate. These are published regression
relations/derived strengths, not a held-out raw measurement set; reproducing
their tables would verify transcription and units, not independent material
prediction. A material card must preserve that distinction and the underlying
heat/condition scope before any calibration is admitted.

1. Select **one** material heat/condition and isothermal test temperature;
   acquire compatible creep, continuous cycling and hold-cycle data with
   documented rights, revision, units and failure definition.
2. Calibrate deformation using creep curves, cyclic loops and relaxation;
   reserve a different amplitude/hold duration for prediction. Never fit and
   score on the same curves, or mix design lower bounds with mean test lives.
3. Compare full stress relaxation and hysteresis before assessing life. Add
   static recovery, cyclic softening or temperature-dependent properties only
   when these comparisons identify the missing mechanism.
4. Independently assess fatigue and rupture calibration, then test interaction
   on held-out hold cycles. Do not double-count hold effects already included
   in a strain-life calibration. Record scatter and censored tests.
5. Only then progress to a nonuniform specimen and component mesh/time checks.
   Multiaxial and nonisothermal life, global damage feedback and regularization
   remain separate gates. ASME/RCC-MRx curves and their safety factors require
   a reviewed, versioned project input; no universal curve is silently supplied.
