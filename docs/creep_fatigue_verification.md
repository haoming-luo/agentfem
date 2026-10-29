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

Do not conflate the separately published
[INL/EXT-21-64314 Alloy 709 report](https://inldigitallibrary.inl.gov/sites/sti/sti/Sort_52942.pdf)
with Alloy 617 data. A similar test protocol is not the same material.

## Executable analytical references

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
