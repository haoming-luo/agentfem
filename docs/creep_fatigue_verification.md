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

Do not conflate the separately published
[INL/EXT-21-64314 Alloy 709 report](https://inldigitallibrary.inl.gov/sites/sti/sti/Sort_52942.pdf)
with Alloy 617 data. A similar test protocol is not the same material.

## Executable analytical references

Run the focused checks in the scientific environment:

```bash
python -m pytest -q tests/test_creep_fatigue_references.py tests/test_assessments.py tests/test_strain_life.py
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
