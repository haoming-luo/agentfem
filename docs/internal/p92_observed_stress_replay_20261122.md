# Separate stress-path feedback from constitutive-history error

The signed frozen-path budget is evaluated on the predicted stress path.
Its late flow deficit cannot by itself prove that the material needs a
larger rate at the *observed* stress: predicted SLC stress is already lower.
Close this distinction before designing another rate modifier.

Prespecify a diagnostic with the completed same-budget original card, all
three first-cycle recorded stress histories, and no parameter changes.
Prescribe the complete piecewise-linear recorded stress; predict strain and
internal variables continuously, using the existing audited stress driver.
The initial observed stress/strain set the reference, with zero internal
memory explicitly assumed rather than inferred from the data. Do not reset
states at control switches or prescribe subsequent observed strain.

Compare strain increments over exactly the existing early and late hold
windows. Report the initial-reference assumption and full-path strain error
separately. Both measured channels participate, so this is a conditional
diagnostic, not independent validation or a replacement loading protocol.
The earlier recorded stress-ripple experiment only perturbed the nominal
stress-controlled dwell; this experiment instead prescribes stress everywhere.

Require Radau/BDF agreement in all twelve states, plus subdivision of the
same piecewise-linear input, mechanical work balance and unchanged source/card
identity. Do not bridge missing cycles, tune to this diagnostic, or count
prescribed stress agreement as a response pass. Keep all original failures.

## First-cycle result

The unchanged additional-budget original card gives the following conditional
strain-increment residuals (prediction minus observation):

| Protocol/window | Residual strain |
| --- | ---: |
| SLC creep, early | +3.595e-5 |
| SLC post-creep relaxation, early | +2.13249e-4 |
| SLC post-creep relaxation, late | -2.58967e-6 |
| SLR positive relaxation, early | -2.21404e-5 |
| SLR positive relaxation, late | -2.08272e-5 |

All three first-cycle independent-integration/refined-input checks pass, as
does signed strain accounting. Reference-arm terminal mechanical imbalance
is below 1.3e-13 MPa; the independent arm must also pass separately. These
are numerical checks of the conditional calculation.

In the SLC early post-creep interval, the predicted strain change is
1.83249e-4, versus -3e-5 recorded. Its components are elastic -2.27096e-4,
recoverable +4.82171e-5, branch A +1.72979e-4 and branch B +1.89149e-4.
Both irreversible branches matter; deleting the recoverable contribution
algebraically would not remove the excess. Such deletion is not a causal
material simulation and is not proposed as a model.

The small late conditional residual qualifies the earlier forward-path
interpretation: a deficient predicted late stress rate at an already too-low
stress does not demonstrate deficient kinetics at the observed stress.
It does not establish a late experimental pass, unique microscopic cause,
or correct initial internal memory. Continuous-cycle checks below retain
this distinction; no per-cycle state reset is allowed.

Evidence: `evidence/creep/2026-11-22-observed-stress-replay.json`.
The scalar/vector dense-evaluation adapter is explicitly tested; the first
attempt to use a vector-only evaluator in scalar quadrature failed and
published no result. Only the corrected complete run is archived.

## Observation readiness before more fitting

`tools/p92_observation_readiness.py` checks the existing observation windows
without solving or choosing a material. It leaves the response policy intact.
Across ten recorded cycles, RF has 40 resolved secants and SLR 60; SLC has
60 secants, of which all ten late-creep strain secants are unresolved under
the existing five-recording-quantum requirement. Their endpoint changes are
only two to four strain quanta. Additional samples are not independent repeats
and do not manufacture resolution. This is not a noise distribution, a reason
to waive the large resolved relaxation failures, or a material validation.

Evidence: `evidence/creep/2026-11-22-observation-readiness.json`.

### Recorded temperature is context, not an automatic correction

The same observation audit retains temperature over every window, using all
recorded samples between its endpoints. Across the first ten cycles, RF,
SLR and SLC span 2.7, 9.6 and 11.1 K, respectively. Do not assign the full
test span to every hold. For the first positive early hold, SLR is
620.4 -> 620.0 C (range 619.9--621.3 C), and SLC is 619.8 -> 619.9 C
(range 619.7--620.7 C). These records do not establish a temperature-only
explanation for the opposing conditional flow deficits.

The [data article's methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC10319173/)
describe mechanical extensometer strain and controlled nominal temperature;
they do not establish the exact incremental thermal-compensation algorithm
needed to justify a new subtraction from these archived strains. Do not add
or subtract thermal strain, fit an activation energy from these small
excursions, or infer measurement uncertainty from their range. The present
card remains an isothermal approximation. Its residuals are not proven to
come exclusively from material kinetics; inherited temperature effects and
measurement/prehistory uncertainty remain separate questions.

## Literature cross-check

The original [BAM data article](https://doi.org/10.1016/j.dib.2023.109333)
supports distinguishing the recorded stress and strain channels under the
three mixed-control protocols. The
[P92 primary-creep/relaxation study](https://doi.org/10.1016/j.engfracmech.2017.02.008)
supports checking the early transient and stress-dependent kinetics, but its
650 C heat is not this 620 C material. The
[P92 anelastic-recovery experiments](https://doi.org/10.1016/j.ijmecsci.2017.01.018)
support retaining recoverable deformation, not transferring fitted constants.
These sources are already in the literature ledger; this recheck is not a new
independent validation dataset or a reason to add another unidentifiable term.

## Continuous-cycle diagnostic

The three-cycle run retains the same card, complete recorded stress path,
zero-memory starting assumption, and independent numerical checks. SLC
positive-hold early strain-increment
residuals are +2.13249e-4, +2.15285e-4 and +1.85815e-4 across cycles 1--3;
negative-hold early residuals are -1.13620e-4, -2.52748e-4 and -2.90865e-4.
The discrepancy therefore persists beyond the first startup, although this
does not prove complete loss of initial-state influence.

The unchanged ten-cycle extension retains this early excess: positive-hold
early residuals range from +1.76365e-4 to +2.34156e-4, while negative-hold
early residuals range from -2.98189e-4 to -1.13620e-4. No fit, state reset,
altered window or new coefficient is used. Later cycles were already
inspected in development and are not blind data. This extension cannot
substitute for failed original mixed-control forward-response screens.

### Independent energy check caught an additional numerical failure

The initial three- and ten-cycle runs agreed in all twelve state comparisons,
but checking the independent BDF arm's terminal energy separately exposed an
accumulated error: about 1.53e-8--1.90e-8 MPa at three cycles, and
1.00e-7--1.16e-7 MPa at ten. Both exceed the prespecified absolute 1e-8 MPa
energy-accounting limit. The reference Radau arm's small balance does not
cancel that failure. Both complete failed records are retained with the
`-energy-failure.json` suffix; their `numerical_passed` is false.

The follow-up tightens BDF from rtol=1e-10 to 1e-12 and scales its absolute
integration tolerances by 0.01. This changes neither the material card,
recorded controls, state-comparison limits nor energy acceptance limit.
The completed tightened one-, three- and ten-cycle runs pass all three
protocols' full-state, signed-budget and energy checks. Ten-cycle independent
terminal imbalances are 7.655e-10 MPa (RF), 5.997e-10 MPa (SLR), and
8.272e-10 MPa (SLC), below the unchanged 1e-8 MPa limit. The complete early
prefixes agree exactly across requested cycle counts. Regression checks also
confirm the reference predicted increments, observations and material card
are unchanged from the retained failed runs. No recalibration or
experimental-response acceptance follows from this numerical success.
Even the retained ten-cycle BDF failure differs from the reference strain by
less than 3.5e-11, whereas the positive early-hold experimental residual is
about 2e-4. The extra energy failure therefore does not explain away the
large experimental mismatch. It exposes a separate numerical-accounting
gate that must be satisfied on its own.

## What a simple rate multiplier cannot repair

The archived signed budgets permit a small exact accounting check, without
rerunning or fitting a new material. Hold elastic and recoverable increments
and the integrated histories fixed. For cycle 1's positive early holds,
the SLR and SLC irreversible branch increments form the rows of

```text
       A             B
SLR    1.912886e-4    1.637019e-4
SLC    1.729790e-4    1.891494e-4
```

After subtracting the fixed elastic and recoverable increments, the recorded
strain requires totals 3.771308e-4 and 1.488791e-4, respectively. A common
scalar multiplier would need to be both 1.06237 and 0.41112. Allowing a
separate nonnegative multiplier for each branch does not fix this: their
SLR/SLC ratio must lie between the column ratios 0.86546 and 1.10585,
but the required ratio is 2.53313. The unconstrained algebraic solution
would need a negative B multiplier, incompatible with this proposed scaling.

The same cone exclusion holds in all ten cycles. Even allowing an accounting
band of +/-2e-5 on each required total leaves the minimum ratio above the
cone maximum. This band is a sensitivity check, not experimental uncertainty
or a replacement acceptance criterion. Regression tests recompute the
inequality from the signed archives rather than trusting rounded numbers.

This excludes only a **frozen-history positive-prefactor repair**. Changing
parameters in a real simulation also changes internal history, so it is not
a proof that the nonlinear material family has no acceptable parameters.
The next replacement must demonstrate a different history-dependent response
in these holds while preserving creep accumulation and recovery. A blanket
rate acceleration/deceleration is not an adequate design argument.

### Next experiment admission

Before another optimizer run, a replacement must predict the *contrast*
between SLR and post-creep SLC early holds with one material, not merely
improve the isolated SLC endpoint. Keep the two recoverable modes and the
same number of independent fitted coordinates. First verify the proposed
equations and power accounting, then test a continuous early-cycle path
covering creep accumulation, hold entry, both signed holds and reversal.
Only a supported discriminator proceeds to a same-start, same-budget fit
against the unchanged control; freeze before checking later cycles. Reusing
the same examined ten cycles cannot be labelled independent blind validation.
Unresolved observation secants remain unresolved, not silently dropped passes.

## Reproduce the diagnostic

Use `fenicsx-env` from the repository root. The source directory must contain
the original `Data_RF.lis`, `Data_SLR.lis` and `Data_SLC.lis` files matching
the frozen card's hashes; output paths must be new. Replace the two example
paths below with the source directory and a new report path.

```bash
python -m tools.p92_observed_stress_replay \
  evidence/creep/2026-11-22-recovery-shape-control-frozen.json \
  /path/to/public-data /path/to/new-replay.json --cycles 10
python -m tools.p92_observation_readiness \
  /path/to/public-data /path/to/new-observations.json --cycles 10
python -m pytest -q tests/test_p92_observed_stress_replay.py \
  tests/test_p92_observation_readiness.py tests/test_p92_stress_ripple.py
```

An output file or successful process exit is not an acceptance decision:
inspect each protocol's `numerical_passed`. The report deliberately keeps
`forward_validation`, `engineering_qualified` and `life_qualified` false.
The original mixed-control response screens, not these conditional replays,
remain the material admission checks.

Final local batch verification: **1143 P92 and CI-scope tests passed in
65.31 seconds** in `fenicsx-env`; changed-file Ruff, generated-document
consistency and whitespace checks also passed. This is targeted engineering
verification, not an experimental response or life qualification. No new
production material, parameter count, response threshold or release is added.
