# P92: response, calibration and life evidence

AgentFEM now has a reproducible P92 research baseline and reusable history/life
assessment infrastructure. The remaining engineering task is to establish
prediction on compatible independent material data and define the intended
component assessment. More solver iterations cannot substitute for that data.

A [coherent-cohort source audit](internal/p92_db_cohort_candidate_20261107.md)
now includes visually checked creep, fatigue and dwell observations. Fixed
printed creep coefficients have unresolved strain-scale discrepancies;
the data are not an installed material card or a qualified coupled-life model.

## What can be reproduced now

These are repository research tools, not installed material cards. Run them
from a source checkout with AgentFEM and its scientific dependencies installed.
Each command emits JSON, keeps source identity and states its validation role.

```bash
python tools/p92_life_screen.py
python tools/p92_pure_fatigue_screen.py
python tools/p92_rupture_screen.py
python tools/p92_prior_damage_audit.py
python tools/p92_rate_fatigue_screen.py
python tools/p92_db_cohort_screen.py
```

The cyclic figure screen additionally needs an existing continuous-reference
campaign: `python tools/p92_cyclic_figure_screen.py campaign.json`. Its
[reproduction record](internal/p92_cyclic_screen_20261106.md) explains generation
and reuse; do not start a long campaign just to inspect the archived conclusion.

| Question | Evidence and result | Meaning |
| --- | --- | --- |
| Can a bounded combined-life model be calibrated without fitting its holdout? | Six training observations, two fixed held-out observations; predictions 961.892 and 788.484 cycles versus 1190 and 853 | Within-source interpolation baseline, not an independent batch or allowable |
| Do the published response equations reproduce cyclic observations? | Fixed-parameter first- and long-cycle experimental figure screens retain reading sensitivity and residual discrepancies | Response evidence; an observed failure count used to place a plotted marker is not predicted failure |
| Can supplied rupture observations enter a reusable assessment? | Four completed rupture tests, split into unaged/aged conditions; exact node recovery and half-life time fraction 0.5 | Numerical interface verification, not validation of interpolation or extrapolation |
| Can a published pure-fatigue relation use the fatigue API? | Separate source-defined strain-life parameters and tests | Reproduction, not a new independent fit and not permission to combine incompatible sources |

The source audits and full numerical records are in the
[life screen](internal/p92_life_screen_20261106.md),
[cyclic screen](internal/p92_cyclic_screen_20261106.md), and
[external-source admission audit](internal/p92_external_source_admission_20261106.md).
The generic `assessments.TabulatedRuptureCurve` retains stress/time units,
stress measure, material condition, temperature and failure definition. It
interpolates only within its declared stress interval at its supplied
temperature; it does not select a P92 curve for the user.

The [prior-damage audit](internal/p92_prior_damage_20261106.md) adds a traceable
remaining-creep-life dataset and audits published predictions, including their
large high-damage errors. It is not a newly executed material model. This offers
a concrete next reproduction route without pooling unrelated P92 conditions.

The [equation reference](internal/p92_creep_equations_20261106.md) now executes
the bounded as-received creep law and agrees with an independent time-domain
integration. The published backstress equations are inconsistent, and the
dimensionally consistent reading still differs from the tabulated prediction
by approximately 3.67%. This is numerical research evidence, not a promoted
P92 material card or engineering lifetime validation.

## Two life routes, not one mixed formula

The [service-aged rate screen](internal/p92_rate_fatigue_20261107.md) adds a
separate 600 C cohort and reproduces published fatigue coefficients without
refitting. Cross-rate errors remain visible (up to +32.56% among the cited
admitted high-to-low-rate comparisons); no blind validation is claimed.

**Combined test-life route:** a fitted creep-fatigue life relation predicts
the source-defined combined test outcome for an admitted waveform. Do not
feed it into a pure-fatigue damage term and add creep damage again.

**Separated assessment route:** compatible pure-fatigue and creep-rupture or
ductility information supports separately defined damage measures. The chosen
interaction envelope, stress/strain measures and design margins must come from
the assessment being followed. A linear sum is a modeling choice, not a
universal validated P92 rule.

The [public scope of ASTM E2714-26](https://store.astm.org/e2714-26.html)
recommends complementary continuous-cycling and creep data for the same
material and temperatures. Its crack-formation outcome is not automatically
the published P92 paper's peak-stress-drop criterion. This reference guides
data selection; it is not a claim of compliance with the full standard.

## Smallest next engineering validation package

Use the [engineering-life uncertainty guide](internal/engineering_life_uncertainty_20261107.md)
to distinguish population scatter, parameter/model uncertainty, predictive
intervals and design margins. The target is a decision with explicit uncertainty,
not exact agreement with every specimen or a probability band around an
unqualified model. The guide records reusable methods and proposed next steps;
it does not add a statistical-life API.

1. Choose the application and failure outcome: parent metal or weld, temperature
   range, loading rate, tensile/compressive holds, environment and whether the
   target is specimen failure, crack initiation, or a code-based component margin.
2. Supply specimen-level observations with heat/product/heat-treatment identity,
   units and stress/strain conventions. Retain interrupted/runout tests as such;
   do not replace them with rupture events. Repeated tests quantify scatter.
3. Freeze a calibration subset and model choices before evaluating independent
   observations. Check for reused specimens across publications. The current
   eight-row table cannot create an independent validation set by being split
   differently or cited from a second paper.
4. Pass numerical checks with those frozen parameters: material point,
   homogeneous structure, time refinement, failed-trial rollback and restart.
   Use co-located accepted stress/strain/temperature histories, not extrema from
   different locations. Then assess a nonuniform component with spatial/time
   refinement and the actual boundary conditions.
5. Report prediction error and scatter separately from design margin. Select
   application-specific acceptance criteria before inspecting validation errors;
   do not promote an arbitrary factor-of-two plot to an engineering guarantee.

[NIMS sheet 48B](https://www.nims.go.jp/eng/press/2018/04/201804050.html)
is a valuable next source: it includes several T92/P92 heats and long-duration
rupture observations. Access and use are through the
[NIMS data service](https://cds.nims.go.jp/en/); the announcement alone does not
supply the numerical curves. Different heats are not silently pooled.

### Compatible does not mean necessarily the same heat

Same-heat tests are useful, not an absolute software admission rule. A
cross-heat research assessment may proceed with explicit transfer assumptions,
stratified validation and reported scatter. It must not be labelled same-heat
validation or a component allowable. Preserve individual source identities
instead of rewriting their material-condition labels to make a combination fit.

The public [EPRI report 1019778 (2010)](https://restservice.epri.com/publicdownload/000000000001019778/0/Product),
Table 5-1 and sections 5.2–5.3, calls for pedigree/distribution/scatter review
and distinguishes deformation interaction from damage interaction. Our
cross-heat research policy is an implementation decision informed by that
guidance, not a claim that the report supplies P92 design curves.

## Integrated software foundation

The [nonuniform bending screen](internal/p92_nonuniform_structure_20261107.md)
adds four mesh levels and two time grids using the ordinary structural
workflow. Total-reaction screening and a small-model restart pass; relaxation
drop and clamp hotspot spatial convergence remain open. This separates a
useful numerical composition test from experimental multiaxial validation.

Small-strain selected histories and prescribed temperature, structural cyclic
checks, and thermal clock/restart safeguards are merged through
PRs #152, #155, #158 and #160–#163. The
[combined installed-use acceptance](internal/p92_composition_acceptance_20261106.md)
includes restart comparison and real VTK reading. The
[structural cycle record](internal/p92_structural_cycles_20261106.md) includes
435 imposed cycles at 650 °C with time refinement, not a prediction of failure.
No release or engineering qualification follows automatically from merging
these interfaces.

Stop retuning this baseline when the missing information is experimental.
Continue useful software work on reusable material/state/result contracts;
resume engineering calibration when the selected application and compatible
validation observations are available.
