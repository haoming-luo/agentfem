# P92 problem re-audit: response kinetics before another fit

## What is being tested

The BAM records are three service-like, mixed-control material experiments,
not a published constitutive card with a known passing solver implementation.
One common card must predict continuous RF, SLR and SLC histories at 620 C.
The [source description](https://pmc.ncbi.nlm.nih.gov/articles/PMC10319173/)
defines engineering stress using initial area and distinguishes imposed strain
from force control. SLC inherits the force at the zero-strain switch; its
subsequent strain is an output, not an extra prescribed loading history.
The tests use specimens from a common pipe, but do not provide replicate-based
scatter sufficient to identify a statistical acceptance band.

No unit or control-definition error has been established by this re-audit.
That is not proof that controller approximation has no contribution. The
existing measured-waveform versus linear and recorded-prefix experiments
remain necessary evidence; do not reset state or prescribe measured SLC
response to improve agreement.

The strategic correction is important: software verification, a predictive
material card, and engineering life qualification are three separate tasks.
Early cyclic softening is not automatically fatigue damage. A numerically
verified material-point trajectory does not qualify a life prediction.

## Exact residual accounting, with unchanged weights

`tools/p92_residual_conflicts.py` partitions the existing cycles 1--3 objective
by protocol, loading phase and hold window (first min(10 s, 10% duration),
middle, last 20%). It reproduces the archived matched costs exactly:
0.0023607800272608744 versus 0.0023195577138414862. No fitting, weights,
parameters or response tolerances changed.

The 1.75% aggregate gain includes an SLC reversal cost reduction of about
0.0000323 and an SLR reversal reduction of 0.0000158, but an RF continuing-ramp
increase of 0.0000159 and an SLC positive-hold early-window increase of
0.0000053. These are finite prediction differences, not parameter gradients
or causal proof of a microscopic mechanism.

Each segment is additionally decomposed as r(t)=a+s(t), where a is its first
observed residual, not an invented control-switch value. The exact cost is
offset cost + shape cost + their signed cross term. Anchoring is diagnostic
only: the original unshifted prediction remains the acceptance input.

| First positive hold | Baseline anchor (MPa) | History anchor (MPa) | Baseline early shape RMS (MPa) | History early shape RMS (MPa) |
| --- | ---: | ---: | ---: | ---: |
| RF | -2.734 | -3.241 | 3.057 | 3.029 |
| SLR | 3.510 | 3.091 | 1.467 | 1.507 |
| SLC, after creep/reload | -0.302 | -0.161 | 5.756 | 6.144 |

The SLC first sample is 0.045 s after the inferred switch. Its entry stress
is already close, while subsequent relaxation diverges; the late shape
residual magnitude grows from 13.850 to 14.025 MPa. Thus merely correcting
the inherited stress/peak cannot explain this particular hold error. This
is consistent with the earlier same-stress diagnostic, which retains
measured strain drift and finds excessive predicted post-creep relaxation.
It does not alone exclude every loading approximation or identify a unique law.

## Literature and transfer limits

* [Jurgens et al., Metals 2019, 9, 99](https://doi.org/10.3390/met9010099):
  low-amplitude P92 tests show initial tensile/compressive relaxation
  asymmetry and hold-related initial hardening. This supports treating
  initial/reversed histories explicitly, not fitting a single monotonic
  softening curve. It does not provide this mixed-control material card.
* [Khayatzadeh et al., Engineering Fracture Mechanics 2017](https://doi.org/10.1016/j.engfracmech.2017.02.008):
  primary plus secondary creep and strain-hardening descriptions matter
  for P92 relaxation. The study concerns 650 C and different stress ranges
  and elastic-follow-up tests. Neither its constants nor its rig compliance
  may be copied into the BAM strain-controlled experiment. Its monotonic
  primary-creep formula is not by itself a reversal-history law.
* [Ohno and Wang, 1993, part I](https://doi.org/10.1016/0749-6419(93)90042-O):
  critical-state dynamic recovery is a relevant alternative for reversal
  behavior. Our current magnitude-powered recovery must not be called a
  complete Ohno--Wang implementation merely because it has a power exponent.
  Exact directional evolution and its scalar reduction require verification
  before implementation; the abstract alone does not establish them.
* [NEML Chaboche documentation](https://neml.readthedocs.io/en/dev/hardening/non/chaboche.html):
  static recovery, accumulated-strain-dependent dynamic recovery and
  isotropic strength are distinct mechanisms. Its backstress sign convention
  differs from ours; copying an expression without conversion is unsafe.

The 2023 BAM main-paper abstract motivates history-sensitive creep regimes;
full text was not obtained in this audit. It is not evidence for a specific
new differential equation or parameters.

## Next decision gate, not a promise of successful calibration

1. Audit directional dynamic recovery against an accessible primary equation,
   retaining the current card as control. Test reversal, zero flow, saturation,
   positive dissipation and time-integration convergence before data fitting.
2. Separate that no-new-parameter law substitution from primary-creep strain
   hardening. Do not simultaneously change kinetics, objective weights and
   history coordinates: that would obscure the cause of any improvement.
3. A justified candidate uses the same 22 candidate/10 retained directions,
   norm limit 0.3 and 40-evaluation ceiling, with the two-scale sensitivity
   gate unchanged. If two new history coordinates are needed, replace weak
   ones; do not add them. Record actual residual evaluations as well.
4. Freeze after cycles 1--3, check cycles 4--10 with independent integration,
   all original peak/hold/strain gates and this segment accounting. Previously
   inspected later cycles are not a blind validation set. All six existing
   response screens still fail; none is relabelled as passing.

This batch implements and validates the diagnostic, not a new constitutive
fit. No new mechanism has passed selection yet. Keep PR #197 open, without
merge, release or production material promotion.

Local acceptance: 96 targeted tests passed (residual accounting, prior matched
identification, same-stress hold audit and CI scope); correctness lint and
`build_docs.py --check` passed. The evidence is
`evidence/creep/2026-11-22-early-residual-conflicts.json`, including frozen-input
and source hashes. These checks validate the diagnostic, not the material.
