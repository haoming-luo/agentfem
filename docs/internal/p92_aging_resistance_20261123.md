# Rate-target resistance memory: prospective replacement

## Evidence and distinction from previous experiments

The previous normalized-rate backstress-exponent replacement lowered matched
cost by only 0.3544% and worsened the SLC hold entry and endpoint. Do not repeat
that fit. Instead, distinguish signed kinematic memory from positive flow
resistance with a rate-dependent target and finite evolution in inelastic
strain. Equal accumulated strain need not imply equal resistance after a
slow creep path and a fast loading path.

[Arai and Motoki (2025), DOI 10.1299/mej.24-00369](https://doi.org/10.1299/mej.24-00369)
describes aging stress separately from backstress for P91. The accessible
primary advance-publication text gives evolution toward a rate-dependent
target in Eqs. 19–21. Its 923 K material card and damage equations are not
admitted as BAM P92 data or copied into this reference. Final-publication PDF
downloads were incomplete and visual screenshots unavailable; the accessible
advance text is the equation source, not a verified numerical reproduction.
The earlier [Yaguchi and Takahashi (1999)](https://doi.org/10.1299/jsmea.42.1)
also distinguishes aging resistance in modified 9Cr–1Mo steel. Neither source
establishes this mechanism for the present P92 cohort at 620 C.

Table 1 of the advance text has a negative logarithmic slope (`B=-17.3`),
supporting the *sign* of the present low-rate resistance target. Neither that
slope nor its intercept is imported as a P92 parameter. The equilibrated
aging-resistance contribution has negative rate dependence; this is not a
claim of negative instantaneous sensitivity of the complete model at fixed
internal state. Total rate sensitivity must include the positive overstress
contribution and evolving history. A scalar material-point screen does not
establish spatial well-posedness or localization behavior.

The original institution's [CRIEPI T96017 report abstract](https://criepi.denken.or.jp/hokokusho/pb/reportDetail?reportNoUkCode=T96017)
identifies strain-rate changes and strain-dip tests as measurements separating
backstress, overstress and aging stress. This supports discriminating tests,
not unique identification from the present three cyclic protocols. If the
replacement remains confounded, record that limitation instead of changing
the observability threshold or declaring its state to be measured aging.

A useful subsequent discriminator would compare matched accumulated
inelastic strain reached at different prior rates, followed by identical
reload/hold conditions. Free recovery at zero stress separately constrains
the retained recoverable modes. These are proposed discriminating paths,
not additional measured observations or new acceptance evidence.

## Reduced hypothesis and parameter accounting

Retain the two recoverable strains, all backstresses, and both accumulated
irreversible branch strains. Add one **state**, `z`, with `z(0)=0`. Replace
constant branch thresholds by `Y_i (1+z)` and define

`v = p_dot_A + p_dot_B`, `v_ref = v0_A + v0_B`,

`z_target = max(-1, c log10(v_ref/v))`,

`z_dot = b v (z_target-z)`.

At zero flow use the exact limit `z_dot=0`, not a fitted numerical rate floor.
This is a shared normalized reduction, not the published single-flow P91 law:
its common reference rate, equal build/decay rate, zero initial increment and
mapping onto two thresholds are explicit assumptions. In particular it does
not add purely thermal aging at zero flow. The known original coefficients
`A.static1` and `A.static2` are removed (set exactly to zero), and their two
coordinate slots are used for `c` and `b`. No new independent coefficient is
introduced, but the additional internal state is reported rather than hidden.
The two recoverable modes and their four coefficients remain unchanged.

The predeclared early screen has exactly three arms: original (`c=0`),
deletion-only (`c=0`, both A static rates zero), and rate-target replacement
(`c=0.1`, `b=250`, both A static rates zero). No loading-protocol labels,
measured-response corrections or hold resets enter the material update.

For constant branch rate, the memory approaches its target exponentially in
accumulated strain. This yields an analytic mechanism test separating two
histories with the same accumulated strain but different rates. At `z=-1`
the vector field is nonnegative, preserving nonnegative thresholds. Stored
energy is unchanged: this added variable changes dissipative resistance, not
elastic stiffness or recoverable energy. For nonnegative thresholds the
existing irreversible-flow and recovery dissipation terms remain nonnegative.
Check power balance, sign symmetry, zero-mechanism reduction, independent
integration and split continuation before interpreting measured responses.

## Identification and freeze admission

First compare early SLC creep accumulation, hold entry, early/late relaxation
and SLR peaks. If the predeclared substitution offers no compatible response
direction, retain that failure and do not spend a fitting budget changing its
initial coefficients repeatedly. If admitted, use the existing 22 candidate /
ten retained directions, radius 0.3 and ceiling 40, with the two weak-coordinate
slots replaced, not added. Run an original same-start/same-budget control.
Only cycles 1–3 may enter identification. Freeze both cards before continuous
ten-cycle development checks, independent BDF integration and waveform
refinement. The new state is dimensionless, not a recoverable strain or a
monotonically accumulated damage variable. All original response and
observation-resolution gates remain unchanged. PR #197 only; no merge/release.

## Early three-arm result and bounded admission

The immutable `2026-11-23-aging-resistance-probe.json` uses only cycles 1–3.
Original/deletion-only/replacement costs are respectively 0.002239828765,
0.002238959861 and 0.002005285012. Deleting the weak static rates alone does
not explain the improvement. SLR cycle-3 peak relative error changes from
3.6536% to 3.6341%. The first SLC post-creep hold endpoint changes from
163.0703 to 165.6539 MPa (observation 174.88 MPa), and its early secant from
-5.3703 to -5.2896 MPa/s (observation -3.7937 MPa/s).

This is mixed, not acceptance: hold entry residual grows from +1.1160 to
+1.5687 MPa, late secant moves away from the observation, and preceding
creep endpoint strain falls from 0.00095556 to 0.00095060 against 0.00106.
The favorable endpoint/early-rate direction with nearly unchanged SLR peak
warrants **one** bounded identifiability/identification comparison, not a
sequence of altered starts. Keep `c=0.1`, `b=250`, all prior tolerances and
budget limits fixed. Abort optimization if substituted directions fail the
existing observability gate. Freeze a successful fit before later-cycle use;
do not promote an aggregate cost improvement to response or life acceptance.

## Completed same-budget identification

Both fits share the source/seed/driver identities and original budget: 22
candidate coordinates, ten retained directions, radius 0.3 and at most 40
optimizer evaluations. The original converges in 3 evaluations / 153 residual
calls to 0.0022088403742. The replacement converges in 5 / 195 to
0.0019456348666: an 11.9160% lower objective, not an 11.9160% response-accuracy
or life improvement. No restart, enlarged range or additional fitted
coefficient was used. Its frozen memory coefficients are `c=0.09618610917`
and `b=242.79122374`.

The replacement's relative Jacobian difference is 5.0337e-5 and retained
subspace distance 1.8169e-4. The residualized history-direction singular
ratio is 0.15530 and minimum retained-coordinate Gram eigenvalue 0.0012957,
passing the unchanged local numerical observability gate. This is not
statistical identifiability or confirmation of a physical aging mechanism.

## Frozen ten-cycle result and decision

Both cards were frozen before replaying ten continuous cycles under RF, SLR
and SLC. All six independent BDF / refined-waveform numerical comparisons
pass; all six original response screens fail. Later cycles are already
inspected development data, not blind validation.

| Observable | Original fit | Replacement fit | Observation |
|---|---:|---:|---:|
| SLR cycle-10 peak relative error | 7.6481% | 7.7751% | 0% |
| First SLC post-creep hold entry residual, MPa | +2.3747 | +2.5219 | 0 |
| First SLC post-creep hold endpoint, MPa | 163.2263 | 164.4016 | 174.88 |
| First SLC post-creep early secant, MPa/s | -5.5074 | -5.4657 | -3.7937 |
| First SLC post-creep late secant, MPa/s | -0.08101 | -0.07749 | -0.18193 |
| Preceding SLC creep endpoint strain | 0.00093897 | 0.00094566 | 0.00106 |

The identity-checked `2026-11-23-aging-resistance-paired.json` retains every
original gate, including unresolved observation-resolution checks. A lower
training objective does not resolve the requested conflict: the SLR peak,
SLC hold entry and late relaxation worsen. **Do not select or promote this
bounded replacement, continue its fit, enlarge its bounds or change the
weights to chase the endpoints.** This rejects this reduced substitution,
not every published aging model or all possible P92 constitutive laws.

Next research admission requires a mechanism or independent observation that
separates post-creep hold kinetics from reverse-loading strength, with an
explicit discriminator before fitting. Retain recoverable deformation and
the existing budget/gates. An additional resistance-memory state alone has
not supplied that separation. No engineering/life qualification, merge or
release follows from this batch.

## Engineering checks

In `fenicsx-env`, the batched `tests/test_p92*.py` and
`tests/test_ci_validation_scope.py` run passes **1194 tests in 65.10 s**.
Changed-file Ruff, generated-documentation consistency and whitespace checks
pass. The paired-report regression re-evaluates the original response checks
from saved observations and verifies the frozen-card/source identities.
These engineering checks do not change the failed scientific disposition.
