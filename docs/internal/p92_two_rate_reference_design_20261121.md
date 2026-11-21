# Bounded two-rate reference: design before implementation

The existing mixed hypothesis has not passed BAM response screening. Its four
tested directions jointly reduce early cost but do not remove hold-shape and
cross-protocol discrepancies. Do not continue widening that fit.

This next reference is research-only, uniaxial, small-strain and isothermal.
It is not a published P92 parameter card or a transcription of a tensor model.
It uses the established parallel-flow / Chaboche-Jiang structure, but states
its scalar conventions explicitly. Each branch has two independently specified
backstress recovery times. No temperature, damage or lifetime is inferred.

## Defined scalar equations

For branch A, e_A = sigma - sum(x_Ai), and

    f_A = q_A * <(|e_A|-k_A)/eta_ref>^n_A * sign(e_A)
    xdot_Ai = H_Ai*f_A - (H_Ai/a_Ai)*W_Ai*x_Ai*|f_A| - r_Ai*x_Ai
    W_Ai = (|x_Ai|/a_Ai)^chi_Ai
    pdot_A = |f_A|
    sigmadot = E*(epsdot - sum(f_A))

eta_ref = 120 MPa is an explicit normalization. E,H,a,q are positive;
k,r,chi are nonnegative and n >= 1. All stresses use MPa and time seconds.
For a constant-stress hold, sigmadot=0 and epsdot=sum(f_A).
The shared elastic stress is evaluated once, not once per branch.

For fixed material coefficients define

    psi = sigma^2/(2E) + sum(x_Ai^2/(2H_Ai))
    D = sum(e_A*f_A + [(H_Ai/a_Ai)*W_Ai*|f_A|+r_Ai]*x_Ai^2/H_Ai)

Then D >= 0 and sigma*epsdot = psidot + D. This is a proved mechanical
identity for this specified reference, not a general first law or a formula
for the earlier softening hypothesis. At x=+/-a the vector field points inward
(or is tangent), so initial |x|<=a is invariant in the continuous equations.

## Verification before fitting

- Exact elastic limit and exponential static recovery with inactive flow.
- Instantaneous mechanical balance and nonnegative dissipation for signed
  loading, reversal and both control modes.
- Continuous cycle / split-step agreement and independent Radau/BDF checks.
- Integrated mechanical work = stored-energy change + dissipation.
- Positive-domain checks, no mutation of committed input, branch permutation.

Only after these pass may an explicitly identified parameter set be fitted to
cycles 1–3. All older candidates remain unchanged. A different state layout
gets a distinct schema; no old checkpoint is reinterpreted.

## First data-facing experiment

The independent driver uses the same measured switches, ramp deviations and
observation weights. It stores strain at index 7 and mechanical accounting at
8/9; it never routes those states through the old index-9 strain driver.
An explicit unfitted bridge seed matches initial threshold, flow normalization,
initial kinematic slopes and nonlinear static recovery at nominal saturation.
It removes the old isotropic softening. This is only an initialization recipe,
not equivalence of the models or a transferable published card.

Before fitting, perturb each component's static recovery by one decade in
either direction, using only cycles 1–3 of all three protocols. Compare costs
and the singular values of the response differences. If these directions are
weak or nearly dependent, do not fit four unconstrained recovery coefficients.
Do not use later cycles to choose the new coefficients. This experiment is a
mechanism discriminator, not a passing P92 material card.

The unfitted bridge completes three protocols, with cost 0.04403. Recovery
secants have singular values 0.06088, 0.01091, 0.0003313, 0.000005057.
Branch-A recovery is weak at this seed; therefore the next bounded test fits
only branch-B's two recovery rates, within the tested one-decade bounds, at
most 15 objective iterations. Initial strength, flow and all other coefficients
stay fixed. A bound hit is evidence against this conditional remedy, not
permission to expand the search. Four recovery parameters are not fitted.

The conditional fit reached cost 0.03938 (10.6% below the unfitted seed), with
the second branch-B recovery at its lower bound. This does not solve the
response mismatch and is not promoted; no boundary expansion is authorized
by this numerical result. Both the unfitted and conditional records are retained.

## Subsequent hypothesis, not yet admitted

If constant-coefficient recovery does not explain the cycles, test positive
backstress saturation evolution while keeping H fixed:

    a_i(p) = a_i,infinity + (a_i,0-a_i,infinity)*exp(-b*p)
    0 < a_i,infinity <= a_i,0, b >= 0

Use each branch's own accumulated inelastic strain, not an explicit cycle
counter or protocol label. The same free energy remains valid because H is
constant; positive a keeps the dissipative recovery term nonnegative.
The invariant envelope is the original a_i,0, not the shrinking a_i(p).
Do not incorrectly reject a legitimate state merely because it lies above
the current saturation target. This is a proposed scalar hypothesis, not an
equation claimed from an inaccessible paper or a qualified P92 model.

The primary-paper abstract on [evolving kinematic hardening in P92](https://www.sciencedirect.com/science/article/abs/pii/S0997753820305118)
supports investigating inelastic-history-dependent recovery; its full text
was unavailable here, so no coefficients or equations are imported.
The [BAM service-like study](https://www.sciencedirect.com/science/article/pii/S0142112322006314)
reports different relaxation/creep effects on subsequent load shifts. This
motivates testing history, not assigning a different law to each protocol.

Before any data fit require the zero-softening limit to reproduce the existing
reference, nonnegative instantaneous dissipation, exact exponential saturation,
continuous split-path agreement, and independent integration across reversal.
Only one shared saturation ratio and one evolution rate for a selected branch
would be considered initially; later cycles remain outside identification.

The optional positive saturation evolution is now implemented in the research
reference (model identity v2). Defaults reproduce the constant-target model.
Its zero-evolution limit, exponential targets, continuous cycles, independent
Radau/BDF response and integrated mechanical balance pass synthetic tests.
It has not been calibrated to BAM or admitted as a P92 material model.

The four prespecified history probes increased early cost (0.04417–0.04597,
versus 0.04403). Do not fit the history term at this seed. This rejects only
the conditional perturbations, not all possible calibrations of that law.
Next establish the flow baseline: vary only branch-A's threshold (30–160 MPa)
and reference-flow multiplier (0.1–10), retaining all other unfitted bridge
coefficients. Use at most 20 iterations on the same early observations.
Do not combine this with recovery or history fitting in this experiment.

The conditional flow fit reduced cost to 0.009210 but reached the lower
threshold bound (30 MPa). Frozen ten-cycle replay passed independent numerical
checks in RF/SLR/SLC, but failed the unchanged response screens. SLC first
hold ends at predicted strain 0.001019 versus 0.00106 measured, while its
inherited stress is still 134.80 MPa versus 121.70 MPa. SLR cycle-10 peak
is 8.87% high. Thus a nearly correct creep endpoint does not close reversal
or cross-protocol response. Do not expand the threshold bound.

Next test the existing dynamic-recovery shape exponent, not another flow
parameter: common branch-A chi in {0,1,3,6}, with that conditional card frozen
otherwise. All four values preserve positive recovery and the mechanical
identity. Use only early three cycles to discriminate reversal curvature;
do not select chi from the ten-cycle results. This is a fixed diagnostic grid,
not a parameter search to continue until a desired answer appears.

The chi grid costs were 0.01307, 0.01011, 0.009210, 0.009068. At chi=6 the
first inherited SLC stress remained 132.80 MPa, versus 121.70 MPa observed.
The approximately 1.5% cost improvement over chi=3 does not close this gap;
no expanded chi search is justified by this discriminator.

An independent recorded-ramp slope audit then checked the base elastic scale.
First-cycle apparent slopes over a 0.0004 strain window were approximately
149.6–159.4 GPa across the three protocols, consistent in scale with the
155.18 GPa reference modulus. Narrower windows have very few recorded points
and appreciable strain quantization. These slopes include viscosity/control
effects and are not a Young's-modulus calibration or confidence interval.
They do not support a large arbitrary modulus correction to remove the
13 MPa reversal error. Source indices and window definitions are retained.

Current outcome: loading-shape discrimination and numerical verification are
closed for the examined scope; cross-protocol material-response admission
remains failed. Stop these bounded conditional searches. The next milestone
requires an identifiable joint reversal/hold model, with its parameter choices
justified by early full hysteresis histories rather than adding corrections
to individual protocol endpoints. No engineering-life claim follows from
any of the cost reductions reported here.

## Joint identification gate

The next bounded experiment combines six already discriminated directions:
branch-A threshold (30–160 MPa), log reference flow (+/-0.5 decade), exponent
(3–8), second saturation (+/-0.3 decade, fixed H), and the two branch-B
recoveries (+/-1 decade). It starts from the explicit unfitted bridge, not
from a selected late-cycle endpoint. All three protocols contribute cycles
1–3 with unchanged weights. No saturation-memory coefficients are fitted.
Before optimization compute the central-difference Jacobian: require smallest
singular value >=1e-5 and condition ratio <=1e4 in these declared dimensionless
coordinates. This is a local numerical screen, not statistical identifiability.
If it passes, allow at most 15 iterations, retain bounds/rank/evaluation count,
then freeze before later-cycle screening. A budget-exhausted fit is not called
converged; a failed screen stops this batch rather than widening the domain.

The six-coordinate experiment passed its local rank screen and converged in
13 iterations (157 residual evaluations), cost 0.007893. The threshold and
second branch-B recovery reached bounds. Frozen cycle-10 peak errors were
RF +3.46%, SLR +9.07%, SLC -4.14%; all three response screens failed. SLC
first hold remained 137.03 MPa versus 121.70 MPa, and its end strain was
0.001234 versus 0.00106. Full state, independent-integration and identical
waveform subdivision checks all passed. No fitted candidate is promoted.

The frozen-reference numerical screen was strengthened after review to cover
every backstress and accumulated-strain endpoint, work/dissipation, and
identical-waveform knot refinement, not only plotted stress/strain samples.
The same flow card passed the expanded checks in all protocols: maximum
stress-state difference below 6.8e-7 MPa, strain-state below 5.5e-11,
work/dissipation below 1.4e-7 MPa. The original field-only record remains;
the complete-numerics record supersedes it for numerical admission. Response
failures are unchanged. Tests deliberately perturb hidden states while
leaving plotted fields identical to ensure the expanded gate rejects them.

## Distinct low-stress creep hypothesis

The original branch-B threshold is about 123 MPa, comparable to the observed
SLC hold stress. It can suppress that branch once its effective stress falls
below threshold. This is consistent with the earlier audit showing 93–97%
of hold strain coming from the other branch; it does not by itself prove an
incorrect physical mechanism. [NEML power-law creep](https://neml.readthedocs.io/en/main/creep/s_power.html)
has rate A*sigma_eq^n without a finite threshold. This motivates a distinct
fixed-zero-threshold branch-B test, not widening branch-A's fitted bound.
Retain kinematic recovery, so this is not claimed to be a pure Norton-law
transcription or a NEML numerical comparison. Freeze the joint early card
otherwise; test branch-B rates 1e-9, 1e-8, 1e-7, 1e-6 /s at 120 MPa effective
stress, using only cycles 1–3. Invalid control paths are explicit failures
with no aggregate cost, not silently shortened successful simulations.

This fixed experiment also failed to improve early cost: the first three
rates gave 0.01343, 0.01319 and 0.01267 versus the joint card's 0.007893.
The 1e-6 /s case left the declared SLC post-hold loading direction and is
recorded as failed with no aggregate cost. Do not infer that every possible
zero-threshold model is disproved; only these frozen conditional changes
were tested. Stop this batch of conditional searches and retain its evidence.

The initial nine tests now pass: analytic elastic/static-recovery checks,
instantaneous signed-work identities, cycling/reversal/hold restart, independent
Radau/BDF integration, branch permutation, invalid-domain and failed-solve
input preservation, and interior accumulated-state drawdown rejection.
The initial strict one-local-error-scale monotonicity check rejected BDF's
inactive-flow plateau: the largest observed strain drawdown was 2.0644e-13
at rtol=1e-10, falling to 2.2629e-14 at rtol=1e-12. The numerical guard
uses ten local error scales, without clipping; independent response and
mechanical-balance checks remain unchanged. It rejects interior decreases,
not just endpoint decreases. These use synthetic parameters and are numerical
verification only. No BAM fit or engineering prediction is claimed.

## Sources and scope

[NEML's Chaboche documentation](https://neml.readthedocs.io/en/main/hardening/non/chaboche.html)
separates component-specific dynamic and static recovery; its stress-like
history sign convention differs from the convention above. No library output
is claimed without actually running it.
The [FVWHT two-rate contribution](https://opus.hs-offenburg.de/files/11628/SeifertSchlayerMaier_PlastCrpFat_FVWHT_2025.pdf)
motivates response-time separation and ratcheting control, not transferable
BAM coefficients. The scalar H/a convention above is defined independently;
do not infer tensor normalization from extracted PDF typography.
