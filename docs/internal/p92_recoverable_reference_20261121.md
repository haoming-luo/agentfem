# Reversal error budget and recoverable-strain discriminator

PR #197 remains open; no merge, release or P92 catalog admission.

The frozen joint card is not refitted for the initial discriminator. At actual
aligned observations, first-cycle SLR/SLC compression-to-zero ramps accumulate
approximately 31.28/24.92 MPa of stress-bias change attributable algebraically
to the nonelastic increment error after excluding the switch ambiguity brackets.
Strain-input error is retained separately. These are interval error budgets, not experimentally identified strain
partitions. They exclude unobserved endpoints and do not establish causation.

## Literature decision

[Kazemi et al. (2023)](https://doi.org/10.1016/j.euromechsol.2022.104821)
separate elastic, recoverable viscoelastic and viscoplastic strain for 10%Cr
steel at 600 C, with Kelvin-Voigt components and biaxial evidence. The accessible
abstract motivates testing a recoverable component; it does not provide a BAM
620 C calibration or justify transferring parameters.

[Li et al. (2021)](https://doi.org/10.1016/j.ijmecsci.2020.106044)
instead describe primary creep regeneration using a shared effective-stress
two-power flow law and evolving back/drag stress. Their 10%Cr 600 C work
shows why a reversal-triggered history reset is inadequate. This is an
alternative mechanism, not equivalent to our independent two-rate branches.

[P92 recovery experiments](https://www.sciencedirect.com/science/article/abs/pii/S0020740316305690)
at 898 K support investigating recoverability, not importing life constants.
Accessible abstracts are distinguished from complete equation verification.

[Benaarbia et al. (2018)](https://nottingham-repository.worktribe.com/output/960836)
provide a thermodynamic viscoelastic/viscoplastic decomposition for P91 at
600 C, addressing relaxation both below and above yielding. Their distinct
strain-rate/range verification is a useful validation design; P91 constants
and their success are not P92 evidence. Our single recoverable element is
deliberately smaller and must stand on its own tests and BAM residuals.

## Minimal original reference, before fitting

Append one recoverable scalar strain a to the existing isothermal reference:
`a_dot = (sigma/H - a)/tau`, with positive H and tau. Total strain is
`sigma/E + plastic_strain_A + plastic_strain_B + a`. Existing branch updates
remain unchanged; strain-control stress rate subtracts `a_dot`, while stress
control total strain rate includes it. No switch-dependent correction occurs.
Add `H*a*a/2` to the specified stored energy and `H*tau*a_dot*a_dot` to
dissipation. Hence the existing mechanical accounting remains closed.
This is a research discriminator, not a transcription of the cited model.

First verify constant-stress exponential creep/recovery, instantaneous power
identity, nonnegative dissipation, sign symmetry, split-path continuation and
Radau/BDF agreement. Then use a predetermined compliance/time grid with all
old coefficients frozen. Retain every failed candidate and all three protocols.
Only a coherent improvement warrants joint early-cycle identification; cycles
4–10 remain inspected development data, not independent validation. Neither
lower aggregate cost nor numerical success changes the response gates.

## Prespecified joint experiment

The fixed grid uses H/E = 5, 10, 20 and tau = 0.2, 2, 20, 200 s.
All 12 paths complete. Only H/E=20, tau=0.2 improves the joint early cost:
0.00789337 to 0.00756068. It changes first SLC hold stress from 137.03 to
128.12 MPa and terminal strain from 0.0012339 to 0.0010404, compared with
121.70 MPa and 0.001060. No response admission follows.

Use that point as the initial hypothesis for joint identification with the
same six backbone coordinates plus log H and log tau, all protocols and early
three cycles jointly. Before optimization require the scaled sensitivity
matrix smallest singular value >=1e-5 and condition <=1e4. These are numerical
screening criteria, not statistical identifiability. Bounds set before this
run: A threshold 1–160 MPa, exponent 2–10, flow multiplier 10^±0.5,
second saturation multiplier 10^±0.3, B recovery multipliers 10^±1;
H multiplier 10^±0.6 around 20E and tau multiplier 10^±1 around 0.2 s.
The threshold search is broader than the previous conditional experiment;
this is a new jointly calibrated model, not a controlled one-parameter
ablation. Limit to 20 optimizer evaluations, retain failed/nonconverged fits.
Do not change the response acceptance thresholds or residual weighting.

## Frozen outcome and next discriminator

The eight-coordinate fit converged in 14 optimizer / 222 residual evaluations,
cost 0.00376539. Flow A reached its lower multiplier bound and saturation A2
its upper bound. This is not a unique material identification. Frozen ten-cycle
Radau/BDF and identical-waveform refinement checks pass all 11 states.

First SLC hold stress is 121.166 MPa (observed 121.70), early strain rate
3.757e-5/s (3.568e-5/s), terminal strain 0.00099784 (0.001060). Its late rate
6.089e-8/s is too low against 1.134e-7/s. Tenth SLC hold stress is 123.389
MPa (121.87). However tenth RF/SLR peak errors rise to 8.97%/12.70%; SLC
peak error is -1.18%. All three complete response screens still fail.

The switch-excluded interval budget localizes the improvement: first
compression-to-zero nonelastic stress-equivalent error falls from 31.28 to
8.74 MPa (SLR) and from 24.92 to 2.24 MPa (SLC). SLR's subsequent reload
still adds approximately 8.83 MPa of that error. These are differences of
aligned response increments, not experimental measurements of the new state.

The observed RF peak falls from 292.91 MPa at cycle 2 to 276.94 at cycle 10;
this reference instead rises slightly. The SLR first-peak bias is already
16.11 MPa, increasing to 33.44 by cycle 10. A cross-cycle mechanism cannot
be assumed to repair that first-cycle bias. Next use the already-tested
positive accumulated-strain-dependent saturation rule on branch A only,
with ratio .8/.5 and rate 100/1000 (plus unchanged baseline), while holding
all fitted coefficients fixed. This small mechanism discriminator precedes
any further joint identification; do not tune against later cycles.

All four saturation-memory grid points worsen early cost (0.003844–0.005686)
and do not reduce RF/SLR third-cycle peaks. Preserve this negative result.
Next discriminate dissipative resistance from kinematic saturation: set only
A strength to `k0*(r+(1-r)*exp(-b*p_A))`, r=.7/.3, b=100/1000, plus unchanged
baseline. Positive strength is retained and effective-stress flow dissipation
remains nonnegative; the specified stored energy does not change. This is a
continuous strain-history law, never a cycle counter or protocol correction.
It tests the direction of a dissipative cyclic-softening mechanism before
considering additional fitting coordinates.

## Matched identification-objective control

The four resistance-softening grid points reduce third-cycle RF/SLR peaks
but damage SLC peaks and raise early cost (0.003833–0.007625). Do not promote
or jointly fit these memory parameters yet. First separate a model limitation
from the existing least-squares objective, which can improve average sampled
response while tolerating peak and secant violations.

Run matched six-coordinate backbone and eight-coordinate recoverable models
from the same archived backbone, with the previously listed bounds, first
three cycles only, and 20 optimizer evaluations. Use the unchanged response
screen's resolved error/limit values as residuals squared (L4 objective).
This deterministic tradeoff emphasizes large normalized violations; it is not
a measurement likelihood, confidence interval or proof of feasibility.
Unresolved secants remain explicitly unresolved and do not contribute false
passes. Failed coverage/path checks reject the candidate, not a finite penalty.
Retain the original sampled-response experiments unchanged. Later-cycle data
and acceptance tolerances never enter the objective selection or optimization.

Multi-objective identification is established for multiphase histories
([2014 study](https://www.sciencedirect.com/science/article/abs/pii/S0045794914000868));
the specific L4 tolerance scaling here is our declared diagnostic design, not
that paper's algorithm or an industrial standard.

Both matched L4 runs exhaust the predeclared 20-evaluation budget without
convergence. The backbone/recoverable costs are 2839.77/1765.09; maximum
normalized early violations are RF 2.156/1.970, SLR 6.288/5.574, SLC
3.975/3.559. All three SLC late creep secants remain unresolved. These are
unfinished optimization candidates, not fitted optima or frozen validation.

## Full-coordinate sensitivity discriminator

Before adding another state, audit twenty positive coefficients: both branches'
strength, flow, exponent-minus-one, two hardening coefficients, two saturation
coefficients, two recovery rates, and recoverable modulus/time. Keep E fixed.
Use dimensionless log10 perturbations, the same three early protocols and
sampled-response residuals, with central steps 0.001 and 0.0005. Their Jacobians
differ by 0.0448% in relative Frobenius norm. A relative singular-value cutoff
of 1e-3 gives rank 16 versus 15: this borderline rank is not robust and must
not be sold as statistical identifiability. B strength, previously frozen,
has column norm 1.247, versus A recovery norms 6.15e-6 and 2.37e-4.

Next retain only the leading eight singular directions of the finer audit,
requiring projector distance <=0.02 between difference scales and overall
Jacobian difference <=0.002. Freeze the other directions. Identify all three
early paths jointly, with a total log-coordinate radius bounded by 0.3
(box half-width 0.3/sqrt(8)), central derivatives 0.0005 and at most twenty
optimizer evaluations. The fixed original sampled-response objective is used;
all unchanged response gates are checked separately. This bounded experiment
tests whether the previously frozen constitutive directions excluded a useful
solution; it is not evidence that a new mechanism or conditional fitting is
unnecessary. No late-cycle observations select the basis or its bounds.

Ill-conditioning in Chaboche-type identification is documented in a
[2024 copper-alloy study](https://link.springer.com/article/10.1007/s11340-024-01057-x).
That paper motivates distinguishing response fit from parameter recovery;
its material constants and surrogate method are not used here.

The eight-direction fit converges in six optimizer evaluations, cost 0.00347497,
with one active bound and projector distance 0.001055. Frozen ten-cycle
independent integration and waveform refinement pass all states, but response
screens still fail. Tenth peak errors are RF +8.34%, SLR +11.98%, SLC -0.71%.
The first SLC hold stress is 122.300 MPa versus 121.70 observed. Further local
backbone tuning is not enough evidence to claim recovery of the missing physics.

## Two-time recoverability discriminator

The cited Kazemi publisher formulation excerpt explicitly describes two
Kelvin-Voigt components in series, not only one. Extend the original scalar
reference to two positive moduli/times, with independent recoverable strains,
summed stored energies H_i*a_i^2/2 and dissipations H_i*tau_i*adot_i^2.
This remains our minimal scalar discriminator, not their published tensor
model or calibrated 10%Cr card. The new 12-state identity rejects old 11-state
restart input. Before data use, verify exact two-exponential stress-controlled
creep/recovery, equal-time reduction to one harmonic-compliance element,
active-flow power identity, split continuation and Radau/BDF agreement.

With the sensitive-direction frozen card unchanged, add slow H/E in {20,100}
and time in {20,200,2000} seconds, retaining the existing fast element and
an unchanged one-element control. These six scales are a declared mechanism
grid, not measured coefficients or a protocol-specific fit. Evaluate early
three cycles of all protocols and preserve every response screen. This asks
whether an independent hold-scale recoverable mode improves the tradeoff;
it does not assume that it explains long-term cyclic softening or fatigue life.

All six fixed probes complete. H/E=20, tau=200 s reduces early cost from
0.00347497 to 0.00341878 and failed resolved checks from RF/SLR/SLC 4/12/14
to 4/9/13. This small improvement is not acceptance. Next jointly identify
the same eight sensitive directions plus the slow modulus/time, using the
same early data and weights. Start from that declared probe; slow coordinates
are bounded to +/-0.5 decades, the existing sensitive-coordinate box remains
unchanged. Before fitting require the ten-coordinate initial central Jacobian
smallest singular value >=1e-5 and condition <=1e4. Allow at most twenty
optimizer evaluations; retain bound hits and failed fits. A converged fit
must then pass frozen full-state Radau/BDF and waveform subdivision, including
both recoverable states, before its unchanged response gates are interpreted.

An additional non-fitted slope audit uses exactly the recorded reversal-window
indices, expands only output observations (never the input path), and reports
slopes against both model strain and recorded strain. The single-component
sensitive card predicts about 148 GPa in most initial reversal windows versus
roughly 150–160 GPa observed. Some windows contain prior plasticity and have
lower apparent slopes. This does not justify replacing the instantaneous
elastic modulus by a fitted secant, nor does it establish a new acceptance
tolerance. Keep it alongside stress/strain path and hold checks to expose any
tradeoff between recovered hold stress and ramp stiffness.

`tools.p92_response_panels` produces source-aligned cycle-three hysteresis and
first-hold panels from a frozen record. Curves use recorded/predicted samples,
without smoothing. The first-hold time axis is symmetric-log with a one-second
linear region; its label is elapsed physical time, not a load factor.

Execution incident: the first ten-cycle spectrum verification finished its
numerical calculations but rejected publication because its driver was
refactored during execution. The before/after code-identity guard correctly
raised `Frozen card or driver changed during replay`. No result file from
that attempt is admitted. Repeat the complete verification after freezing the
decoder and driver; this incident is not a constitutive integration failure.

The repeated spectrum replay passes all-state numerical checks. Joint cost is
0.00257608 (nine optimizer evaluations, three active sensitive-direction bounds).
SLR peak errors at cycles 1/3/10 are +2.74/+5.84/+10.18%; the first long hold
ends at 44.45 MPa versus 45.30 observed. First SLC hold stress is 123.65 MPa
versus 121.70, but subsequent positive hold ends at 159.02 versus 174.88 MPa.
All full response gates still fail. This separates improved initial response
from the remaining cyclic evolution and post-creep reload/relaxation deficit.

Next test positive dissipative resistance evolution on each flow branch:
`k_j(p_j)=k_j0*(r_j+(1-r_j)*exp(-100*p_j))`. The rate 100 is fixed before
identification; ratios are bounded 0.5–1. This extends the already tested
single-branch resistance hypothesis to the two physical flow branches, not
to protocol-specific parameters. Each branch uses its own accumulated strain;
stored energy is unchanged and effective-stress flow remains dissipative.
First verify zero-evolution equivalence, signed power identity and independent
integration. Probe r_A/r_B independently at 0.8 with the spectrum card frozen
before deciding which amplitudes the early histories can constrain. Do not
add a cycle counter, damage multiplier or experimental state reset.

All four branch-resistance probes completed. Relative to cost 0.00257608,
A-only resistance softening gives 0.00259833, B-only 0.00303291, and both
0.00304548. B softening increases the SLR third-peak error from 15.92 to
17.86 MPa and worsens the SLC peak substantially. These are negative controls,
not parameter seeds for an unrestricted optimizer. A-only softening reduces
the SLR third-peak error to 15.01 MPa but worsens SLC; it does not resolve the
shared-history constraint.

## Kinematic-modulus evolution: declared discriminator, not a fitted fix

The publisher abstract of [the 2021 P92 study](https://www.sciencedirect.com/science/article/abs/pii/S0997753820305118)
explicitly reports accumulated-inelastic-strain-dependent kinematic evolution;
its experiments are at 650 C, not the BAM cohort. The full equations were not
retrieved. The open [2020 P92 paper](https://www.mdpi.com/2075-4701/10/3/307)
also distinguishes changing hold relaxation from changing subsequent peak
evolution. Neither establishes the following law or validates it for BAM.

Test an original minimal thermodynamically consistent scalar discriminator:
`s_j(p)=r_j+(1-r_j)*exp(-b_j*p)`, `H_ji=H_ji0*s_j`,
`a_ji=a_ji0*s_j`, keeping the yield resistance fixed. With internal strain
`alpha_ji=X_ji/H_ji`, retain the existing alpha evolution. Consequently
`Xdot=H*flow-decay*X+(Hdot/H)*X`. The changing stored energy
`X^2/(2H)` requires the additional dissipation `-Hdot*X^2/(2H^2)`.
Since Hdot<=0, the extra contribution is nonnegative. Omitting either the
stress-rate or energy term would be inconsistent. Instantaneous elastic E is
unchanged; this is neither elastic damage nor a lifetime law.

Before data replay, require zero-evolution equivalence, a finite-difference
stored-energy derivative including H(p), signed symmetry, independent Radau/
BDF and split-path checks. Probe branch A/B independently with r=0.8 and
b=100, using the same frozen spectrum card, early cycles and existing gates.
No optimization or later-cycle selection is part of this first discriminator.

The twelve spectrum tests pass, including the changing-H energy derivative.
The fixed kinematic-modulus probes give costs 0.00269527 (A), 0.00257969 (B),
and 0.00271133 (both), all above the unchanged 0.00257608. A reduces SLR
third-peak error from 15.92 to 14.90 MPa, but increases the magnitude of the
SLC third-peak error from 3.19 to 9.81 MPa. Do not promote or jointly fit this
mechanism on this evidence. Keep its verified zero-default research option
and negative-control record separate from the accepted software material API.

## Reversal-sensitive dynamic recovery

Read the complete open [Li et al. 2021 formulation](https://www.dora.lib4ri.ch/empa/islandora/object/empa%3A24827/datastream/PDF/Li-2021-The_investigation_of_primary_creep-%28published_version%29.pdf),
in particular scalar equations 13–17. It distinguishes signed viscoplastic
strain magnitude in kinematic evolution from accumulated strain in drag
evolution. Its 10%Cr, 600 C card is not transferable to BAM P92 at 620 C.
We test that distinction alone, not reproduce their full shared-backstress/
drag model: multiply each existing positive dynamic recovery coefficient by
`m=1+(r-1)*(1-exp(-b*z))`, where z is either absolute signed total flow strain
or the sum of branch accumulated strains. H, a, resistance, static recovery,
and recoverable spectrum remain frozen. Both choices stay dissipative.

A new thirteenth state integrates signed total flow from zero. Do not derive
it from experimental strain minus elastic strain: the measured initial strain
offset differs across specimens. For signed memory a load reversal can undo
part of this history, whereas accumulated memory cannot. No event-triggered
reset or protocol label enters the rates. Verify the extra state, its bound by
total accumulated flow, power balance, zero-effect limit, reversal symmetry,
split continuation and independent integration before data replay.

Prespecify r_A/r_B = (1,1), (2,1), (1,2), (0.5,1), (1,0.5), b=1000;
compare signed versus accumulated history on all first-three-cycle paths.
This is a matched mechanism discriminator, not an inferred microstructure
measurement. Keep the same response gates and no late-cycle fitting.

The nineteen joint spectrum/memory tests pass. The first synthetic reversal
fixture initially did not drive the signed flow through zero; extending that
fixture's reverse ramp corrected the test stimulus without changing tolerances.
All nine measured-path probes complete. The best signed probe lowers aggregate
cost by only 0.071%, while increasing the SLR third-peak error from 15.92 to
16.56 MPa. The post-creep SLC positive-hold endpoint remains about 159 MPa
against 174.88 MPa measured. Do not select this as a successful mechanism or
start an unrestricted fit on the tiny aggregate-cost reduction.
