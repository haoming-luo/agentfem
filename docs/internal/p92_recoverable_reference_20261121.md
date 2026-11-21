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
