# P92 recovery and softening discriminator

## Question and sources

The frozen early-cycle fit fails SLC at cycle 5. This experiment separates
backstress time recovery, isotropic time evolution, and the negative
accumulation-dependent isotropic targets. It changes no solver, data split or
acceptance threshold and performs **no parameter optimization**.

[Zhang et al., Metals 2020, equations 9–13](https://doi.org/10.3390/met10030307)
distinguish backstress recovery from isotropic evolution. Their calibration
uses hysteresis, cyclic peaks and dwell relaxation in stages. Their 600/650 C
strain-controlled evidence is not validation of BAM's 620 C mixed-control
tests. The paper explicitly notes the added complexity of stress control.
[NEML's Chaboche documentation](https://neml.readthedocs.io/en/dev/hardening/non/chaboche.html)
likewise separates strain- and time-dependent terms, with an opposite
hardening-stress sign convention. We retain our existing convention rather
than copying coefficients across formulations.

Our two branches remain a research hypothesis, not an exact reproduction of
a published two-mechanism law. In particular, the previous six-parameter fit
did not identify separate isotropic time-recovery coefficients for each branch.

## Prespecified experiment

Six variants, each replayed on RF/SLR/SLC with Radau at 1e-8 and BDF at 1e-10:

1. unchanged frozen baseline;
2. fast branch static backstress recovery off;
3. fast branch time-dependent isotropic recovery off;
4. slow branch static backstress recovery off;
5. slow branch time-dependent isotropic recovery off;
6. both branches' negative accumulation targets removed (`Q=H=0`).

The last intervention retains `-b*R*p_dot`; it does not freeze isotropic state.
All other parameters, initial-state assumptions and measured control paths
remain unchanged. These are diagnostic ablations, not recalibrated materials.
Partial failed cycles remain excluded, and no reverse ramp is invented.

The signed rate ledger reconstructs the implemented equations exactly:
backstress hardening + dynamic recovery + static recovery, and isotropic
accumulation + time evolution. Its units are MPa/s. Endpoint flow rates and
branch accumulated-strain increments are not integrated signed creep shares,
stored energy or dissipated energy.

## Results

RF and SLR RMS below use identical cycles 4–10 and both extrema.

| Variant | RF RMS, MPa | SLR RMS, MPa | First SLC failed cycle |
|---|---:|---:|---:|
| Frozen baseline | 16.485 | 26.706 | 5 |
| Fast static backstress off | 16.487 | 26.708 | 5 |
| Fast time-isotropic off | 5.654 | 17.296 | 7 |
| Slow static backstress off | 16.825 | 26.553 | 9 |
| Slow time-isotropic off | 19.577 | 30.821 | None through 10 |
| Both negative accumulation targets removed | 15.740 | 26.106 | 5 |

All 18 trajectories agree on the failure cycle between the two integrators.
The maximum stress-like state difference is 3.28e-6 MPa and the maximum
strain-like state difference is 1.85e-10, within the existing numerical
screening limits. This is numerical agreement, not experimental admission.

In the baseline's fifth SLC hold, the slow branch's strength `k+R` has fallen
to 62.34 MPa. Its exit flow rate is 4.30e-7/s versus the fast branch's
5.32e-8/s. Disabling slow time-isotropic evolution keeps that branch's strength
at 122.13 MPa and reduces its exit flow to 2.12e-9/s. The total fifth-hold strain
increment falls from 0.002097 to 0.000966. These are counterfactual diagnostics,
not proof that the real material has no time-dependent isotropic evolution.

The direction across cycles also matters. The last recorded SLC hold strains
are 0.00106, 0.00099 and 0.00096 in cycles 1, 3 and 5, respectively (approximately
1794 s into each 1800 s hold, not exact endpoint measurements). The baseline
predicts increasing endpoint increments, starting at only 0.000157 in cycle 1.
The slow-time-isotropic ablation still predicts only 0.000120 in cycle 1.
Thus preventing the late envelope violation does not fix the early transient.
The same ablation worsens RF and SLR: **none is admitted as a replacement**.

## Next bounded identification decision

Do not add a broad parameter sweep or accept the ten-cycle ablation as a fix.
The nearly inactive fast static-recovery scale from the previous fit is a
candidate to freeze; replace that fitting direction with a slow isotropic
time-recovery scale, retaining six unknowns rather than increasing dimension.
Also inspect fast isotropic time recovery because its ablation improves RF/SLR.
Before fitting, retain separate early-hold strain and stress residual reports,
so a smaller aggregate objective cannot conceal a poor first hold.

Use the same early cycles 1–3, declare bounds before optimization, freeze once,
then check subsequent cycles and switch-bracket sensitivity. All inspected
cycles remain development data. Independent specimen validation, a supported
initial-state/preconditioning history and a compatible life law remain separate
gates. Do not invent a thermal soak or initial internal state to repair the fit:
the reviewed BAM data description does not specify that history sufficiently.

## Evidence and reproduction

```sh
python -m tools.p92_recovery_discriminator evidence/creep/2026-11-17-measured-controls-fit.json /path/to/BAM recovery-screen.json
```

`evidence/creep/2026-11-17-recovery-discriminator.json` retains parameter cards,
source identities, all protocol comparisons, signed entry/exit rate ledgers and
independent-integration differences. No raw personal metadata is included.
The entire experiment is one local development batch and one targeted CI gate.
#196 was merged after all latest applicable CI passed; no version was released.

## Branch-time identification follow-up

PR #197 remains open by explicit user request, even when CI is green. Work is
batched on that PR until milestone review; no automatic merge or release.

The next experiment replaces the weak fast-static-recovery coordinate with a
slow-branch isotropic time-recovery scale. A second bounded experiment also
fixes the weak fast strength margin and replaces that coordinate with a fast
isotropic time-recovery scale. Both retain **six unknowns**, start from the
same previous early-cycle parameter card, and exactly reproduce it at unit
time scales. Neither adds later cycles to the objective. Bounds on each time
scale are 1e-5 to 10; these are search bounds, not measured physical limits.
All other coordinate bounds, objective weights and integration checks remain.

Each hold now reports stress and strain RMS, last sampled prediction versus
observation, and an early secant over recorded points within the first 10 s
(or the first tenth of a shorter hold). These are finite-interval secants,
not noise-free instantaneous rates. During imposed constant stress, observed
stress fluctuations diagnose the controller approximation; they are not a
constitutive stress-relaxation target. Sample indices and relative times are
retained. The diagnostics do not change fitting weights or accept a model.

The single slow-time fit converges in nine optimizer / 81 residual evaluations,
with cost 0.02428884. Its time scale is approximately 1.045e-5 and the fast
strength-margin coordinate hits its lower bound; weak identifiability remains.
Frozen SLC reaches ten cycles under measured, nominal and both switch-bracket
paths. However, at the first hold's last recorded sample its strain is 0.000407
versus observed 0.00106, and stress is 136.52 versus 121.70 MPa. RF/SLR/SLC
extrema RMS on cycles 4–10 are 17.24/30.92/40.50 MPa. This closes the previous
envelope failure for that path, not the response-accuracy or engineering gate.

The dual-time fit converges in 18 optimizer / 186 residual evaluations, with
cost 0.02224865. Fast time recovery reaches its upper search bound (10), slow
time recovery its lower bound (1e-5), and the slow second-backstress scale its
lower bound. The Jacobian singular-value ratio falls from about 6.04e6 in the
single-time experiment to 9.39e4, but this is not a confidence interval or proof
of identifiability. No bounds were expanded after seeing these results.

Frozen RF/SLR/SLC cycles 4–10 extrema RMS are 13.76/15.32/42.97 MPa; cycle-10
peak errors are -4.14%/-3.38%/-22.84%. All three complete ten cycles and pass
numerical refinement; SLC also completes the nominal and both timing-bracket
paths. These are development checks, not an independent specimen validation.

The first SLC hold remains decisive: last sampled strain is 0.000596 versus
0.00106 observed, stress 134.07 versus 121.70 MPa, and early strain secant
5.35e-6/s versus 3.57e-5/s. By cycle 10 the sampled strain is 0.001571 versus
0.00087. Thus the early transient and cycle trend are still wrong despite
removing the fifth-cycle envelope failure. Engineering and life gates remain
closed. Next discriminate primary creep and reversal/backstress history using
the early observations; do not silently add later data or alter acceptance.

The two `2026-11-17-{slow,both}-time-identification.json` evidence files retain
six-coordinate names, frozen cards and hashes, hold samples, failed acceptance
flags, and numerical/timing checks. Reproduce the bounded experiments with:

```sh
python -m tools.p92_measured_fit evidence/creep/2026-11-16-bam-positive-strength.json evidence/creep/2026-11-17-measured-controls-fit.json /path/to/BAM --parameter-direction slow_time_isotropic --max-nfev 40 --output slow-time.json
python -m tools.p92_measured_fit evidence/creep/2026-11-16-bam-positive-strength.json evidence/creep/2026-11-17-measured-controls-fit.json /path/to/BAM --parameter-direction both_time_isotropic --max-nfev 40 --output both-time.json
```
