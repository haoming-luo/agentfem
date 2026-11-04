# Static recovery: mechanism discriminator before material admission

## Decision and scope

Test whether time-dependent strength loss can reactivate plastic flow during
a strain hold. This is a synthetic uniaxial tensile material-point experiment,
not a new FEM solver, a fitted P92 law or a life prediction. Keep it under
`tools/`, outside the runtime constitutive catalogue.

NEML's [ChabocheVoceRecovery description](https://neml.readthedocs.io/en/dev/hardening/non/special_chaboche.html)
distinguishes flow-driven hardening from static recovery. Its stress signs and
backstress normalization differ from common conventions; no parameters or
tensor implementation are copied here. The discriminator uses a simpler,
explicitly stated positive tensile strength convention.

## Model and exact solution

Let `sigma = E (epsilon - p)`, yield `sigma <= Y + R`, and
`R_dot = H p_dot - k R`, with positive E, H, Y and nonnegative k.
Initial tensile stress satisfies `Y < sigma0 < Y + R0`. Hold total strain.
Before plastic activation, stress is constant and `R = R0 exp(-k t)`.
Activation occurs at `ta = log(R0 / (sigma0 - Y)) / k`.
After activation, consistency gives
`R_dot = -k E R / (E + H)`, `sigma = Y + R`, and
`p - p0 = (sigma0 - sigma) / E`. For k=0, no hold evolution occurs.

With stored energy `psi = sigma^2/(2E) + R^2/(2H)`, the dissipation rate is
`Y p_dot + k R^2/H >= 0` on the tensile branch. This is a model-specific
thermodynamic statement, not a general creep-fatigue energy law.

## Numerical experiment and admission gates

Backward Euler eliminates `R_new = (R_old + H delta_p)/(1 + k dt)`.
Use the elastic predictor if its yield function is nonpositive; otherwise
solve the scalar consistency equation exactly. Compare time refinements with
the independent closed-form hold, and verify fixed-old-state tangent, yield
admissibility, nonnegative plastic increments and untouched input state.

This isolates a necessary mechanism distinction: strength may decay while
stress remains unchanged; stress relaxation begins only after yield
reactivation. Static recovery is therefore not automatically a successful
explanation of the P92 low-stress hold. Real-data admission still needs full
paths, temperature conventions, multiaxial formulation, calibrated parameters
and a new independent validation condition. Do not retune on SLR and call it
unseen validation.

## Executed evidence

`python tools/static_recovery_discriminator.py` uses E=200000, H=10000,
Y=100, initial stress=150, initial strength=100 and k=0.02 in consistent
stress/time units. Exact activation is 34.657359. Maximum stress errors for
100/200/400/800 increments are 0.347572/0.174461/0.087414/0.043750;
halving time increments halves the error. Twelve tests pass, including
fixed-old-state tangents, inactive holds, energy decay, invalid time inputs,
initially active states and scalar exact queries. No experimental fit occurs.

## Public material candidate for the next gate

Zhang et al., [Metals 10 (2020) 307](https://doi.org/10.3390/met10030307),
provide a modified unified viscoplastic model, parameters at 600 and 650 C,
and tensile-dwell comparisons. This is a useful next reproduction source,
not a parameter card for the BAM 620 C dataset. Its nonlinear kinematic and
isotropic recovery/softening terms exceed this simple discriminator.

Before implementation, retrieve and check the complete equations and loading
waveform, pin the parameter transcription, and establish quantitative reference
data or digitization uncertainty. Then compare independent time integrators,
material tangents and structural reactions. A paper's reported agreement is
not AgentFEM validation; interpolating its parameters to another material batch
and temperature is a separate modelling assumption. Keep this as a proposed
source until those checks are executable.

## Published-equation reproduction, separate from validation

The version-2 PDF was subsequently downloaded and its equations (9)--(13),
Table 4 and loading description checked against rendered pages 9--10 and
page 3. SHA-256:
`c00a71167969edaab6a1b52641140cb5a3f241e82a90b74b464f293f46fcd604`.
`tools/p92_published_recovery_reference.py` transcribes the uniaxial equations
and both parameter sets, without fitting or temperature interpolation. The
startup is declared explicitly: zero state to positive half-range; successive
tensile dwells are separated by compression and tension ramps.

The reference remains outside the runtime package. Radau and BDF agree on
two-cycle histories at both temperatures and 60/600 s dwells within 0.001 MPa
at the sampled hold stresses. Seven source/equation/integrator tests plus the
twelve analytical discriminator tests pass. Two integrators sharing one rate
function check integration, not independent transcription or experiment.

A 200-cycle, 600 C, 0.8% range, 60 s dwell run gives hold-end stresses
239.8993, 229.4435 and 186.0506 MPa at cycles 1, 20 and 200. Figures 11--14
were visually inspected, but no numerical experimental acceptance is claimed.
The article does not supply a machine-readable comparison here; startup and
plot digitization uncertainty must be resolved before declaring reproduction
of its curves. The public reference now supplies a useful independent local
target for a future material provider, not a substitute for that provider or
for structural verification.

The same reference now includes a local backward-Euler corrector solving all
five variables simultaneously. Its axial algorithmic tangent comes from the
implicit-function Jacobian, not a frozen-history elastic modulus. Analytical
rate derivatives pass centered differences in tensile, compressive and
inactive states at both temperatures; evolved-state tangents pass independent
strain differences. Hold-only and full loading/reversal/hold refinements
converge toward the continuous reference. The published-model suite passes
18 tests (plus 12 separate analytical-discriminator tests). This is still a
uniaxial local reference, not an unverified automatic 3D embedding.

## Long-path integration check

The `--published-campaign` option runs the eight Table 2 conditions to the
integer floor of their reported half-lives. Reported lives select duration;
they are explicitly inputs, never predicted fatigue lives. Each integration
method covered 3267 cycles, retaining the complete cycle-to-cycle state.
Radau and BDF maximum sampled hold-stress differences were:

| Temperature, C | Strain range | Dwell, s | Cycles | Maximum difference, MPa |
| --- | --- | --- | --- | --- |
| 600 | 0.008 | 60 | 651 | 1.968e-5 |
| 600 | 0.008 | 180 | 595 | 1.366e-5 |
| 600 | 0.008 | 600 | 337 | 1.171e-5 |
| 600 | 0.012 | 180 | 270 | 1.038e-5 |
| 650 | 0.008 | 60 | 435 | 1.505e-5 |
| 650 | 0.008 | 180 | 426 | 1.029e-5 |
| 650 | 0.008 | 600 | 312 | 8.825e-6 |
| 650 | 0.012 | 180 | 241 | 9.560e-6 |

Commands: run `python tools/p92_published_recovery_reference.py
--published-campaign --method Radau --output <new-json-path>` and repeat with
`--method BDF` and a different output path. Both use rtol=1e-8. The executed
tool SHA-256 was
`38d20a1530c9391f7a98c9e226de0c74928a820c6fb7b342a993539d0da51a55`.
Local totals were 166.54 s and 119.16 s respectively, not FEM performance
evidence. Including new-file-only evidence output checks, 31 targeted tests
pass. Experimental data comparison and structural integration remain open.
