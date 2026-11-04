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
