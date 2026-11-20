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
