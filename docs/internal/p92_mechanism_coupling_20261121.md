# Minimal energetic coupling: prospective research design

## Why investigate coupling

[ANL-ART-167 (2019)](https://publications.anl.gov/anlpubs/2020/02/158587.pdf)
discusses Grade 91 inelastic model qualification and proposed ASME guidance,
not a current P92 standard. It emphasizes cyclic hysteresis, holds, ratcheting,
creep/plasticity interaction, heat variability and independent validation.
Long-term and multiaxial evidence cannot be inferred from short uniaxial tests.
It also distinguishes the use of constitutive softening from a separate damage
assessment. Thus a damage variable is not automatically the cure for early
P92 response mismatch, and agreement with three protocols is not code approval.

[Wolff, Bohm and Taleb (2011)](https://journals.ub.ovgu.de/index.php/techmech/article/download/766/743)
derives thermodynamically consistent two-mechanism classes with coupled
kinematic energies and evolution. It motivates checking coupling and its
dissipation explicitly, not copying a coupling coefficient into existing
backstress equations. Its mathematical examples do not validate P92.

## Independent scalar construction

For each existing backstress component i, retain conjugate forces
`x_i=(X_Ai,X_Bi)` and define
`C_i=[[H_Ai,kappa_i*sqrt(H_Ai*H_Bi)],
      [kappa_i*sqrt(H_Ai*H_Bi),H_Bi]]`, with abs(kappa_i)<1.
The quadratic energy is `0.5*x_i.T*inverse(C_i)*x_i`. This positive-definite
matrix changes the physical meaning of internal states; use a distinct model
identity and never reinterpret an existing restart as a coupled state.

Each branch keeps its original flow law with effective stress
`sigma-sum(X_branch)`. Write the internal strain rate as
`qdot_ji=f_j-d_ji*X_ji/H_ji`, where d is the nonnegative existing dynamic plus
static recovery coefficient. Evolve `xdot_i=C_i*qdot_i`. Keep the two Kelvin
components unchanged. All coefficients are constant in this first construction;
reject evolving moduli/saturation histories rather than omit their derivatives.

The energy derivative is exactly `sum_i x_i.T*qdot_i`. Therefore dissipation
is `sum_j (sigma-sum_i X_ji)*f_j + sum_ji d_ji*X_ji^2/H_ji` plus Kelvin
dissipation. Every term is nonnegative for the retained monotone flow laws.
This is our isothermal scalar derivation, not a claim of reproducing every
equation in the cited paper. It is not a finite-strain material or a life law.

## Verification and experiment

Require zero-coupling rate/energy/trajectory equivalence; sign symmetry;
branch permutation; finite-difference energy derivative; inactive-flow linear
matrix-exponential agreement; independent Radau/BDF and split continuation.
The old uncoupled rectangular backstress bound is not valid with coupling and
must not be reused. Nonnegative energy/dissipation and state identities remain
checked, without clipping conjugate forces to old saturation values.

After verification, compare kappa=(0,0), (.1,.1), (.25,.25), (.5,.5) on the
same frozen spectrum and early three-protocol histories. No fit or late-cycle
selection in this discriminator. Record control-path failure as failure.
Proceed to identification only if compatible early observables support it;
keep the ongoing matched static-recovery comparison independent and unchanged.

## Fixed-card result (2026-11-21)

The independent reference passed zero-coupling equivalence, signed and branch
permutation symmetry, finite-difference power balance, inactive-flow matrix
exponential, Radau/BDF and split continuation checks. These are numerical
verification, not material validation.

All four early paths completed. Their unchanged aggregate costs were
0.00257608, 0.00274245, 0.00330060 and 0.00483841 for increasing coupling.
At kappa=0.25, SLR third-cycle peak error fell from 15.915 to 13.398 MPa and
SLC first creep-hold stress moved from 123.651 to 121.015 MPa (observed
121.70). However the subsequent SLC tensile-hold endpoint only moved from
159.025 to 160.159 MPa (observed 174.88), and SLC peak error worsened.
Increasing coupling to 0.5 worsened the overall error further. Thus the
fixed-card probe does not support promoting or launching an unrestricted fit
of this coupled family. Retain the tested reference and negative evidence;
the independently planned static-recovery comparison remains the next gate.

Evidence: `evidence/creep/2026-11-21-energetic-coupling.json`. Repeated sample
index/time arrays are omitted from that compact record; source and code
identities, observed/predicted diagnostics, and response screens remain.
