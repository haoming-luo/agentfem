# P92 tensor admission: a research gate, not a material card

## Source and decision

The version-2 PDF of [Zhang et al., Metals 10 (2020) 307](https://doi.org/10.3390/met10030307)
was visually rechecked on printed pages 7-10. Table 3 and equations 9-13 use
scalar stress, backstress and inelastic strain; Table 4 supplies E, but not a
Poisson ratio. Its SHA-256 remains
`c00a71167969edaab6a1b52641140cb5a3f241e82a90b74b464f293f46fcd604`.
The existing published-figure screen stays unchanged.

A three-dimensional extension therefore needs explicit assumptions: isotropic
elasticity with a separately supplied Poisson ratio, pressure-independent J2
flow, deviatoric backstresses and the stress-norm convention below. Those
assumptions are not additional experimental facts from the source.

## Candidate normalization

With `s = dev(sigma)`, `eta = s - X1 - X2`, define
`q = sqrt(3/2 * eta:eta)`, `v = (<q-R-k>/Z)^n`, and
`ep_dot = (3/2) * v * eta/q`, with zero flow on the inactive branch.
Use `sigma_dot = C:(strain_rate-ep_dot)` and

```text
Xi_dot = (2/3) Ci ai ep_dot - Ci v Xi - gamma(p) J(Xi)^(m-1) Xi
J(Xi) = sqrt(3/2 * Xi:Xi)
R_dot = b(Q-R)v + gamma_r(p)|Qr-R|^(mr-1)(Qr-R) + H v
p_dot = v
```

The source's scalar chi maps to `diag(2 chi/3, -chi/3, -chi/3)`, not
`diag(chi, 0, 0)`. Consequently `J(Xi)=abs(chi)` and multiplying the axial
backstress equation by 3/2 recovers the source equation exactly. Substituting
the raw Frobenius norm would multiply the static-recovery coefficient by
`(2/3)^((m-1)/2)`, a substantial change at the published exponents.

[NEML's Chaboche documentation](https://neml.readthedocs.io/en/dev/hardening/non/chaboche.html)
provides a useful implementation comparison, but explicitly uses an opposite
hardening sign convention and its own norm factors. Copying its parameters
without a convention map would not reproduce this source model.

## Numerical implementation and evidence

First verify rates: exact tension/compression/inactive reduction to the existing
scalar reference, hydrostatic independence, traceless inelastic/backstress
rates, and covariance under a constant change of Cartesian basis. The latter
is small-strain tensor covariance, not finite-rotation objectivity.

Then solve the complete coupled backward-Euler local system in Mandel stress
coordinates and an orthonormal five-component deviatoric backstress basis.
The unknown vector has 18 entries: stress(6), X1(5), X2(5), R, p. Its
algorithmic tangent must differentiate the converged update at fixed old state:
`B = solve(I-dt*df/dy, [C; 0])`, taking the first six rows. Independent strain
differences and the stress-free lateral condensation must recover the existing
uniaxial tangent. A last-substep tangent is not a complete substepped tangent.

All experiments stay in `tools/`, outside the constitutive catalogue. Effective
stress power `eta:ep_dot=q*v >= 0` is checked, but is not claimed to be the full
thermodynamic dissipation of a softening/recovery model. Stored energy and
general first-law closure are not invented.

The coupled local update is now implemented in
`tools/p92_tensor_normalization.py`. Fifty numerical tests cover the rate
reduction, analytic 18-by-18 Jacobian, complete six-direction strain tangent,
stress-free lateral condensation, stress-unit scaling, constant-basis rotation
covariance, and time refinement against independent Radau integration.

`tools/p92_tensor_material_reference.py` is a research-only local provider for
the bounded physical-time consumer (PR #150). It declares its supplied Poisson
ratio, fixed published temperature, source parameters and named state. It does
not contain a global nonlinear solver. Mandel shear scaling and component
permutation are explicitly converted to the public tensor-Voigt convention.
Temperature fields and other external fields are rejected rather than ignored.

Ten FEniCSx/provider tests verify the public tangent checker with all shear
components, rejection of unsupported fields, 600/650 Celsius homogeneous
structures with free lateral contraction against the independent scalar
backward-Euler history, restart at the start of a hold, and all reversal/hold
endpoints of a tension-hold-compression-hold-unload cycle. The expanded suite
including scalar references, physical-time and atomic-restart regressions,
and CI routing passed **171 tests**, with one dedicated two-rank admission test
skipped in the serial run (already run separately for PR #150).
This is numerical composition evidence, not a multiaxial
experimental validation or a complete fatigue simulation.

Structural refinement, nonproportional material data, temperature interpolation,
cyclic-life prediction and a production material catalogue entry remain
separate. The source does not supply a unique multiaxial model or Poisson ratio;
no claim of calibrated three-dimensional P92 behavior follows from this gate.

## Integration follow-up

PR #150 is merged. This research branch is now based on that verified main,
including the independent scalar figure screen and co-located finite-strain
history consumer; the latter is not automatically available on this small-strain
route. CI changes to the upstream scalar reference now also select the tensor
and structural tests, preventing an unnoticed dependency regression.

At both published temperatures, structural runs with 16, 32 and 64 increments
approach an independently integrated continuous-time scalar Radau reference.
Each refinement reduces final axial-stress error by at least 30% in the tested
ramp/hold path. This is time-discretization evidence for a homogeneous patch,
not spatial convergence or experimental multiaxial validation. Invalid complex,
boolean, array-valued and out-of-range Poisson assumptions are rejected at
construction.

Force-controlled 300 MPa ramp/hold structures at both temperatures also show
continued inelastic accumulation during the hold. The accumulated strain
approaches an independently integrated stress-controlled scalar Radau history
under 16/32/64 increments, with at least 25% error reduction on each refinement.
This complements the displacement-controlled relaxation checks; it is not
experimental creep-life validation.

The integrated local regression is **245 passed, 1 skipped** (the serial run
skips the dedicated two-rank admission check). Generated documentation and
targeted lint checks pass. No new release or material maturity promotion is
made by this integration.

Next gate: retain fixed published parameters and test additional full cyclic
experimental histories with explicit digitization uncertainty. Separately,
extend co-located accepted histories to the small-strain consumer before using
structural paths as life-assessment inputs. Temperature interpolation and
multiaxial calibration need independent evidence, not inferred constants.
