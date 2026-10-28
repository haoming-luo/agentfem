# Thermal inelasticity and continuation: bounded development plan

This record contains generic software requirements only. No private proposal,
personal information, proprietary material data or customer geometry is retained.

## Order and gates

1. Maintenance and ordinary use: PR #132 merged; add a checked zero-copy XDMF
   continuation view. A view is not a restart or a scientific continuity proof.
   Fix complete-Step restoration if corrupt later payloads can leave earlier
   fields changed. Keep existing checkpoints and core state ownership.
2. Thermal inelasticity: reuse accepted temperature histories and the physical
   creep clock; add the monotone Sinh branch to the existing creep integrator.
   Then compose creep and cyclic plasticity within one local increment, with
   one accepted state and one complete consistent tangent. Stage structural
   admission only after local tests, combined state rollback/restart and energy
   bookkeeping pass. Do not use the rate-independent provider with artificial
   normalized time to bypass physical-time semantics.
3. Damage/life: start strain-life postprocessing with explicit strain convention,
   reviewed curve identity and units; keep lifetime assessment distinct from
   constitutive stiffness degradation. Global K-R damage and mesh regularization
   remain separate future gates. No crack-length or industrial-life claim.

## Simultaneous creep/plasticity design

Retain the existing complete J2/Chaboche return map `P` and its tangent `Cp`.
At fixed previously accepted histories solve the six-component stress equation

`R(sigma) = sigma - P(eps_mech - CE_old - dt flow(sigma), plastic_old) = 0`.

The Mandel basis is orthonormal: no hidden engineering-shear scaling. With
`L = d flow / d sigma`, the local Jacobian is `I + dt Cp L`; the consistent
strain tangent solves `(I + dt Cp L) C = Cp`. Backstress makes the two flow
directions generally different, so no common radial direction is assumed.
Each line-search probe re-evaluates from *old* plastic history. Failure returns
no new state. This condensed simultaneous solve is not a one-pass split.

Initial boundary: small strain; constant shared elastic moduli; isotropic J2
or Armstrong--Frederick/Chaboche; power-law or Sinh creep; optional shared secant
thermal expansion and normalized Arrhenius power-law rate. Temperature-dependent
plastic parameters, static backstress recovery, finite strain and contact are
not inferred. A material-point prototype does not imply global admission.

Global integration must preserve separate CE/CEEQ and PE/PEEQ/backstress fields,
bind every field to checkpoint identity, and include plastic hardening storage,
material dissipation and backward-Euler contributions without calling plastic
work stored energy. Do not graft the coupled state onto a CE-only checkpoint.

## References and independent checks

- [MOOSE simultaneous creep/plasticity](https://mooseframework.inl.gov/source/materials/ComputeCreepPlasticityStress.html)
  motivates solving both mechanisms within one increment. Its default finite
  incremental formulation is not the total-small-strain formulation used here.
- [Abaqus creep integration theory](https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-ratedepplast.htm)
  distinguishes rate integration from instantaneous plastic response.
- Tests use an independent scalar root for coaxial isotropic hardening, exact
  n=1 continuous Sinh relaxation under refinement, nonproportional Chaboche
  histories, fixed-old-state tangent differences, zero-mechanism limits, and
  forced failures. None is an independent commercial structural comparison.

No new full composite runtime, MPI composition, release or installer claim is
part of this record. Local targeted counts are recorded at each verified commit;
skipped rank-specific tests are not counted as passed.

## Integrated slice

The serial homogeneous 3D composition now reuses the existing implicit creep
Step rather than creating a second nonlinear solver. The joined quadrature
transaction saves CE/CEEQ and PE/PEEQ/all backstresses. Accepted plastic work,
hardening storage, material loss and discrete contributions are separate.
Both serial and portable file formats have serial interruption/continuation
checks; MPI execution remains rejected. The ordinary
`examples/thermal_cyclic_creep` project completes through `agentfem run` with
computed (not independently validated) result status.

Promotion remains gated on broader structural time/mesh convergence and an
independent multiaxial reference. Static backstress recovery and temperature-
dependent plastic parameters are not implemented. The mechanical energy
ledger excludes imposed thermal transfer; the isolated plastic energy
decomposition is not promoted into a full thermal first-law claim.

Local verification: 156 targeted tests passed in 100.98 s across coupled
creep/plasticity, Sinh creep, platform/constitutive models, checkpoint
capabilities and the generic small-strain Step. The ordinary source CLI example
completed 16 accepted increments in approximately 4.4 s; this is a workflow
smoke check, not a speed benchmark or an installed-wheel test.

Follow-up checks add an independent continuous uniaxial solution for linear
creep plus isotropic plasticity: 16/32/64-step error ratios verify first-order
time convergence. A 1/2-subdivision homogeneous patch gives matching reactions
and state; it is deliberately not called nonuniform spatial convergence.
The broad affected static/dynamic/operator/stateful selection passed 193 tests
in 109.69 s. It exposed and now guards against an old duck-typing hazard:
having E/nu must not advertise a history material as stateless elasticity.
Explicit supplied-operator workflows remain distinct and supported.

The ordinary project now solves homogeneous insulated heat first, transfers
accepted temperature frames on the same mesh, and completes the 16-increment
mechanical path in approximately 4.2 s. This records a source-CLI workflow
check only, not an installed-wheel or thermal feedback/first-law validation.
The project declares its required public API, and `agentfem check --runtime`
passes. A separate final selection of flow/platform, continuation output and
assessment regressions passed 178 tests in 11.20 s. Counts overlap earlier
selections and must not be summed as unique test coverage. Ruff, generated
documentation checks, strict MkDocs and the 57-card knowledge check pass.
Local REUSE lint was unavailable because that tool is absent in the numerical
environment; repository CI retains that gate.

## First damage/life assessment slice

The bounded scalar Coffin-Manson-Basquin postprocessor adds explicit
calibration range, source, cycle/reversal convention and an uncorrected-mean
decision. It reuses rainflow and the existing creep/fatigue interaction;
no stiffness damage or life-driven state mutation is introduced. All 68 tests
in the strain-life, existing fatigue/fracture and assessment selection passed.
The [public contract](../strain_life_assessment.md) explains scope, theory
source and the risk of dwell double-counting. This starts stage three; it does
not close global damage, regularization, multiaxial life or material calibration.
