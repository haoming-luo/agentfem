---
title: "Thermal stress, creep and solution procedures"
description: "Choose a numerical procedure for an AgentFEM study, transfer temperature fields and understand the implicit creep and dynamic routes."
---

# Solution Procedures, Thermal Stress, and the Creep Route

## Problem type is not a solver algorithm

`Study` answers what is being modeled: solid mechanics or heat transfer,
static or transient, dimension, and kinematic assumptions.
`SolutionProcedure` answers how it is solved:

| Physical equation | Standard route | Explicit route |
| --- | --- | --- |
| \(K u=F\) | linear solve | not applicable |
| \(R(u)=0\) | incremental Newton | not yet available |
| \(C\dot T+KT=Q\) | implicit Euler | not yet available |
| \(R(u,CE,t)=0\) | backward-Euler creep + Newton | not applicable |
| \(M\ddot u+C\dot u+Ku=F\) | Newmark or generalized-\(\alpha\) | central difference |

This keeps `second_order_dynamics` from meaning “central difference” merely
because that was the first implemented algorithm. It also keeps
`incrementation`, Newton settings, time integration, and output cadence as
separate decisions.

The implicit structural route currently assumes linear operators and
time-invariant prescribed displacement supports. Newmark uses average
acceleration by default. Generalized-\(\alpha\) derives
\(\alpha_m,\alpha_f,\beta,\gamma\) from the spectral radius at infinite
frequency and provides controllable high-frequency damping. Nonlinear
implicit dynamics and moving supports require additional residual,
linearization, and prescribed-kinematics work.

For linear implicit dynamics, the effective operator is fixed when the mesh,
spaces, mass, damping, stiffness, time step, integration parameters, and
constrained dof set remain unchanged. AgentFEM therefore uses
`operator_policy="auto"` by default: it prepares the matrix and PETSc solver
once, then assembles only the evolving right-hand side. `"reuse"` makes that
contract explicit, while `"refresh_each_step"` retains conservative repeated
assembly for externally mutated operators. A change to a guarded runtime
invariant fails closed rather than continuing with a stale matrix. Step and
result summaries record the selected policy, matrix and right-hand-side
assembly counts, solve counts, KSP iterations, and the reason for the
selection. Partial `run(until_step=...)` calls retain the prepared allocation;
terminal completion or failure closes it deterministically. Checkpoints store
scientific state rather than backend handles, so the receiving Step prepares
one new operator safely under its declared policy.

Time-dependent inputs carry an explicit invalidation contract. Built-in load
amplitudes and supported ambient-temperature histories declare that they
change only the right-hand side. A custom callback is conservative by default,
so `operator_policy="auto"` refreshes the effective matrix rather than risking
a stale solve. When the callback is known to update forcing only, declare it:

```python
from agentfem import time

update_force = time.input_update(
    update_my_constant,
    effects="right_hand_side",  # `"rhs"` is accepted as a short alias
    name="pulse_force",
    identity={"kind": "pulse_force", "revision": 1},
)

step = model.step(
    target=u,
    dt=dt,
    steps=steps,
    update_load=update_force,
)
```

Use `effects="operator"` when stiffness, mass, damping, or another bilinear
coefficient changes; use `"state"` for accepted history that can alter a
residual or tangent, and `"output"` for observation-only changes. Forcing
`operator_policy="reuse"` against an operator/state declaration is rejected
before the first solve. Step and result evidence record every declaration and
whether the plan has a restart identity. Transient checkpoint schema v5 binds
the complete plan and rejects a changed or anonymous callback before restoring
any field. A v4 archive remains readable by the AgentFEM version that created
it, but current AgentFEM does not guess whether its omitted load history is
compatible.

The same policy now governs linear implicit-Euler heat transfer. A fixed
capacity/conduction operator is prepared once while the history and source
vectors are assembled at every accepted physical time. Operator- or
state-changing inputs select per-step matrix refresh, and a bare callback takes
that conservative path. Temperature-dependent conductivity or heat capacity
uses a nonlinear residual and therefore remains per-step assembled; requesting
`operator_policy="reuse"` for that route is rejected rather than interpreted
as nonlinear Jacobian reuse.

## The useful first thermal-mechanical route

Many component analyses do not need a monolithic temperature-displacement
solve. If temperature affects stress but deformation and dissipation do not
meaningfully affect heat transfer, the transparent route is:

1. solve \(\rho c_p \dot T-\nabla\cdot(k\nabla T)=Q\);
2. retain the temperature history;
3. evaluate
   \(\epsilon_{\mathrm{th}}=\alpha(T-T_{\mathrm{ref}})I\);
4. solve mechanical equilibrium with the named equivalent operator
   \(F_{\mathrm{thermal}}\).

AgentFEM stores \(E,\nu,\rho,\alpha,k,c_p,T_{\mathrm{ref}}\) in one
thermoelastic property object. Constants may be replaced by inspectable
temperature tables. Tabulated conductivity or specific heat automatically
selects a nonlinear implicit-Euler residual using the conservative enthalpy
difference \([h(T_{n+1})-h(T_n)]/\Delta t\), with \(dh/dT=\rho c_p(T)\).
The thermal step can capture its
accepted fields in a `FieldHistory`; the receiving creep step samples that
history at its own physical increment endpoints. Interpolation, range policy,
units, time coordinates, content hash, source Study, source procedure, and
accepted-time transfer role remain part of the transfer contract. Saved nodal
histories are keyed by physical DOF coordinates and can
be restored across changed MPI partitions and rank counts. Plane strain keeps
the constrained out-of-plane thermal strain; plane stress uses the reduced
constitutive relation.

This follows the same modeling distinction documented by Abaqus: fully
coupled analysis is needed when stress and temperature mutually influence
each other; sequential analysis is appropriate when the coupling is
effectively one way. The current route does not claim monolithic coupling.

## The first global creep state consumer

### Hyperbolic-sine branch

`constitutive.isotropic_sinh(young=..., poisson=..., density=...,
coefficient=..., stress_scale=..., exponent=...)` now uses the same
`studies.creep_solid()` and `model.step(...)` route. Its equivalent rate is
`coefficient * sinh(q / stress_scale)**exponent / reference_time`.
The bounded global integrator requires `exponent >= 1`. A safeguarded local
solve and log-domain rate evaluation handle large *trial* stresses without
clipping the physical rate. The consistent tangent differentiates the full
backward-Euler update, holding the previously accepted state fixed.

This branch is **experimental global**, not a newly validated engineering law.
`tests/test_sinh_creep.py` checks scalar/batch histories, fixed-old-state tangent,
the zero-rate limit, local relaxation convergence to an independent analytical
solution, serial 3D held stress, and checkpoint continuation. Existing power-law
checks remain separate. Distributed and independent structural validation have
not been added for Sinh. Shared `elastic=...` thermal properties are accepted,
but the Sinh rate parameters are temperature independent; no Arrhenius factor
is inferred. Coupled plasticity, damage and contact are not enabled by this
addition. The historical quadrature storage schema name is retained for archive
compatibility; full-Step identity also binds the complete material law.

### Explicit creep/plasticity composition

`constitutive.CreepPlasticityMaterial(creep=..., plastic=...)` combines an
implicit creep material with `J2LinearIsotropicHardening` or
`ChabocheCombinedHardening`. Register this one material and retain the ordinary
`studies.creep_solid()` → `model.step(duration=..., temperature=...)` workflow.
The two mechanisms share one physical increment and one accepted state; this
is not two sequential Steps and does not reset plastic history during a hold.

The first admitted scope is experimental serial homogeneous 3D small strain
with constant shared elastic moduli. Thermal expansion and the Arrhenius
power-law rate use the existing temperature-history input. Temperature-dependent
plastic parameters, static recovery, regional composition, axisymmetry, MPI,
finite strain and contact are not admitted. The local quadrature loop currently
prioritizes correctness; no throughput improvement is claimed.

CE/CEEQ, PE/PEEQ and every backstress branch remain distinct restart fields.
The maximum inelastic increment limit checks both creep and plastic increments.
Energy histories distinguish plastic work, hardening storage, reference-yield/
dynamic-recovery dissipation and the backward-Euler contribution. The mechanical
energy residual still excludes imposed thermal energy transfer; it is not a
general first-law certificate. V1 and V2 checkpoint formats pass serial joint
continuation, but V2 file layout alone does not qualify MPI execution.

`examples/thermal_cyclic_creep` is an ordinary project with synthetic parameters.
Independent uniaxial equations, nonproportional material paths and tangent
differences form the current evidence. Independent multiaxial structural
validation and broader time/mesh studies remain promotion gates.

### Power-law and temperature dependence

The local `ArrheniusPowerLawCreep` relation is normalized at a declared
reference temperature:

\[
A(T)=A_{\mathrm{ref}}
\exp\left[-\frac{Q}{R}
\left(\frac{1}{T}-\frac{1}{T_{\mathrm{ref}}}\right)\right].
\]

The implementation uses the equivalent exponent form in code; all
temperatures are absolute. The global three-dimensional and axisymmetric Mises
power-law route now accepts either an isothermal material or the same
thermoelastic property asset used by heat transfer. A scalar, finite-element
field, or accepted `FieldHistory` supplies \(T\) at every creep quadrature
point. The local update consumes the normalized Arrhenius rate, \(E(T)\),
\(\nu(T)\), and the endpoint secant thermal strain
\(\alpha(T)[T-T_{\mathrm{ref}}]I\) through one consistent material contract.
It provides:

- committed/trial `CE` and `CEEQ` at every quadrature point;
- a safeguarded backward-Euler local scalar solve over \(\Delta t\);
- an analytical algorithmic consistent tangent consumed by global Newton;
- atomic commit after acceptance and rollback after Newton, local, or
  maximum-creep-increment failure;
- complete regional material assignment, fixed or automatic physical-time
  increments, natural-load work, prescribed-motion work, elastic energy,
  creep dissipation, total internal energy, a mechanical energy residual, and
  portable full-Step checkpoint/restart;
- MPI-safe regional constitutive dispatch and a portable quadrature-state
  archive keyed by original physical cell, quadrature point, rule, material
  contract, and mesh fingerprint;
- integration-point temperature consumption, `TEMP` output, increment-wise
  temperature evidence, and temperature identity in checkpoints;
- scalar, finite-element, or physical-time `FieldHistory` temperature input;
  attempted increments move the live temperature to their endpoint, while
  rollback and restart restore it to the accepted physical time;
- a three-dimensional homogeneous stress-relaxation Golden contract.

The mechanical residual compares external mechanical work with stored elastic
energy plus creep dissipation. Under a changing prescribed temperature it also
contains the unreported thermal/material energy transfer, so it is evidence for
auditing the mechanical ledger rather than a full thermo-mechanical
conservation error.

`assessments.sequential_energy_ledger(thermal_result, mechanical_result,
field_history=temperature_history)` records this boundary explicitly. It keeps
the thermal heat-balance residual and the receiving mechanical work/energy
residual in separate channels. It deliberately does not add them or claim a
monolithic thermo-mechanical conservation equation.

This is implemented by reusing `QuadratureTransaction`; creep does not own a
second private state store. Checkpoint recovery reconstructs accepted stress
from displacement and committed `CE` without advancing a fictitious time
increment. The route follows the same consistent-linearization principle that
underpins robust Newton convergence in inelastic finite elements.

Arrhenius power-law creep is therefore a global consumer, while Sinh and
Kachanov--Rabotnov remain material-point capabilities. The accepted transient
temperature history is now an explicit transfer object; it does not silently
promote damage, mesh regularization, or structural rupture prediction.

## Engineering creep--fatigue assessment

`agentfem.assessments` is a postprocessing layer, not a new constitutive law.
It combines three independently reviewable assets:

1. source-identified dwell blocks evaluated by the time-fraction rule
   \(D_c=\sum_i n_i t_i/t_{r,i}\);
2. an existing `FatigueAssessment` produced from a declared stress history,
   cycle-counting policy, mean-stress policy and S--N curve;
3. an explicit interaction diagram whose points and source are supplied by
   the project.

AgentFEM includes only the transparent reference \(D_c+D_f=1\). It does not
embed ASME, R5, company, or material-specific allowable curves. Licensed and
current normative data therefore stays at the reviewed project boundary,
while the software provides one stable decision margin and structured result
record.

For a result-driven assessment, `DwellInterval` declares each physical hold
and `creep_fatigue_from_result(...)` extracts governing stress and temperature
from named scalar histories. The supplied rupture-time callable and its source
remain project assets. Out-of-range dwells, missing histories, non-scalar
histories, and nonphysical rupture times fail explicitly.

For a resolved relaxation history, use `creep_damage_from_result` instead of
reducing an entire hold to one governing value. This integrates the **damage
rate** `1 / rupture_time(stress, temperature)` on the union of the stress and
temperature sample times, with clipped dwell endpoints and trapezoidal
quadrature. Both inputs must declare physical time and the same time unit;
stress units must match explicitly and temperature must be in kelvin.

```python
from agentfem import assessments

# result contains source-identified scalar stress (MPa) and temperature (K)
# histories on physical time in seconds. The project supplies a reviewed
# rupture_time(sigma_MPa, temperature_K) callable returning seconds and
# rejecting states outside its own calibration domain.
creep = assessments.creep_damage_from_result(
    result,
    stress_history="axial_stress",
    temperature_history="temperature",
    dwells=[assessments.DwellInterval(1.0, 2.0, label="tensile_hold")],
    rupture_time=rupture_time,
    rupture_source="project material relation, revision and calibration domain",
    stress_unit="MPa", time_unit="s", stress_measure="signed_axial",
    maximum_source_interval=0.05,
)
# Combine explicitly with a separately assessed stress- or strain-life history.
combined = assessments.creep_fatigue(fatigue=fatigue, creep=creep,
                                    interaction=reviewed_interaction)
combined.attach(result)
```

The result retains each segment's stress, temperature, rates and damage,
plus source IDs and original sampling gaps. No implicit absolute stress,
unit conversion, temperature conversion or history extrapolation is applied.
Positive infinite rupture time means explicitly zero damage rate. Missing
coverage, overlapping/out-of-order dwells and invalid callback output fail
without attaching a partial assessment. `repetitions` assumes a repeated
response; it does not simulate cyclic evolution. The maximum source interval
is checked on **each original history**, so a dense temperature grid cannot
hide an under-resolved stress history. It is a sampling guard, not an error
certificate: refine the histories to establish quadrature accuracy.

Stress and temperature histories must describe the **same material location**
and loading path. Pairing separate spatial maxima can combine values that
never occurred together; that is an explicitly chosen envelope assessment,
not a local material-life prediction. This scalar interface does not infer
point identity from history names or certify such an envelope as conservative.

Nonisothermal integration is mathematically supported when the supplied
relation covers every evaluated state, but this does not validate a material's
nonisothermal rupture or creep-fatigue behavior. The existing block interface
and stress-life convenience entrypoint remain unchanged.

## Current boundary and next gates

Implemented now:

- explicit central difference and implicit Newmark/generalized-\(\alpha\);
- implicit-Euler heat transfer as a model-owned Step;
- conservative state-dependent conductivity/capacity using the same Step,
  Result, progress, heat-ledger, rollback, checkpoint, and MPI contracts;
- accepted-time field capture and temperature-history transfer to creep;
- coordinate-keyed field-history persistence verified in both two-rank-to-one
  and one-rank-to-two-rank directions;
- bounded temperature-property tables for sequential thermoelastic stiffness
  and thermal expansion;
- sequential isotropic thermal expansion as a visible vector operator;
- normalized Arrhenius temperature dependence at material-point and global
  integration-point level;
- global 3D and axisymmetric J2 quadrature state with regional materials, analytical tangent,
  and portable full-Step restart;
- cumulative J2 restart history, analytical uniaxial Golden verification,
  quadrature S/PE/PEEQ/MISES and nodal RF result fields, cyclic amplitude, physical-increment cutback,
  prescribed work, and internal-energy decomposition;
- exact material-point Kachanov-Rabotnov creep-damage updates, Sinh creep,
  and modified-theta curve projection;
- a sequential hot-wall FEM-to-creep assessment example with explicit
  calibration and maturity boundaries.
- global 3D and axisymmetric power-law creep with backward Euler, analytical tangent, shared
  transaction, automatic cutback, CE/CEEQ/S/MISES/RF/TEMP, prescribed
  Arrhenius temperature fields, dissipation history, regional materials,
  portable full-Step restart, MPI-portable quadrature state, and a
  relaxation Golden contract;
- a 3D component contract in which accepted transient heat states drive the
  global Arrhenius creep step on the same physical clock and through the same
  temperature-dependent thermoelastic material asset;
- a source-preserving engineering creep--fatigue assessment contract with
  named-history dwell extraction, explicit interaction data, and no hidden
  design-code constants;
- an experimental three-dimensional Chaboche combined-hardening route with
  multiple Armstrong--Frederick backstresses, global cyclic equilibrium,
  quadrature rollback and restart, and standard `ALPHA` output;
- the published NAFEMS R0027 Test 7 thick-cylinder stress oracle and a native
  Q2 axisymmetric elastic-preload-to-creep route. One-, two-, and four-cell
  radial meridians reduce the maximum radial/hoop/axial stress error from
  `1.40%` to `0.203%` to `0.0268%`. On the four-cell mesh the individual
  errors are `0.0141%`, `0.0145%`, and `0.0268%`, measured directly from
  integration-point stress with the full-revolution volume weight. The
  `0.5%` automated acceptance criterion is AgentFEM's verification contract,
  not an official NAFEMS tolerance. The public Abaqus reproduction reports
  differences from the reference solution rather than a pass/fail threshold:
  its largest tabulated radial-stress difference is `1.84%` for CAX8R
  (`1.85%` for CCL24R), and its largest hoop-stress difference is `1.00%`.
  The former 3D quarter-sector formulation remains available as an explicit
  comparison route.

For creep, nonlinear equilibrium convergence and time-integration accuracy
are separate decisions. `creep_strain_error_tolerance` limits

\[
\max_q\left|\dot{\bar\varepsilon}^{cr}_{q,n+1}
-\dot{\bar\varepsilon}^{cr}_{q,n}\right|\Delta t,
\]

at owned integration points. An otherwise converged increment that exceeds
the tolerance is rolled back and cut back atomically. This follows the public
NAFEMS/Abaqus `CETOL` meaning; `maximum_inelastic_increment` remains the
separate bound on the magnitude of the accepted CEEQ increment.

The J2 and creep Steps compile their live displacement-to-quadrature strain
expression once. Newton iterations, line-search trials, rollback recovery and
result reconstruction reevaluate its coefficients without repeatedly asking
FFCx to rediscover the same compiled expression. This is an implementation
detail below the public workflow, but it is important for long adaptive paths.

Homogeneous isothermal creep regions also use a vectorized quadrature update.
This is an execution optimization, not a second constitutive model: the batch
follows the same safeguarded backward-Euler equation and returns the same
algorithmic tangent as the scalar material-point update. Temperature-dependent
and multi-material regions retain pointwise material dispatch so scientific
semantics take precedence over batching.

Next gates:

1. improve the global Newton/increment controller toward the public deck's
   40-increment execution efficiency without weakening
   the scientific gates, and add intermediate-time observables when a
   transient rather than terminal benchmark contract is available;
2. exercise tabulated heat properties and accepted history transfer on an
   external 3D power-component benchmark with time-step convergence;
3. finish field/energy/checkpoint products on top of the common complete
   execution-event trace, accepted-increment histories, status files, and
   result manifest across heat, implicit dynamics, and explicit dynamics;
4. exercise the portable full-Step archive on larger partition changes and
   scheduled HPC restart campaigns;
5. add multi-element and external power-component benchmarks, then promote
   the experimental global Newton MPI path;
6. introduce K-R/Liu--Murakami damage only with near-failure time control and
   a declared mesh-regularization policy;
7. add a distributed archive backend only when field histories outgrow the
   current compact root-gathered format;
8. monolithic coupling only after a real case demonstrates two-way feedback.

References:

- [Abaqus fully coupled thermal-stress analysis](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEANLRefMap/simaanl-c-couptempdisp.htm)
- [Abaqus heat-transfer procedures](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEANLRefMap/simaanl-c-heatproc.htm)
- [Abaqus CREEP user subroutine](https://docs.software.vt.edu/abaqusv2024/English/SIMACAESUBRefMap/simasub-c-creep.htm)
- [Chung and Hulbert generalized-\(\alpha\) paper](https://deepblue.lib.umich.edu/bitstream/handle/2027.42/50422/1640100803_ftp.pdf?isAllowed=y&sequence=1)
- [Simo and Taylor, consistent tangent operators](https://escholarship.org/uc/item/9cp19009)
- [Duxbury, Crook, and Lyons, consistent plasticity/creep integration](https://doi.org/10.1002/nme.1620370803)
- [NAFEMS R0027 Test 7 as reproduced in the Abaqus verification guide](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-nafemscreep.htm)
